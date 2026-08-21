from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from app.services.bakery_inference_service import (
    BakeryInferenceService,
    _foundation_candidate_floor,
)
from app.services.hybrid_errors import FoundationNotReadyError
from app.services.hybrid_inference_service import HybridInferenceService
from app.services.product_mapping_service import ProductMappingService


MAPPING_PATH = Path(__file__).resolve().parents[1] / "config" / "product_mapping.json"


def result(engine: str, decision: str, code: str = "BR-1", count: int = 4, confidence: float = 0.9):
    return {
        "image_name": "tray.jpg",
        "total_detections": count,
        "avg_confidence": confidence,
        "decision": {
            "decision": decision,
            "product_code": code if decision == "DIRECT" else None,
            "product_name": code,
            "dominant_class": code,
            "display_name": code,
            "count": count,
            "purity": 1.0,
            "avg_confidence": confidence,
            "confidence_threshold": 0.25,
        },
        "objects": [],
        "engine": engine,
    }


class FakeEngine:
    def __init__(self, payload, ready=True):
        self.payload = payload
        self.ready = ready
        self.calls = 0
        self.batch_calls = 0

    def infer_bytes(self, *args, **kwargs):
        self.calls += 1
        return deepcopy(self.payload)

    def infer_batch(self, items):
        self.batch_calls += 1
        outputs = []
        for item in items:
            payload = deepcopy(self.payload)
            payload["image_name"] = item["image_name"]
            outputs.append(payload)
        return outputs

    def health(self):
        return {"ready": self.ready}

    def is_ready(self):
        return self.ready


class BoxAwareYolo(FakeEngine):
    def __init__(self, payload):
        super().__init__(payload)
        self.apply_calls = 0

    def apply_foundation_box_rescue(self, payload, verification, **_kwargs):
        self.apply_calls += 1
        updated = deepcopy(payload)
        accepted = int(verification.get("accepted") or 0)
        if accepted:
            updated["decision"]["count"] += accepted
            updated["total_detections"] += accepted
        updated["foundation_box_rescue"] = {
            "attempted": verification.get("attempted", 0),
            "accepted": accepted,
            "kept_after_deduplication": accepted,
            "class_disagreements": verification.get("disagreements", 0),
        }
        return updated


class BoxAwareFoundation(FakeEngine):
    def __init__(self, payload, verification):
        super().__init__(payload)
        self.verification = verification
        self.verify_calls = 0

    def verify_yolo_candidates(self, _raw_bytes, candidates, **_kwargs):
        self.verify_calls += 1
        output = deepcopy(self.verification)
        output["attempted"] = len(candidates)
        return output


def low_candidate(raw_class="BR-1", confidence=0.20, box=None):
    return {
        "raw_class": raw_class,
        "confidence": confidence,
        "confidence_threshold": 0.25,
        "box_xyxy": box or [100, 100, 150, 150],
    }


def test_candidate_floor_exposes_subthreshold_band_without_raising_low_classes():
    assert _foundation_candidate_floor(0.50) == pytest.approx(0.25)
    assert _foundation_candidate_floor(0.15) == pytest.approx(0.075)
    assert _foundation_candidate_floor(0.02) == pytest.approx(0.02)


def test_auto_keeps_confident_yolo_without_loading_foundation():
    yolo = FakeEngine(result("YOLO", "DIRECT"))
    foundation = FakeEngine(result("FOUNDATION", "DIRECT"))
    service = HybridInferenceService(yolo, foundation)

    output = service.infer_bytes(b"image", image_name="tray.jpg", mode="AUTO")

    assert output["hybrid"]["selected_engine"] == "YOLO"
    assert yolo.calls == 1
    assert foundation.calls == 0


def test_auto_uses_foundation_when_yolo_is_uncertain():
    yolo = FakeEngine(result("YOLO", "NO_DETECTION", count=0, confidence=0.0))
    foundation = FakeEngine(result("FOUNDATION", "DIRECT", code="BR-NEW"))
    service = HybridInferenceService(yolo, foundation)

    output = service.infer_bytes(b"image", image_name="tray.jpg", mode="AUTO")

    assert output["decision"]["product_code"] == "BR-NEW"
    assert output["hybrid"]["selected_engine"] == "FOUNDATION"


def test_compare_disagreement_requires_operator_review():
    yolo = FakeEngine(result("YOLO", "DIRECT", code="BR-1", count=4))
    foundation = FakeEngine(result("FOUNDATION", "DIRECT", code="BR-2", count=5))
    service = HybridInferenceService(yolo, foundation)

    output = service.infer_bytes(b"image", image_name="tray.jpg", mode="COMPARE")

    assert output["decision"]["decision"] == "REVIEW"
    assert {item["product_code"] for item in output["decision"]["candidates"]} == {"BR-1", "BR-2"}
    assert output["decision"]["count"] == 5
    assert output["decision"]["preferred_source"] == "FOUNDATION"


def test_review_prefers_safe_foundation_direct_over_yolo_family():
    yolo_payload = result("YOLO", "FAMILY", count=1, confidence=0.87)
    yolo_payload["decision"]["members"] = [
        {"product_code": "OTHER-1", "product_name": "Other"}
    ]
    foundation_payload = result(
        "FOUNDATION", "DIRECT", code="BR-SD-0000133", count=6, confidence=0.86
    )
    foundation_payload["objects"] = [{"foundation_mask_index": index} for index in range(6)]
    service = HybridInferenceService(
        FakeEngine(yolo_payload), FakeEngine(foundation_payload)
    )

    output = service.infer_bytes(b"image", image_name="tray.jpg", mode="COMPARE")

    assert output["decision"]["decision"] == "REVIEW"
    assert output["decision"]["count"] == 6
    assert output["decision"]["preferred_source"] == "FOUNDATION"
    assert output["total_detections"] == 6
    assert len(output["objects"]) == 6


def test_compare_fails_actionably_when_foundation_is_not_provisioned():
    yolo = FakeEngine(result("YOLO", "DIRECT"))
    foundation = FakeEngine(result("FOUNDATION", "DIRECT"), ready=False)
    service = HybridInferenceService(yolo, foundation)

    with pytest.raises(FoundationNotReadyError, match="COMPARE requires"):
        service.infer_bytes(b"image", image_name="tray.jpg", mode="COMPARE")


def test_two_unsafe_results_do_not_create_a_confirmable_review():
    yolo = FakeEngine(result("YOLO", "AMBIGUOUS", count=2, confidence=0.2))
    foundation = FakeEngine(result("FOUNDATION", "AMBIGUOUS", count=3, confidence=0.3))
    service = HybridInferenceService(yolo, foundation)

    output = service.infer_bytes(b"image", image_name="tray.jpg", mode="COMPARE")

    assert output["decision"]["decision"] == "AMBIGUOUS"
    assert output["hybrid"]["selected_engine"] == "FOUNDATION"


def test_auto_batches_confident_yolo_images_without_foundation_calls():
    yolo = FakeEngine(result("YOLO", "DIRECT"))
    foundation = FakeEngine(result("FOUNDATION", "DIRECT"))
    service = HybridInferenceService(yolo, foundation)
    items = [
        {"raw_bytes": b"one", "image_name": "one.jpg", "annotated_path": None},
        {"raw_bytes": b"two", "image_name": "two.jpg", "annotated_path": None},
    ]

    outputs = service.infer_batch(items, mode="AUTO")

    assert [item["image_name"] for item in outputs] == ["one.jpg", "two.jpg"]
    assert yolo.batch_calls == 1
    assert yolo.calls == 0
    assert foundation.calls == 0


def test_auto_recovers_unique_subthreshold_box_without_full_foundation_scan():
    yolo_payload = result("YOLO", "DIRECT", code="BR-1", count=4)
    yolo_payload["candidate_objects"] = [low_candidate()]
    yolo = BoxAwareYolo(yolo_payload)
    foundation = BoxAwareFoundation(
        result("FOUNDATION", "DIRECT", code="BR-1", count=5),
        {"accepted": 1, "disagreements": 0, "verdicts": []},
    )
    service = HybridInferenceService(yolo, foundation)

    output = service.infer_bytes(b"image", image_name="tray.jpg", mode="AUTO")

    assert output["decision"]["count"] == 5
    assert output["hybrid"]["selected_engine"] == "FOUNDATION_BOX_RESCUE"
    assert foundation.verify_calls == 1
    assert foundation.calls == 0
    assert yolo.apply_calls == 1


def test_box_class_disagreement_escalates_to_full_image_comparison():
    yolo_payload = result("YOLO", "DIRECT", code="BR-1", count=4)
    yolo_payload["candidate_objects"] = [low_candidate()]
    yolo = BoxAwareYolo(yolo_payload)
    foundation = BoxAwareFoundation(
        result("FOUNDATION", "DIRECT", code="BR-2", count=5),
        {"accepted": 0, "disagreements": 1, "verdicts": []},
    )
    service = HybridInferenceService(yolo, foundation)

    output = service.infer_bytes(b"image", image_name="tray.jpg", mode="AUTO")

    assert foundation.verify_calls == 1
    assert foundation.calls == 1
    assert output["decision"]["decision"] == "REVIEW"
    assert output["hybrid"]["selected_engine"] == "REVIEW"


def test_box_rescue_skips_candidate_overlapping_an_accepted_detection():
    yolo_payload = result("YOLO", "DIRECT", code="BR-1", count=1)
    yolo_payload["objects"] = [
        {"raw_class": "BR-1", "box_xyxy": [0, 0, 100, 100], "confidence": 0.9}
    ]
    yolo_payload["candidate_objects"] = [
        low_candidate(box=[10, 10, 90, 90])
    ]
    foundation = BoxAwareFoundation(
        result("FOUNDATION", "DIRECT", code="BR-1", count=1),
        {"accepted": 1, "disagreements": 0, "verdicts": []},
    )
    service = HybridInferenceService(BoxAwareYolo(yolo_payload), foundation)

    output = service.infer_bytes(b"image", image_name="tray.jpg", mode="AUTO")

    assert output["hybrid"]["selected_engine"] == "YOLO"
    assert foundation.verify_calls == 0
    assert foundation.calls == 0


def test_box_rescue_limits_foundation_work_to_highest_confidence_candidates():
    yolo_payload = result("YOLO", "DIRECT", code="BR-1", count=1)
    yolo_payload["candidate_objects"] = [
        low_candidate(confidence=0.13, box=[0, 0, 10, 10]),
        low_candidate(confidence=0.22, box=[20, 0, 30, 10]),
        low_candidate(confidence=0.19, box=[40, 0, 50, 10]),
    ]
    service = HybridInferenceService(
        BoxAwareYolo(yolo_payload),
        BoxAwareFoundation(
            result("FOUNDATION", "DIRECT"),
            {"accepted": 0, "disagreements": 0, "verdicts": []},
        ),
        box_rescue_max_candidates=2,
    )

    candidates, stats = service._box_rescue_candidates(yolo_payload)

    assert [item["confidence"] for item in candidates] == [0.22, 0.19]
    assert stats["truncated"] == 1


def test_yolo_merger_rebuilds_count_from_only_accepted_foundation_boxes():
    service = object.__new__(BakeryInferenceService)
    service.mapping = ProductMappingService(MAPPING_PATH)
    service.duplicate_iou = 0.85
    service.cross_class_duplicate_coverage = 0.85
    service.min_purity = 0.90
    service.conflict_review_min_dominant_count = 2
    raw_class = service.mapping.class_names()[0]
    mapping = service.mapping.resolve(raw_class)
    candidate = {
        "class_id": 0,
        "raw_class": raw_class,
        "mapping_type": mapping["type"],
        "display_name": mapping["display_name"],
        "confidence_threshold": mapping["confidence_threshold"],
        "confidence": 0.20,
        "box_xyxy": [10.0, 10.0, 40.0, 40.0],
        "product_code": mapping.get("product_code"),
        "product_name": mapping.get("product_name"),
        "purchase_price": 0.0,
    }
    payload = {
        "objects": [],
        "candidate_objects": [candidate],
        "detections_removed_by_class_filter": 1,
    }
    verification = {
        "attempted": 2,
        "accepted": 1,
        "disagreements": 1,
        "verdicts": [
            {
                "accepted": True,
                "candidate": candidate,
                "foundation_similarity": 0.88,
                "foundation_margin": 0.20,
                "refined_box_xyxy": [11.0, 11.0, 39.0, 39.0],
                "mask_refined": True,
            },
            {
                "accepted": False,
                "candidate": candidate,
                "foundation_similarity": 0.91,
                "foundation_margin": 0.30,
                "class_disagreement": True,
            },
        ],
    }

    merged = service.apply_foundation_box_rescue(payload, verification)

    assert merged["decision"]["count"] == 1
    assert merged["total_detections"] == 1
    assert merged["objects"][0]["confidence"] == pytest.approx(0.88)
    assert merged["objects"][0]["yolo_confidence"] == pytest.approx(0.20)
    assert merged["foundation_box_rescue"]["class_disagreements"] == 1


def test_compare_keeps_countable_foundation_review_over_larger_unsafe_yolo():
    yolo_payload = result("YOLO", "AMBIGUOUS", count=3, confidence=0.78)
    foundation_payload = result(
        "FOUNDATION", "REVIEW", count=2, confidence=0.91
    )
    foundation_payload["decision"]["candidates"] = [
        {"product_code": "BR-FOUNDATION", "product_name": "Foundation SKU"}
    ]
    service = HybridInferenceService(
        FakeEngine(yolo_payload), FakeEngine(foundation_payload)
    )

    output = service.infer_bytes(b"image", image_name="tray.jpg", mode="COMPARE")

    assert output["decision"]["decision"] == "REVIEW"
    assert output["decision"]["count"] == 2
    assert output["decision"]["candidates"][0]["product_code"] == "BR-FOUNDATION"
    assert output["hybrid"]["selected_engine"] == "FOUNDATION"


def test_compare_disagreement_preserves_all_family_members():
    yolo_payload = result("YOLO", "FAMILY", count=4, confidence=0.9)
    yolo_payload["decision"]["members"] = [
        {"product_code": "FAMILY-A", "product_name": "Family A"},
        {"product_code": "FAMILY-B", "product_name": "Family B"},
    ]
    foundation_payload = result(
        "FOUNDATION", "DIRECT", code="OTHER", count=4, confidence=0.9
    )
    service = HybridInferenceService(
        FakeEngine(yolo_payload), FakeEngine(foundation_payload)
    )

    output = service.infer_bytes(b"image", image_name="tray.jpg", mode="COMPARE")

    assert output["decision"]["decision"] == "REVIEW"
    assert {item["product_code"] for item in output["decision"]["candidates"]} == {
        "FAMILY-A",
        "FAMILY-B",
        "OTHER",
    }
