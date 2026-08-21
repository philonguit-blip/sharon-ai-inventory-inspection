from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import torch

from app.services.foundation_inference_service import FoundationInferenceService


CURRENT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "hybrid_reference_registry.json"
)


def test_common_cookies_reference_is_a_two_member_family():
    registry = json.loads(CURRENT_REGISTRY_PATH.read_text(encoding="utf-8"))
    cookies = registry["products"]["CA-COO-0000040_112"]

    assert cookies["type"] == "family"
    assert {member["product_code"] for member in cookies["members"]} == {
        "CA-COO-0000040",
        "CA-COO-0000112",
    }


def _service(tmp_path, **overrides) -> FoundationInferenceService:
    defaults = {
        "sam_model_path": tmp_path / "sam.pt",
        "reference_path": tmp_path / "references.npz",
        "registry_path": tmp_path / "registry.json",
        "device": "cpu",
    }
    defaults.update(overrides)
    return FoundationInferenceService(**defaults)


def test_sam_resize_preserves_aspect_ratio(tmp_path):
    service = _service(tmp_path, sam_max_side=1280)
    image = np.zeros((3060, 4080, 3), dtype=np.uint8)

    resized, scale = service._resize_for_sam(image)

    assert resized.shape[:2] == (960, 1280)
    assert scale == 1280 / 4080


def test_segment_scales_boxes_back_to_original_image(tmp_path):
    service = _service(
        tmp_path,
        sam_max_side=200,
        min_area_ratio=0.001,
        max_area_ratio=0.5,
        max_box_area_ratio=0.5,
        edge_margin_ratio=0.0,
    )
    mask = np.zeros((150, 200), dtype=np.float32)
    mask[30:50, 50:80] = 1

    class FakeSam:
        source_shape = None

        def predict(self, **kwargs):
            self.source_shape = kwargs["source"].shape
            class FakeBoxes:
                conf = torch.tensor([0.9])

                def __len__(self):
                    return 1

            result = SimpleNamespace(
                masks=SimpleNamespace(data=torch.from_numpy(mask[None, ...])),
                boxes=FakeBoxes(),
            )
            return [result]

    service.sam = FakeSam()
    segments = service._segment(np.zeros((300, 400, 3), dtype=np.uint8))

    assert service.sam.source_shape[:2] == (150, 200)
    assert len(segments) == 1
    assert segments[0]["box_xyxy"] == [100.0, 60.0, 160.0, 100.0]
    assert segments[0]["area"] == 2400.0


def test_prompted_sam_refines_all_candidate_boxes_in_one_call(tmp_path):
    service = _service(tmp_path, sam_max_side=200)

    class FakeSam:
        calls = 0
        source_shape = None
        prompted_boxes = None

        def predict(self, **kwargs):
            self.calls += 1
            self.source_shape = kwargs["source"].shape
            self.prompted_boxes = kwargs["bboxes"]
            boxes = SimpleNamespace(
                xyxy=torch.tensor([[25.0, 20.0, 75.0, 50.0]]),
                conf=torch.tensor([0.93]),
                cls=torch.tensor([0.0]),
            )
            return [SimpleNamespace(boxes=boxes)]

    sam = FakeSam()
    service.sam = sam
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    candidates = [
        {"raw_class": "A", "box_xyxy": [40.0, 30.0, 160.0, 110.0]}
    ]

    segments = service._prompt_candidate_segments(image, candidates, 0.0)

    assert sam.calls == 1
    assert sam.source_shape[:2] == (150, 200)
    assert sam.prompted_boxes == [[20.0, 15.0, 80.0, 55.0]]
    assert segments[0]["box_xyxy"] == [50.0, 40.0, 150.0, 100.0]
    assert segments[0]["mask_refined"] is True


def test_instance_semantic_gate_uses_target_score_and_competitor_margin(tmp_path):
    service = _service(
        tmp_path,
        instance_similarity_threshold=0.60,
        instance_similarity_margin=0.02,
    )
    service.reference_keys = np.asarray(["A", "A", "B"])
    similarities = np.asarray(
        [
            [0.88, 0.84, 0.40],
            [0.64, 0.62, 0.63],
            [0.58, 0.55, 0.20],
        ],
        dtype=np.float32,
    )

    scores, margins, keep = service._instance_semantic_scores(similarities, "A")

    np.testing.assert_allclose(scores, [0.88, 0.64, 0.58])
    np.testing.assert_allclose(margins, [0.48, 0.01, 0.38], atol=1e-6)
    assert keep.tolist() == [True, False, False]


def test_candidate_verification_batches_crops_and_never_substitutes_class(tmp_path):
    service = _service(
        tmp_path,
        instance_similarity_threshold=0.60,
        instance_similarity_margin=0.02,
    )
    service._loaded = True
    service.reference_embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    service.reference_keys = np.asarray(["A", "B"])
    service.registry = {
        "A": {"type": "direct", "product_code": "A", "product_name": "A"},
        "B": {"type": "direct", "product_code": "B", "product_name": "B"},
    }
    service._prompt_candidate_segments = lambda _image, candidates, _ratio: [
        {
            "box_xyxy": list(candidate["box_xyxy"]),
            "area": 400.0,
            "score": 0.9,
            "prompt_index": index,
            "mask_refined": True,
        }
        for index, candidate in enumerate(candidates)
    ]

    class FakeEncoder:
        calls = 0
        batch_sizes = []

        def encode_rgb(self, images, batch_size=16):
            self.calls += 1
            self.batch_sizes.append((len(images), batch_size))
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    encoder = FakeEncoder()
    service.encoder = encoder
    ok, encoded = cv2.imencode(".jpg", np.zeros((80, 100, 3), dtype=np.uint8))
    assert ok
    candidates = [
        {"raw_class": "A", "confidence": 0.4, "box_xyxy": [5, 5, 25, 25]},
        {"raw_class": "A", "confidence": 0.4, "box_xyxy": [40, 5, 60, 25]},
    ]

    verification = service.verify_yolo_candidates(
        encoded.tobytes(), candidates, batch_size=8
    )

    assert encoder.calls == 1
    assert encoder.batch_sizes == [(2, 8)]
    assert verification["accepted"] == 1
    assert verification["disagreements"] == 1
    assert verification["verdicts"][0]["status"] == "ACCEPTED"
    assert verification["verdicts"][1]["status"] == "CLASS_DISAGREEMENT"
    assert verification["verdicts"][1]["accepted"] is False


def test_area_filter_removes_tiny_and_huge_semantic_outliers(tmp_path):
    service = _service(
        tmp_path,
        instance_min_area_factor=0.35,
        instance_max_area_factor=2.5,
    )
    segments = [{"area": area} for area in (10, 100, 110, 900)]

    keep, median = service._robust_area_keep(
        segments, np.asarray([True, True, True, True])
    )

    assert median == 105.0
    assert keep.tolist() == [False, True, True, False]


def test_semantic_duplicate_filter_keeps_stronger_overlapping_candidate(tmp_path):
    service = _service(tmp_path, instance_box_coverage_nms=0.45)
    segments = [
        {"box_xyxy": [0, 0, 100, 20]},
        {"box_xyxy": [30, 5, 70, 25]},
        {"box_xyxy": [120, 0, 160, 20]},
    ]

    keep, removed = service._semantic_duplicate_keep(
        segments,
        np.asarray([0.91, 0.99, 0.88], dtype=np.float32),
        np.asarray([True, True, True]),
    )

    assert removed == 1
    assert keep.tolist() == [False, True, True]


def test_infer_counts_only_per_instance_valid_masks(tmp_path):
    service = _service(
        tmp_path,
        similarity_threshold=0.72,
        similarity_margin=0.04,
        instance_similarity_threshold=0.60,
        instance_similarity_margin=0.02,
    )
    service._loaded = True
    service.reference_embeddings = np.asarray(
        [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]], dtype=np.float32
    )
    service.reference_keys = np.asarray(["A", "A", "B"])
    service.registry = {
        "A": {
            "type": "direct",
            "display_name": "Product A",
            "product_code": "A",
            "product_name": "Product A",
        }
    }
    service._segment = lambda _image: [
        {"area": 100.0, "score": 0.9, "box_xyxy": [10, 10, 30, 30]},
        {"area": 110.0, "score": 0.9, "box_xyxy": [40, 10, 60, 32]},
        {"area": 105.0, "score": 0.9, "box_xyxy": [70, 10, 90, 31]},
    ]
    service._last_segmentation_stats = {"duplicates_removed": 2}

    class FakeEncoder:
        def encode_rgb(self, _images):
            return np.asarray([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float32)

    service.encoder = FakeEncoder()
    ok, encoded = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    assert ok

    result = service.infer_bytes(encoded.tobytes(), image_name="tray.jpg")

    assert result["decision"]["decision"] == "DIRECT"
    assert result["decision"]["count"] == 2
    assert result["total_detections"] == 2
    assert result["detections_removed_by_class_filter"] == 1
    assert result["detections_removed_as_duplicates"] == 2
    assert result["foundation_filtering"]["semantic_kept"] == 2
    assert all("foundation_instance_margin" in item for item in result["objects"])
