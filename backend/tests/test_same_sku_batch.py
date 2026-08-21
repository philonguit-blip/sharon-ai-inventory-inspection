from __future__ import annotations

from pathlib import Path
from typing import Any

from app.routes import local_jobs


def _result(image_name: str, code: str, count: int) -> dict[str, Any]:
    return {
        "image_name": image_name,
        "total_detections": count,
        "avg_confidence": 0.91,
        "inference_ms": 10.0,
        "decision": {
            "decision": "DIRECT",
            "product_code": code,
            "product_name": code,
            "display_name": code,
            "count": count,
            "purity": 1.0,
            "avg_confidence": 0.91,
            "requires_confirmation": True,
        },
    }


def test_aggregate_batch_adds_counts_for_the_same_direct_sku() -> None:
    decision = local_jobs._aggregate_batch_decision(
        [_result("tray-1.jpg", "SKU-1", 4), _result("tray-2.jpg", "SKU-1", 6)]
    )

    assert decision["decision"] == "DIRECT"
    assert decision["product_code"] == "SKU-1"
    assert decision["count"] == 10
    assert [item["count"] for item in decision["per_image"]] == [4, 6]


def test_aggregate_batch_routes_class_disagreement_to_manual_review() -> None:
    decision = local_jobs._aggregate_batch_decision(
        [_result("tray-1.jpg", "SKU-1", 4), _result("tray-2.jpg", "SKU-2", 6)]
    )

    assert decision["decision"] == "REVIEW"
    assert decision["requires_confirmation"] is True
    assert decision["requires_user_selection"] is True
    assert decision["count"] == 10
    assert {item["product_code"] for item in decision["candidates"]} == {
        "SKU-1",
        "SKU-2",
    }
    assert "không tự chọn" in decision["message"]


def test_aggregate_batch_keeps_common_family_members() -> None:
    members = [
        {"product_code": "SKU-A", "product_name": "A"},
        {"product_code": "SKU-B", "product_name": "B"},
    ]
    results = []
    for index, count in enumerate((3, 5), start=1):
        result = _result(f"tray-{index}.jpg", "unused", count)
        result["decision"] = {
            "decision": "FAMILY",
            "dominant_class": "FAMILY-1",
            "display_name": "Family 1",
            "members": members,
            "count": count,
            "purity": 1.0,
            "avg_confidence": 0.88,
        }
        results.append(result)

    decision = local_jobs._aggregate_batch_decision(results)

    assert decision["decision"] == "FAMILY"
    assert decision["count"] == 8
    assert {item["product_code"] for item in decision["members"]} == {
        "SKU-A",
        "SKU-B",
    }


def test_aggregate_batch_uses_foundation_review_count_and_excludes_unsafe_count() -> None:
    unsafe = {
        "image_name": "unsafe.jpg",
        "total_detections": 1,
        "decision": {
            "decision": "AMBIGUOUS",
            "dominant_class": "CLASS-1",
            "display_name": "SKU-1",
            "count": 1,
            "avg_confidence": 0.85,
            "purity": 1.0,
        },
    }
    direct = _result("yolo.jpg", "SKU-1", 4)
    review = {
        "image_name": "foundation-review.jpg",
        "total_detections": 6,
        "decision": {
            "decision": "REVIEW",
            "dominant_class": "CLASS-1",
            "display_name": "Hybrid disagreement",
            "count": 6,
            "avg_confidence": 0.86,
            "purity": 0.0,
            "preferred_source": "FOUNDATION",
            "candidates": [
                {"product_code": "SKU-1", "product_name": "SKU-1", "count": 6}
            ],
        },
    }

    decision = local_jobs._aggregate_batch_decision([unsafe, direct, review])

    assert decision["decision"] == "REVIEW"
    assert decision["count"] == 10
    assert decision["unconfirmed_count_excluded"] == 1
    assert decision["valid_image_count"] == 2
    assert decision["invalid_images"] == ["unsafe.jpg"]
    assert [item["count"] for item in decision["per_image"]] == [1, 4, 6]


def test_batch_context_recovers_validated_foundation_object_count() -> None:
    raw_class = "CLASS-1"
    foundation = {
        "image_name": "foundation-margin-review.jpg",
        "engine": "FOUNDATION",
        "total_detections": 1,
        "decision": {
            "decision": "AMBIGUOUS",
            "dominant_class": raw_class,
            "display_name": "SKU-1",
            "count": 1,
            "avg_confidence": 0.85,
            "purity": 1.0,
        },
        "objects": [
            {
                "raw_class": raw_class,
                "product_code": "SKU-1",
                "product_name": "SKU-1",
                "confidence": 0.867,
                "confidence_threshold": 0.6,
                "foundation_instance_margin": 0.079,
            }
        ],
        "foundation_filtering": {"instance_similarity_margin": 0.02},
        "hybrid": {"selected_engine": "FOUNDATION"},
    }
    direct = _result("yolo.jpg", "SKU-1", 4)
    direct["decision"]["dominant_class"] = raw_class
    review = {
        "image_name": "foundation-review.jpg",
        "total_detections": 6,
        "decision": {
            "decision": "REVIEW",
            "dominant_class": raw_class,
            "display_name": "Hybrid disagreement",
            "count": 6,
            "avg_confidence": 0.86,
            "purity": 0.0,
            "preferred_source": "FOUNDATION",
            "candidates": [
                {"product_code": "SKU-1", "product_name": "SKU-1", "count": 6}
            ],
        },
    }
    results = [foundation, direct, review]
    pending = [
        {"source_path": Path("unused.jpg"), "image_name": item["image_name"]}
        for item in results
    ]

    class NoYoloRescue:
        def rescue_result_for_class(self, *args: Any, **kwargs: Any) -> None:
            return None

    rescue = local_jobs._attempt_batch_rescue(
        results, pending, NoYoloRescue()
    )
    decision = local_jobs._aggregate_batch_decision(results)

    assert rescue["recovered"] == 1
    assert rescue["foundation_context_recovered"] == 1
    assert decision["decision"] == "REVIEW"
    assert decision["count"] == 11
    assert decision["unconfirmed_count_excluded"] == 0
    assert decision["valid_image_count"] == 3
    assert decision["per_image"][0]["resolved"] is True
    assert decision["per_image"][0]["recovered_by_batch_context"] is True


def test_process_job_persists_aggregate_and_all_image_summaries(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(local_jobs, "JOBS_ROOT", tmp_path.resolve())
    job_id = "a" * 32
    job_directory = local_jobs._job_directory(job_id)
    original = job_directory / "original"
    original.mkdir(parents=True)
    paths = [original / "001_tray-1.jpg", original / "002_tray-2.jpg"]
    for path in paths:
        path.write_bytes(b"image")
    local_jobs._write_json_atomic(
        local_jobs._job_state_path(job_id),
        local_jobs._initial_job_state(job_id, 2),
    )

    class FakeInference:
        def infer_bytes(self, raw_bytes: bytes, **kwargs: Any) -> dict[str, Any]:
            annotated_path = Path(kwargs["annotated_path"])
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_path.write_bytes(b"annotated")
            count = 4 if kwargs["image_name"] == "tray-1.jpg" else 6
            return _result(kwargs["image_name"], "SKU-1", count)

    saved = [
        {"display_name": f"tray-{index}.jpg", "path": str(path)}
        for index, path in enumerate(paths, start=1)
    ]
    local_jobs._process_job(job_id, saved, FakeInference(), None, "AUTO")
    state = local_jobs._read_job(job_id)

    assert state["status"] == "AWAITING_CONFIRMATION"
    assert state["processed_images"] == 2
    assert state["total_quantity"] == 10
    assert state["decision"]["count"] == 10
    assert len(state["images"]) == 2
