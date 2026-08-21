from copy import deepcopy

import pytest

from app.config import MAPPING_PATH
from app.services.bakery_inference_service import BakeryInferenceService
from app.services.product_mapping_service import ProductMappingService


WHOLEMEAL = "BR-SD-0000133_SourdoughWholemealBread_Pc"
CIABATTA = "BR-SD-0000123_SourdoughSharonCiabatta_Pc"


def _service() -> BakeryInferenceService:
    service = object.__new__(BakeryInferenceService)
    service.mapping = ProductMappingService(MAPPING_PATH)
    service.duplicate_iou = 0.85
    service.cross_class_duplicate_coverage = 0.85
    service.min_purity = 0.90
    service.conflict_review_min_dominant_count = 2
    service.edge_outlier_enabled = True
    service.edge_outlier_margin_ratio = 0.01
    service.edge_outlier_confidence_gap = 0.10
    service.edge_outlier_min_dominant_count = 2
    return service


def _object(raw_class: str, confidence: float, box: list[float]) -> dict:
    mapping = ProductMappingService(MAPPING_PATH).resolve(raw_class)
    return {
        "raw_class": raw_class,
        "display_name": mapping["display_name"],
        "mapping_type": mapping["type"],
        "confidence_threshold": mapping["confidence_threshold"],
        "confidence": confidence,
        "box_xyxy": box,
        "product_code": mapping.get("product_code"),
        "product_name": mapping.get("product_name"),
        "purchase_price": mapping.get("purchase_price", 0.0),
    }


@pytest.fixture
def reported_pdf_objects() -> list[dict]:
    return [
        _object(
            WHOLEMEAL,
            0.9487700462,
            [88.97, 605.14, 607.94, 1080.01],
        ),
        _object(
            WHOLEMEAL,
            0.9411641955,
            [417.34, 906.94, 920.68, 1443.53],
        ),
        _object(
            CIABATTA,
            0.7815313935,
            [521.57, 1759.34, 1080.0, 1918.31],
        ),
    ]


def test_pdf_edge_false_positive_is_removed_and_counted(
    reported_pdf_objects: list[dict],
):
    service = _service()

    kept, removed = service._remove_edge_class_outliers(
        reported_pdf_objects,
        image_width=1080,
        image_height=1920,
    )
    decision = service._build_decision(kept)

    assert len(removed) == 1
    assert removed[0]["raw_class"] == CIABATTA
    assert removed[0]["removal_reason"] == (
        "isolated_weaker_class_clipped_by_image_edge"
    )
    assert decision["decision"] == "DIRECT"
    assert decision["dominant_class"] == WHOLEMEAL
    assert decision["count"] == 2
    assert decision["purity"] == 1.0


def test_interior_conflicting_class_routes_to_operator_review(
    reported_pdf_objects: list[dict],
):
    service = _service()
    objects = deepcopy(reported_pdf_objects)
    objects[-1]["box_xyxy"] = [300.0, 300.0, 600.0, 600.0]

    kept, removed = service._remove_edge_class_outliers(
        objects,
        image_width=1080,
        image_height=1920,
    )

    assert removed == []
    decision = service._build_decision(kept)
    assert decision["decision"] == "REVIEW"
    assert decision["count"] == 3
    assert decision["requires_confirmation"] is True


def test_strong_edge_conflict_is_not_silently_removed(
    reported_pdf_objects: list[dict],
):
    service = _service()
    objects = deepcopy(reported_pdf_objects)
    objects[-1]["confidence"] = 0.90

    kept, removed = service._remove_edge_class_outliers(
        objects,
        image_width=1080,
        image_height=1920,
    )

    assert removed == []
    assert service._build_decision(kept)["decision"] == "REVIEW"


def test_multiple_minority_objects_are_preserved_for_review(
    reported_pdf_objects: list[dict],
):
    service = _service()
    objects = deepcopy(reported_pdf_objects)
    objects.append(
        _object(CIABATTA, 0.76, [0.0, 1000.0, 180.0, 1300.0])
    )

    kept, removed = service._remove_edge_class_outliers(
        objects,
        image_width=1080,
        image_height=1920,
    )

    assert removed == []
    assert service._build_decision(kept)["decision"] == "REVIEW"


def test_one_to_one_conflict_remains_blocked_as_ambiguous():
    service = _service()
    objects = [
        _object(WHOLEMEAL, 0.94, [100.0, 100.0, 300.0, 300.0]),
        _object(CIABATTA, 0.90, [500.0, 500.0, 700.0, 700.0]),
    ]

    assert service._build_decision(objects)["decision"] == "AMBIGUOUS"


def test_nested_cross_class_prediction_is_removed_as_duplicate():
    service = _service()
    objects = [
        _object(WHOLEMEAL, 0.95, [100.0, 100.0, 500.0, 500.0]),
        _object(CIABATTA, 0.70, [150.0, 150.0, 300.0, 300.0]),
    ]

    kept = service._remove_duplicate_objects(objects)

    assert len(kept) == 1
    assert kept[0]["raw_class"] == WHOLEMEAL
