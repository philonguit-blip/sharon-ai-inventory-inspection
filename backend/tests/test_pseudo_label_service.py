from __future__ import annotations

import json
from pathlib import Path

from app.services.pseudo_label_service import PseudoLabelService


def inference(objects):
    return {
        "width": 100,
        "height": 50,
        "engine": "YOLO",
        "objects": objects,
    }


def test_confirmed_boxes_become_train_ready_yolo_labels(tmp_path: Path):
    source = tmp_path / "tray.jpg"
    source.write_bytes(b"test-image")
    service = PseudoLabelService(tmp_path / "dataset")

    output = service.capture(
        job_id="a" * 32,
        source_path=source,
        inference_result=inference(
            [
                {"box_xyxy": [10, 5, 30, 25]},
                {"box_xyxy": [50, 10, 90, 40]},
            ]
        ),
        confirmed_product={"product_code": "BR-1", "product_name": "Bread", "quantity": 2},
    )

    assert output["train_ready"] is True
    label = Path(output["label_path"])
    assert len(label.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert json.loads((tmp_path / "dataset" / "classes.json").read_text())["classes"] == {"BR-1": 0}


def test_corrected_count_is_audit_only_not_box_training_data(tmp_path: Path):
    source = tmp_path / "tray.jpg"
    source.write_bytes(b"test-image")
    service = PseudoLabelService(tmp_path / "dataset")

    output = service.capture(
        job_id="b" * 32,
        source_path=source,
        inference_result=inference([{"box_xyxy": [10, 5, 30, 25]}]),
        confirmed_product={"product_code": "BR-2", "product_name": "Bread", "quantity": 3},
    )

    assert output["train_ready"] is False
    assert output["review_status"] == "VERIFIED_COUNT_ONLY"
    assert not list((tmp_path / "dataset" / "labels").glob("*.txt"))
