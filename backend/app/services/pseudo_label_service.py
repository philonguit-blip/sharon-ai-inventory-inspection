"""Capture operator-confirmed detections as auditable training candidates."""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import PSEUDO_LABEL_ENABLED, PSEUDO_LABEL_ROOT


class PseudoLabelService:
    def __init__(
        self,
        root: Path | str = PSEUDO_LABEL_ROOT,
        *,
        enabled: bool = PSEUDO_LABEL_ENABLED,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.enabled = bool(enabled)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.metadata = self.root / "metadata"
        self.registry_path = self.root / "classes.json"
        self._lock = threading.Lock()
        if self.enabled:
            for directory in (self.images, self.labels, self.metadata):
                directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def health(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "root": str(self.root),
            "class_count": len(self._read_registry()),
        }

    def _read_registry(self) -> dict[str, int]:
        if not self.registry_path.is_file():
            return {}
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        classes = payload.get("classes", {})
        return {str(key): int(value) for key, value in classes.items()}

    @staticmethod
    def _write_atomic(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    def _class_id(self, product_code: str) -> int:
        classes = self._read_registry()
        if product_code not in classes:
            classes[product_code] = max(classes.values(), default=-1) + 1
            self._write_atomic(
                self.registry_path,
                json.dumps(
                    {"schema_version": 1, "classes": classes},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        return classes[product_code]

    def capture(
        self,
        *,
        job_id: str,
        source_path: Path | str,
        inference_result: dict[str, Any],
        confirmed_product: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"captured": False, "reason": "Pseudo-label capture is disabled."}

        source = Path(source_path).expanduser().resolve()
        width = int(inference_result.get("width") or 0)
        height = int(inference_result.get("height") or 0)
        quantity = int(confirmed_product.get("quantity") or 0)
        product_code = str(confirmed_product.get("product_code") or "").strip()
        objects = [
            item
            for item in (inference_result.get("objects") or [])
            if isinstance(item, dict) and len(item.get("box_xyxy") or []) == 4
        ]
        train_ready = bool(
            source.is_file()
            and width > 0
            and height > 0
            and product_code
            and quantity > 0
            and len(objects) == quantity
        )
        record = {
            "schema_version": 1,
            "job_id": job_id,
            "captured_at": self._now(),
            "product_code": product_code,
            "product_name": confirmed_product.get("product_name"),
            "confirmed_quantity": quantity,
            "detected_box_count": len(objects),
            "engine": inference_result.get("engine") or "YOLO",
            "hybrid": inference_result.get("hybrid"),
            "source_image": source.name,
            "train_ready": train_ready,
            "review_status": "VERIFIED_BOXES" if train_ready else "VERIFIED_COUNT_ONLY",
        }

        with self._lock:
            stem = f"{job_id}_{source.stem}"
            image_target = self.images / f"{stem}{source.suffix.lower()}"
            metadata_target = self.metadata / f"{stem}.json"
            if source.is_file() and not image_target.exists():
                shutil.copy2(source, image_target)
            record["image_path"] = str(image_target)

            if train_ready:
                class_id = self._class_id(product_code)
                lines: list[str] = []
                for item in objects:
                    x1, y1, x2, y2 = [float(v) for v in item["box_xyxy"]]
                    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
                    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width <= 0 or box_height <= 0:
                        train_ready = False
                        break
                    lines.append(
                        f"{class_id} {((x1 + x2) / 2) / width:.8f} "
                        f"{((y1 + y2) / 2) / height:.8f} "
                        f"{box_width / width:.8f} {box_height / height:.8f}"
                    )
                if train_ready:
                    label_target = self.labels / f"{stem}.txt"
                    self._write_atomic(label_target, "\n".join(lines) + "\n")
                    record["label_path"] = str(label_target)
                    record["class_id"] = class_id
                else:
                    record["train_ready"] = False
                    record["review_status"] = "INVALID_BOX_GEOMETRY"

            self._write_atomic(
                metadata_target,
                json.dumps(record, ensure_ascii=False, indent=2),
            )
            record["metadata_path"] = str(metadata_target)
        return {"captured": True, **record}

    def stats(self) -> dict[str, Any]:
        records = []
        if self.metadata.is_dir():
            for path in self.metadata.glob("*.json"):
                try:
                    records.append(json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    continue
        return {
            **self.health(),
            "records": len(records),
            "train_ready": sum(bool(item.get("train_ready")) for item in records),
            "review_required": sum(not bool(item.get("train_ready")) for item in records),
            "by_product": {
                code: sum(item.get("product_code") == code for item in records)
                for code in sorted({str(item.get("product_code") or "") for item in records} - {""})
            },
        }
