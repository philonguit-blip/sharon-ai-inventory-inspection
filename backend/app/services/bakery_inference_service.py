"""Production YOLO inference for the Sharon Bakery inventory workflow.

This service handles uploaded bakery images with the production PyTorch model
and converts every model class to a KiotViet product through
``product_mapping.json``.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    MAPPING_PATH,
    MAX_DETECTIONS,
    MODEL_PATH,
)
from app.services.product_mapping_service import (
    ProductMappingError,
    ProductMappingService,
)


MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "60000000"))
ANNOTATED_JPEG_QUALITY = int(os.getenv("ANNOTATED_JPEG_QUALITY", "92"))

# BGR colours used by OpenCV. More classes can safely reuse the palette.
BOX_COLORS = (
    (34, 197, 94),
    (249, 115, 22),
    (59, 130, 246),
    (168, 85, 247),
    (236, 72, 153),
)


class BakeryInferenceError(RuntimeError):
    """Base error raised by the bakery inference service."""


class ModelLoadError(BakeryInferenceError):
    """Raised when the production model cannot be loaded or validated."""


class ImageDecodeError(BakeryInferenceError):
    """Raised when uploaded bytes are not a valid supported image."""


def _normalise_model_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    if isinstance(names, (list, tuple)):
        return {class_id: str(name) for class_id, name in enumerate(names)}
    raise ModelLoadError("The model does not expose a valid class-name table.")


def _resolve_device(value: int | str | None) -> int | str:
    if value is None:
        value = os.getenv("BAKERY_DEVICE", "auto")
    if isinstance(value, int):
        return value

    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return 0 if torch.cuda.is_available() else "cpu"
    if text.isdigit():
        return int(text)
    return text


class BakeryInferenceService:
    """Load the production bakery model once and count products in images."""

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        mapping_path: Path | str = MAPPING_PATH,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        iou: float = DEFAULT_IOU,
        max_detections: int = MAX_DETECTIONS,
        device: int | str | None = None,
        show_confidence: bool = True,
        line_width: int = 2,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.mapping_path = Path(mapping_path).expanduser().resolve()
        self.confidence = float(confidence)
        if not 0.0 < self.confidence <= 1.0:
            raise ModelLoadError("Confidence threshold must be between 0 and 1.")
        self.image_size = int(image_size)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self.device = _resolve_device(device)
        self.show_confidence = bool(show_confidence)
        self.line_width = max(1, int(line_width))
        self._lock = threading.Lock()

        if not self.model_path.is_file():
            raise ModelLoadError(f"Production model not found: {self.model_path}")

        try:
            self.mapping = ProductMappingService(self.mapping_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Cannot load product mapping: {self.mapping_path}"
            ) from exc

        try:
            self.model = YOLO(str(self.model_path))
            self.raw_names = _normalise_model_names(self.model.names)
        except Exception as exc:
            raise ModelLoadError(
                f"Cannot load production model: {self.model_path}"
            ) from exc

        self.products_by_class_id = self._validate_product_mapping()
        self.candidate_confidence = max(
            self.confidence,
            min(
                float(product["confidence_threshold"])
                for product in self.products_by_class_id.values()
            ),
        )

    def _validate_product_mapping(self) -> dict[int, dict[str, Any]]:
        products: dict[int, dict[str, Any]] = {}
        unmapped: list[str] = []

        for class_id, raw_class in self.raw_names.items():
            try:
                products[class_id] = self.mapping.resolve(raw_class)
            except ProductMappingError:
                unmapped.append(raw_class)

        if unmapped:
            missing = ", ".join(sorted(unmapped))
            raise ModelLoadError(
                "Every model class must be mapped before inference. "
                f"Missing mapping for: {missing}"
            )
        if not products:
            raise ModelLoadError("The production model contains no classes.")
        return products

    def health(self) -> dict[str, Any]:
        """Return non-secret model metadata for the API health endpoint."""
        return {
            "ready": True,
            "model_path": str(self.model_path),
            "device": self.device,
            "image_size": self.image_size,
            "iou": self.iou,
            "confidence_floor": self.confidence,
            "candidate_confidence": self.candidate_confidence,
            "classes": self.raw_names,
        }

    def infer_path(
        self,
        image_path: Path | str,
        *,
        annotated_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run inference on a local image path."""
        source = Path(image_path).expanduser().resolve()
        if not source.is_file():
            raise ImageDecodeError(f"Image not found: {source}")
        return self.infer_bytes(
            source.read_bytes(),
            image_name=source.name,
            annotated_path=annotated_path,
        )

    def infer_bytes(
        self,
        raw_bytes: bytes,
        *,
        image_name: str,
        annotated_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Decode, detect, filter, count and optionally annotate one image."""
        started = time.perf_counter()
        image_bgr = self._decode_image(raw_bytes)

        try:
            with self._lock:
                with torch.inference_mode():
                    results = self.model.predict(
                        source=image_bgr,
                        imgsz=self.image_size,
                        conf=self.candidate_confidence,
                        iou=self.iou,
                        max_det=self.max_detections,
                        end2end=False,
                        agnostic_nms=False,
                        rect=False,
                        augment=False,
                        device=self.device,
                        save=False,
                        verbose=False,
                    )
        except Exception as exc:
            raise BakeryInferenceError(
                f"Model inference failed for image: {image_name}"
            ) from exc

        if not results:
            raise BakeryInferenceError(
                f"The model returned no result for image: {image_name}"
            )

        result = results[0]
        raw_count, kept_count = self._filter_by_class_threshold(result)
        objects = self._extract_objects(result)
        products = self._aggregate_products(objects)
        confidence_sum = sum(float(item["confidence"]) for item in objects)

        saved_annotation: str | None = None
        if annotated_path is not None:
            output = Path(annotated_path).expanduser().resolve()
            annotated = self._draw_annotations(image_bgr, objects)
            self._write_jpeg(output, annotated)
            saved_annotation = str(output)

        height, width = image_bgr.shape[:2]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "image_name": image_name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "width": int(width),
            "height": int(height),
            "raw_detections_before_class_filter": raw_count,
            "detections_removed_by_class_filter": raw_count - kept_count,
            "total_detections": kept_count,
            "confidence_sum": confidence_sum,
            "avg_confidence": confidence_sum / kept_count if kept_count else 0.0,
            "inference_ms": elapsed_ms,
            "products": products,
            "objects": objects,
            "annotated_path": saved_annotation,
            "status": "SUCCESS",
            "error": "",
        }

    @staticmethod
    def _decode_image(raw_bytes: bytes) -> np.ndarray:
        if not raw_bytes:
            raise ImageDecodeError("The uploaded image is empty.")

        encoded = np.frombuffer(raw_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ImageDecodeError(
                "The image cannot be decoded; it may be damaged or unsupported."
            )

        height, width = image_bgr.shape[:2]
        if height <= 0 or width <= 0:
            raise ImageDecodeError("The image dimensions are invalid.")
        if height * width > MAX_IMAGE_PIXELS:
            raise ImageDecodeError(
                f"The image exceeds the pixel limit: {width}x{height}."
            )
        return image_bgr

    def _filter_by_class_threshold(self, result: Any) -> tuple[int, int]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return 0, 0

        raw_count = len(boxes)
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy().astype(float)
        keep_numpy = np.asarray(
            [
                confidence
                >= max(
                    self.confidence,
                    float(self.products_by_class_id[int(class_id)]["confidence_threshold"]),
                )
                for class_id, confidence in zip(class_ids, confidences)
            ],
            dtype=bool,
        )

        box_data = boxes.data
        if isinstance(box_data, torch.Tensor):
            keep_mask = torch.as_tensor(
                keep_numpy, dtype=torch.bool, device=box_data.device
            )
        else:
            keep_mask = keep_numpy
        result.update(boxes=box_data[keep_mask])
        return raw_count, int(keep_numpy.sum())

    def _extract_objects(self, result: Any) -> list[dict[str, Any]]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy_values = boxes.xyxy.detach().cpu().numpy().astype(float)
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy().astype(float)
        objects: list[dict[str, Any]] = []

        for xyxy, class_id, confidence in zip(
            xyxy_values, class_ids, confidences
        ):
            product = self.products_by_class_id[int(class_id)]
            objects.append(
                {
                    "class_id": int(class_id),
                    "raw_class": self.raw_names[int(class_id)],
                    "product_code": product["product_code"],
                    "product_name": product["product_name"],
                    "purchase_price": float(product["purchase_price"]),
                    "confidence_threshold": max(
                        self.confidence,
                        float(product["confidence_threshold"]),
                    ),
                    "confidence": float(confidence),
                    "box_xyxy": [float(value) for value in xyxy],
                }
            )
        return objects

    @staticmethod
    def _aggregate_products(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in objects:
            grouped[str(item["product_code"])].append(item)

        rows: list[dict[str, Any]] = []
        for product_code in sorted(grouped):
            items = grouped[product_code]
            confidences = [float(item["confidence"]) for item in items]
            rows.append(
                {
                    "product_code": product_code,
                    "product_name": items[0]["product_name"],
                    "purchase_price": float(items[0]["purchase_price"]),
                    "quantity": len(items),
                    "raw_classes": sorted(
                        {str(item["raw_class"]) for item in items}
                    ),
                    "confidence_threshold": min(
                        float(item["confidence_threshold"]) for item in items
                    ),
                    "confidence_sum": sum(confidences),
                    "avg_confidence": sum(confidences) / len(confidences),
                    "min_confidence": min(confidences),
                    "max_confidence": max(confidences),
                }
            )
        return rows

    def _draw_annotations(
        self, image_bgr: np.ndarray, objects: list[dict[str, Any]]
    ) -> np.ndarray:
        canvas = image_bgr.copy()
        image_height, image_width = canvas.shape[:2]
        scale = max(1.0, max(image_height, image_width) / 1200.0)
        box_thickness = max(2, int(round(self.line_width * scale)))
        font_scale = max(0.7, min(2.2, 0.7 * scale))
        text_thickness = max(1, int(round(box_thickness * 0.55)))
        padding = max(4, int(round(4 * scale)))

        for item in objects:
            x1, y1, x2, y2 = (
                int(round(value)) for value in item["box_xyxy"]
            )
            x1 = min(max(x1, 0), image_width - 1)
            y1 = min(max(y1, 0), image_height - 1)
            x2 = min(max(x2, 0), image_width - 1)
            y2 = min(max(y2, 0), image_height - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            color = BOX_COLORS[int(item["class_id"]) % len(BOX_COLORS)]
            label = str(item["product_code"])
            if self.show_confidence:
                label += f" {float(item['confidence']):.2f}"

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                color,
                thickness=box_thickness,
                lineType=cv2.LINE_AA,
            )
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )
            label_top = max(0, y1 - text_height - baseline - 2 * padding)
            label_right = min(image_width - 1, x1 + text_width + 2 * padding)
            cv2.rectangle(
                canvas, (x1, label_top), (label_right, y1), color, thickness=-1
            )
            cv2.putText(
                canvas,
                label,
                (x1 + padding, max(text_height + padding, y1 - padding - baseline)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
        return canvas

    @staticmethod
    def _write_jpeg(path: Path, image_bgr: np.ndarray) -> None:
        success, encoded = cv2.imencode(
            ".jpg",
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, ANNOTATED_JPEG_QUALITY],
        )
        if not success:
            raise BakeryInferenceError(f"Cannot encode annotated image: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.tobytes())
