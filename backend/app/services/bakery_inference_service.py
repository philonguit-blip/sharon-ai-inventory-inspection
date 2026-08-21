"""Production YOLO inference for the Sharon Bakery single-product intake flow.

The detector exposes the visual classes declared by the production model and mapping. Some
visual classes are SKU families and therefore require an explicit user choice
before a KiotViet receipt can be created.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import Counter, defaultdict
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
    CONFLICT_REVIEW_MIN_DOMINANT_COUNT,
    CROSS_CLASS_DUPLICATE_COVERAGE,
    DUPLICATE_BOX_IOU,
    EDGE_CLASS_OUTLIER_CONFIDENCE_GAP,
    EDGE_CLASS_OUTLIER_ENABLED,
    EDGE_CLASS_OUTLIER_MARGIN_RATIO,
    EDGE_CLASS_OUTLIER_MIN_DOMINANT_COUNT,
    HYBRID_BOX_RESCUE_ENABLED,
    HYBRID_BOX_RESCUE_MIN_CONFIDENCE,
    HYBRID_BOX_RESCUE_MIN_THRESHOLD_RATIO,
    MAPPING_PATH,
    MAX_DETECTIONS,
    MIN_DOMINANT_PURITY,
    MODEL_PATH,
)
from app.services.product_mapping_service import (
    ProductMappingError,
    ProductMappingService,
)


MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "60000000"))
ANNOTATED_JPEG_QUALITY = int(os.getenv("ANNOTATED_JPEG_QUALITY", "92"))

# Batch rescue is intentionally conservative. It never lowers the normal
# production threshold globally; it is used only after the other images in
# the same job strongly agree on one visual class.
BATCH_RESCUE_THRESHOLD_RATIO = float(
    os.getenv("BAKERY_BATCH_RESCUE_THRESHOLD_RATIO", "0.70")
)
BATCH_RESCUE_MIN_CONFIDENCE = float(
    os.getenv("BAKERY_BATCH_RESCUE_MIN_CONFIDENCE", "0.20")
)

# Garlic Grissini consists of long, thin pieces that are often placed almost
# edge-to-edge. Standard NMS at 0.50 can suppress neighbouring physical sticks
# whose axis-aligned boxes overlap strongly. Run the model with the highest NMS
# IoU needed by any special class, then re-apply class-aware NMS below so every
# other class keeps the normal production IoU.
GARLIC_GRISSINI_CLASS = "CA-GRI-0000041_GarlicGrissini"
GARLIC_GRISSINI_NMS_IOU = float(
    os.getenv("BAKERY_GRISSINI_NMS_IOU", "0.75")
)
CLASS_NMS_IOU_OVERRIDES: dict[str, float] = {
    GARLIC_GRISSINI_CLASS: GARLIC_GRISSINI_NMS_IOU,
}


def _foundation_candidate_floor(threshold: float) -> float:
    """Return the lowest YOLO score worth sending to box verification."""
    threshold = max(0.0, min(1.0, float(threshold)))
    if not HYBRID_BOX_RESCUE_ENABLED:
        return threshold
    ratio = max(
        0.05,
        min(1.0, float(HYBRID_BOX_RESCUE_MIN_THRESHOLD_RATIO)),
    )
    minimum = max(0.0, min(1.0, float(HYBRID_BOX_RESCUE_MIN_CONFIDENCE)))
    return min(threshold, max(minimum, threshold * ratio))

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


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _box_coverage(box_a: list[float], box_b: list[float]) -> float:
    """Intersection divided by the smaller box area."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller > 0.0 else 0.0


class BakeryInferenceService:
    """Load the production bakery model once and analyse one product image."""

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        mapping_path: Path | str = MAPPING_PATH,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        iou: float = DEFAULT_IOU,
        max_detections: int = MAX_DETECTIONS,
        duplicate_iou: float = DUPLICATE_BOX_IOU,
        cross_class_duplicate_coverage: float = CROSS_CLASS_DUPLICATE_COVERAGE,
        min_purity: float = MIN_DOMINANT_PURITY,
        conflict_review_min_dominant_count: int = (
            CONFLICT_REVIEW_MIN_DOMINANT_COUNT
        ),
        edge_outlier_enabled: bool = EDGE_CLASS_OUTLIER_ENABLED,
        edge_outlier_margin_ratio: float = EDGE_CLASS_OUTLIER_MARGIN_RATIO,
        edge_outlier_confidence_gap: float = EDGE_CLASS_OUTLIER_CONFIDENCE_GAP,
        edge_outlier_min_dominant_count: int = (
            EDGE_CLASS_OUTLIER_MIN_DOMINANT_COUNT
        ),
        device: int | str | None = None,
        show_confidence: bool = True,
        line_width: int = 2,
        confidence_overrides: dict[str, float] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.mapping_path = Path(mapping_path).expanduser().resolve()
        self.confidence = float(confidence)
        if not 0.0 < self.confidence <= 1.0:
            raise ModelLoadError("Confidence floor must be between 0 and 1.")
        self.image_size = int(image_size)
        self.iou = float(iou)
        self.class_nms_iou_overrides = {
            str(raw_class): float(value)
            for raw_class, value in CLASS_NMS_IOU_OVERRIDES.items()
        }
        self.inference_iou = max(
            [self.iou, *self.class_nms_iou_overrides.values()]
        )
        self.max_detections = int(max_detections)
        self.duplicate_iou = float(duplicate_iou)
        self.cross_class_duplicate_coverage = float(
            cross_class_duplicate_coverage
        )
        self.min_purity = float(min_purity)
        self.conflict_review_min_dominant_count = max(
            2, int(conflict_review_min_dominant_count)
        )
        self.edge_outlier_enabled = bool(edge_outlier_enabled)
        self.edge_outlier_margin_ratio = float(edge_outlier_margin_ratio)
        self.edge_outlier_confidence_gap = float(edge_outlier_confidence_gap)
        self.edge_outlier_min_dominant_count = max(
            2, int(edge_outlier_min_dominant_count)
        )
        if not 0.0 < self.iou <= 1.0:
            raise ModelLoadError("NMS IoU must be between 0 and 1.")
        invalid_class_nms = {
            raw_class: value
            for raw_class, value in self.class_nms_iou_overrides.items()
            if not 0.0 < float(value) <= 1.0
        }
        if invalid_class_nms:
            raise ModelLoadError(
                "Class-specific NMS IoU must be between 0 and 1: "
                + ", ".join(
                    f"{raw_class}={value}"
                    for raw_class, value in sorted(invalid_class_nms.items())
                )
            )
        if not 0.0 < self.duplicate_iou <= 1.0:
            raise ModelLoadError("Duplicate IoU must be between 0 and 1.")
        if not 0.0 < self.cross_class_duplicate_coverage <= 1.0:
            raise ModelLoadError(
                "Cross-class duplicate coverage must be between 0 and 1."
            )
        if not 0.0 < self.min_purity <= 1.0:
            raise ModelLoadError("Minimum purity must be between 0 and 1.")
        if not 0.0 <= self.edge_outlier_margin_ratio <= 0.10:
            raise ModelLoadError(
                "Edge-outlier margin ratio must be between 0 and 0.10."
            )
        if not 0.0 <= self.edge_outlier_confidence_gap <= 1.0:
            raise ModelLoadError(
                "Edge-outlier confidence gap must be between 0 and 1."
            )

        self.device = _resolve_device(device)
        self.show_confidence = bool(show_confidence)
        self.line_width = max(1, int(line_width))
        self._lock = threading.Lock()

        if not self.model_path.is_file():
            raise ModelLoadError(f"Production model not found: {self.model_path}")

        try:
            self.mapping = ProductMappingService(
                self.mapping_path,
                confidence_overrides=confidence_overrides,
            )
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

        # Candidate inference must expose the sub-threshold proposal band used
        # by Foundation box rescue. These proposals are never counted directly;
        # normal per-class filtering still runs before the business decision.
        # Keeping the floor per class also ensures a very low production
        # threshold is never accidentally raised by the rescue configuration.
        candidate_floors = []
        for product in self.products_by_class_id.values():
            threshold = float(product["confidence_threshold"])
            candidate_floors.append(_foundation_candidate_floor(threshold))
        self.candidate_confidence = min(
            self.confidence,
            min(candidate_floors),
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
        return {
            "ready": True,
            "model_path": str(self.model_path),
            "device": self.device,
            "image_size": self.image_size,
            "iou": self.iou,
            "model_nms_iou": self.inference_iou,
            "class_nms_iou_overrides": dict(self.class_nms_iou_overrides),
            "confidence_floor": self.confidence,
            "candidate_confidence": self.candidate_confidence,
            "duplicate_iou": self.duplicate_iou,
            "cross_class_duplicate_coverage": (
                self.cross_class_duplicate_coverage
            ),
            "min_purity": self.min_purity,
            "conflict_review_min_dominant_count": (
                self.conflict_review_min_dominant_count
            ),
            "edge_outlier_enabled": self.edge_outlier_enabled,
            "edge_outlier_margin_ratio": self.edge_outlier_margin_ratio,
            "edge_outlier_confidence_gap": self.edge_outlier_confidence_gap,
            "edge_outlier_min_dominant_count": (
                self.edge_outlier_min_dominant_count
            ),
            "batch_rescue_threshold_ratio": BATCH_RESCUE_THRESHOLD_RATIO,
            "batch_rescue_min_confidence": BATCH_RESCUE_MIN_CONFIDENCE,
            "visual_class_count": len(self.raw_names),
            "classes": self.raw_names,
        }

    def infer_path(
        self,
        image_path: Path | str,
        *,
        annotated_path: Path | str | None = None,
    ) -> dict[str, Any]:
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
        started = time.perf_counter()
        image_bgr = self._decode_image(raw_bytes)

        try:
            with self._lock:
                with torch.inference_mode():
                    results = self.model.predict(
                        source=image_bgr,
                        imgsz=self.image_size,
                        conf=self.candidate_confidence,
                        iou=self.inference_iou,
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
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return self._build_result_payload(
            raw_bytes=raw_bytes,
            image_name=image_name,
            image_bgr=image_bgr,
            result=results[0],
            annotated_path=annotated_path,
            inference_ms=elapsed_ms,
        )

    def infer_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run one YOLO forward pass for a small group of decoded images."""
        if not items:
            return []
        started = time.perf_counter()
        decoded: list[np.ndarray] = []
        for item in items:
            decoded.append(self._decode_image(bytes(item["raw_bytes"])))
        try:
            with self._lock:
                with torch.inference_mode():
                    predictions = self.model.predict(
                        source=decoded,
                        imgsz=self.image_size,
                        conf=self.candidate_confidence,
                        iou=self.inference_iou,
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
            # Some CPU/provider combinations cannot allocate a batched tensor
            # at the configured inference size. Fall back to the proven single-image path instead of
            # failing the entire user job.
            fallback = [
                self.infer_bytes(
                    bytes(item["raw_bytes"]),
                    image_name=str(item["image_name"]),
                    annotated_path=item.get("annotated_path"),
                )
                for item in items
            ]
            for payload in fallback:
                payload["batch_fallback"] = True
                payload["batch_fallback_reason"] = type(exc).__name__
            return fallback
        if len(predictions) != len(items):
            fallback = [
                self.infer_bytes(
                    bytes(item["raw_bytes"]),
                    image_name=str(item["image_name"]),
                    annotated_path=item.get("annotated_path"),
                )
                for item in items
            ]
            for payload in fallback:
                payload["batch_fallback"] = True
                payload["batch_fallback_reason"] = "INCOMPLETE_BATCH"
            return fallback
        elapsed_per_image = (
            (time.perf_counter() - started) * 1000.0 / len(items)
        )
        return [
            self._build_result_payload(
                raw_bytes=bytes(item["raw_bytes"]),
                image_name=str(item["image_name"]),
                image_bgr=image_bgr,
                result=prediction,
                annotated_path=item.get("annotated_path"),
                inference_ms=elapsed_per_image,
            )
            for item, image_bgr, prediction in zip(
                items, decoded, predictions, strict=True
            )
        ]

    def _build_result_payload(
        self,
        *,
        raw_bytes: bytes,
        image_name: str,
        image_bgr: np.ndarray,
        result: Any,
        annotated_path: Path | str | None,
        inference_ms: float,
    ) -> dict[str, Any]:
        # The model first runs with the most permissive NMS IoU required by
        # any special class. Re-apply class-aware NMS before thresholding so
        # Garlic Grissini can use 0.75 while all other classes keep the normal
        # production IoU.
        (
            class_nms_input_count,
            class_nms_kept_count,
        ) = self._apply_class_specific_nms(result)

        # Preserve the low-confidence candidate pool before the normal
        # per-class threshold mutates result.boxes. A later batch-consensus
        # rescue can inspect only the already-computed YOLO candidates without
        # running the model a second time or lowering production thresholds.
        candidate_objects = self._extract_objects(result)
        raw_count, threshold_kept_count = self._filter_by_class_threshold(result)
        threshold_objects = self._extract_objects(result)
        objects = self._remove_duplicate_objects(threshold_objects)
        duplicate_removed = len(threshold_objects) - len(objects)
        height, width = image_bgr.shape[:2]
        objects, edge_outliers = self._remove_edge_class_outliers(
            objects,
            image_width=int(width),
            image_height=int(height),
        )
        decision = self._build_decision(objects)
        products = self._products_for_direct_decision(decision)

        confidence_sum = sum(float(item["confidence"]) for item in objects)

        saved_annotation: str | None = None
        if annotated_path is not None:
            output = Path(annotated_path).expanduser().resolve()
            annotated = self._draw_annotations(image_bgr, objects)
            self._write_jpeg(output, annotated)
            saved_annotation = str(output)

        return {
            "image_name": image_name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "width": int(width),
            "height": int(height),
            "detections_after_model_nms": class_nms_input_count,
            "detections_removed_by_class_specific_nms": (
                class_nms_input_count - class_nms_kept_count
            ),
            "raw_detections_before_class_filter": raw_count,
            "detections_removed_by_class_filter": raw_count - threshold_kept_count,
            "detections_removed_as_duplicates": duplicate_removed,
            "detections_removed_as_edge_class_outliers": len(edge_outliers),
            "edge_class_outliers": edge_outliers,
            "total_detections": len(objects),
            "confidence_sum": confidence_sum,
            "avg_confidence": (
                confidence_sum / len(objects) if objects else 0.0
            ),
            "inference_ms": float(inference_ms),
            "decision": decision,
            # Preserved for compatibility. Family/ambiguous/no-detection results
            # intentionally do not expose an exact KiotViet product yet.
            "products": products,
            "objects": objects,
            # Internal recovery evidence. These are YOLO candidates after model
            # NMS at candidate_confidence but before per-class thresholds.
            # They are never counted unless a same-job batch consensus safely
            # requests a rescue for one specific visual class.
            "candidate_objects": candidate_objects,
            "annotated_path": saved_annotation,
            "status": "SUCCESS",
            "error": "",
        }

    def rescue_result_for_class(
        self,
        payload: dict[str, Any],
        raw_class: str,
        *,
        raw_bytes: bytes | None = None,
        annotated_path: Path | str | None = None,
        threshold_ratio: float = BATCH_RESCUE_THRESHOLD_RATIO,
        min_confidence: float = BATCH_RESCUE_MIN_CONFIDENCE,
    ) -> dict[str, Any] | None:
        """Recover one failed image using same-job visual-class consensus.

        This method does not run YOLO again. It reuses the candidate boxes
        produced by the original forward pass before per-class filtering and
        only accepts boxes for ``raw_class``. The rescue threshold is lower
        than the normal class threshold, but never changes production config.
        """
        target = str(raw_class or "").strip()
        if not target:
            return None

        try:
            mapping = self.mapping.resolve(target)
        except ProductMappingError:
            return None

        normal_threshold = float(mapping["confidence_threshold"])
        ratio = max(0.05, min(1.0, float(threshold_ratio)))
        minimum = max(0.0, min(1.0, float(min_confidence)))
        rescue_threshold = min(
            normal_threshold,
            max(
                self.candidate_confidence,
                minimum,
                normal_threshold * ratio,
            ),
        )

        candidate_pool = payload.get("candidate_objects")
        if not isinstance(candidate_pool, list):
            candidate_pool = payload.get("yolo_candidate_objects")
        if not isinstance(candidate_pool, list):
            return None

        target_candidates = [
            dict(item)
            for item in candidate_pool
            if isinstance(item, dict)
            and str(item.get("raw_class") or "") == target
            and float(item.get("confidence") or 0.0) >= rescue_threshold
        ]
        objects = self._remove_duplicate_objects(target_candidates)
        if not objects:
            return None

        confidences = [float(item["confidence"]) for item in objects]
        count = len(objects)
        base = {
            "dominant_class": target,
            "display_name": str(mapping["display_name"]),
            "count": count,
            "physical_count": count,
            "dominant_count": count,
            "total_detections": count,
            "purity": 1.0,
            "classification_purity": 1.0,
            "avg_confidence": sum(confidences) / count,
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "confidence_threshold": normal_threshold,
            "rescue_confidence_threshold": rescue_threshold,
            "class_breakdown": [
                {
                    "raw_class": target,
                    "display_name": str(mapping["display_name"]),
                    "count": count,
                    "avg_confidence": sum(confidences) / count,
                }
            ],
            "recovered_by_batch_context": True,
        }

        if mapping["type"] == "family":
            decision = {
                **base,
                "decision": "FAMILY",
                "members": [dict(member) for member in mapping["members"]],
                "requires_user_selection": True,
                "requires_confirmation": True,
                "message": (
                    "Recovered from low-confidence YOLO candidates because the "
                    "other images in this same-SKU batch agree on this visual family. "
                    "Verify the SKU and count before confirmation."
                ),
            }
        else:
            decision = {
                **base,
                "decision": "DIRECT",
                "product_code": str(mapping["product_code"]),
                "product_name": str(mapping["product_name"]),
                "requires_user_selection": False,
                "requires_confirmation": True,
                "message": (
                    "Recovered from low-confidence YOLO candidates because the "
                    "other images in this same-SKU batch agree on this class. "
                    "Verify the count before confirmation."
                ),
            }

        rescued = dict(payload)
        rescued.update(
            {
                "total_detections": count,
                "confidence_sum": sum(confidences),
                "avg_confidence": sum(confidences) / count,
                "decision": decision,
                "products": self._products_for_direct_decision(decision),
                "objects": objects,
                "status": "SUCCESS",
                "error": "",
                "batch_rescue": {
                    "applied": True,
                    "raw_class": target,
                    "normal_threshold": normal_threshold,
                    "rescue_threshold": rescue_threshold,
                    "candidate_count": len(target_candidates),
                    "kept_count": count,
                },
            }
        )

        if raw_bytes is not None and annotated_path is not None:
            image_bgr = self._decode_image(raw_bytes)
            output = Path(annotated_path).expanduser().resolve()
            annotated = self._draw_annotations(image_bgr, objects)
            self._write_jpeg(output, annotated)
            rescued["annotated_path"] = str(output)

        return rescued

    def apply_foundation_box_rescue(
        self,
        payload: dict[str, Any],
        verification: dict[str, Any],
        *,
        raw_bytes: bytes | None = None,
        annotated_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Merge only same-class Foundation-verified YOLO proposals.

        Foundation similarity becomes the recovered object's confidence, while
        the original YOLO score remains attached as audit evidence. Confident
        class disagreements are deliberately not merged; the hybrid router can
        escalate those images to a full-image comparison.
        """
        recovered: list[dict[str, Any]] = []
        for verdict in verification.get("verdicts") or []:
            if not isinstance(verdict, dict) or not verdict.get("accepted"):
                continue
            candidate = verdict.get("candidate")
            if not isinstance(candidate, dict):
                continue
            item = dict(candidate)
            item["yolo_confidence"] = float(item.get("confidence") or 0.0)
            item["confidence"] = float(
                np.clip(verdict.get("foundation_similarity") or 0.0, 0.0, 1.0)
            )
            item["box_xyxy"] = [
                float(value)
                for value in (
                    verdict.get("refined_box_xyxy")
                    or item.get("box_xyxy")
                    or []
                )
            ]
            item["foundation_similarity"] = float(
                verdict.get("foundation_similarity") or 0.0
            )
            item["foundation_margin"] = float(
                verdict.get("foundation_margin") or 0.0
            )
            item["foundation_mask_refined"] = bool(verdict.get("mask_refined"))
            item["recovered_by_foundation_box"] = True
            if len(item["box_xyxy"]) == 4:
                recovered.append(item)

        existing = [
            dict(item)
            for item in (payload.get("objects") or [])
            if isinstance(item, dict)
        ]
        objects = self._remove_duplicate_objects(existing + recovered)
        kept_recovered = sum(
            1 for item in objects if item.get("recovered_by_foundation_box")
        )
        decision = self._build_decision(objects)
        confidences = [float(item.get("confidence") or 0.0) for item in objects]
        result = dict(payload)
        result.update(
            {
                "total_detections": len(objects),
                "confidence_sum": sum(confidences),
                "avg_confidence": (
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
                "decision": decision,
                "products": self._products_for_direct_decision(decision),
                "objects": objects,
                "detections_removed_by_class_filter": max(
                    0,
                    int(payload.get("detections_removed_by_class_filter") or 0)
                    - kept_recovered,
                ),
                "foundation_box_rescue": {
                    "attempted": int(verification.get("attempted") or 0),
                    "accepted": int(verification.get("accepted") or 0),
                    "kept_after_deduplication": kept_recovered,
                    "class_disagreements": int(
                        verification.get("disagreements") or 0
                    ),
                    "timings_ms": dict(verification.get("timings_ms") or {}),
                    "verdicts": [
                        {
                            key: value
                            for key, value in verdict.items()
                            if key != "candidate"
                        }
                        for verdict in (verification.get("verdicts") or [])
                        if isinstance(verdict, dict)
                    ],
                },
            }
        )
        if raw_bytes is not None and annotated_path is not None:
            image_bgr = self._decode_image(raw_bytes)
            output = Path(annotated_path).expanduser().resolve()
            annotated = self._draw_annotations(image_bgr, objects)
            self._write_jpeg(output, annotated)
            result["annotated_path"] = str(output)
        return result

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

    def _apply_class_specific_nms(self, result: Any) -> tuple[int, int]:
        """Apply per-class NMS after the model's permissive first-pass NMS.

        Ultralytics accepts only one IoU value for ``predict``. To preserve
        neighbouring Garlic Grissini sticks we therefore run the model at the
        most permissive configured class IoU (0.75 by default), then restore
        the normal production IoU for every other class here.
        """
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return 0, 0

        xyxy_values = boxes.xyxy.detach().cpu().numpy().astype(float)
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy().astype(float)
        total = len(class_ids)

        keep_numpy = np.zeros(total, dtype=bool)
        confidence_order = np.argsort(-confidences)

        for class_id in sorted(set(class_ids.tolist())):
            raw_class = self.raw_names.get(int(class_id), "")
            nms_iou = float(
                self.class_nms_iou_overrides.get(raw_class, self.iou)
            )
            pending = [
                int(index)
                for index in confidence_order
                if int(class_ids[int(index)]) == int(class_id)
            ]

            while pending:
                current = pending.pop(0)
                keep_numpy[current] = True
                current_box = [
                    float(value) for value in xyxy_values[current]
                ]
                pending = [
                    index
                    for index in pending
                    if _box_iou(
                        current_box,
                        [float(value) for value in xyxy_values[index]],
                    )
                    < nms_iou
                ]

        box_data = boxes.data
        if isinstance(box_data, torch.Tensor):
            keep_mask = torch.as_tensor(
                keep_numpy,
                dtype=torch.bool,
                device=box_data.device,
            )
        else:
            keep_mask = keep_numpy
        result.update(boxes=box_data[keep_mask])
        return total, int(keep_numpy.sum())

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
                >= float(
                    self.products_by_class_id[int(class_id)][
                        "confidence_threshold"
                    ]
                )
                for class_id, confidence in zip(class_ids, confidences)
            ],
            dtype=bool,
        )

        box_data = boxes.data
        if isinstance(box_data, torch.Tensor):
            keep_mask = torch.as_tensor(
                keep_numpy,
                dtype=torch.bool,
                device=box_data.device,
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
            xyxy_values,
            class_ids,
            confidences,
        ):
            mapping = self.products_by_class_id[int(class_id)]
            item = {
                "class_id": int(class_id),
                "raw_class": self.raw_names[int(class_id)],
                "mapping_type": mapping["type"],
                "display_name": mapping["display_name"],
                "confidence_threshold": float(mapping["confidence_threshold"]),
                "confidence": float(confidence),
                "box_xyxy": [float(value) for value in xyxy],
            }
            if mapping["type"] == "direct":
                item.update(
                    {
                        "product_code": mapping["product_code"],
                        "product_name": mapping["product_name"],
                        "purchase_price": float(mapping["purchase_price"]),
                    }
                )
            objects.append(item)
        return objects

    def _remove_duplicate_objects(
        self,
        objects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove near-identical boxes across classes while preserving crowded trays."""
        ordered = sorted(
            objects,
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )
        kept: list[dict[str, Any]] = []
        for candidate in ordered:
            if any(
                (
                    _box_iou(
                        candidate["box_xyxy"],
                        accepted["box_xyxy"],
                    )
                    >= self.duplicate_iou
                    or (
                        str(candidate["raw_class"])
                        != str(accepted["raw_class"])
                        and _box_coverage(
                            candidate["box_xyxy"],
                            accepted["box_xyxy"],
                        )
                        >= self.cross_class_duplicate_coverage
                    )
                )
                for accepted in kept
            ):
                continue
            kept.append(candidate)

        # Stable visual order makes output easier to inspect and test.
        return sorted(
            kept,
            key=lambda item: (
                float(item["box_xyxy"][1]),
                float(item["box_xyxy"][0]),
            ),
        )

    def _build_decision(
        self,
        objects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a business decision while keeping physical count separate from class identity.

        ``count`` is the number of distinct retained physical objects after
        thresholding/deduplication. ``dominant_count`` is only classification
        evidence for the winning visual class. This distinction is important
        for same-SKU jobs: a misclassified loaf is still a real loaf and must not
        disappear from the quantity merely because its class label disagrees.
        """
        if not objects:
            return {
                "decision": "NO_DETECTION",
                "count": 0,
                "physical_count": 0,
                "dominant_count": 0,
                "total_detections": 0,
                "purity": 0.0,
                "classification_purity": 0.0,
                "avg_confidence": 0.0,
                "physical_avg_confidence": 0.0,
                "requires_user_selection": False,
                "requires_confirmation": False,
                "message": (
                    "No bakery product was detected. Retake the image with the "
                    "product/tray fully visible, sharp and well lit."
                ),
            }

        by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in objects:
            by_class[str(item["raw_class"])].append(item)

        ranked = sorted(
            by_class.items(),
            key=lambda pair: (
                len(pair[1]),
                sum(float(item["confidence"]) for item in pair[1]),
            ),
            reverse=True,
        )
        dominant_class, dominant_objects = ranked[0]
        dominant_count = len(dominant_objects)
        physical_count = len(objects)
        purity = dominant_count / physical_count
        dominant_confidences = [
            float(item["confidence"]) for item in dominant_objects
        ]
        all_confidences = [float(item["confidence"]) for item in objects]
        mapping = self.mapping.resolve(dominant_class)

        class_breakdown: list[dict[str, Any]] = []
        for raw_class, items in ranked:
            class_mapping = self.mapping.resolve(raw_class)
            class_breakdown.append(
                {
                    "raw_class": raw_class,
                    "mapping_type": class_mapping["type"],
                    "display_name": items[0]["display_name"],
                    "count": len(items),
                    "avg_confidence": (
                        sum(float(item["confidence"]) for item in items)
                        / len(items)
                    ),
                    "confidence_threshold": float(
                        class_mapping["confidence_threshold"]
                    ),
                }
            )

        base = {
            "dominant_class": dominant_class,
            "display_name": mapping["display_name"],
            "count": physical_count,
            "physical_count": physical_count,
            "dominant_count": dominant_count,
            "total_detections": physical_count,
            "purity": purity,
            "classification_purity": purity,
            "avg_confidence": (
                sum(dominant_confidences) / len(dominant_confidences)
            ),
            "physical_avg_confidence": sum(all_confidences) / physical_count,
            "min_confidence": min(dominant_confidences),
            "max_confidence": max(dominant_confidences),
            "confidence_threshold": float(mapping["confidence_threshold"]),
            "class_breakdown": class_breakdown,
        }

        if purity < self.min_purity:
            if dominant_count >= self.conflict_review_min_dominant_count:
                candidate_by_code: dict[str, dict[str, Any]] = {}
                for raw_class, items in ranked:
                    class_mapping = self.mapping.resolve(raw_class)
                    detected_quantity = len(items)
                    class_avg_confidence = (
                        sum(float(item["confidence"]) for item in items)
                        / detected_quantity
                    )
                    if class_mapping["type"] == "family":
                        raw_candidates = [dict(member) for member in class_mapping["members"]]
                    else:
                        raw_candidates = [
                            {
                                "product_code": class_mapping["product_code"],
                                "product_name": class_mapping["product_name"],
                                "display_name": class_mapping["display_name"],
                            }
                        ]

                    for raw_candidate in raw_candidates:
                        code = str(raw_candidate.get("product_code") or "").strip()
                        if not code:
                            continue
                        key = code.casefold()
                        candidate = candidate_by_code.get(key)
                        evidence = {
                            **raw_candidate,
                            "source": "YOLO_CLASS_CONFLICT",
                            "visual_class": raw_class,
                            "count": physical_count,
                            "detected_quantity": detected_quantity,
                            "avg_confidence": class_avg_confidence,
                        }
                        if candidate is None or (
                            detected_quantity, class_avg_confidence
                        ) > (
                            int(candidate.get("detected_quantity") or 0),
                            float(candidate.get("avg_confidence") or 0.0),
                        ):
                            candidate_by_code[key] = evidence

                candidates = sorted(
                    candidate_by_code.values(),
                    key=lambda item: (
                        -int(item.get("detected_quantity") or 0),
                        -float(item.get("avg_confidence") or 0.0),
                        str(item.get("product_code") or ""),
                    ),
                )
                return {
                    **base,
                    "decision": "REVIEW",
                    "candidates": candidates,
                    "preferred_product_code": (
                        candidates[0]["product_code"]
                        if len(candidates) == 1
                        else None
                    ),
                    "requires_user_selection": True,
                    "requires_confirmation": True,
                    "message": (
                        "YOLO retained multiple strong visual classes for the same "
                        "tray. The physical object count is preserved, while class "
                        "identity remains under review. Select the correct SKU and "
                        "verify the quantity before creating a KiotViet document."
                    ),
                }
            return {
                **base,
                "decision": "AMBIGUOUS",
                "requires_user_selection": False,
                "requires_confirmation": False,
                "message": (
                    "The image contains conflicting visual classes. "
                    "Retake the image with only one product type, or send it "
                    "for manual review."
                ),
            }

        if mapping["type"] == "family":
            return {
                **base,
                "decision": "FAMILY",
                "members": [dict(member) for member in mapping["members"]],
                "requires_user_selection": True,
                "requires_confirmation": True,
                "message": (
                    "The visual family is reliable, but the exact business SKU "
                    "cannot be determined from appearance alone. Select the SKU."
                ),
            }

        return {
            **base,
            "decision": "DIRECT",
            "product_code": mapping["product_code"],
            "product_name": mapping["product_name"],
            "requires_user_selection": False,
            "requires_confirmation": True,
            "message": "Direct SKU detected. Confirm the product and count.",
        }

    def _remove_edge_class_outliers(
        self,
        objects: list[dict[str, Any]],
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Discard one isolated, weaker class prediction clipped by the frame.

        This is intentionally stricter than ordinary confidence filtering.  It
        only handles the known failure mode where at least two strong objects
        agree, while exactly one object from another class is cut by an image
        edge and is clearly weaker.  Real mixed-SKU evidence remains untouched
        and is still rejected by the normal purity gate.
        """
        if (
            not self.edge_outlier_enabled
            or len(objects) < self.edge_outlier_min_dominant_count + 1
            or image_width <= 0
            or image_height <= 0
        ):
            return objects, []

        by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in objects:
            by_class[str(item["raw_class"])].append(item)
        if len(by_class) < 2:
            return objects, []

        ranked = sorted(
            by_class.items(),
            key=lambda pair: (
                len(pair[1]),
                sum(float(item["confidence"]) for item in pair[1]),
            ),
            reverse=True,
        )
        dominant_class, dominant_objects = ranked[0]
        minority_objects = [
            item
            for item in objects
            if str(item["raw_class"]) != dominant_class
        ]
        if (
            len(dominant_objects) < self.edge_outlier_min_dominant_count
            or len(minority_objects) != 1
        ):
            return objects, []

        candidate = minority_objects[0]
        dominant_min_confidence = min(
            float(item["confidence"]) for item in dominant_objects
        )
        if (
            dominant_min_confidence - float(candidate["confidence"])
            < self.edge_outlier_confidence_gap
        ):
            return objects, []

        x1, y1, x2, y2 = [float(value) for value in candidate["box_xyxy"]]
        margin_x = image_width * self.edge_outlier_margin_ratio
        margin_y = image_height * self.edge_outlier_margin_ratio
        touches_edge = (
            x1 <= margin_x
            or y1 <= margin_y
            or x2 >= image_width - margin_x
            or y2 >= image_height - margin_y
        )
        if not touches_edge:
            return objects, []

        removed = {
            **candidate,
            "removal_reason": "isolated_weaker_class_clipped_by_image_edge",
            "dominant_class": dominant_class,
            "dominant_min_confidence": dominant_min_confidence,
            "confidence_gap": (
                dominant_min_confidence - float(candidate["confidence"])
            ),
        }
        return [item for item in objects if item is not candidate], [removed]

    @staticmethod
    def _products_for_direct_decision(
        decision: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if decision.get("decision") != "DIRECT":
            return []
        return [
            {
                "product_code": str(decision["product_code"]),
                "product_name": str(decision["product_name"]),
                "purchase_price": 0,
                "quantity": int(decision["count"]),
                "confidence_threshold": float(
                    decision["confidence_threshold"]
                ),
                "avg_confidence": float(decision["avg_confidence"]),
                "min_confidence": float(decision["min_confidence"]),
                "max_confidence": float(decision["max_confidence"]),
            }
        ]

    def _draw_annotations(
        self,
        image_bgr: np.ndarray,
        objects: list[dict[str, Any]],
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
            label = str(item["display_name"])
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
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_thickness,
            )
            label_top = max(
                0,
                y1 - text_height - baseline - 2 * padding,
            )
            label_right = min(
                image_width - 1,
                x1 + text_width + 2 * padding,
            )
            cv2.rectangle(
                canvas,
                (x1, label_top),
                (label_right, y1),
                color,
                thickness=-1,
            )
            cv2.putText(
                canvas,
                label,
                (
                    x1 + padding,
                    max(
                        text_height + padding,
                        y1 - padding - baseline,
                    ),
                ),
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
            raise BakeryInferenceError(
                f"Cannot encode annotated image: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.tobytes())
