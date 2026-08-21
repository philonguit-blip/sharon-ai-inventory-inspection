"""Open-vocabulary Foundation fallback for bakery tray counting.

The service deliberately loads SAM2 and DINOv2 lazily.  A normal AUTO request
therefore pays no Foundation startup/RAM cost while the YOLO result is clear.
Reference embeddings are local business data and never leave the workstation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from app.services.hybrid_errors import FoundationInferenceError, FoundationNotReadyError

from app.config import (
    FOUNDATION_DINO_MODEL,
    FOUNDATION_INSTANCE_BOX_COVERAGE_NMS,
    FOUNDATION_INSTANCE_MAX_AREA_FACTOR,
    FOUNDATION_INSTANCE_MIN_AREA_FACTOR,
    FOUNDATION_INSTANCE_SIMILARITY_MARGIN,
    FOUNDATION_INSTANCE_SIMILARITY_THRESHOLD,
    FOUNDATION_MASK_NMS_IOU,
    FOUNDATION_MASK_QUALITY,
    FOUNDATION_MAX_BOX_AREA_RATIO,
    FOUNDATION_MAX_MASK_AREA_RATIO,
    FOUNDATION_EDGE_MARGIN_RATIO,
    FOUNDATION_MIN_MASK_AREA_RATIO,
    FOUNDATION_POINTS_STRIDE,
    FOUNDATION_REFERENCE_PATH,
    FOUNDATION_REGISTRY_PATH,
    FOUNDATION_SAM_MAX_SIDE,
    FOUNDATION_SAM_MODEL_PATH,
    FOUNDATION_SIMILARITY_MARGIN,
    FOUNDATION_SIMILARITY_THRESHOLD,
)


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class DinoEmbeddingEncoder:
    """Small reusable DINOv2 encoder used by inference and reference tooling."""

    def __init__(self, model_name: str = FOUNDATION_DINO_MODEL, device: str = "cpu"):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise FoundationNotReadyError(
                "transformers is not installed; run setup_windows.ps1 again."
            ) from exc
        self.device = device
        # Pin the processor implementation so a future transformers default
        # change cannot silently alter reference/runtime embeddings.
        # Runtime and reference tooling must remain deterministic/offline after
        # setup. Without local_files_only Transformers performs network HEAD
        # retries even when the model is already cached, making auto-crop look
        # like it has stalled on a machine with restricted internet access.
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                use_fast=False,
                local_files_only=True,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=True,
            ).to(device)
        except OSError as exc:
            raise FoundationNotReadyError(
                "DINOv2 is not cached locally; run setup_windows.ps1 once "
                "with internet access before starting the offline runtime."
            ) from exc
        self.model.eval()

    def encode_rgb(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        batches: list[np.ndarray] = []
        for offset in range(0, len(images), max(1, int(batch_size))):
            chunk = images[offset : offset + batch_size]
            inputs = self.processor(images=chunk, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
                vectors = outputs.last_hidden_state[:, 0, :]
                vectors = torch.nn.functional.normalize(vectors, dim=1)
            batches.append(vectors.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(batches, axis=0)


class FoundationInferenceService:
    """Segment tray items with SAM2 and classify crops with private DINO refs."""

    def __init__(
        self,
        *,
        sam_model_path: Path | str = FOUNDATION_SAM_MODEL_PATH,
        reference_path: Path | str = FOUNDATION_REFERENCE_PATH,
        registry_path: Path | str = FOUNDATION_REGISTRY_PATH,
        dino_model: str = FOUNDATION_DINO_MODEL,
        points_stride: int = FOUNDATION_POINTS_STRIDE,
        min_area_ratio: float = FOUNDATION_MIN_MASK_AREA_RATIO,
        max_area_ratio: float = FOUNDATION_MAX_MASK_AREA_RATIO,
        max_box_area_ratio: float = FOUNDATION_MAX_BOX_AREA_RATIO,
        edge_margin_ratio: float = FOUNDATION_EDGE_MARGIN_RATIO,
        mask_nms_iou: float = FOUNDATION_MASK_NMS_IOU,
        mask_quality: float = FOUNDATION_MASK_QUALITY,
        similarity_threshold: float = FOUNDATION_SIMILARITY_THRESHOLD,
        similarity_margin: float = FOUNDATION_SIMILARITY_MARGIN,
        instance_similarity_threshold: float = FOUNDATION_INSTANCE_SIMILARITY_THRESHOLD,
        instance_similarity_margin: float = FOUNDATION_INSTANCE_SIMILARITY_MARGIN,
        instance_min_area_factor: float = FOUNDATION_INSTANCE_MIN_AREA_FACTOR,
        instance_max_area_factor: float = FOUNDATION_INSTANCE_MAX_AREA_FACTOR,
        instance_box_coverage_nms: float = FOUNDATION_INSTANCE_BOX_COVERAGE_NMS,
        sam_max_side: int = FOUNDATION_SAM_MAX_SIDE,
        device: str | None = None,
    ) -> None:
        self.sam_model_path = Path(sam_model_path).expanduser().resolve()
        self.reference_path = Path(reference_path).expanduser().resolve()
        self.registry_path = Path(registry_path).expanduser().resolve()
        self.dino_model = str(dino_model)
        self.points_stride = max(16, int(points_stride))
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.max_box_area_ratio = float(max_box_area_ratio)
        self.edge_margin_ratio = float(edge_margin_ratio)
        self.mask_nms_iou = float(mask_nms_iou)
        self.mask_quality = float(mask_quality)
        self.similarity_threshold = float(similarity_threshold)
        self.similarity_margin = float(similarity_margin)
        self.instance_similarity_threshold = float(instance_similarity_threshold)
        self.instance_similarity_margin = float(instance_similarity_margin)
        self.instance_min_area_factor = max(0.01, float(instance_min_area_factor))
        self.instance_max_area_factor = max(
            self.instance_min_area_factor, float(instance_max_area_factor)
        )
        self.instance_box_coverage_nms = float(
            np.clip(instance_box_coverage_nms, 0.0, 1.0)
        )
        self.sam_max_side = max(64, int(sam_max_side))
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error = ""
        self.sam: Any = None
        self.encoder: DinoEmbeddingEncoder | None = None
        self.reference_embeddings = np.empty((0, 0), dtype=np.float32)
        self.reference_keys = np.empty((0,), dtype=str)
        self.registry: dict[str, Any] = {}
        self._last_segmentation_stats: dict[str, Any] = {}

    def health(self) -> dict[str, Any]:
        dependency_ready = importlib.util.find_spec("transformers") is not None
        assets = {
            "sam_model": self.sam_model_path.is_file(),
            "reference_embeddings": self.reference_path.is_file(),
            "reference_registry": self.registry_path.is_file(),
            "transformers": dependency_ready,
        }
        ready = all(assets.values())
        return {
            "ready": ready,
            "loaded": self._loaded,
            "load_error": self._load_error,
            "device": self.device,
            "sam_model_path": str(self.sam_model_path),
            "reference_path": str(self.reference_path),
            "registry_path": str(self.registry_path),
            "dino_model": self.dino_model,
            "points_stride": self.points_stride,
            "sam_max_side": self.sam_max_side,
            "mask_quality": self.mask_quality,
            "similarity_threshold": self.similarity_threshold,
            "similarity_margin": self.similarity_margin,
            "instance_similarity_threshold": self.instance_similarity_threshold,
            "instance_similarity_margin": self.instance_similarity_margin,
            "instance_area_factors": [
                self.instance_min_area_factor,
                self.instance_max_area_factor,
            ],
            "instance_box_coverage_nms": self.instance_box_coverage_nms,
            "assets": assets,
            "reference_count": int(len(self.reference_keys)) if self._loaded else None,
        }

    def is_ready(self) -> bool:
        return bool(self.health()["ready"])

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.is_ready():
            missing = [
                key for key, present in self.health()["assets"].items() if not present
            ]
            raise FoundationNotReadyError(
                "Foundation engine is not provisioned. Missing: " + ", ".join(missing)
            )
        try:
            from ultralytics import SAM

            payload = np.load(self.reference_path, allow_pickle=False)
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            keys = np.asarray(payload["reference_keys"]).astype(str)
            if embeddings.ndim != 2 or len(embeddings) == 0:
                raise ValueError("reference embeddings are empty")
            if len(keys) != len(embeddings):
                raise ValueError("reference_keys length does not match embeddings")
            registry_payload = json.loads(
                self.registry_path.read_text(encoding="utf-8")
            )
            products = registry_payload.get("products")
            if not isinstance(products, dict):
                raise ValueError("registry.products must be an object")

            self.sam = SAM(str(self.sam_model_path))
            self.encoder = DinoEmbeddingEncoder(self.dino_model, self.device)
            self.reference_embeddings = _normalise_rows(embeddings)
            self.reference_keys = keys
            self.registry = products
            self._loaded = True
            self._load_error = ""
        except Exception as exc:
            self._load_error = str(exc)
            raise FoundationInferenceError(
                f"Cannot load Foundation engine: {exc}"
            ) from exc

    @staticmethod
    def _decode(raw_bytes: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FoundationInferenceError("Uploaded image cannot be decoded.")
        return image

    def _resize_for_sam(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Resize only the SAM branch and return its original-to-SAM scale."""
        height, width = image_bgr.shape[:2]
        longest = max(height, width)
        if longest <= self.sam_max_side:
            return image_bgr, 1.0
        scale = self.sam_max_side / float(longest)
        resized = cv2.resize(
            image_bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _segment(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        assert self.sam is not None
        source_height, source_width = image_bgr.shape[:2]
        sam_image, sam_scale = self._resize_for_sam(image_bgr)
        height, width = sam_image.shape[:2]

        # Stride is expressed in SAM-input pixels. This produces about 11x11
        # prompts for a 1024/1280 input at the default stride of 96.
        grid_x = max(4, int(round(width / self.points_stride)))
        grid_y = max(4, int(round(height / self.points_stride)))
        xs = np.linspace(width / (2 * grid_x), width * (1 - 1 / (2 * grid_x)), grid_x)
        ys = np.linspace(height / (2 * grid_y), height * (1 - 1 / (2 * grid_y)), grid_y)
        points = np.asarray([[[x, y]] for y in ys for x in xs], dtype=np.float32)
        labels = np.ones((len(points), 1), dtype=np.int32)
        results = self.sam.predict(
            source=sam_image,
            points=points,
            labels=labels,
            conf=self.mask_quality,
            device=self.device,
            save=False,
            verbose=False,
        )
        stats: dict[str, Any] = {
            "sam_input_width": int(width),
            "sam_input_height": int(height),
            "sam_scale": float(sam_scale),
            "prompt_count": int(len(points)),
            "raw_masks": 0,
            "basic_filter_kept": 0,
            "duplicates_removed": 0,
            "segments_after_nms": 0,
        }
        if not results or results[0].masks is None:
            self._last_segmentation_stats = stats
            return []
        result = results[0]
        masks = result.masks.data.detach().cpu().numpy() > 0.5
        stats["raw_masks"] = int(len(masks))
        scores = np.ones((len(masks),), dtype=np.float32)
        if result.boxes is not None and len(result.boxes) == len(masks):
            scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

        image_area = float(height * width)
        inverse_scale = 1.0 / max(sam_scale, 1e-12)
        candidates: list[dict[str, Any]] = []
        for mask, score in zip(masks, scores):
            if float(score) < self.mask_quality:
                continue
            mask_area = int(mask.sum())
            ratio = mask_area / image_area
            if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
                continue
            mask_ys, mask_xs = np.where(mask)
            if not len(mask_xs):
                continue
            x1, y1 = int(mask_xs.min()), int(mask_ys.min())
            x2, y2 = int(mask_xs.max() + 1), int(mask_ys.max() + 1)
            box_width, box_height = x2 - x1, y2 - y1
            box_ratio = (box_width * box_height) / image_area
            edge_x = width * self.edge_margin_ratio
            edge_y = height * self.edge_margin_ratio
            if (
                x1 <= edge_x
                or y1 <= edge_y
                or x2 >= width - edge_x
                or y2 >= height - edge_y
                or box_ratio > self.max_box_area_ratio
                or box_width < width * 0.03
                or box_height < height * 0.03
            ):
                continue
            candidates.append(
                {
                    "mask": mask,
                    "score": float(score),
                    # Convert area and boxes to original-image coordinates. DINO
                    # crops and annotations are always made from the original.
                    "area": float(mask_area * inverse_scale * inverse_scale),
                    "box_xyxy": [
                        float(np.clip(x1 * inverse_scale, 0, source_width)),
                        float(np.clip(y1 * inverse_scale, 0, source_height)),
                        float(np.clip(x2 * inverse_scale, 0, source_width)),
                        float(np.clip(y2 * inverse_scale, 0, source_height)),
                    ],
                }
            )
        stats["basic_filter_kept"] = int(len(candidates))

        kept: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            duplicate = False
            for accepted in kept:
                intersection = np.logical_and(candidate["mask"], accepted["mask"]).sum()
                union = np.logical_or(candidate["mask"], accepted["mask"]).sum()
                if union and float(intersection / union) >= self.mask_nms_iou:
                    duplicate = True
                    break
                smaller = min(candidate["area"], accepted["area"])
                scaled_intersection = float(intersection) * inverse_scale * inverse_scale
                if smaller and scaled_intersection / smaller >= 0.70:
                    duplicate = True
                    break
                ax1, ay1, ax2, ay2 = accepted["box_xyxy"]
                bx1, by1, bx2, by2 = candidate["box_xyxy"]
                box_intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
                    0.0, min(ay2, by2) - max(ay1, by1)
                )
                box_union = (
                    (ax2 - ax1) * (ay2 - ay1)
                    + (bx2 - bx1) * (by2 - by1)
                    - box_intersection
                )
                if box_union and box_intersection / box_union >= 0.60:
                    duplicate = True
                    break
            if duplicate:
                stats["duplicates_removed"] += 1
            else:
                kept.append(candidate)
        # The full masks are no longer needed after NMS. Dropping them keeps
        # memory bounded for large mobile uploads.
        for candidate in kept:
            candidate.pop("mask", None)
        kept.sort(key=lambda item: (item["box_xyxy"][1], item["box_xyxy"][0]))
        stats["segments_after_nms"] = int(len(kept))
        self._last_segmentation_stats = stats
        return kept

    @staticmethod
    def _masked_crops(image_bgr: np.ndarray, masks: list[dict[str, Any]]) -> list[np.ndarray]:
        crops: list[np.ndarray] = []
        height, width = image_bgr.shape[:2]
        for item in masks:
            x1, y1, x2, y2 = [float(v) for v in item["box_xyxy"]]
            pad_x = (x2 - x1) * 0.06
            pad_y = (y2 - y1) * 0.06
            x1 = max(0, int(round(x1 - pad_x)))
            y1 = max(0, int(round(y1 - pad_y)))
            x2 = min(width, int(round(x2 + pad_x)))
            y2 = min(height, int(round(y2 + pad_y)))
            crop = image_bgr[y1:y2, x1:x2].copy()
            # Preserve the real local tray/background context. Reference crops
            # are produced from bounding boxes too; replacing the background
            # with a flat colour created a large embedding domain shift.
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return crops

    @staticmethod
    def _expanded_box(
        image_bgr: np.ndarray,
        box_xyxy: list[float] | tuple[float, ...],
        expansion_ratio: float,
    ) -> list[float]:
        """Clamp and expand one YOLO proposal while retaining local context."""
        height, width = image_bgr.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in box_xyxy]
        if x2 <= x1 or y2 <= y1:
            raise FoundationInferenceError("YOLO candidate contains an invalid box.")
        ratio = float(np.clip(expansion_ratio, 0.0, 0.50))
        pad_x = (x2 - x1) * ratio
        pad_y = (y2 - y1) * ratio
        expanded = [
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(width), x2 + pad_x),
            min(float(height), y2 + pad_y),
        ]
        if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
            raise FoundationInferenceError("YOLO candidate falls outside the image.")
        return expanded

    def _prompt_candidate_segments(
        self,
        image_bgr: np.ndarray,
        candidates: list[dict[str, Any]],
        expansion_ratio: float,
    ) -> list[dict[str, Any]]:
        """Refine every YOLO proposal in one prompted SAM2 forward pass.

        Ultralytics preserves each prompt index in ``boxes.cls``. If SAM drops
        an individual prompt, the expanded YOLO crop remains usable by DINO;
        this avoids converting one missing mask into an all-image failure.
        """
        expanded_boxes = [
            self._expanded_box(
                image_bgr,
                list(candidate.get("box_xyxy") or []),
                expansion_ratio,
            )
            for candidate in candidates
        ]
        segments = [
            {
                "box_xyxy": box,
                "area": max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]),
                "score": 0.0,
                "prompt_index": index,
                "mask_refined": False,
            }
            for index, box in enumerate(expanded_boxes)
        ]
        if not expanded_boxes:
            return segments

        resized, scale = self._resize_for_sam(image_bgr)
        prompted_boxes = [
            [float(value) * scale for value in box] for box in expanded_boxes
        ]
        assert self.sam is not None
        results = self.sam.predict(
            source=resized,
            bboxes=prompted_boxes,
            device=self.device,
            verbose=False,
            retina_masks=True,
            conf=0.0,
        )
        if not results or results[0].boxes is None:
            return segments

        boxes = results[0].boxes
        refined_values = boxes.xyxy.detach().cpu().numpy().astype(float)
        scores = boxes.conf.detach().cpu().numpy().astype(float)
        prompt_indices = boxes.cls.detach().cpu().numpy().astype(int)
        height, width = image_bgr.shape[:2]
        inverse_scale = 1.0 / max(scale, 1e-12)
        for refined, score, prompt_index in zip(
            refined_values,
            scores,
            prompt_indices,
        ):
            if prompt_index < 0 or prompt_index >= len(segments):
                continue
            x1, y1, x2, y2 = [float(value) * inverse_scale for value in refined]
            refined_box = [
                max(0.0, min(float(width), x1)),
                max(0.0, min(float(height), y1)),
                max(0.0, min(float(width), x2)),
                max(0.0, min(float(height), y2)),
            ]
            if refined_box[2] <= refined_box[0] or refined_box[3] <= refined_box[1]:
                continue
            segments[prompt_index] = {
                "box_xyxy": refined_box,
                "area": (
                    (refined_box[2] - refined_box[0])
                    * (refined_box[3] - refined_box[1])
                ),
                "score": float(score),
                "prompt_index": prompt_index,
                "mask_refined": True,
            }
        return segments

    def _similarities(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = _normalise_rows(embeddings)
        return embeddings @ self.reference_embeddings.T

    def _classify_instances_from_similarities(
        self,
        similarities: np.ndarray,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Classify independent prompted crops without tray-level voting."""
        keys = sorted(set(self.reference_keys.tolist()))
        if not keys or similarities.size == 0:
            count = int(similarities.shape[0])
            return (
                [""] * count,
                np.zeros((count,), dtype=np.float32),
                np.zeros((count,), dtype=np.float32),
            )
        class_scores = np.column_stack(
            [
                similarities[:, np.where(self.reference_keys == key)[0]].max(axis=1)
                for key in keys
            ]
        )
        best_indices = class_scores.argmax(axis=1)
        best_scores = class_scores[np.arange(len(class_scores)), best_indices]
        if class_scores.shape[1] > 1:
            runner_up = np.partition(class_scores, -2, axis=1)[:, -2]
        else:
            runner_up = np.full_like(best_scores, -1.0)
        return (
            [keys[int(index)] for index in best_indices],
            best_scores.astype(np.float32),
            (best_scores - runner_up).astype(np.float32),
        )

    def verify_yolo_candidates(
        self,
        raw_bytes: bytes,
        candidates: list[dict[str, Any]],
        *,
        expansion_ratio: float = 0.12,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        """Verify low-confidence YOLO boxes with one SAM2 pass and batched DINO.

        A candidate is accepted only when Foundation independently selects the
        same raw class and clears both the per-instance similarity and margin
        gates. A confident alternative class is reported as disagreement and
        is never silently substituted into the YOLO count.
        """
        started = time.perf_counter()
        if not candidates:
            return {
                "attempted": 0,
                "accepted": 0,
                "disagreements": 0,
                "verdicts": [],
                "timings_ms": {"total": 0.0},
            }
        image_bgr = self._decode(raw_bytes)
        decoded_at = time.perf_counter()
        with self._lock:
            self._ensure_loaded()
            sam_started = time.perf_counter()
            segments = self._prompt_candidate_segments(
                image_bgr,
                candidates,
                expansion_ratio,
            )
            sam_finished = time.perf_counter()
            assert self.encoder is not None
            embeddings = self.encoder.encode_rgb(
                self._masked_crops(image_bgr, segments),
                batch_size=max(1, int(batch_size)),
            )
            encoded_at = time.perf_counter()
            similarities = self._similarities(embeddings)
            keys, scores, margins = self._classify_instances_from_similarities(
                similarities
            )

        verdicts: list[dict[str, Any]] = []
        accepted = 0
        disagreements = 0
        for index, (candidate, segment, key, score, margin) in enumerate(
            zip(candidates, segments, keys, scores, margins, strict=True)
        ):
            expected = str(candidate.get("raw_class") or "")
            semantically_safe = bool(
                float(score) >= self.instance_similarity_threshold
                and float(margin) >= self.instance_similarity_margin
                and key in self.registry
            )
            agrees = bool(semantically_safe and key == expected)
            disagrees = bool(semantically_safe and key != expected)
            if agrees:
                status = "ACCEPTED"
                accepted += 1
            elif disagrees:
                status = "CLASS_DISAGREEMENT"
                disagreements += 1
            else:
                status = "LOW_SIMILARITY"
            registry_item = dict(self.registry.get(key) or {})
            verdicts.append(
                {
                    "candidate_index": index,
                    "status": status,
                    "accepted": agrees,
                    "class_disagreement": disagrees,
                    "expected_raw_class": expected,
                    "foundation_raw_class": key,
                    "foundation_similarity": float(score),
                    "foundation_margin": float(margin),
                    "similarity_threshold": self.instance_similarity_threshold,
                    "margin_threshold": self.instance_similarity_margin,
                    "refined_box_xyxy": list(segment["box_xyxy"]),
                    "sam_score": float(segment.get("score") or 0.0),
                    "mask_refined": bool(segment.get("mask_refined")),
                    "foundation_product_code": registry_item.get("product_code"),
                    "foundation_product_name": registry_item.get("product_name"),
                    "candidate": dict(candidate),
                }
            )
        finished = time.perf_counter()
        return {
            "attempted": len(candidates),
            "accepted": accepted,
            "disagreements": disagreements,
            "verdicts": verdicts,
            "timings_ms": {
                "decode": (decoded_at - started) * 1000.0,
                "sam_prompted": (sam_finished - sam_started) * 1000.0,
                "dino_batched": (encoded_at - sam_finished) * 1000.0,
                "total": (finished - started) * 1000.0,
            },
        }

    def _classify_tray_from_similarities(
        self, similarities: np.ndarray
    ) -> tuple[str, float, float]:
        scores: list[tuple[str, float]] = []
        for key in sorted(set(self.reference_keys.tolist())):
            positions = np.where(self.reference_keys == key)[0]
            # Best matching reference for each instance, then robust tray median.
            per_instance = similarities[:, positions].max(axis=1)
            scores.append((key, float(np.median(per_instance))))
        scores.sort(key=lambda item: item[1], reverse=True)
        best_key, best_score = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else -1.0
        return best_key, best_score, best_score - runner_up

    def _classify_tray(self, embeddings: np.ndarray) -> tuple[str, float, float]:
        """Compatibility wrapper used by the Foundation diagnostic app."""
        return self._classify_tray_from_similarities(self._similarities(embeddings))

    def _instance_semantic_scores(
        self, similarities: np.ndarray, reference_key: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate each SAM candidate against the selected tray SKU.

        A candidate must resemble the target SKU and beat its best alternative
        by a small margin. This is the primary protection against reflective
        tray/background masks being counted as products.
        """
        target_positions = np.where(self.reference_keys == reference_key)[0]
        if not len(target_positions):
            count = similarities.shape[0]
            return (
                np.full((count,), -1.0, dtype=np.float32),
                np.full((count,), -1.0, dtype=np.float32),
                np.zeros((count,), dtype=bool),
            )
        target_scores = similarities[:, target_positions].max(axis=1)
        other_positions = np.where(self.reference_keys != reference_key)[0]
        if len(other_positions):
            other_scores = similarities[:, other_positions].max(axis=1)
        else:
            other_scores = np.full_like(target_scores, -1.0)
        margins = target_scores - other_scores
        keep = np.logical_and(
            target_scores >= self.instance_similarity_threshold,
            margins >= self.instance_similarity_margin,
        )
        return target_scores, margins, keep

    def _robust_area_keep(
        self, segments: list[dict[str, Any]], semantic_keep: np.ndarray
    ) -> tuple[np.ndarray, float | None]:
        """Apply a secondary robust size gate after semantic validation."""
        keep = np.asarray(semantic_keep, dtype=bool).copy()
        indices = np.flatnonzero(keep)
        if not len(indices):
            return keep, None
        areas = np.asarray([float(segments[index]["area"]) for index in indices])
        median_area = float(np.median(areas))
        if median_area <= 0:
            return keep, median_area
        minimum = median_area * self.instance_min_area_factor
        maximum = median_area * self.instance_max_area_factor
        for index in indices:
            area = float(segments[index]["area"])
            if area < minimum or area > maximum:
                keep[index] = False
        return keep, median_area

    def _semantic_duplicate_keep(
        self,
        segments: list[dict[str, Any]],
        instance_scores: np.ndarray,
        candidate_keep: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Suppress overlapping semantic candidates, retaining the best match.

        SAM can describe one product with a tight mask and a second, much wider
        partial/background mask. Mask IoU may be low even when their boxes cover
        the same product, so this check runs after DINO and prefers the stronger
        semantic candidate. Production photos are specified as non-occluded.
        """
        keep = np.zeros((len(segments),), dtype=bool)
        accepted: list[int] = []
        ordered = sorted(
            np.flatnonzero(candidate_keep).tolist(),
            key=lambda index: float(instance_scores[index]),
            reverse=True,
        )
        removed = 0
        for index in ordered:
            x1, y1, x2, y2 = [float(value) for value in segments[index]["box_xyxy"]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            duplicate = False
            for accepted_index in accepted:
                ax1, ay1, ax2, ay2 = [
                    float(value) for value in segments[accepted_index]["box_xyxy"]
                ]
                accepted_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                intersection = max(0.0, min(x2, ax2) - max(x1, ax1)) * max(
                    0.0, min(y2, ay2) - max(y1, ay1)
                )
                smaller_box = min(area, accepted_area)
                if (
                    smaller_box > 0
                    and intersection / smaller_box
                    >= self.instance_box_coverage_nms
                ):
                    duplicate = True
                    break
            if duplicate:
                removed += 1
            else:
                keep[index] = True
                accepted.append(index)
        return keep, removed

    def infer_bytes(
        self,
        raw_bytes: bytes,
        *,
        image_name: str,
        annotated_path: Path | str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        image_bgr = self._decode(raw_bytes)
        decoded_at = time.perf_counter()
        with self._lock:
            self._ensure_loaded()
            segmentation_started = time.perf_counter()
            segments = self._segment(image_bgr)
            segmentation_finished = time.perf_counter()
            segmentation_stats = dict(self._last_segmentation_stats)
            if segments:
                assert self.encoder is not None
                embeddings = self.encoder.encode_rgb(
                    self._masked_crops(image_bgr, segments)
                )
                encoded_at = time.perf_counter()
                similarities = self._similarities(embeddings)
                reference_key, similarity, margin = (
                    self._classify_tray_from_similarities(similarities)
                )
                instance_scores, instance_margins, semantic_keep = (
                    self._instance_semantic_scores(similarities, reference_key)
                )
                area_keep, median_area = self._robust_area_keep(
                    segments, semantic_keep
                )
                final_keep, semantic_duplicates_removed = (
                    self._semantic_duplicate_keep(
                        segments, instance_scores, area_keep
                    )
                )
            else:
                encoded_at = segmentation_finished
                reference_key, similarity, margin = "", 0.0, 0.0
                instance_scores = np.empty((0,), dtype=np.float32)
                instance_margins = np.empty((0,), dtype=np.float32)
                semantic_keep = np.empty((0,), dtype=bool)
                area_keep = np.empty((0,), dtype=bool)
                final_keep = np.empty((0,), dtype=bool)
                median_area = None
                semantic_duplicates_removed = 0

        registry_item = dict(self.registry.get(reference_key) or {})
        display_name = str(
            registry_item.get("display_name")
            or registry_item.get("product_name")
            or reference_key
            or "Unknown"
        )
        tray_confident = bool(
            segments
            and similarity >= self.similarity_threshold
            and margin >= self.similarity_margin
            and registry_item
        )
        objects: list[dict[str, Any]] = []
        for index in np.flatnonzero(final_keep).tolist():
            segment = segments[index]
            instance_confidence = float(np.clip(instance_scores[index], 0.0, 1.0))
            item = {
                "class_id": -1,
                "raw_class": reference_key,
                "mapping_type": str(registry_item.get("type") or "direct"),
                "display_name": display_name,
                "confidence_threshold": self.instance_similarity_threshold,
                "confidence": instance_confidence,
                "box_xyxy": segment["box_xyxy"],
                "foundation_mask_index": index,
                "foundation_instance_margin": float(instance_margins[index]),
                "foundation_tray_similarity": float(similarity),
            }
            if registry_item.get("product_code"):
                item.update(
                    {
                        "product_code": registry_item["product_code"],
                        "product_name": registry_item.get("product_name", display_name),
                        "purchase_price": 0.0,
                    }
                )
            objects.append(item)

        object_confidences = [float(item["confidence"]) for item in objects]
        mapping_type = str(registry_item.get("type") or "direct").lower()

        # A Foundation result has two independent safety layers:
        #
        # 1) tray-level class separation (similarity + top-two margin);
        # 2) per-instance semantic validation for every retained SAM object.
        #
        # A weak tray-level margin must continue to block DIRECT/FAMILY, but it
        # should not discard an otherwise valid object count.  When the selected
        # reference is registered and every retained object already passed the
        # per-instance semantic/area/duplicate filters, expose the result as
        # REVIEW so the operator can verify the SKU and quantity explicitly.
        review_candidates: list[dict[str, Any]] = []
        if registry_item and objects:
            if mapping_type == "family":
                for member in registry_item.get("members") or []:
                    if not isinstance(member, dict):
                        continue
                    product_code = str(member.get("product_code") or "").strip()
                    if not product_code:
                        continue
                    review_candidates.append(
                        {
                            "source": "FOUNDATION_TRAY_REVIEW",
                            "product_code": product_code,
                            "product_name": str(
                                member.get("product_name")
                                or member.get("display_name")
                                or product_code
                            ),
                            "count": len(objects),
                        }
                    )
            else:
                product_code = str(registry_item.get("product_code") or "").strip()
                if product_code:
                    review_candidates.append(
                        {
                            "source": "FOUNDATION_TRAY_REVIEW",
                            "product_code": product_code,
                            "product_name": str(
                                registry_item.get("product_name")
                                or registry_item.get("display_name")
                                or product_code
                            ),
                            "count": len(objects),
                        }
                    )

        if not segments:
            decision = {
                "decision": "NO_DETECTION",
                "count": 0,
                "purity": 0.0,
                "avg_confidence": 0.0,
                "requires_confirmation": False,
                "message": "Foundation engine could not isolate bakery items.",
            }
        elif not objects:
            decision = {
                "decision": "AMBIGUOUS",
                "dominant_class": reference_key,
                "display_name": display_name,
                "count": 0,
                "total_detections": 0,
                "purity": 0.0,
                "avg_confidence": 0.0,
                "tray_similarity": float(similarity),
                "similarity_margin": float(margin),
                "requires_confirmation": False,
                "message": (
                    "Foundation identified the tray SKU, but per-instance semantic "
                    "validation rejected every SAM mask. Manual review is required."
                ),
            }
        elif not tray_confident and review_candidates:
            decision = {
                "decision": "REVIEW",
                "dominant_class": reference_key,
                "display_name": display_name,
                "count": len(objects),
                "total_detections": len(objects),
                "purity": 1.0,
                "avg_confidence": float(np.mean(object_confidences)),
                "min_confidence": float(np.min(object_confidences)),
                "max_confidence": float(np.max(object_confidences)),
                "confidence_threshold": self.instance_similarity_threshold,
                "instance_similarity_margin": self.instance_similarity_margin,
                "tray_similarity": float(similarity),
                "similarity_margin": float(margin),
                "tray_similarity_threshold": self.similarity_threshold,
                "tray_similarity_margin_threshold": self.similarity_margin,
                "candidates": review_candidates,
                "preferred_source": "FOUNDATION",
                "preferred_product_code": (
                    review_candidates[0]["product_code"]
                    if len(review_candidates) == 1
                    else None
                ),
                "requires_user_selection": True,
                "requires_confirmation": True,
                "message": (
                    "Foundation retained valid per-instance product detections, "
                    "but the tray-level class separation did not clear the automatic "
                    "safety gate. Verify the SKU and quantity before creating the "
                    "KiotViet document."
                ),
            }
        elif not tray_confident:
            # Keep the original hard-stop behaviour when Foundation cannot map
            # the weak tray-level result to a safe closed-catalog candidate.
            decision = {
                "decision": "AMBIGUOUS",
                "dominant_class": reference_key,
                "display_name": display_name,
                "count": len(objects),
                "total_detections": len(objects),
                "purity": 1.0,
                "avg_confidence": float(max(0.0, min(1.0, similarity))),
                "tray_similarity": float(similarity),
                "similarity_margin": float(margin),
                "requires_confirmation": False,
                "message": (
                    "Foundation similarity is below the safe threshold and no "
                    "confirmable catalog candidate could be produced. Manual "
                    "retake/review is required."
                ),
            }
        else:
            decision = {
                "decision": "FAMILY" if mapping_type == "family" else "DIRECT",
                "dominant_class": reference_key,
                "display_name": display_name,
                "product_code": registry_item.get("product_code"),
                "product_name": registry_item.get("product_name", display_name),
                "members": list(registry_item.get("members") or []),
                "count": len(objects),
                "total_detections": len(objects),
                "purity": 1.0,
                "avg_confidence": float(np.mean(object_confidences)),
                "min_confidence": float(np.min(object_confidences)),
                "max_confidence": float(np.max(object_confidences)),
                "confidence_threshold": self.instance_similarity_threshold,
                "instance_similarity_margin": self.instance_similarity_margin,
                "tray_similarity": float(similarity),
                "similarity_margin": float(margin),
                "requires_user_selection": mapping_type == "family",
                "requires_confirmation": True,
                "message": (
                    "Foundation tray and per-instance reference matches are ready "
                    "for operator confirmation."
                ),
            }

        saved_annotation: str | None = None
        if annotated_path is not None:
            annotated = image_bgr.copy()
            for object_index, item in enumerate(objects, start=1):
                x1, y1, x2, y2 = [int(v) for v in item["box_xyxy"]]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 133, 86), 2)
                cv2.putText(
                    annotated,
                    f"{object_index}: {display_name[:28]} {item['confidence']:.2f}",
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 90, 58),
                    2,
                    cv2.LINE_AA,
                )
            output = Path(annotated_path).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise FoundationInferenceError(f"Cannot write annotation: {output}")
            saved_annotation = str(output)

        height, width = image_bgr.shape[:2]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        products = []
        if decision["decision"] == "DIRECT":
            products = [{
                "product_code": decision["product_code"],
                "product_name": decision["product_name"],
                "purchase_price": 0,
                "quantity": decision["count"],
            }]
        return {
            "image_name": image_name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "width": int(width),
            "height": int(height),
            "raw_detections_before_class_filter": len(segments),
            "detections_removed_by_class_filter": int(len(segments) - len(objects)),
            "detections_removed_as_duplicates": int(
                segmentation_stats.get("duplicates_removed", 0)
            ) + int(semantic_duplicates_removed),
            "total_detections": len(objects),
            "confidence_sum": float(sum(item["confidence"] for item in objects)),
            "avg_confidence": float(decision.get("avg_confidence") or 0.0),
            "inference_ms": elapsed_ms,
            "decision": decision,
            "products": products,
            "objects": objects,
            "annotated_path": saved_annotation,
            "foundation_filtering": {
                "tray_reference_key": reference_key,
                "tray_similarity": float(similarity),
                "tray_margin": float(margin),
                "sam_segments": int(len(segments)),
                "semantic_kept": int(np.count_nonzero(semantic_keep)),
                "semantic_removed": int(len(segments) - np.count_nonzero(semantic_keep)),
                "geometry_kept": int(np.count_nonzero(area_keep)),
                "geometry_removed": int(
                    np.count_nonzero(semantic_keep) - np.count_nonzero(area_keep)
                ),
                "semantic_duplicates_removed": int(semantic_duplicates_removed),
                "final_kept": int(np.count_nonzero(final_keep)),
                "median_instance_area": median_area,
                "instance_similarity_threshold": self.instance_similarity_threshold,
                "instance_similarity_margin": self.instance_similarity_margin,
                "instance_area_factors": [
                    self.instance_min_area_factor,
                    self.instance_max_area_factor,
                ],
                "instance_box_coverage_nms": self.instance_box_coverage_nms,
                "segmentation": segmentation_stats,
            },
            "timings_ms": {
                "decode": (decoded_at - started) * 1000.0,
                "sam": (segmentation_finished - segmentation_started) * 1000.0,
                "dino": (encoded_at - segmentation_finished) * 1000.0,
                "total": elapsed_ms,
            },
            "engine": "FOUNDATION",
            "status": "SUCCESS",
            "error": "",
        }
