"""Authenticated developer-only runtime controls for detector configuration."""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.config import DEVELOPER_SETTINGS_KEY
from app.services.bakery_inference_service import BakeryInferenceService
from app.services.developer_settings_service import (
    DeveloperSettingsError,
    DeveloperSettingsService,
)
from app.services.hybrid_inference_service import HybridInferenceService


router = APIRouter(
    prefix="/api/v1/bakery/developer",
    tags=["Bakery Developer Settings"],
    include_in_schema=False,
)


class DeveloperAccess(BaseModel):
    developer_key: str = Field(min_length=1, max_length=256)


class DeveloperSettingsUpdate(DeveloperAccess):
    active_model: str = Field(min_length=1, max_length=260)
    thresholds: dict[str, float]


def _authorise(provided: str) -> None:
    if not DEVELOPER_SETTINGS_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Developer settings are disabled. Configure DEVELOPER_SETTINGS_KEY.",
        )
    if not hmac.compare_digest(str(provided), DEVELOPER_SETTINGS_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Developer key is invalid.",
        )


def _settings_service(request: Request) -> DeveloperSettingsService:
    service = getattr(request.app.state, "developer_settings_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Developer settings are unavailable.")
    return service


def _current_yolo(request: Request) -> BakeryInferenceService:
    inference = getattr(request.app.state, "bakery_inference_service", None)
    yolo = getattr(inference, "yolo", None)
    if yolo is None:
        raise HTTPException(status_code=503, detail="YOLO service is unavailable.")
    return yolo


def _snapshot(request: Request) -> dict[str, Any]:
    settings = _settings_service(request)
    yolo = _current_yolo(request)
    persisted = settings.load()
    return {
        "ready": True,
        "active_model": settings._relative_model_name(Path(yolo.model_path)),
        "available_models": settings.list_models(),
        "classes": [
            {
                "raw_class": item["raw_class"],
                "display_name": item["display_name"],
                "confidence_threshold": float(item["confidence_threshold"]),
            }
            for item in yolo.mapping.class_settings()
        ],
        "updated_at": persisted.get("updated_at"),
        "note": "Changes apply to new jobs; jobs already running keep their original model.",
    }


@router.post("/settings/query")
def query_developer_settings(
    payload: DeveloperAccess,
    request: Request,
) -> dict[str, Any]:
    _authorise(payload.developer_key)
    return _snapshot(request)


@router.put("/settings")
def update_developer_settings(
    payload: DeveloperSettingsUpdate,
    request: Request,
) -> dict[str, Any]:
    _authorise(payload.developer_key)
    settings = _settings_service(request)
    current = _current_yolo(request)
    lock = request.app.state.developer_settings_lock

    try:
        model_path = settings.resolve_model(payload.active_model)
        thresholds = {str(key): float(value) for key, value in payload.thresholds.items()}
        if any(not 0.0 < value <= 1.0 for value in thresholds.values()):
            raise DeveloperSettingsError("Thresholds must be greater than 0 and at most 1.")

        with lock:
            # Constructing the candidate performs model loading, model-class to
            # SKU validation and threshold validation before touching live state.
            candidate_yolo = BakeryInferenceService(
                model_path=model_path,
                mapping_path=current.mapping_path,
                confidence=current.confidence,
                image_size=current.image_size,
                iou=current.iou,
                max_detections=current.max_detections,
                duplicate_iou=current.duplicate_iou,
                cross_class_duplicate_coverage=(
                    current.cross_class_duplicate_coverage
                ),
                min_purity=current.min_purity,
                conflict_review_min_dominant_count=(
                    current.conflict_review_min_dominant_count
                ),
                edge_outlier_enabled=current.edge_outlier_enabled,
                edge_outlier_margin_ratio=current.edge_outlier_margin_ratio,
                edge_outlier_confidence_gap=(
                    current.edge_outlier_confidence_gap
                ),
                edge_outlier_min_dominant_count=(
                    current.edge_outlier_min_dominant_count
                ),
                device=current.device,
                show_confidence=current.show_confidence,
                line_width=current.line_width,
                confidence_overrides=thresholds,
            )
            old_hybrid = request.app.state.bakery_inference_service
            candidate_hybrid = HybridInferenceService(
                candidate_yolo,
                old_hybrid.foundation,
                enabled=old_hybrid.enabled,
                default_mode=old_hybrid.default_mode,
                fallback_margin=old_hybrid.fallback_margin,
                box_rescue_enabled=old_hybrid.box_rescue_enabled,
                box_rescue_min_threshold_ratio=(
                    old_hybrid.box_rescue_min_threshold_ratio
                ),
                box_rescue_min_confidence=old_hybrid.box_rescue_min_confidence,
                box_rescue_max_candidates=old_hybrid.box_rescue_max_candidates,
                box_rescue_expansion_ratio=old_hybrid.box_rescue_expansion_ratio,
                box_rescue_existing_coverage=(
                    old_hybrid.box_rescue_existing_coverage
                ),
            )
            settings.save(model_path=model_path, thresholds=thresholds)
            request.app.state.bakery_inference_service = candidate_hybrid
            request.app.state.product_mapping_service = candidate_yolo.mapping
            request.app.state.bakery_startup_error = ""
    except (DeveloperSettingsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Model/configuration validation failed: {exc}",
        ) from exc

    return _snapshot(request)
