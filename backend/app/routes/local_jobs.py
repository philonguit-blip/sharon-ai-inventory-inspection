"""Local asynchronous job API for bakery image counting and Excel export."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import (
    INFERENCE_BATCH_SIZE,
    KIOTVIET_AUTO_CREATE_DRAFT,
    MAX_FILE_SIZE_BYTES,
    MAX_IMAGES_PER_JOB,
    MAX_JOB_UPLOAD_SIZE_BYTES,
    R2_TRANSFER_WORKERS,
    RUNTIME_PATH,
)
from app.schemas.jobs import (
    BakeryHealthResponse,
    CreateR2JobRequest,
    JobAccepted,
    JobStatusResponse,
    KiotVietSubmitRequest,
    PresignUploadsRequest,
    PresignUploadsResponse,
)
from app.services.product_mapping_service import ProductMappingError
if TYPE_CHECKING:
    from app.services.bakery_inference_service import BakeryInferenceService
    from app.services.excel_service import ExcelService
    from app.services.kiotviet_service import KiotVietService
    from app.services.storage_service import R2StorageService
else:
    BakeryInferenceService = Any
    ExcelService = Any
    KiotVietService = Any
    R2StorageService = Any


router = APIRouter(prefix="/api/v1/bakery", tags=["Bakery Counting"])

JOBS_ROOT = (RUNTIME_PATH / "jobs").resolve()
JOBS_ROOT.mkdir(parents=True, exist_ok=True)
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SAFE_STEM_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
KIOTVIET_SUBMIT_LOCK = threading.Lock()
UPLOAD_URL_EXPIRES_SECONDS = 900

# KiotViet may return non-standard 4xx validation responses (notably HTTP 420).
# These responses prove that KiotViet received and rejected the write, so retrying
# the same job is safe after the underlying payload/configuration is corrected.
# Network errors and 5xx responses remain uncertain and must still reconcile
# before any second write is attempted.
DEFINITIVE_KIOTVIET_REJECTION_STATUSES = {400, 401, 403, 404, 409, 420, 422, 429}


def _kiotviet_error_status(value: Any) -> int | None:
    status_code = getattr(value, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    match = re.search(r"\bHTTP\s+(\d{3})\b", str(value or ""), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _kiotviet_failure_is_retry_safe(value: Any) -> bool:
    status_code = _kiotviet_error_status(value)
    return status_code in DEFINITIVE_KIOTVIET_REJECTION_STATUSES

# Batch recovery is intentionally contextual: normal per-image thresholds stay
# unchanged. Rescue is attempted only when multiple valid images in the same
# user-declared same-SKU job strongly agree on one visual class.
BATCH_RESCUE_ENABLED = os.getenv("BAKERY_BATCH_RESCUE_ENABLED", "true").strip().lower() not in {
    "0", "false", "no", "off"
}
BATCH_RESCUE_MIN_VALID_IMAGES = max(2, int(os.getenv("BAKERY_BATCH_RESCUE_MIN_VALID_IMAGES", "2")))
BATCH_RESCUE_MIN_IMAGE_AGREEMENT = float(
    os.getenv("BAKERY_BATCH_RESCUE_MIN_IMAGE_AGREEMENT", "0.75")
)


class ConfirmJobRequest(BaseModel):
    """Explicit user confirmation before any KiotViet write."""

    confirm: bool = True
    product_code: str | None = None
    quantity: int | None = Field(default=None, ge=1, le=5000)
    document_type: Literal["PURCHASE_RECEIPT", "MANUFACTURING"] = "PURCHASE_RECEIPT"



def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _job_directory(job_id: str) -> Path:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    candidate = (JOBS_ROOT / job_id).resolve()
    if candidate.parent != JOBS_ROOT:
        raise HTTPException(status_code=404, detail="Job not found.")
    return candidate


def _job_state_path(job_id: str) -> Path:
    return _job_directory(job_id) / "job.json"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover_interrupted_jobs(jobs_root: Path = JOBS_ROOT) -> int:
    """Mark jobs whose background task was lost during a backend restart."""
    recovered = 0
    if not jobs_root.is_dir():
        return recovered
    for state_path in jobs_root.glob("*/job.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if str(state.get("status") or "").upper() not in {"QUEUED", "PROCESSING"}:
                continue
            state.update(
                {
                    "status": "ERROR",
                    "updated_at": _now_iso(),
                    "error": (
                        "Backend restarted before this job completed. "
                        "Please submit the image batch again."
                    ),
                }
            )
            _write_json_atomic(state_path, state)
            recovered += 1
        except (OSError, ValueError, TypeError):
            continue
    return recovered


def _read_job(job_id: str) -> dict[str, Any]:
    state_path = _job_state_path(job_id)
    if not state_path.is_file():
        raise HTTPException(status_code=404, detail="Job not found.")
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail="The job state cannot be read."
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="The job state is invalid.")
    return payload


def _service_from_app(request: Request, name: str) -> Any:
    service = getattr(request.app.state, name, None)
    if service is None:
        error = getattr(request.app.state, "bakery_startup_error", "")
        detail = "Bakery pipeline is not ready."
        if error:
            detail += f" {error}"
        raise HTTPException(status_code=503, detail=detail)
    return service


def _product_catalog(request: Request) -> list[dict[str, Any]]:
    service = getattr(request.app.state, "product_mapping_service", None)
    if service is None:
        return []
    return service.all_business_products()


def _job_response(state: dict[str, Any], request: Request) -> dict[str, Any]:
    response = dict(state)
    response["product_catalog"] = _product_catalog(request)
    return response


def _safe_upload_name(index: int, original_name: str) -> str:
    path_name = Path(original_name).name
    extension = Path(path_name).suffix.lower()
    stem = Path(path_name).stem
    safe_stem = SAFE_STEM_PATTERN.sub("_", stem).strip("._-") or "image"
    return f"{index:03d}_{safe_stem[:80]}{extension}"


def _remove_incomplete_job(job_directory: Path) -> None:
    resolved = job_directory.resolve()
    if resolved.parent == JOBS_ROOT and resolved.exists():
        shutil.rmtree(resolved)


def _initial_job_state(
    job_id: str,
    total_images: int,
    inference_mode: str = "AUTO",
) -> dict[str, Any]:
    now = _now_iso()
    return {
        "job_id": job_id,
        "status": "QUEUED",
        "created_at": now,
        "updated_at": now,
        "total_images": total_images,
        "processed_images": 0,
        "inference_mode": str(inference_mode).upper(),
        "error": "",
        "message": "",
        "product_count": 0,
        "total_quantity": 0,
        "products": [],
        "images": [],
        "decision": None,
        "confirmed_product": None,
        "excel_filename": None,
        "excel_url": None,
        "excel_error": "",
        "r2_objects": [],
        # Receipt creation is intentionally deferred until explicit confirmation.
        "kiotviet": None,
        "confirmation_retry_safe": False,
        "pseudo_label": None,
    }

def _validate_upload_metadata(files: list[Any]) -> None:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    if len(files) > MAX_IMAGES_PER_JOB:
        raise HTTPException(
            status_code=400,
            detail=f"A job accepts at most {MAX_IMAGES_PER_JOB} images of the same SKU.",
        )

    total_size = 0
    for item in files:
        filename = Path(str(item.filename)).name
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format for {filename}: {extension or 'none'}.",
            )
        if int(item.size_bytes) > MAX_FILE_SIZE_BYTES:
            limit_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the {limit_mb:g} MB limit: {filename}.",
            )
        total_size += int(item.size_bytes)

    if total_size > MAX_JOB_UPLOAD_SIZE_BYTES:
        total_limit_mb = MAX_JOB_UPLOAD_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {total_limit_mb:g} MB job limit.",
        )

def _manifest_path(job_id: str) -> Path:
    return _job_directory(job_id) / "upload_manifest.json"


def _read_upload_manifest(job_id: str) -> dict[str, Any]:
    path = _manifest_path(job_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Upload session not found.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Upload session is invalid.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise HTTPException(status_code=500, detail="Upload session is invalid.")
    return payload


def _download_r2_uploads(
    job_id: str,
    requested_keys: list[str],
    storage_service: R2StorageService,
) -> list[dict[str, str]]:
    manifest = _read_upload_manifest(job_id)
    expected = _validate_r2_object_keys(manifest, requested_keys)

    original_directory = _job_directory(job_id) / "original"
    if original_directory.exists():
        shutil.rmtree(original_directory)
    original_directory.mkdir(parents=True, exist_ok=False)
    saved: list[dict[str, str]] = []
    try:
        def download_one(object_key: str) -> dict[str, str]:
            item = expected[object_key]
            info = storage_service.object_info(object_key)
            expected_size = int(item["size_bytes"])
            if info["size"] != expected_size:
                raise HTTPException(
                    status_code=409,
                    detail=f"R2 upload size mismatch: {item['filename']}.",
                )
            destination = original_directory / str(item["safe_name"])
            storage_service.download_file(object_key, destination)
            digest = _sha256_file(destination)
            return {
                "display_name": str(item["filename"]),
                "safe_name": str(item["safe_name"]),
                "path": str(destination),
                "sha256": digest,
            }

        worker_count = min(R2_TRANSFER_WORKERS, len(requested_keys))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            saved = list(executor.map(download_one, requested_keys))

        hashes: set[str] = set()
        for item in saved:
            digest = item["sha256"]
            if digest in hashes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Duplicate image in the same job: {item['display_name']}.",
                )
            hashes.add(digest)
    except Exception:
        if original_directory.exists():
            shutil.rmtree(original_directory)
        raise
    return saved


def _validate_r2_object_keys(
    manifest: dict[str, Any], requested_keys: list[str]
) -> dict[str, dict[str, Any]]:
    expected = {str(item["object_key"]): item for item in manifest["files"]}
    if len(requested_keys) != len(set(requested_keys)):
        raise HTTPException(status_code=400, detail="Duplicate R2 object key.")
    if set(requested_keys) != set(expected):
        raise HTTPException(
            status_code=400,
            detail="Uploaded R2 objects do not match the upload session.",
        )
    return expected


async def _save_uploads(
    files: list[UploadFile], job_directory: Path
) -> list[dict[str, str]]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    if len(files) > MAX_IMAGES_PER_JOB:
        raise HTTPException(
            status_code=400,
            detail=f"A job accepts at most {MAX_IMAGES_PER_JOB} images of the same SKU.",
        )

    original_directory = job_directory / "original"
    original_directory.mkdir(parents=True, exist_ok=False)
    saved: list[dict[str, str]] = []
    hashes: set[str] = set()
    total_size_bytes = 0

    for index, upload in enumerate(files, start=1):
        display_name = Path(upload.filename or f"image_{index}.jpg").name
        extension = Path(display_name).suffix.lower()
        if extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format for {display_name}: {extension or 'none'}.",
            )

        raw_bytes = await upload.read(MAX_FILE_SIZE_BYTES + 1)
        await upload.close()
        if not raw_bytes:
            raise HTTPException(
                status_code=400, detail=f"Uploaded image is empty: {display_name}."
            )
        if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
            limit_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the {limit_mb:g} MB limit: {display_name}.",
            )
        total_size_bytes += len(raw_bytes)
        if total_size_bytes > MAX_JOB_UPLOAD_SIZE_BYTES:
            total_limit_mb = MAX_JOB_UPLOAD_SIZE_BYTES / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Total upload exceeds the {total_limit_mb:g} MB limit. "
                    "Split the images into smaller jobs."
                ),
            )

        digest = hashlib.sha256(raw_bytes).hexdigest()
        if digest in hashes:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate image in the same job: {display_name}.",
            )
        hashes.add(digest)

        safe_name = _safe_upload_name(index, display_name)
        destination = original_directory / safe_name
        destination.write_bytes(raw_bytes)
        saved.append(
            {
                "display_name": display_name,
                "safe_name": safe_name,
                "path": str(destination),
                "sha256": digest,
            }
        )
    return saved


def _image_summary(
    job_id: str,
    result: dict[str, Any],
    annotated_filename: str,
) -> dict[str, Any]:
    decision = result.get("decision")
    return {
        "image_name": str(result["image_name"]),
        "status": "SUCCESS",
        "total_detections": int(result["total_detections"]),
        "avg_confidence": float(result["avg_confidence"]),
        "inference_ms": float(result["inference_ms"]),
        "decision": decision if isinstance(decision, dict) else None,
        "annotated_filename": annotated_filename,
        "annotated_url": (
            f"/api/v1/bakery/jobs/{job_id}/annotated/{annotated_filename}"
        ),
        "error": "",
    }


CONFIRMABLE_DECISIONS = {"DIRECT", "FAMILY", "REVIEW"}


def _decision_choices(decision: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return business SKUs represented by one image decision, with evidence."""
    decision_type = str(decision.get("decision") or "").upper()
    if decision_type == "DIRECT":
        code = str(decision.get("product_code") or "").strip()
        if not code:
            return {}
        return {
            code.casefold(): {
                "product_code": code,
                "product_name": str(decision.get("product_name") or decision.get("display_name") or code),
                "display_name": str(decision.get("display_name") or decision.get("product_name") or code),
                "visual_class": decision.get("dominant_class"),
                "detected_quantity": int(decision.get("dominant_count") or decision.get("count") or 0),
                "avg_confidence": float(decision.get("avg_confidence") or 0.0),
            }
        }

    source_key = "members" if decision_type == "FAMILY" else "candidates"
    if decision_type not in {"FAMILY", "REVIEW"}:
        return {}
    choices: dict[str, dict[str, Any]] = {}
    for item in decision.get(source_key) or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("product_code") or "").strip()
        if not code:
            continue
        candidate = {
            "product_code": code,
            "product_name": str(item.get("product_name") or item.get("display_name") or code),
            "display_name": str(item.get("display_name") or item.get("product_name") or code),
            "visual_class": item.get("visual_class") or decision.get("dominant_class"),
            "detected_quantity": int(item.get("detected_quantity") or decision.get("dominant_count") or 0),
            "avg_confidence": float(item.get("avg_confidence") or decision.get("avg_confidence") or 0.0),
        }
        existing = choices.get(code.casefold())
        if existing is None or (candidate["detected_quantity"], candidate["avg_confidence"]) > (int(existing.get("detected_quantity") or 0), float(existing.get("avg_confidence") or 0.0)):
            choices[code.casefold()] = candidate
    return choices


def _result_is_confirmable(result: dict[str, Any]) -> bool:
    decision = result.get("decision")
    if not isinstance(decision, dict):
        return False
    decision_type = str(decision.get("decision") or "").upper()
    return bool(
        decision_type in CONFIRMABLE_DECISIONS
        and _decision_choices(decision)
        and int(decision.get("count") or 0) > 0
    )


def _rescue_foundation_count_from_batch_context(
    result: dict[str, Any],
    raw_class: str,
) -> dict[str, Any] | None:
    """Promote semantically validated Foundation objects using batch consensus.

    Foundation can reject the tray-level identity when its top-two reference
    margin is narrowly below the global safety threshold even though every
    retained instance already passed the stricter per-object semantic filters.
    For a user-declared same-SKU batch, two or more independently valid images
    establish the visual class. In that narrow case the retained object count
    is safe to include, while the whole job remains REVIEW for the operator.
    """
    decision = result.get("decision")
    if not isinstance(decision, dict):
        return None
    if str(decision.get("decision") or "").upper() != "AMBIGUOUS":
        return None

    selected_engine = str((result.get("hybrid") or {}).get("selected_engine") or "")
    if str(result.get("engine") or "").upper() != "FOUNDATION" and selected_engine.upper() != "FOUNDATION":
        return None

    expected_class = str(raw_class or "").strip()
    count = max(0, int(decision.get("count") or 0))
    objects = result.get("objects")
    if not expected_class or count <= 0 or not isinstance(objects, list):
        return None
    if len(objects) != count or int(result.get("total_detections") or 0) != count:
        return None

    filtering = result.get("foundation_filtering") or {}
    required_margin = max(0.0, float(filtering.get("instance_similarity_margin") or 0.0))
    product_codes: set[str] = set()
    product_names: dict[str, str] = {}
    confidences: list[float] = []
    for item in objects:
        if not isinstance(item, dict):
            return None
        if str(item.get("raw_class") or "").strip() != expected_class:
            return None
        confidence = float(item.get("confidence") or 0.0)
        threshold = float(item.get("confidence_threshold") or 1.0)
        instance_margin = item.get("foundation_instance_margin")
        if confidence < threshold or instance_margin is None:
            return None
        if float(instance_margin) < required_margin:
            return None
        product_code = str(item.get("product_code") or "").strip()
        if not product_code:
            return None
        product_codes.add(product_code)
        product_names[product_code] = str(
            item.get("product_name")
            or item.get("display_name")
            or product_code
        )
        confidences.append(confidence)

    if len(product_codes) != 1:
        return None
    product_code = next(iter(product_codes))
    rescued = dict(result)
    rescued["decision"] = {
        **decision,
        "decision": "REVIEW",
        "count": count,
        "total_detections": count,
        "purity": 1.0,
        "avg_confidence": sum(confidences) / len(confidences),
        "candidates": [
            {
                "source": "FOUNDATION_BATCH_CONTEXT",
                "product_code": product_code,
                "product_name": product_names[product_code],
                "count": count,
            }
        ],
        "preferred_source": "FOUNDATION",
        "preferred_product_code": product_code,
        "recovered_by_batch_context": True,
        "requires_user_selection": True,
        "requires_confirmation": True,
        "message": (
            "Foundation object count passed per-instance validation and was "
            "recovered because the other images in this same-SKU batch agreed "
            "on the same visual class. Verify before confirmation."
        ),
    }
    rescued["batch_rescue"] = {
        "applied": True,
        "source": "FOUNDATION_OBJECTS",
        "raw_class": expected_class,
        "count": count,
    }
    return rescued


def _consensus_visual_class(results: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    """Find a strong visual-class winner among already-valid images."""
    valid = [result for result in results if _result_is_confirmable(result)]
    votes: Counter[str] = Counter()
    weighted: Counter[str] = Counter()

    for result in valid:
        decision = result.get("decision") or {}
        raw_class = str(decision.get("dominant_class") or "").strip()
        if not raw_class:
            continue
        votes[raw_class] += 1
        weighted[raw_class] += max(1, int(decision.get("count") or 0))

    if len(valid) < BATCH_RESCUE_MIN_VALID_IMAGES or not votes:
        return None, {
            "valid_images": len(valid),
            "reason": "not_enough_valid_images",
        }

    winner, winner_votes = votes.most_common(1)[0]
    vote_total = sum(votes.values())
    image_agreement = winner_votes / max(1, vote_total)
    weighted_total = sum(weighted.values())
    count_agreement = weighted[winner] / max(1, weighted_total)
    required = max(0.5, min(1.0, BATCH_RESCUE_MIN_IMAGE_AGREEMENT))

    if image_agreement < required or count_agreement < required:
        return None, {
            "valid_images": len(valid),
            "winner": winner,
            "image_agreement": image_agreement,
            "count_agreement": count_agreement,
            "reason": "weak_consensus",
        }

    return winner, {
        "valid_images": len(valid),
        "winner": winner,
        "winner_votes": winner_votes,
        "image_agreement": image_agreement,
        "count_agreement": count_agreement,
        "reason": "strong_consensus",
    }


def _attempt_batch_rescue(
    results: list[dict[str, Any]],
    pending_images: list[dict[str, Any]],
    inference_service: BakeryInferenceService,
) -> dict[str, Any]:
    """Recover failed images from low-confidence YOLO candidates when safe.

    No model rerun occurs here. The YOLO service retains its low-confidence
    candidate pool from the original forward pass; this function merely asks it
    to re-evaluate the one visual class already established by the rest of the
    same-SKU batch.
    """
    summary: dict[str, Any] = {
        "enabled": BATCH_RESCUE_ENABLED,
        "attempted": 0,
        "recovered": 0,
        "recovered_images": [],
        "remaining_invalid_images": [],
    }
    if not BATCH_RESCUE_ENABLED or not results:
        return summary

    invalid_indices = [
        index for index, result in enumerate(results) if not _result_is_confirmable(result)
    ]
    if not invalid_indices:
        return summary

    raw_class, consensus = _consensus_visual_class(results)
    summary["consensus"] = consensus
    if not raw_class:
        summary["remaining_invalid_images"] = [
            str(results[index].get("image_name") or f"Image {index + 1}")
            for index in invalid_indices
        ]
        return summary

    rescue = getattr(inference_service, "rescue_result_for_class", None)
    if not callable(rescue):
        summary["reason"] = "inference_service_has_no_rescue_method"
        summary["remaining_invalid_images"] = [
            str(results[index].get("image_name") or f"Image {index + 1}")
            for index in invalid_indices
        ]
        return summary

    summary["raw_class"] = raw_class
    for index in invalid_indices:
        if index >= len(pending_images):
            continue
        item = pending_images[index]
        source_path = Path(item["source_path"])
        summary["attempted"] += 1
        rescued = _rescue_foundation_count_from_batch_context(
            results[index], raw_class
        )
        if rescued is not None:
            summary.setdefault("foundation_context_recovered", 0)
            summary["foundation_context_recovered"] += 1
        else:
            try:
                rescued = rescue(
                    results[index],
                    raw_class,
                    raw_bytes=source_path.read_bytes(),
                    annotated_path=item.get("annotated_path"),
                )
            except Exception as exc:
                summary.setdefault("errors", []).append(
                    {
                        "image_name": str(item.get("image_name") or source_path.name),
                        "error": str(exc),
                    }
                )
                rescued = None

        if rescued is not None and _result_is_confirmable(rescued):
            results[index] = rescued
            summary["recovered"] += 1
            summary["recovered_images"].append(
                str(rescued.get("image_name") or item.get("image_name") or source_path.name)
            )

    summary["remaining_invalid_images"] = [
        str(result.get("image_name") or f"Image {index + 1}")
        for index, result in enumerate(results)
        if not _result_is_confirmable(result)
    ]
    return summary


def _aggregate_batch_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate same-SKU images without conflating physical count and class."""
    if not results:
        raise ValueError("A batch must contain at least one inference result.")

    per_image: list[dict[str, Any]] = []
    valid_entries: list[tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]] = []
    invalid_images: list[str] = []
    total_count = 0
    unconfirmed_count = 0
    total_detections = 0
    weighted_confidence = 0.0
    weighted_count = 0
    global_class_counts: Counter[str] = Counter()
    global_class_confidence_sum: Counter[str] = Counter()
    global_class_display: dict[str, str] = {}
    dominant_image_votes: Counter[str] = Counter()

    for result in results:
        decision = result.get("decision")
        if not isinstance(decision, dict):
            decision = {}
        decision_type = str(decision.get("decision") or "UNKNOWN").upper()
        physical_count = max(0, int(decision.get("physical_count") or result.get("total_detections") or decision.get("count") or 0))
        breakdown_counts = [int(item.get("count") or 0) for item in (decision.get("class_breakdown") or []) if isinstance(item, dict)]
        dominant_count = max(0, int(decision.get("dominant_count") or max(breakdown_counts or [decision.get("count") or 0])))
        confidence = float(result.get("avg_confidence") if result.get("avg_confidence") is not None else decision.get("avg_confidence") or 0.0)
        purity = float(decision.get("classification_purity") if decision.get("classification_purity") is not None else decision.get("purity") or 0.0)
        choices = _decision_choices(decision)
        image_name = str(result.get("image_name") or "Image")
        is_valid = bool(decision_type in CONFIRMABLE_DECISIONS and choices and physical_count > 0)

        per_image.append({
            "image_name": image_name,
            "decision": decision_type,
            "product_code": decision.get("product_code"),
            "display_name": decision.get("display_name"),
            "dominant_class": decision.get("dominant_class"),
            "count": physical_count,
            "physical_count": physical_count,
            "dominant_count": dominant_count,
            "purity": purity,
            "avg_confidence": confidence,
            "recovered_by_batch_context": bool(decision.get("recovered_by_batch_context") or (result.get("batch_rescue") or {}).get("applied")),
            "resolved": is_valid,
        })

        result_total = max(0, int(result.get("total_detections") or physical_count))
        total_detections += result_total
        if is_valid:
            total_count += physical_count
            valid_entries.append((decision, choices, result))
            weight = max(1, physical_count)
            weighted_count += weight
            weighted_confidence += confidence * weight
            raw_class = str(decision.get("dominant_class") or "").strip()
            if raw_class:
                dominant_image_votes[raw_class] += 1
            objects = result.get("objects")
            if isinstance(objects, list):
                for item in objects:
                    if not isinstance(item, dict):
                        continue
                    item_class = str(item.get("raw_class") or "").strip()
                    if not item_class:
                        continue
                    global_class_counts[item_class] += 1
                    global_class_confidence_sum[item_class] += float(item.get("confidence") or 0.0)
                    global_class_display.setdefault(item_class, str(item.get("display_name") or item_class))
            else:
                for item in decision.get("class_breakdown") or []:
                    if not isinstance(item, dict):
                        continue
                    item_class = str(item.get("raw_class") or "").strip()
                    item_count = max(0, int(item.get("count") or 0))
                    if not item_class or item_count <= 0:
                        continue
                    global_class_counts[item_class] += item_count
                    global_class_confidence_sum[item_class] += float(item.get("avg_confidence") or 0.0) * item_count
                    global_class_display.setdefault(item_class, str(item.get("display_name") or item_class))
        else:
            unconfirmed_count += physical_count
            invalid_images.append(image_name)

    average_confidence = weighted_confidence / weighted_count if weighted_count else 0.0
    classified_total = sum(global_class_counts.values())
    dominant_batch_class = None
    dominant_batch_count = 0
    if global_class_counts:
        dominant_batch_class, dominant_batch_count = global_class_counts.most_common(1)[0]
    global_purity = dominant_batch_count / classified_total if classified_total else 0.0
    class_breakdown = [{
        "raw_class": raw_class,
        "display_name": global_class_display.get(raw_class, raw_class),
        "count": count,
        "avg_confidence": global_class_confidence_sum[raw_class] / count if count else 0.0,
    } for raw_class, count in global_class_counts.most_common()]
    image_vote_total = sum(dominant_image_votes.values())
    dominant_image_class = None
    dominant_image_vote_count = 0
    if dominant_image_votes:
        dominant_image_class, dominant_image_vote_count = dominant_image_votes.most_common(1)[0]
    image_consensus = dominant_image_vote_count / image_vote_total if image_vote_total else 0.0

    base = {
        "count": total_count,
        "physical_count": total_count,
        "total_detections": total_detections,
        "purity": global_purity,
        "classification_purity": global_purity,
        "avg_confidence": average_confidence,
        "dominant_class": dominant_batch_class,
        "dominant_count": dominant_batch_count,
        "class_breakdown": class_breakdown,
        "image_consensus": image_consensus,
        "dominant_image_class": dominant_image_class,
        "dominant_image_votes": dominant_image_vote_count,
        "per_image": per_image,
        "batch_size": len(results),
        "valid_image_count": len(valid_entries),
        "invalid_image_count": len(invalid_images),
        "invalid_images": invalid_images,
        "count_is_partial": bool(invalid_images),
        "unconfirmed_count_excluded": unconfirmed_count,
    }

    if not valid_entries:
        return {**base, "decision": "AMBIGUOUS", "display_name": "Không đủ dữ liệu nhận diện", "requires_user_selection": False, "requires_confirmation": False, "message": "Không có ảnh nào trong batch nhận diện đủ an toàn để suy ra SKU. Hãy chụp lại ít nhất một ảnh rõ hơn trước khi xác nhận."}

    common_codes = set(valid_entries[0][1])
    for _, choices, _ in valid_entries[1:]:
        common_codes.intersection_update(choices)

    if not common_codes:
        candidate_by_code: dict[str, dict[str, Any]] = {}
        image_votes: Counter[str] = Counter()
        detected_quantities: Counter[str] = Counter()
        for _, choices, _ in valid_entries:
            for code_key, item in choices.items():
                candidate_by_code.setdefault(code_key, dict(item))
                image_votes[code_key] += 1
                detected_quantities[code_key] += max(0, int(item.get("detected_quantity") or 0))
        candidates = [{**candidate_by_code[key], "source": "BATCH_CLASS_DISAGREEMENT", "count": total_count, "image_votes": image_votes[key], "detected_quantity": detected_quantities[key]} for key in sorted(candidate_by_code, key=lambda key: (-image_votes[key], -detected_quantities[key], key))]
        return {**base, "decision": "REVIEW", "display_name": "AI dự đoán nhiều class trong batch", "candidates": candidates, "requires_user_selection": True, "requires_confirmation": True, "message": "Các ảnh trong batch có bằng chứng class khác nhau. Vì job được khai báo là cùng một SKU, hệ thống giữ tổng số physical objects nhưng không tự chọn sản phẩm. Hãy chọn đúng SKU và kiểm tra số lượng trước khi xác nhận KiotViet."}

    merged_choices: list[dict[str, Any]] = []
    for code_key in sorted(common_codes):
        evidence_items = [choices[code_key] for _, choices, _ in valid_entries if code_key in choices]
        selected = max(evidence_items, key=lambda item: (int(item.get("detected_quantity") or 0), float(item.get("avg_confidence") or 0.0)))
        merged = dict(selected)
        merged["detected_quantity"] = sum(max(0, int(item.get("detected_quantity") or 0)) for item in evidence_items)
        merged_choices.append(merged)

    if invalid_images:
        names = ", ".join(invalid_images)
        return {**base, "decision": "REVIEW", "display_name": "Batch cùng SKU cần bổ sung count", "candidates": [{**item, "source": "BATCH_VALID_IMAGES", "count": total_count} for item in merged_choices], "requires_user_selection": True, "requires_confirmation": True, "message": f"{len(valid_entries)}/{len(results)} ảnh có SKU/family tương thích, nhưng {len(invalid_images)} ảnh chưa được đếm an toàn: {names}. Tổng AI hiện tại là {total_count} và chỉ cộng các ảnh đã nhận diện. Hãy kiểm tra ảnh chưa nhận diện và sửa tổng số lượng nếu cần trước khi xác nhận."}

    decision_types = {str(decision.get("decision") or "").upper() for decision, _, _ in valid_entries}
    if decision_types == {"DIRECT"} and len(common_codes) == 1:
        first = valid_entries[0][0]
        selected = merged_choices[0]
        return {**first, **base, "decision": "DIRECT", "product_code": selected["product_code"], "product_name": selected["product_name"], "requires_user_selection": False, "requires_confirmation": True, "message": f"{len(results)} ảnh cùng SKU; tổng số lượng bằng tổng physical count từng ảnh. Hãy kiểm tra và xác nhận trước khi tạo phiếu."}

    if decision_types == {"FAMILY"}:
        first = valid_entries[0][0]
        return {**first, **base, "decision": "FAMILY", "members": merged_choices, "requires_user_selection": True, "requires_confirmation": True, "message": f"{len(results)} ảnh cùng một visual family. Chọn SKU nghiệp vụ rồi xác nhận tổng physical count."}

    return {**base, "decision": "REVIEW", "display_name": "Batch cùng SKU cần xác nhận", "candidates": [{**item, "source": "BATCH", "count": total_count} for item in merged_choices], "requires_user_selection": True, "requires_confirmation": True, "message": "Các ảnh có ít nhất một SKU tương thích chung nhưng class/decision chưa đồng thuận hoàn toàn. Hệ thống vẫn giữ tổng physical count; hãy chọn đúng SKU và kiểm tra số lượng trước khi xác nhận."}


def _process_job(
    job_id: str,
    saved_uploads: list[dict[str, str]],
    inference_service: BakeryInferenceService,
    storage_service: R2StorageService | None,
    inference_mode: str = "AUTO",
) -> None:
    """Detect a same-SKU image batch and stop before any business-side write.

    DIRECT/FAMILY/REVIEW -> AWAITING_CONFIRMATION
    AMBIGUOUS/NO_DETECTION -> NEEDS_RETAKE

    Excel and KiotViet are intentionally deferred until explicit confirmation.
    """
    if not saved_uploads or len(saved_uploads) > MAX_IMAGES_PER_JOB:
        state = _read_job(job_id)
        state.update(
            {
                "status": "ERROR",
                "updated_at": _now_iso(),
                "error": (
                    f"A same-SKU batch requires 1 to {MAX_IMAGES_PER_JOB} images."
                ),
            }
        )
        _write_json_atomic(_job_state_path(job_id), state)
        return

    job_directory = _job_directory(job_id)
    state = _read_job(job_id)
    state.update({"status": "PROCESSING", "updated_at": _now_iso()})
    _write_json_atomic(_job_state_path(job_id), state)

    full_results: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    annotated_directory = job_directory / "annotated"

    try:
        detections_path = job_directory / "detections.json"
        annotated_paths: list[Path] = []
        source_paths: list[Path] = []
        pending_images: list[dict[str, Any]] = []
        for upload in saved_uploads:
            source_path = Path(upload["path"])
            source_paths.append(source_path)
            annotated_filename = f"{source_path.stem}_annotated.jpg"
            annotated_path = annotated_directory / annotated_filename
            annotated_paths.append(annotated_path)
            pending_images.append(
                {
                    "source_path": source_path,
                    "image_name": upload["display_name"],
                    "annotated_path": annotated_path,
                    "annotated_filename": annotated_filename,
                }
            )

        for offset in range(0, len(pending_images), INFERENCE_BATCH_SIZE):
            chunk = pending_images[offset : offset + INFERENCE_BATCH_SIZE]
            chunk_inputs = [
                {
                    **item,
                    "raw_bytes": Path(item["source_path"]).read_bytes(),
                }
                for item in chunk
            ]
            infer_batch = getattr(inference_service, "infer_batch", None)
            if callable(infer_batch):
                chunk_results = infer_batch(chunk_inputs, mode=inference_mode)
            else:
                chunk_results = [
                    inference_service.infer_bytes(
                        item["raw_bytes"],
                        image_name=item["image_name"],
                        annotated_path=item["annotated_path"],
                        mode=inference_mode,
                    )
                    for item in chunk_inputs
                ]
            if len(chunk_results) != len(chunk):
                raise ValueError("Inference returned an incomplete image batch.")
            for item, result in zip(chunk, chunk_results, strict=True):
                full_results.append(result)
                image_summaries.append(
                    _image_summary(
                        job_id,
                        result,
                        str(item["annotated_filename"]),
                    )
                )
                _write_json_atomic(detections_path, full_results)
                processed = len(full_results)
                state.update(
                    {
                        "status": "PROCESSING",
                        "updated_at": _now_iso(),
                        "processed_images": processed,
                        "images": image_summaries,
                        "message": (
                            f"Đã xử lý {processed}/{len(saved_uploads)} ảnh."
                        ),
                        "error": "",
                    }
                )
                _write_json_atomic(_job_state_path(job_id), state)

        # A same-SKU batch may contain one difficult image. Before declaring
        # the entire job incomplete, reuse that image's low-confidence YOLO
        # candidates for the visual class strongly established by the other
        # images. This does not re-run the model and does not change thresholds.
        rescue_summary = _attempt_batch_rescue(
            full_results,
            pending_images,
            inference_service,
        )
        if rescue_summary.get("attempted"):
            image_summaries = [
                _image_summary(
                    job_id,
                    result,
                    str(item["annotated_filename"]),
                )
                for item, result in zip(pending_images, full_results, strict=True)
            ]
            _write_json_atomic(detections_path, full_results)

        decision = _aggregate_batch_decision(full_results)
        decision["batch_rescue"] = rescue_summary

        decision_type = str(decision.get("decision") or "").upper()
        if decision_type in {"DIRECT", "FAMILY", "REVIEW"}:
            next_status = "AWAITING_CONFIRMATION"
            message = str(
                decision.get("message")
                or "Detection is ready. Confirm before creating the KiotViet receipt."
            )
        elif decision_type in {"AMBIGUOUS", "NO_DETECTION"}:
            next_status = "NEEDS_RETAKE"
            message = str(
                decision.get("message")
                or "Retake the image or review it manually."
            )
        else:
            raise ValueError(f"Unsupported inference decision: {decision_type!r}.")

        r2_objects: list[dict[str, Any]] = []
        if storage_service is not None:
            artifacts = (
                [("original", path) for path in source_paths]
                + [("annotated", path) for path in annotated_paths]
                + [("metadata", detections_path)]
            )

            def upload_artifact(item: tuple[str, Path]) -> dict[str, Any]:
                category, artifact_path = item
                object_key = storage_service.job_key(
                    job_id,
                    category,
                    artifact_path.name,
                )
                return storage_service.upload_file(artifact_path, object_key)

            worker_count = min(R2_TRANSFER_WORKERS, len(artifacts))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                r2_objects = list(executor.map(upload_artifact, artifacts))

        state.update(
            {
                "status": next_status,
                "updated_at": _now_iso(),
                "processed_images": len(full_results),
                "product_count": 1 if decision_type in {"DIRECT", "FAMILY", "REVIEW"} else 0,
                "total_quantity": int(decision.get("count") or 0),
                "products": [],
                "images": image_summaries,
                "decision": decision,
                "inference_mode": str(inference_mode).upper(),
                "r2_objects": r2_objects,
                "message": message,
                "kiotviet": None,
                "error": "",
            }
        )
        _write_json_atomic(_job_state_path(job_id), state)

    except Exception as exc:
        state.update(
            {
                "status": "ERROR",
                "updated_at": _now_iso(),
                "processed_images": len(full_results),
                "images": image_summaries,
                "error": str(exc),
            }
        )
        _write_json_atomic(_job_state_path(job_id), state)


def _download_and_process_r2_job(
    job_id: str,
    object_keys: list[str],
    inference_service: BakeryInferenceService,
    storage_service: R2StorageService,
    inference_mode: str,
) -> None:
    """Download an accepted R2 batch in the background, then run inference."""
    state = _read_job(job_id)
    state.update(
        {
            "status": "PROCESSING",
            "updated_at": _now_iso(),
            "message": f"Đang tải {len(object_keys)} ảnh từ R2 về máy AI.",
            "error": "",
        }
    )
    _write_json_atomic(_job_state_path(job_id), state)
    try:
        saved_uploads = _download_r2_uploads(
            job_id,
            object_keys,
            storage_service,
        )
        _process_job(
            job_id,
            saved_uploads,
            inference_service,
            storage_service,
            inference_mode,
        )
    except Exception as exc:
        state = _read_job(job_id)
        state.update(
            {
                "status": "ERROR",
                "updated_at": _now_iso(),
                "error": str(exc),
                "message": "Không thể tải hoặc xử lý batch ảnh từ R2.",
            }
        )
        _write_json_atomic(_job_state_path(job_id), state)


@router.get("/health", response_model=BakeryHealthResponse)
def bakery_health(request: Request) -> BakeryHealthResponse:
    inference_service = getattr(
        request.app.state, "bakery_inference_service", None
    )
    excel_service = getattr(request.app.state, "excel_service", None)
    startup_error = str(
        getattr(request.app.state, "bakery_startup_error", "")
    )
    ready = inference_service is not None and excel_service is not None
    return BakeryHealthResponse(
        ready=ready,
        template_ready=(
            excel_service.template_path.is_file()
            if excel_service is not None
            else False
        ),
        model=inference_service.health() if inference_service is not None else None,
        r2_configured=(
            getattr(request.app.state, "r2_storage_service", None) is not None
        ),
        kiotviet_configured=(
            getattr(request.app.state, "kiotviet_service", None) is not None
        ),
        manufacturing_configured=(
            getattr(request.app.state, "manufacturing_service", None) is not None
        ),
        kiotviet_auto_create_draft=KIOTVIET_AUTO_CREATE_DRAFT,
        max_images_per_job=MAX_IMAGES_PER_JOB,
        max_image_size_mb=round(MAX_FILE_SIZE_BYTES / (1024 * 1024), 2),
        max_job_upload_size_mb=round(
            MAX_JOB_UPLOAD_SIZE_BYTES / (1024 * 1024), 2
        ),
        allowed_image_extensions=sorted(ALLOWED_IMAGE_EXTENSIONS),
        pseudo_label=(
            request.app.state.pseudo_label_service.health()
            if getattr(request.app.state, "pseudo_label_service", None) is not None
            else None
        ),
        error=startup_error,
    )


@router.post(
    "/uploads/presign",
    response_model=PresignUploadsResponse,
)
def presign_uploads(
    payload: PresignUploadsRequest,
    request: Request,
) -> PresignUploadsResponse:
    storage_service: R2StorageService | None = getattr(
        request.app.state, "r2_storage_service", None
    )
    if storage_service is None:
        raise HTTPException(status_code=503, detail="R2 storage is not configured.")
    _validate_upload_metadata(payload.files)

    job_id = payload.job_id or uuid4().hex
    job_directory = _job_directory(job_id)
    if _job_state_path(job_id).exists():
        raise HTTPException(
            status_code=409,
            detail="This upload session has already been submitted.",
        )

    if job_directory.exists():
        manifest = _read_upload_manifest(job_id)
        requested = [
            {
                "filename": Path(item.filename).name,
                "content_type": item.content_type,
                "size_bytes": item.size_bytes,
            }
            for item in payload.files
        ]
        existing = [
            {
                "filename": str(item["filename"]),
                "content_type": str(item["content_type"]),
                "size_bytes": int(item["size_bytes"]),
            }
            for item in manifest["files"]
        ]
        if requested != existing:
            raise HTTPException(
                status_code=409,
                detail="Upload session metadata does not match the original request.",
            )
        existing_mode = str(manifest.get("inference_mode") or "AUTO").upper()
        if existing_mode != payload.inference_mode:
            raise HTTPException(
                status_code=409,
                detail="Upload session inference mode does not match the original request.",
            )
        uploads = []
        for item in manifest["files"]:
            uploads.append(
                {
                    "filename": item["filename"],
                    "content_type": item["content_type"],
                    "size_bytes": item["size_bytes"],
                    "object_key": item["object_key"],
                    "upload_url": storage_service.presign_upload(
                        str(item["object_key"]),
                        str(item["content_type"]),
                        expires_in=UPLOAD_URL_EXPIRES_SECONDS,
                    ),
                    "method": "PUT",
                    "headers": {"Content-Type": item["content_type"]},
                }
            )
        return PresignUploadsResponse(
            job_id=job_id,
            expires_in=UPLOAD_URL_EXPIRES_SECONDS,
            uploads=uploads,
            submit_url="/api/v1/bakery/jobs/from-r2",
        )

    job_directory.mkdir(parents=True, exist_ok=False)
    manifest_files: list[dict[str, Any]] = []
    uploads: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(payload.files, start=1):
            filename = Path(item.filename).name
            safe_name = _safe_upload_name(index, filename)
            object_key = storage_service.job_key(job_id, "incoming", safe_name)
            upload_url = storage_service.presign_upload(
                object_key,
                item.content_type,
                expires_in=UPLOAD_URL_EXPIRES_SECONDS,
            )
            entry = {
                "filename": filename,
                "safe_name": safe_name,
                "content_type": item.content_type,
                "size_bytes": item.size_bytes,
                "object_key": object_key,
            }
            manifest_files.append(entry)
            uploads.append(
                {
                    "filename": filename,
                    "content_type": item.content_type,
                    "size_bytes": item.size_bytes,
                    "object_key": object_key,
                    "upload_url": upload_url,
                    "method": "PUT",
                    "headers": {"Content-Type": item.content_type},
                }
            )
        _write_json_atomic(
            _manifest_path(job_id),
            {
                "job_id": job_id,
                "created_at": _now_iso(),
                "expires_in": UPLOAD_URL_EXPIRES_SECONDS,
                "inference_mode": payload.inference_mode,
                "files": manifest_files,
            },
        )
    except Exception:
        _remove_incomplete_job(job_directory)
        raise

    return PresignUploadsResponse(
        job_id=job_id,
        expires_in=UPLOAD_URL_EXPIRES_SECONDS,
        uploads=uploads,
        submit_url="/api/v1/bakery/jobs/from-r2",
    )


@router.post(
    "/jobs/from-r2",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_job_from_r2(
    payload: CreateR2JobRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JobAccepted:
    inference_service: BakeryInferenceService = _service_from_app(
        request, "bakery_inference_service"
    )
    storage_service: R2StorageService | None = getattr(
        request.app.state, "r2_storage_service", None
    )
    if storage_service is None:
        raise HTTPException(status_code=503, detail="R2 storage is not configured.")
    if not payload.files or len(payload.files) > MAX_IMAGES_PER_JOB:
        raise HTTPException(
            status_code=400,
            detail=(
                f"A same-SKU batch requires 1 to {MAX_IMAGES_PER_JOB} uploaded images."
            ),
        )

    job_id = payload.job_id
    if _job_state_path(job_id).exists():
        existing = _read_job(job_id)
        return JobAccepted(
            job_id=job_id,
            status=existing.get("status", "QUEUED"),
            total_images=int(existing.get("total_images") or len(payload.files)),
            status_url=f"/api/v1/bakery/jobs/{job_id}",
            message="Job was already submitted; the existing state was returned.",
        )
    manifest = _read_upload_manifest(payload.job_id)
    manifest_mode = str(manifest.get("inference_mode") or "AUTO").upper()
    if manifest_mode != payload.inference_mode:
        raise HTTPException(
            status_code=409,
            detail="Submitted inference mode does not match the upload session.",
        )
    object_keys = [item.object_key for item in payload.files]
    _validate_r2_object_keys(manifest, object_keys)
    state = _initial_job_state(job_id, len(object_keys), payload.inference_mode)
    _write_json_atomic(_job_state_path(job_id), state)
    background_tasks.add_task(
        _download_and_process_r2_job,
        job_id,
        object_keys,
        inference_service,
        storage_service,
        payload.inference_mode,
    )
    return JobAccepted(
        job_id=job_id,
        status="QUEUED",
        total_images=len(object_keys),
        status_url=f"/api/v1/bakery/jobs/{job_id}",
        message="Batch accepted. R2 download and detection are running in the background.",
    )


@router.post(
    "/jobs",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_job(
    request: Request,
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="One tray image")],
    inference_mode: Annotated[str, Form()] = "AUTO",
) -> JobAccepted:
    inference_service: BakeryInferenceService = _service_from_app(
        request, "bakery_inference_service"
    )
    storage_service: R2StorageService | None = getattr(
        request.app.state, "r2_storage_service", None
    )
    job_id = uuid4().hex
    job_directory = _job_directory(job_id)

    try:
        saved_uploads = await _save_uploads(files, job_directory)
        inference_mode = str(inference_mode).strip().upper()
        if inference_mode not in {"AUTO", "YOLO", "FOUNDATION", "COMPARE"}:
            raise HTTPException(status_code=422, detail="Invalid inference mode.")
        state = _initial_job_state(job_id, len(saved_uploads), inference_mode)
        _write_json_atomic(_job_state_path(job_id), state)
    except Exception:
        _remove_incomplete_job(job_directory)
        raise

    background_tasks.add_task(
        _process_job,
        job_id,
        saved_uploads,
        inference_service,
        storage_service,
        inference_mode,
    )
    return JobAccepted(
        job_id=job_id,
        status="QUEUED",
        total_images=len(saved_uploads),
        status_url=f"/api/v1/bakery/jobs/{job_id}",
        message="Image accepted. Detection is running; business writes require confirmation.",
    )


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    return _job_response(_read_job(job_id), request)


@router.get("/hybrid/dataset")
def hybrid_dataset_stats(request: Request) -> dict[str, Any]:
    service = getattr(request.app.state, "pseudo_label_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Pseudo-label service is unavailable.")
    return service.stats()


@router.get("/jobs/{job_id}/links")
def get_job_with_r2_links(job_id: str, request: Request) -> dict[str, Any]:
    state = _read_job(job_id)
    storage_service: R2StorageService | None = getattr(
        request.app.state, "r2_storage_service", None
    )
    if storage_service is None:
        return _job_response(state, request)

    objects: list[dict[str, Any]] = []
    links_by_name: dict[tuple[str, str], str] = {}
    for raw_item in state.get("r2_objects") or []:
        item = dict(raw_item)
        object_key = str(item.get("object_key") or "")
        if object_key:
            item["download_url"] = storage_service.presign_download(object_key)
            parts = object_key.split("/")
            if len(parts) >= 4:
                links_by_name[(parts[-2], parts[-1])] = item["download_url"]
        objects.append(item)

    response = dict(state)
    response["r2_objects"] = objects
    excel_filename = str(state.get("excel_filename") or "")
    response["excel_url"] = links_by_name.get(("output", excel_filename))
    images: list[dict[str, Any]] = []
    for raw_image in state.get("images") or []:
        image = dict(raw_image)
        annotated_filename = str(image.get("annotated_filename") or "")
        image["annotated_url"] = links_by_name.get(
            ("annotated", annotated_filename)
        )
        images.append(image)
    response["images"] = images
    response["product_catalog"] = _product_catalog(request)
    return response



def _load_detection_results(job_id: str) -> list[dict[str, Any]]:
    path = _job_directory(job_id) / "detections.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Detection metadata not found.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Detection metadata is invalid.",
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise HTTPException(
            status_code=500,
            detail="Same-SKU batch detection metadata is invalid.",
        )
    return payload


def _confirmed_product_from_decision(
    decision: dict[str, Any],
    selected_code: str | None,
    selected_quantity: int | None = None,
    product_mapping_service: Any | None = None,
) -> dict[str, Any]:
    decision_type = str(decision.get("decision") or "").upper()
    quantity = int(selected_quantity or decision.get("count") or 0)
    if quantity <= 0:
        raise HTTPException(status_code=409, detail="Detected count is not valid.")

    if decision_type not in CONFIRMABLE_DECISIONS:
        raise HTTPException(
            status_code=409,
            detail=f"Decision {decision_type or 'UNKNOWN'} cannot be confirmed.",
        )

    # Production confirmation accepts an explicit operator correction to any
    # product in the supported catalog. The catalog is still closed and
    # validated, so arbitrary product codes can never reach KiotViet.
    if product_mapping_service is not None:
        target = str(selected_code or "").strip()
        if not target and decision_type == "DIRECT":
            target = str(decision.get("product_code") or "").strip()
        if not target:
            raise HTTPException(
                status_code=422,
                detail="Select a supported product before confirming.",
            )
        try:
            catalog_product = product_mapping_service.resolve_product_code(target)
        except ProductMappingError as exc:
            raise HTTPException(
                status_code=409,
                detail="Selected product is not in the supported product catalog.",
            ) from exc
        return {
            "product_code": catalog_product["product_code"],
            "product_name": catalog_product["product_name"],
            "purchase_price": 0,
            "quantity": quantity,
        }

    if decision_type == "DIRECT":
        expected_code = str(decision.get("product_code") or "").strip()
        if not expected_code:
            raise HTTPException(
                status_code=500,
                detail="Direct decision has no product code.",
            )
        if selected_code and selected_code.strip().casefold() != expected_code.casefold():
            raise HTTPException(
                status_code=409,
                detail="Selected product code does not match the detected direct SKU.",
            )
        return {
            "product_code": expected_code,
            "product_name": str(
                decision.get("product_name")
                or decision.get("display_name")
                or expected_code
            ),
            "purchase_price": 0,
            "quantity": quantity,
        }

    if decision_type == "FAMILY":
        target = str(selected_code or "").strip()
        if not target:
            raise HTTPException(
                status_code=422,
                detail="Select one SKU from the detected family before confirming.",
            )

        members = decision.get("members")
        if not isinstance(members, list):
            raise HTTPException(status_code=500, detail="Family members are missing.")

        for member in members:
            if not isinstance(member, dict):
                continue
            code = str(member.get("product_code") or "").strip()
            if code.casefold() == target.casefold():
                return {
                    "product_code": code,
                    "product_name": str(
                        member.get("product_name")
                        or member.get("display_name")
                        or code
                    ),
                    "purchase_price": 0,
                    "quantity": quantity,
                }

        raise HTTPException(
            status_code=409,
            detail="Selected product code is not a member of the detected family.",
        )

    if decision_type == "REVIEW":
        target = str(selected_code or "").strip()
        if not target:
            raise HTTPException(
                status_code=422,
                detail="Select a reviewed product code before confirming.",
            )
        candidates = decision.get("candidates")
        if not isinstance(candidates, list):
            raise HTTPException(status_code=500, detail="Review candidates are missing.")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            code = str(candidate.get("product_code") or "").strip()
            if code.casefold() == target.casefold():
                return {
                    "product_code": code,
                    "product_name": str(candidate.get("product_name") or code),
                    "purchase_price": 0,
                    "quantity": quantity,
                }
        raise HTTPException(status_code=409, detail="Selected product is not a review candidate.")

    raise HTTPException(
        status_code=409,
        detail=f"Decision {decision_type or 'UNKNOWN'} cannot be confirmed.",
    )


@router.post("/jobs/{job_id}/confirm")
def confirm_job(
    job_id: str,
    request_body: ConfirmJobRequest,
    request: Request,
) -> dict[str, Any]:
    """Confirm AI result and create the selected KiotViet business document."""
    if not request_body.confirm:
        raise HTTPException(
            status_code=422,
            detail="Explicit confirmation is required.",
        )

    service: KiotVietService = _service_from_app(request, "kiotviet_service")
    manufacturing_service = getattr(request.app.state, "manufacturing_service", None)
    excel_service: ExcelService | None = getattr(request.app.state, "excel_service", None)
    storage_service: R2StorageService | None = getattr(
        request.app.state, "r2_storage_service", None
    )

    with KIOTVIET_SUBMIT_LOCK:
        state = _read_job(job_id)

        if state.get("status") == "COMPLETED":
            existing = state.get("kiotviet")
            if isinstance(existing, dict) and existing.get("created") is True:
                return state
            raise HTTPException(
                status_code=409,
                detail="Job is already completed.",
            )

        current_status = str(state.get("status") or "").upper()
        if current_status not in {"AWAITING_CONFIRMATION", "CONFIRMING"}:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Job cannot be confirmed. Current status: "
                    f"{state.get('status')}."
                ),
            )

        # A previous explicit KiotViet 4xx response means the write was rejected,
        # not interrupted after an unknown commit. Older jobs did not persist the
        # boolean flag, so also infer it from the stored "HTTP NNN" error text.
        previous_retry_safe = bool(state.get("confirmation_retry_safe")) or (
            current_status == "CONFIRMING"
            and _kiotviet_failure_is_retry_safe(state.get("confirmation_error"))
        )
        if current_status == "CONFIRMING" and previous_retry_safe:
            current_status = "AWAITING_CONFIRMATION"
            state.update(
                {
                    "status": current_status,
                    "updated_at": _now_iso(),
                    "message": (
                        "The previous KiotViet request was explicitly rejected; "
                        "retrying this job is safe."
                    ),
                    "confirmation_retry_safe": False,
                }
            )
            _write_json_atomic(_job_state_path(job_id), state)

        if current_status == "CONFIRMING":
            document_type = str(
                state.get("document_type") or request_body.document_type
            ).upper()
            confirmed_product = state.get("confirmed_product")
            if not isinstance(confirmed_product, dict):
                raise HTTPException(
                    status_code=500,
                    detail="Interrupted confirmation has no confirmed product.",
                )
            live_product_id = confirmed_product.get("product_id")
            try:
                if document_type == "MANUFACTURING":
                    if manufacturing_service is None:
                        raise RuntimeError("KiotViet manufacturing RPA is not configured.")
                    receipt_result = manufacturing_service.reconcile_by_job_id(job_id)
                    if isinstance(receipt_result, dict) and receipt_result.get("retry_safe"):
                        branch = service.resolve_branch()
                        receipt_result = manufacturing_service.create_manufacturing_receipt(
                            confirmed_product,
                            job_id,
                            int(branch["id"]),
                        )
                else:
                    reconcile = getattr(
                        service,
                        "reconcile_purchase_receipt_by_job_id",
                        service.find_purchase_receipt_by_job_id,
                    )
                    recovered_receipt = reconcile(job_id)
                    receipt_result = (
                        {
                            "dry_run": False,
                            "action": "RECOVERED",
                            "document_type": "PURCHASE_RECEIPT",
                            "merged_into_daily_receipt": str(
                                recovered_receipt.get("description") or ""
                            ).startswith("AI inventory inspection daily "),
                            "recovered": True,
                            "validation": {
                                "is_draft": bool(getattr(service, "create_as_draft", False)),
                                "recovered": True,
                            },
                            "receipt": recovered_receipt,
                        }
                        if recovered_receipt is not None
                        else None
                    )
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            if receipt_result is None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "The previous KiotViet confirmation was interrupted and no "
                        "matching document is visible yet. Check KiotViet before "
                        "retrying; the backend will not create a duplicate document."
                    ),
                )
        else:
            document_type = request_body.document_type
            decision = state.get("decision")
            if not isinstance(decision, dict):
                raise HTTPException(status_code=500, detail="Job decision is missing.")

            confirmed_product = _confirmed_product_from_decision(
                decision,
                request_body.product_code,
                request_body.quantity,
                getattr(request.app.state, "product_mapping_service", None),
            )

            # Resolve productCode -> live productId before creating any receipt.
            try:
                live_product = service.get_product_by_code(
                    confirmed_product["product_code"]
                )
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

            live_product_id = live_product.get("id")
            try:
                live_product_id = int(live_product_id)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "KiotViet product has an invalid product ID: "
                        f"{confirmed_product['product_code']}."
                    ),
                ) from exc

            if live_product_id <= 0:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "KiotViet product has an invalid product ID "
                        f"({live_product_id}): "
                        f"{confirmed_product['product_code']}."
                    ),
                )

            if live_product.get("isActive", True) is False:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "KiotViet product is inactive: "
                        f"{confirmed_product['product_code']}."
                    ),
                )

            confirmed_product["product_id"] = live_product_id
            if document_type == "MANUFACTURING" and manufacturing_service is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "KiotViet manufacturing RPA is not configured or not available. "
                        "Start the manufacturing service before confirming."
                    ),
                )
            state.update(
                {
                    "status": "CONFIRMING",
                    "updated_at": _now_iso(),
                    "confirmed_product": confirmed_product,
                    "document_type": document_type,
                    "confirmation_key": f"kiotviet:{job_id}",
                    "confirmation_error": "",
                    "confirmation_retry_safe": False,
                    "message": (
                        "Creating a KiotViet manufacturing receipt."
                        if document_type == "MANUFACTURING"
                        else "Creating or updating today's KiotViet purchase receipt."
                    ),
                }
            )
            _write_json_atomic(_job_state_path(job_id), state)

            # Persisting CONFIRMING before the remote write makes a retry enter
            # reconciliation mode instead of issuing a second POST.
            try:
                if document_type == "MANUFACTURING":
                    branch = service.resolve_branch()
                    receipt_result = manufacturing_service.create_manufacturing_receipt(
                        confirmed_product,
                        job_id,
                        int(branch["id"]),
                    )
                else:
                    receipt_result = service.create_purchase_receipt(
                        [confirmed_product],
                        job_id,
                    )
            except Exception as exc:
                retry_safe = _kiotviet_failure_is_retry_safe(exc)
                state.update(
                    {
                        # A KiotViet 4xx validation/auth/business response is a
                        # definitive rejection: no receipt was committed, so the
                        # same job may be submitted again after correction.  For
                        # network/5xx failures, keep CONFIRMING and reconcile first.
                        "status": (
                            "AWAITING_CONFIRMATION" if retry_safe else "CONFIRMING"
                        ),
                        "updated_at": _now_iso(),
                        "confirmation_error": str(exc),
                        "confirmation_retry_safe": retry_safe,
                        "message": (
                            "KiotViet rejected the confirmation. It is safe to retry "
                            "this job after correcting the reported problem."
                            if retry_safe
                            else (
                                "KiotViet confirmation is uncertain. The next retry "
                                "will reconcile by job ID before any new write."
                            )
                        ),
                    }
                )
                _write_json_atomic(_job_state_path(job_id), state)
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        # Excel is now an optional audit/export artifact, generated only after
        # the exact business SKU is known.
        excel_filename: str | None = None
        excel_url: str | None = None
        excel_error = ""
        new_r2_objects: list[dict[str, Any]] = list(state.get("r2_objects") or [])

        if excel_service is not None and document_type == "PURCHASE_RECEIPT":
            try:
                inference_results = _load_detection_results(job_id)
                # The receipt contains one aggregated product line.  Keep it on
                # the first record only so the Excel aggregator cannot add the
                # same batch total once per image.
                for inference_result in inference_results:
                    inference_result["products"] = []
                inference_results[0]["products"] = [confirmed_product]
                output_directory = _job_directory(job_id) / "output"
                excel_filename = f"MauFileNhapHang_{job_id}.xlsx"
                excel_path = output_directory / excel_filename
                excel_service.create_import_workbook(
                    inference_results,
                    excel_path,
                )
                excel_url = f"/api/v1/bakery/jobs/{job_id}/excel"

                if storage_service is not None:
                    object_key = storage_service.job_key(
                        job_id,
                        "output",
                        excel_filename,
                    )
                    new_r2_objects.append(
                        storage_service.upload_file(excel_path, object_key)
                    )
            except Exception as exc:
                excel_error = str(exc)

        pseudo_label: dict[str, Any] | None = None
        pseudo_service = getattr(request.app.state, "pseudo_label_service", None)
        if pseudo_service is not None:
            try:
                inference_results = _load_detection_results(job_id)
                original_files = sorted((_job_directory(job_id) / "original").glob("*"))
                if len(original_files) != len(inference_results):
                    raise FileNotFoundError("Original batch images are incomplete.")
                detected_total = int((state.get("decision") or {}).get("count") or 0)
                if int(confirmed_product["quantity"]) != detected_total:
                    pseudo_label = {
                        "captured": False,
                        "reason": (
                            "The operator corrected the batch total; the correction "
                            "cannot be allocated safely to individual images."
                        ),
                    }
                else:
                    captures: list[dict[str, Any]] = []
                    for index, (source_path, inference_result) in enumerate(
                        zip(original_files, inference_results, strict=True), start=1
                    ):
                        image_decision = inference_result.get("decision") or {}
                        per_image_product = {
                            **confirmed_product,
                            "quantity": int(image_decision.get("count") or 0),
                        }
                        captures.append(
                            pseudo_service.capture(
                                job_id=f"{job_id}_{index:03d}",
                                source_path=source_path,
                                inference_result=inference_result,
                                confirmed_product=per_image_product,
                            )
                        )
                    pseudo_label = {
                        "captured": all(item.get("captured") for item in captures),
                        "records": captures,
                        "train_ready": sum(
                            bool(item.get("train_ready")) for item in captures
                        ),
                    }
            except Exception as exc:
                pseudo_label = {"captured": False, "error": str(exc)}

        receipt_action = str(receipt_result.get("action") or "RECOVERED").upper()
        merged_daily = bool(receipt_result.get("merged_into_daily_receipt"))
        if document_type == "MANUFACTURING":
            completion_message = "Confirmed and created the KiotViet manufacturing receipt."
        else:
            completion_message = (
                "Confirmed and merged into today's KiotViet purchase receipt."
                if merged_daily or receipt_action in {"UPDATED", "REUSED"}
                else "Confirmed and created today's KiotViet purchase receipt."
            )
        state.update(
            {
                "status": "COMPLETED",
                "updated_at": _now_iso(),
                "confirmed_product": confirmed_product,
                "document_type": document_type,
                "products": [confirmed_product],
                "product_count": 1,
                "total_quantity": int(confirmed_product["quantity"]),
                "excel_filename": excel_filename,
                "excel_url": excel_url,
                "excel_error": excel_error,
                "r2_objects": new_r2_objects,
                "kiotviet": {
                    "created": True,
                    "created_at": _now_iso(),
                    "document_type": document_type,
                    "action": receipt_action,
                    "merged_into_daily_receipt": merged_daily,
                    "resolved_product_id": int(live_product_id),
                    **receipt_result,
                },
                "pseudo_label": pseudo_label,
                "message": completion_message,
                "confirmation_error": "",
                "confirmation_retry_safe": False,
                "error": "",
            }
        )
        _write_json_atomic(_job_state_path(job_id), state)
        return state


@router.get("/jobs/{job_id}/excel")
def download_excel(job_id: str) -> FileResponse:
    state = _read_job(job_id)
    if state.get("status") != "COMPLETED":
        raise HTTPException(
            status_code=409,
            detail=f"Excel is not ready. Current job status: {state.get('status')}.",
        )
    filename = str(state.get("excel_filename") or "")
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=500, detail="Invalid Excel output name.")
    path = _job_directory(job_id) / "output" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Excel output not found.")
    return FileResponse(
        path,
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        filename=filename,
    )


@router.get("/jobs/{job_id}/annotated/{filename}")
def download_annotated_image(job_id: str, filename: str) -> FileResponse:
    _read_job(job_id)
    if Path(filename).name != filename or not filename.endswith("_annotated.jpg"):
        raise HTTPException(status_code=404, detail="Annotated image not found.")
    path = _job_directory(job_id) / "annotated" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Annotated image not found.")
    return FileResponse(path, media_type="image/jpeg", filename=filename)


def _completed_job(job_id: str) -> dict[str, Any]:
    state = _read_job(job_id)
    if state.get("status") != "COMPLETED":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed. Current status: {state.get('status')}.",
        )
    return state


@router.get("/jobs/{job_id}/kiotviet-preview")
def preview_kiotviet(job_id: str, request: Request) -> dict[str, Any]:
    state = _completed_job(job_id)
    service: KiotVietService = _service_from_app(request, "kiotviet_service")
    try:
        return service.preview_purchase_receipt(state["products"], job_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/kiotviet")
def submit_kiotviet(
    job_id: str,
    request_body: KiotVietSubmitRequest,
    request: Request,
) -> dict[str, Any]:
    service: KiotVietService = _service_from_app(request, "kiotviet_service")
    if not request_body.confirm:
        state = _completed_job(job_id)
        try:
            return service.preview_purchase_receipt(state["products"], job_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    with KIOTVIET_SUBMIT_LOCK:
        state = _completed_job(job_id)
        existing = state.get("kiotviet")
        if isinstance(existing, dict) and existing.get("created") is True:
            raise HTTPException(
                status_code=409,
                detail="This job already created a KiotViet purchase receipt.",
            )
        try:
            result = service.create_purchase_receipt(state["products"], job_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        state["kiotviet"] = {
            "created": True,
            "created_at": _now_iso(),
            **result,
        }
        state["updated_at"] = _now_iso()
        _write_json_atomic(_job_state_path(job_id), state)
        return state["kiotviet"]
