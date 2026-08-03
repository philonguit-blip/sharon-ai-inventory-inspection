"""Local asynchronous job API for bakery image counting and Excel export."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from app.config import (
    KIOTVIET_AUTO_CREATE_DRAFT,
    MAX_FILE_SIZE_BYTES,
    MAX_IMAGES_PER_JOB,
    MAX_JOB_UPLOAD_SIZE_BYTES,
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
R2_TRANSFER_WORKERS = 4


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


def _initial_job_state(job_id: str, total_images: int) -> dict[str, Any]:
    now = _now_iso()
    return {
        "job_id": job_id,
        "status": "QUEUED",
        "created_at": now,
        "updated_at": now,
        "total_images": total_images,
        "processed_images": 0,
        "error": "",
        "product_count": 0,
        "total_quantity": 0,
        "products": [],
        "images": [],
        "excel_filename": None,
        "excel_url": None,
        "r2_objects": [],
        "kiotviet": (
            {
                "auto_submit": True,
                "validation_status": "PENDING",
                "created": False,
                "error": "",
            }
            if KIOTVIET_AUTO_CREATE_DRAFT
            else None
        ),
    }


def _validate_upload_metadata(files: list[Any]) -> None:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    if len(files) > MAX_IMAGES_PER_JOB:
        raise HTTPException(
            status_code=400,
            detail=f"A job accepts at most {MAX_IMAGES_PER_JOB} images.",
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
            detail=(
                f"Total upload exceeds the {total_limit_mb:g} MB limit. "
                "Split the images into smaller jobs."
            ),
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
    expected = {str(item["object_key"]): item for item in manifest["files"]}
    if len(requested_keys) != len(set(requested_keys)):
        raise HTTPException(status_code=400, detail="Duplicate R2 object key.")
    if set(requested_keys) != set(expected):
        raise HTTPException(
            status_code=400,
            detail="Uploaded R2 objects do not match the upload session.",
        )

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


async def _save_uploads(
    files: list[UploadFile], job_directory: Path
) -> list[dict[str, str]]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    if len(files) > MAX_IMAGES_PER_JOB:
        raise HTTPException(
            status_code=400,
            detail=f"A job accepts at most {MAX_IMAGES_PER_JOB} images.",
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
    job_id: str, result: dict[str, Any], annotated_filename: str
) -> dict[str, Any]:
    return {
        "image_name": str(result["image_name"]),
        "status": "SUCCESS",
        "total_detections": int(result["total_detections"]),
        "avg_confidence": float(result["avg_confidence"]),
        "inference_ms": float(result["inference_ms"]),
        "annotated_filename": annotated_filename,
        "annotated_url": (
            f"/api/v1/bakery/jobs/{job_id}/annotated/{annotated_filename}"
        ),
        "error": "",
    }


def _auto_create_kiotviet_draft(
    job_id: str,
    products: list[dict[str, Any]],
    service: KiotVietService,
) -> dict[str, Any]:
    """Validate live KiotViet data, then create exactly one draft receipt."""
    try:
        preview = service.preview_purchase_receipt(products, job_id)
        validation = preview.get("validation") if isinstance(preview, dict) else None
        if not isinstance(validation, dict):
            raise ValueError("KiotViet preview returned invalid validation data.")
        if validation.get("is_draft") is not True:
            raise ValueError(
                "Automatic submission is blocked because KiotViet is not configured "
                "to create draft receipts."
            )
    except Exception as exc:
        return {
            "auto_submit": True,
            "validation_status": "FAILED",
            "created": False,
            "error": str(exc),
        }

    try:
        result = service.create_purchase_receipt(products, job_id)
    except Exception as exc:
        return {
            "auto_submit": True,
            "validation_status": "PASSED",
            "validation": validation,
            "created": False,
            "error": str(exc),
        }

    return {
        "auto_submit": True,
        "validation_status": "PASSED",
        "created": True,
        "created_at": _now_iso(),
        "error": "",
        **result,
    }


def _process_job(
    job_id: str,
    saved_uploads: list[dict[str, str]],
    inference_service: BakeryInferenceService,
    excel_service: ExcelService,
    storage_service: R2StorageService | None,
    kiotviet_service: KiotVietService | None,
    auto_create_kiotviet_draft: bool,
) -> None:
    job_directory = _job_directory(job_id)
    state = _read_job(job_id)
    state.update({"status": "PROCESSING", "updated_at": _now_iso()})
    _write_json_atomic(_job_state_path(job_id), state)

    full_results: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    annotated_directory = job_directory / "annotated"
    output_directory = job_directory / "output"

    try:
        for upload in saved_uploads:
            source_path = Path(upload["path"])
            annotated_filename = f"{source_path.stem}_annotated.jpg"
            annotated_path = annotated_directory / annotated_filename
            result = inference_service.infer_bytes(
                source_path.read_bytes(),
                image_name=upload["display_name"],
                annotated_path=annotated_path,
            )
            full_results.append(result)
            image_summaries.append(
                _image_summary(job_id, result, annotated_filename)
            )
            state.update(
                {
                    "processed_images": len(full_results),
                    "images": image_summaries,
                    "updated_at": _now_iso(),
                }
            )
            _write_json_atomic(_job_state_path(job_id), state)

        detections_path = job_directory / "detections.json"
        _write_json_atomic(detections_path, full_results)

        excel_filename = f"MauFileNhapHang_{job_id}.xlsx"
        excel_path = output_directory / excel_filename
        excel_result = excel_service.create_import_workbook(
            full_results, excel_path
        )
        r2_objects: list[dict[str, Any]] = []
        if storage_service is not None:
            artifact_groups = [
                ("original", [Path(item["path"]) for item in saved_uploads]),
                ("annotated", sorted(annotated_directory.glob("*.jpg"))),
                ("output", [excel_path]),
                ("metadata", [detections_path]),
            ]
            artifacts = [
                (category, artifact_path)
                for category, paths in artifact_groups
                for artifact_path in paths
            ]

            def upload_artifact(item: tuple[str, Path]) -> dict[str, Any]:
                category, artifact_path = item
                object_key = storage_service.job_key(
                    job_id, category, artifact_path.name
                )
                return storage_service.upload_file(artifact_path, object_key)

            worker_count = min(R2_TRANSFER_WORKERS, len(artifacts))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                r2_objects = list(executor.map(upload_artifact, artifacts))
        kiotviet_result: dict[str, Any] | None = None
        if auto_create_kiotviet_draft:
            if kiotviet_service is None:
                kiotviet_result = {
                    "auto_submit": True,
                    "validation_status": "FAILED",
                    "created": False,
                    "error": "KiotViet integration is not configured.",
                }
            else:
                kiotviet_result = _auto_create_kiotviet_draft(
                    job_id,
                    excel_result["products"],
                    kiotviet_service,
                )

        state.update(
            {
                "status": "COMPLETED",
                "updated_at": _now_iso(),
                "processed_images": len(full_results),
                "product_count": int(excel_result["product_count"]),
                "total_quantity": int(excel_result["total_quantity"]),
                "products": excel_result["products"],
                "images": image_summaries,
                "excel_filename": excel_filename,
                "excel_url": f"/api/v1/bakery/jobs/{job_id}/excel",
                "r2_objects": r2_objects,
                "kiotviet": kiotviet_result,
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
        kiotviet_auto_create_draft=KIOTVIET_AUTO_CREATE_DRAFT,
        max_images_per_job=MAX_IMAGES_PER_JOB,
        max_image_size_mb=round(MAX_FILE_SIZE_BYTES / (1024 * 1024), 2),
        max_job_upload_size_mb=round(
            MAX_JOB_UPLOAD_SIZE_BYTES / (1024 * 1024), 2
        ),
        allowed_image_extensions=sorted(ALLOWED_IMAGE_EXTENSIONS),
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
    excel_service: ExcelService = _service_from_app(request, "excel_service")
    storage_service: R2StorageService | None = getattr(
        request.app.state, "r2_storage_service", None
    )
    if storage_service is None:
        raise HTTPException(status_code=503, detail="R2 storage is not configured.")
    kiotviet_service: KiotVietService | None = getattr(
        request.app.state, "kiotviet_service", None
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
    saved_uploads = _download_r2_uploads(
        job_id,
        [item.object_key for item in payload.files],
        storage_service,
    )
    state = _initial_job_state(job_id, len(saved_uploads))
    _write_json_atomic(_job_state_path(job_id), state)
    background_tasks.add_task(
        _process_job,
        job_id,
        saved_uploads,
        inference_service,
        excel_service,
        storage_service,
        kiotviet_service,
        KIOTVIET_AUTO_CREATE_DRAFT,
    )
    return JobAccepted(
        job_id=job_id,
        status="QUEUED",
        total_images=len(saved_uploads),
        status_url=f"/api/v1/bakery/jobs/{job_id}",
        message="R2 images accepted. Detection and Excel export are running.",
    )


@router.post(
    "/jobs",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_job(
    request: Request,
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="One or more images")],
) -> JobAccepted:
    inference_service: BakeryInferenceService = _service_from_app(
        request, "bakery_inference_service"
    )
    excel_service: ExcelService = _service_from_app(request, "excel_service")
    storage_service: R2StorageService | None = getattr(
        request.app.state, "r2_storage_service", None
    )
    kiotviet_service: KiotVietService | None = getattr(
        request.app.state, "kiotviet_service", None
    )
    job_id = uuid4().hex
    job_directory = _job_directory(job_id)

    try:
        saved_uploads = await _save_uploads(files, job_directory)
        state = _initial_job_state(job_id, len(saved_uploads))
        _write_json_atomic(_job_state_path(job_id), state)
    except Exception:
        _remove_incomplete_job(job_directory)
        raise

    background_tasks.add_task(
        _process_job,
        job_id,
        saved_uploads,
        inference_service,
        excel_service,
        storage_service,
        kiotviet_service,
        KIOTVIET_AUTO_CREATE_DRAFT,
    )
    return JobAccepted(
        job_id=job_id,
        status="QUEUED",
        total_images=len(saved_uploads),
        status_url=f"/api/v1/bakery/jobs/{job_id}",
        message="Images accepted. Detection and Excel export are running.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    return JobStatusResponse.model_validate(_read_job(job_id))


@router.get("/jobs/{job_id}/links")
def get_job_with_r2_links(job_id: str, request: Request) -> dict[str, Any]:
    state = _read_job(job_id)
    if state.get("status") != "COMPLETED":
        return state
    storage_service: R2StorageService | None = getattr(
        request.app.state, "r2_storage_service", None
    )
    if storage_service is None:
        raise HTTPException(status_code=503, detail="R2 storage is not configured.")

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
    return response


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
