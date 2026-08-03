"""Same-origin browser gateway for n8n orchestration."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.jobs import CreateR2JobRequest, PresignUploadsRequest
from app.services.n8n_service import N8nOrchestratorError, N8nOrchestratorService


router = APIRouter(prefix="/api/v1/orchestrator", tags=["Bakery orchestration"])


def _service(request: Request) -> N8nOrchestratorService:
    service = getattr(request.app.state, "n8n_orchestrator_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="n8n orchestration is not configured.",
        )
    return service


async def _call(operation) -> dict[str, Any]:
    try:
        return await operation
    except N8nOrchestratorError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("/health")
async def orchestrator_health(request: Request) -> dict[str, Any]:
    payload = await _call(_service(request).health())
    payload["orchestration"] = "n8n"
    return payload


@router.post("/uploads/presign")
async def orchestrator_prepare_uploads(
    payload: PresignUploadsRequest,
    request: Request,
) -> dict[str, Any]:
    return await _call(
        _service(request).prepare_uploads(payload.model_dump(mode="json"))
    )


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def orchestrator_submit_job(
    payload: CreateR2JobRequest,
    request: Request,
) -> dict[str, Any]:
    return await _call(_service(request).submit_job(payload.model_dump(mode="json")))


@router.get("/jobs/{job_id}")
async def orchestrator_job_status(job_id: str, request: Request) -> dict[str, Any]:
    if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
        raise HTTPException(status_code=422, detail="Invalid job ID.")
    return await _call(_service(request).job_status(job_id))
