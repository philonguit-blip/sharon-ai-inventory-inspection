from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Environment variables must be loaded before importing application config.
load_dotenv()

from app.config import (
    APP_AUTH_PASSWORD,
    APP_AUTH_USERNAME,
    N8N_REQUEST_TIMEOUT_SECONDS,
    N8N_WEBHOOK_BASE,
    N8N_WEBHOOK_PASSWORD,
    N8N_WEBHOOK_USERNAME,
)
from app.routes.local_jobs import recover_interrupted_jobs, router as local_jobs_router
from app.routes.orchestrator import router as orchestrator_router
from app.security import basic_auth_enabled, basic_auth_matches
from app.services.bakery_inference_service import BakeryInferenceService
from app.services.excel_service import ExcelService
from app.services.kiotviet_service import KiotVietService
from app.services.n8n_service import N8nOrchestratorService
from app.services.storage_service import R2StorageService


app = FastAPI(
    title="Sharon Bakery AI Inventory",
    description="Production image-counting service for bakery inventory intake.",
    version="2.0.0",
)

interrupted_jobs = recover_interrupted_jobs()
if interrupted_jobs:
    print(f"[SYSTEM] Marked {interrupted_jobs} interrupted bakery job(s) as ERROR.")


@app.middleware("http")
async def protect_inventory_app(request, call_next):
    if request.url.path == "/healthz":
        return await call_next(request)
    if not basic_auth_enabled(APP_AUTH_USERNAME, APP_AUTH_PASSWORD):
        return await call_next(request)
    if basic_auth_matches(
        request.headers.get("Authorization"),
        APP_AUTH_USERNAME,
        APP_AUTH_PASSWORD,
    ):
        return await call_next(request)
    return JSONResponse(
        status_code=401,
        content={"detail": "Authentication required."},
        headers={
            "WWW-Authenticate": 'Basic realm="Sharon Bakery Inventory", charset="UTF-8"'
        },
    )


@app.get("/healthz", include_in_schema=False)
def public_healthcheck() -> dict[str, str]:
    return {"status": "ok"}


FRONTEND_ROOT = Path(__file__).resolve().parent.parent / "frontend"
app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_ROOT / "assets"),
    name="bakery-assets",
)


@app.get("/", include_in_schema=False)
@app.get("/bakery", include_in_schema=False)
def bakery_web_app() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html", media_type="text/html")


app.state.bakery_inference_service = None
app.state.excel_service = None
app.state.bakery_startup_error = ""
app.state.r2_storage_service = None
app.state.kiotviet_service = None
app.state.n8n_orchestrator_service = None

try:
    app.state.bakery_inference_service = BakeryInferenceService()
    app.state.excel_service = ExcelService()
    print("[SYSTEM] Bakery image pipeline is ready.")
except Exception as exc:
    app.state.bakery_startup_error = str(exc)
    print(f"[CRITICAL] Bakery image pipeline failed to start: {exc}")

try:
    app.state.r2_storage_service = R2StorageService()
    print("[SYSTEM] R2 artifact storage is ready.")
except Exception as exc:
    print(f"[WARNING] R2 artifact storage is disabled: {exc}")

try:
    app.state.kiotviet_service = KiotVietService()
    print("[SYSTEM] KiotViet integration is configured.")
except Exception as exc:
    print(f"[WARNING] KiotViet integration is disabled: {exc}")

try:
    app.state.n8n_orchestrator_service = N8nOrchestratorService(
        N8N_WEBHOOK_BASE,
        N8N_WEBHOOK_USERNAME,
        N8N_WEBHOOK_PASSWORD,
        N8N_REQUEST_TIMEOUT_SECONDS,
    )
    print("[SYSTEM] n8n orchestration gateway is ready.")
except Exception as exc:
    print(f"[WARNING] n8n orchestration gateway is disabled: {exc}")

app.include_router(local_jobs_router)
app.include_router(orchestrator_router)
print("[WEB] Open http://127.0.0.1:8080 (do not browse to http://0.0.0.0:8080).")


@app.on_event("shutdown")
async def close_orchestrator_client() -> None:
    service = getattr(app.state, "n8n_orchestrator_service", None)
    if service is not None:
        await service.close()
