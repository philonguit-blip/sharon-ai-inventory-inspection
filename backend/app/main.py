from pathlib import Path
import threading

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
from app.routes.developer import router as developer_router
from app.security import basic_auth_enabled, basic_auth_matches
from app.services.bakery_inference_service import BakeryInferenceService
from app.services.excel_service import ExcelService
from app.services.foundation_inference_service import FoundationInferenceService
from app.services.hybrid_inference_service import HybridInferenceService
from app.services.kiotviet_service import KiotVietService
from app.services.manufacturing_service import ManufacturingService
from app.services.n8n_service import N8nOrchestratorService
from app.services.pseudo_label_service import PseudoLabelService
from app.services.storage_service import R2StorageService
from app.services.developer_settings_service import DeveloperSettingsService
from app.config import BACKEND_ROOT, DEVELOPER_SETTINGS_PATH, MODEL_PATH


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
app.state.pseudo_label_service = None
app.state.product_mapping_service = None
app.state.developer_settings_service = DeveloperSettingsService(
    models_root=BACKEND_ROOT / "models",
    settings_path=DEVELOPER_SETTINGS_PATH,
    default_model_path=MODEL_PATH,
)
app.state.developer_settings_lock = threading.Lock()

try:
    runtime_model_path, runtime_thresholds = (
        app.state.developer_settings_service.startup_configuration()
    )
    try:
        yolo_service = BakeryInferenceService(
            model_path=runtime_model_path,
            confidence_overrides=runtime_thresholds,
        )
    except Exception as runtime_exc:
        # A stale/incompatible saved model must never prevent the known-good
        # environment model from starting after a reboot.
        if runtime_model_path == MODEL_PATH and not runtime_thresholds:
            raise
        print(
            "[WARNING] Saved developer settings were rejected; "
            f"falling back to the configured production model: {runtime_exc}"
        )
        yolo_service = BakeryInferenceService()
    foundation_service = FoundationInferenceService()
    app.state.bakery_inference_service = HybridInferenceService(
        yolo_service,
        foundation_service,
    )
    app.state.product_mapping_service = yolo_service.mapping
    app.state.pseudo_label_service = PseudoLabelService()
    app.state.excel_service = ExcelService()
    foundation_status = (
        "ready" if foundation_service.is_ready() else "waiting for assets"
    )
    print(f"[SYSTEM] Hybrid bakery pipeline is ready (Foundation: {foundation_status}).")
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
    from app.config import (
        MANUFACTURING_RPA_BASE_URL,
        MANUFACTURING_RPA_INTERNAL_TOKEN,
        MANUFACTURING_RPA_TIMEOUT_SECONDS,
    )

    app.state.manufacturing_service = ManufacturingService(
        MANUFACTURING_RPA_BASE_URL,
        MANUFACTURING_RPA_INTERNAL_TOKEN,
        MANUFACTURING_RPA_TIMEOUT_SECONDS,
    )
    print("[SYSTEM] KiotViet manufacturing RPA integration is configured.")
except Exception as exc:
    app.state.manufacturing_service = None
    print(f"[WARNING] KiotViet manufacturing RPA is disabled: {exc}")

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
app.include_router(developer_router)
print("[WEB] Open http://127.0.0.1:8080 (do not browse to http://0.0.0.0:8080).")


@app.on_event("shutdown")
async def close_orchestrator_client() -> None:
    service = getattr(app.state, "n8n_orchestrator_service", None)
    if service is not None:
        await service.close()
