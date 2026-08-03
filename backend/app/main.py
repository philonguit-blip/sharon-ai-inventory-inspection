import os
import cv2
import importlib.util
import tempfile
import httpx
import traceback
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load cấu hình từ file .env
load_dotenv()

from app.routes.local_jobs import recover_interrupted_jobs, router as local_jobs_router
from app.routes.orchestrator import router as orchestrator_router
from app.config import (
    APP_AUTH_PASSWORD,
    APP_AUTH_USERNAME,
    N8N_REQUEST_TIMEOUT_SECONDS,
    N8N_WEBHOOK_BASE,
    N8N_WEBHOOK_PASSWORD,
    N8N_WEBHOOK_USERNAME,
)
from app.security import basic_auth_enabled, basic_auth_matches
from app.services.bakery_inference_service import BakeryInferenceService
from app.services.excel_service import ExcelService
from app.services.kiotviet_service import KiotVietService
from app.services.n8n_service import N8nOrchestratorService
from app.services.storage_service import R2StorageService

try:
    import boto3
except ImportError:
    boto3 = None

try:
    from app.services.yolo_service import YoloInferenceService
    legacy_import_error = ""
except Exception as e:
    YoloInferenceService = None
    legacy_import_error = str(e)

# 1. KHỞI TẠO ỨNG DỤNG
app = FastAPI(
    title="Inventory Inspection AI Engine",
    description="Lõi xử lý Computer Vision kiểm định kho bãi cho cả Video và Ảnh tĩnh",
    version="1.1.0"
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

# 2. KHỞI TẠO CLOUDFLARE R2 BOTO3 CLIENT
s3_client = None
if boto3 is not None:
    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{os.getenv('CLOUDFLARE_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
        region_name="auto"
    )
R2_BUCKET = os.getenv('R2_BUCKET_NAME', 'inventory-ai-inspection-mvp')

# 3. KHỞI TẠO AI MODEL (SINGLETON)
print("[SYSTEM] Đang nạp ma trận trọng số YOLOv8 vào bộ nhớ...")
try:
    if YoloInferenceService is None:
        raise RuntimeError(legacy_import_error or "Legacy YOLO dependencies are missing.")
    missing_legacy_packages = [
        package
        for package in ("onnx", "onnxruntime")
        if importlib.util.find_spec(package) is None
    ]
    if missing_legacy_packages:
        raise RuntimeError(
            "Legacy video pipeline dependencies are missing: "
            + ", ".join(missing_legacy_packages)
        )
    ai_service = YoloInferenceService()
    print("[SYSTEM] Khởi động lõi AI thành công!")
except Exception as e:
    print(f"[CRITICAL] Lỗi khởi tạo mô hình AI: {e}")
    ai_service = None

# Bakery image pipeline (separate from the legacy video/ONNX service).
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
except Exception as e:
    app.state.bakery_startup_error = str(e)
    print(f"[CRITICAL] Bakery image pipeline failed to start: {e}")

try:
    app.state.r2_storage_service = R2StorageService()
    print("[SYSTEM] R2 artifact storage is ready.")
except Exception as e:
    print(f"[WARNING] R2 artifact storage is disabled: {e}")

try:
    app.state.kiotviet_service = KiotVietService()
    print("[SYSTEM] KiotViet integration is configured.")
except Exception as e:
    print(f"[WARNING] KiotViet integration is disabled: {e}")

try:
    app.state.n8n_orchestrator_service = N8nOrchestratorService(
        N8N_WEBHOOK_BASE,
        N8N_WEBHOOK_USERNAME,
        N8N_WEBHOOK_PASSWORD,
        N8N_REQUEST_TIMEOUT_SECONDS,
    )
    print("[SYSTEM] n8n orchestration gateway is ready.")
except Exception as e:
    print(f"[WARNING] n8n orchestration gateway is disabled: {e}")

app.include_router(local_jobs_router)
app.include_router(orchestrator_router)
print("[WEB] Mở giao diện tại http://127.0.0.1:8080 (không dùng http://0.0.0.0:8080).")


@app.on_event("shutdown")
async def close_orchestrator_client() -> None:
    service = getattr(app.state, "n8n_orchestrator_service", None)
    if service is not None:
        await service.close()

# 4. SCHEMA DỮ LIỆU ĐẦU VÀO (Tương thích ngược với n8n)
class MediaRequest(BaseModel):
    video_key: str  # Vẫn giữ tên biến này để n8n không bị vỡ luồng
    teams_message_id: str
    callback_url: str

# 5. HÀM XỬ LÝ NGẦM ĐA PHƯƠNG TIỆN (BACKGROUND WORKER)
async def process_media_pipeline(media_key: str, teams_message_id: str, callback_url: str):
    """Tự động định tuyến (Routing) xử lý Ảnh tĩnh hoặc Video dựa trên đuôi file."""
    
    # Phân tích định dạng file
    ext = Path(media_key).suffix.lower()
    is_image = ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    
    # Tạo file tạm với đúng đuôi định dạng để OpenCV nhận diện chuẩn Codec
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext if ext else ".mp4") as tmp_file:
        temp_media_path = tmp_file.name

    try:
        print(f"[PROCESS] Bắt đầu tải luồng dữ liệu từ R2: {media_key}")
        s3_client.download_file(R2_BUCKET, media_key, temp_media_path)

        raw_detections = defaultdict(list)

        # PHÂN NHÁNH 1: XỬ LÝ ẢNH TĨNH
        if is_image:
            print("[PROCESS] Định dạng Ảnh (Image) được phát hiện.")
            img = cv2.imread(temp_media_path)
            if img is None:
                raise ValueError("Không thể đọc file ảnh. Dữ liệu có thể đã bị hỏng trong quá trình upload.")
            
            # Quét AI 1 lần duy nhất
            frame_result = ai_service.analyze_frame(img)
            for cls_name, conf_list in frame_result.items():
                raw_detections[cls_name].extend(conf_list)

        # PHÂN NHÁNH 2: XỬ LÝ VIDEO
        elif is_video:
            print("[PROCESS] Định dạng Video được phát hiện. Đang kích hoạt ByteTrack...")
            
            # Hàm Tracking tự động stream video, nhóm ID vật thể và trả về danh sách Unique
            tracked_results = ai_service.analyze_video_with_tracking(temp_media_path)
            
            # Ghép danh sách Unique vào raw_detections để tương thích hoàn toàn với đoạn code đóng gói bên dưới
            for cls_name, conf_list in tracked_results.items():
                raw_detections[cls_name].extend(conf_list)
            
        else:
            raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")

        # TỔNG HỢP BÁO CÁO DỮ LIỆU
        final_detections = {}
        for obj_class, conf_list in raw_detections.items():
            final_detections[obj_class] = {
                "count": len(conf_list),
                "avg_confidence": round(sum(conf_list) / len(conf_list), 4),
                "min_confidence": round(min(conf_list), 4),
                "max_confidence": round(max(conf_list), 4)
            }

        payload = {
            "status": "SUCCESS",
            "teams_message_id": teams_message_id,
            "raw_video_key": media_key,
            "media_type": "image" if is_image else "video", # Thêm trường Metadata để n8n biết
            "detections": final_detections
        }
        
        # Bắn kết quả Async về Workflow 3 của n8n
        async with httpx.AsyncClient() as client:
            await client.post(callback_url, json=payload, timeout=30.0)
            print(f"[SUCCESS] Đã gửi báo cáo (Type: {'Ảnh' if is_image else 'Video'}) cho tin nhắn: {teams_message_id}")

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Quá trình AI đổ vỡ: {traceback.format_exc()}")
        error_payload = {
            "status": "ERROR",
            "teams_message_id": teams_message_id,
            "error_message": error_msg
        }
        async with httpx.AsyncClient() as client:
            await client.post(callback_url, json=error_payload)
            
    finally:
        # Graceful Cleanup: Giải phóng Ổ cứng
        if os.path.exists(temp_media_path):
            os.remove(temp_media_path)

# 6. ENDPOINT API ĐỊNH TUYẾN
@app.post("/api/v1/process-video", tags=["AI Inference"])
async def trigger_media_processing(req: MediaRequest, bg_tasks: BackgroundTasks):
    if ai_service is None:
        raise HTTPException(status_code=503, detail="Hệ thống AI đang bảo trì.")
    if s3_client is None:
        raise HTTPException(
            status_code=503,
            detail="R2 video pipeline is unavailable because boto3 is not installed.",
        )
        
    bg_tasks.add_task(
        process_media_pipeline, 
        req.video_key, 
        req.teams_message_id, 
        req.callback_url
    )
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"message": "Đã tiếp nhận file. Hệ thống đang phân loại ảnh/video ngầm."}
    )

    # CLI: uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
    # web: http://127.0.0.1:8080/docs
