import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

BACKEND_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.getenv(
        "YOLO_MODEL_PATH",
        "models/best_YOLO26s_PROD_1_SKU_v4.pt",
    )
)

if not MODEL_PATH.is_absolute():
    MODEL_PATH = BACKEND_ROOT / MODEL_PATH

TEMPLATE_PATH = BACKEND_ROOT / "templates" / "MauFileNhapHang.xlsx"
MAPPING_PATH = BACKEND_ROOT / "config" / "product_mapping.json"
RUNTIME_PATH = BACKEND_ROOT / "runtime"

APP_AUTH_USERNAME = os.getenv("APP_AUTH_USERNAME", "").strip()
APP_AUTH_PASSWORD = os.getenv("APP_AUTH_PASSWORD", "").strip()

# The browser never receives these credentials. The same-origin FastAPI gateway
# calls n8n server-side, while large image bytes continue to go directly to R2.
N8N_WEBHOOK_BASE = os.getenv(
    "N8N_WEBHOOK_BASE",
    "https://n8n.sharon-finefoods.com/webhook",
).strip().rstrip("/")
N8N_WEBHOOK_USERNAME = os.getenv(
    "N8N_WEBHOOK_USERNAME", APP_AUTH_USERNAME
).strip()
N8N_WEBHOOK_PASSWORD = os.getenv(
    "N8N_WEBHOOK_PASSWORD", APP_AUTH_PASSWORD
).strip()
N8N_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("N8N_REQUEST_TIMEOUT_SECONDS", "30")
)

DEFAULT_CONFIDENCE = float(
    os.getenv("BAKERY_CONFIDENCE", "0.55")
)
DEFAULT_IOU = float(
    os.getenv("BAKERY_IOU", "0.55")
)
DEFAULT_IMAGE_SIZE = int(
    os.getenv("BAKERY_IMAGE_SIZE", "768")
)
MAX_DETECTIONS = int(
    os.getenv("BAKERY_MAX_DETECTIONS", "300")
)
MAX_IMAGES_PER_JOB = int(
    os.getenv("MAX_IMAGES_PER_JOB", "50")
)
MAX_FILE_SIZE_BYTES = int(
    os.getenv("MAX_IMAGE_SIZE_BYTES", str(50 * 1024 * 1024))
)
MAX_JOB_UPLOAD_SIZE_BYTES = int(
    os.getenv("MAX_JOB_UPLOAD_SIZE_BYTES", str(160 * 1024 * 1024))
)

R2_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "").strip()
R2_PUBLIC_BASE_URL = os.getenv("R2_PUBLIC_BASE_URL", "").strip().rstrip("/")

KIOTVIET_RETAILER = os.getenv("KIOTVIET_RETAILER", "").strip()
KIOTVIET_CLIENT_ID = os.getenv("KIOTVIET_CLIENT_ID", "").strip()
KIOTVIET_CLIENT_SECRET = os.getenv("KIOTVIET_CLIENT_SECRET", "").strip()
KIOTVIET_BRANCH_NAME = os.getenv("KIOTVIET_BRANCH_NAME", "Warehouse").strip()
KIOTVIET_PURCHASE_BY_USERNAME = os.getenv(
    "KIOTVIET_PURCHASE_BY_USERNAME", ""
).strip()
KIOTVIET_SUPPLIER_CODE = os.getenv("KIOTVIET_SUPPLIER_CODE", "").strip()
KIOTVIET_CREATE_AS_DRAFT = env_bool("KIOTVIET_CREATE_AS_DRAFT", True)
KIOTVIET_AUTO_CREATE_DRAFT = env_bool("KIOTVIET_AUTO_CREATE_DRAFT", False)
KIOTVIET_DEFAULT_PURCHASE_PRICE = float(
    os.getenv("KIOTVIET_DEFAULT_PURCHASE_PRICE", "0")
)

RUNTIME_PATH.mkdir(parents=True, exist_ok=True)
