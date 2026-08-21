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
        "models/best_YOLO26s_26SKUs_2656_31SKUs.pt",
    )
)

if not MODEL_PATH.is_absolute():
    MODEL_PATH = BACKEND_ROOT / MODEL_PATH

TEMPLATE_PATH = BACKEND_ROOT / "templates" / "MauFileNhapHang.xlsx"
MAPPING_PATH = BACKEND_ROOT / "config" / "product_mapping.json"
RUNTIME_PATH = BACKEND_ROOT / "runtime"

# Hybrid inference keeps the proven YOLO detector on the fast path and loads
# the Foundation stack only when AUTO routing needs it (or an operator asks for
# FOUNDATION/COMPARE explicitly).  Large models and generated references live
# outside Git and are intentionally resolved relative to the backend folder.
HYBRID_ENABLED = env_bool("HYBRID_ENABLED", True)
HYBRID_DEFAULT_MODE = os.getenv("HYBRID_DEFAULT_MODE", "AUTO").strip().upper()
if HYBRID_DEFAULT_MODE not in {"AUTO", "YOLO", "FOUNDATION", "COMPARE"}:
    HYBRID_DEFAULT_MODE = "AUTO"


def _backend_path(env_name: str, default: str) -> Path:
    value = Path(os.getenv(env_name, default)).expanduser()
    return value if value.is_absolute() else BACKEND_ROOT / value


FOUNDATION_SAM_MODEL_PATH = _backend_path(
    "FOUNDATION_SAM_MODEL_PATH", "models/sam2.1_s.pt"
)
FOUNDATION_REFERENCE_PATH = _backend_path(
    "FOUNDATION_REFERENCE_PATH", "hybrid_data/reference_embeddings.npz"
)
FOUNDATION_REGISTRY_PATH = _backend_path(
    "FOUNDATION_REGISTRY_PATH", "config/hybrid_reference_registry.json"
)
FOUNDATION_DINO_MODEL = os.getenv(
    "FOUNDATION_DINO_MODEL", "facebook/dinov2-small"
).strip()
FOUNDATION_POINTS_STRIDE = max(
    16, int(os.getenv("FOUNDATION_POINTS_STRIDE", "96"))
)
FOUNDATION_SAM_MAX_SIDE = max(
    640, int(os.getenv("FOUNDATION_SAM_MAX_SIDE", "1280"))
)
FOUNDATION_MIN_MASK_AREA_RATIO = float(
    os.getenv("FOUNDATION_MIN_MASK_AREA_RATIO", "0.0015")
)
FOUNDATION_MAX_MASK_AREA_RATIO = float(
    os.getenv("FOUNDATION_MAX_MASK_AREA_RATIO", "0.35")
)
FOUNDATION_MAX_BOX_AREA_RATIO = float(
    os.getenv("FOUNDATION_MAX_BOX_AREA_RATIO", "0.22")
)
FOUNDATION_EDGE_MARGIN_RATIO = float(
    os.getenv("FOUNDATION_EDGE_MARGIN_RATIO", "0.005")
)
FOUNDATION_MASK_NMS_IOU = float(os.getenv("FOUNDATION_MASK_NMS_IOU", "0.72"))
FOUNDATION_MASK_QUALITY = float(os.getenv("FOUNDATION_MASK_QUALITY", "0.55"))
FOUNDATION_SIMILARITY_THRESHOLD = float(
    os.getenv("FOUNDATION_SIMILARITY_THRESHOLD", "0.72")
)
FOUNDATION_SIMILARITY_MARGIN = float(
    os.getenv("FOUNDATION_SIMILARITY_MARGIN", "0.04")
)
FOUNDATION_INSTANCE_SIMILARITY_THRESHOLD = float(
    os.getenv("FOUNDATION_INSTANCE_SIMILARITY_THRESHOLD", "0.60")
)
FOUNDATION_INSTANCE_SIMILARITY_MARGIN = float(
    os.getenv("FOUNDATION_INSTANCE_SIMILARITY_MARGIN", "0.02")
)
FOUNDATION_INSTANCE_MIN_AREA_FACTOR = float(
    os.getenv("FOUNDATION_INSTANCE_MIN_AREA_FACTOR", "0.35")
)
FOUNDATION_INSTANCE_MAX_AREA_FACTOR = float(
    os.getenv("FOUNDATION_INSTANCE_MAX_AREA_FACTOR", "2.50")
)
FOUNDATION_INSTANCE_BOX_COVERAGE_NMS = float(
    os.getenv("FOUNDATION_INSTANCE_BOX_COVERAGE_NMS", "0.45")
)
HYBRID_YOLO_FALLBACK_MARGIN = float(
    os.getenv("HYBRID_YOLO_FALLBACK_MARGIN", "0.04")
)
HYBRID_BOX_RESCUE_ENABLED = env_bool("HYBRID_BOX_RESCUE_ENABLED", True)
HYBRID_BOX_RESCUE_MIN_THRESHOLD_RATIO = float(
    os.getenv("HYBRID_BOX_RESCUE_MIN_THRESHOLD_RATIO", "0.50")
)
HYBRID_BOX_RESCUE_MIN_CONFIDENCE = float(
    os.getenv("HYBRID_BOX_RESCUE_MIN_CONFIDENCE", "0.05")
)
HYBRID_BOX_RESCUE_MAX_CANDIDATES = max(
    1, int(os.getenv("HYBRID_BOX_RESCUE_MAX_CANDIDATES", "30"))
)
HYBRID_BOX_RESCUE_EXPANSION_RATIO = float(
    os.getenv("HYBRID_BOX_RESCUE_EXPANSION_RATIO", "0.12")
)
HYBRID_BOX_RESCUE_EXISTING_COVERAGE = float(
    os.getenv("HYBRID_BOX_RESCUE_EXISTING_COVERAGE", "0.60")
)
PSEUDO_LABEL_ENABLED = env_bool("PSEUDO_LABEL_ENABLED", True)
PSEUDO_LABEL_ROOT = _backend_path(
    "PSEUDO_LABEL_ROOT", "runtime/hybrid_dataset"
)

APP_AUTH_USERNAME = os.getenv("APP_AUTH_USERNAME", "").strip()
APP_AUTH_PASSWORD = os.getenv("APP_AUTH_PASSWORD", "").strip()

# Developer settings are intentionally protected by a second key.  Existing
# installations remain operable by falling back to the application password,
# while production can set a distinct DEVELOPER_SETTINGS_KEY in .env.
DEVELOPER_SETTINGS_KEY = os.getenv(
    "DEVELOPER_SETTINGS_KEY", APP_AUTH_PASSWORD
).strip()
DEVELOPER_SETTINGS_PATH = _backend_path(
    "DEVELOPER_SETTINGS_PATH", "runtime/developer_settings.json"
)

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
    os.getenv("BAKERY_CONFIDENCE", "0.25")
)
DEFAULT_IOU = float(
    os.getenv("BAKERY_IOU", "0.50")
)
DEFAULT_IMAGE_SIZE = int(
    os.getenv("BAKERY_IMAGE_SIZE", "1024")
)
MAX_DETECTIONS = int(
    os.getenv("BAKERY_MAX_DETECTIONS", "300")
)
INFERENCE_BATCH_SIZE = max(
    1, int(os.getenv("BAKERY_INFERENCE_BATCH_SIZE", "4"))
)

# Every image still contains one product family, but one job may now contain a
# batch of different trays/photos belonging to the same business SKU.  The job
# router validates that invariant before it adds the per-image counts.
DUPLICATE_BOX_IOU = float(
    os.getenv("BAKERY_DUPLICATE_BOX_IOU", "0.85")
)
CROSS_CLASS_DUPLICATE_COVERAGE = float(
    os.getenv("BAKERY_CROSS_CLASS_DUPLICATE_COVERAGE", "0.85")
)
MIN_DOMINANT_PURITY = float(
    os.getenv("BAKERY_MIN_DOMINANT_PURITY", "0.90")
)
CONFLICT_REVIEW_MIN_DOMINANT_COUNT = max(
    2, int(os.getenv("BAKERY_CONFLICT_REVIEW_MIN_DOMINANT_COUNT", "2"))
)
# A single clipped detection at the image border can occasionally be assigned
# to another visual class even when the tray itself is unambiguous.  Keep this
# correction deliberately narrow: it never lowers the normal purity gate and
# only removes one isolated, weaker, border-clipped class outlier.
EDGE_CLASS_OUTLIER_ENABLED = env_bool(
    "BAKERY_EDGE_CLASS_OUTLIER_ENABLED", True
)
EDGE_CLASS_OUTLIER_MARGIN_RATIO = float(
    os.getenv("BAKERY_EDGE_CLASS_OUTLIER_MARGIN_RATIO", "0.01")
)
EDGE_CLASS_OUTLIER_CONFIDENCE_GAP = float(
    os.getenv("BAKERY_EDGE_CLASS_OUTLIER_CONFIDENCE_GAP", "0.10")
)
EDGE_CLASS_OUTLIER_MIN_DOMINANT_COUNT = max(
    2, int(os.getenv("BAKERY_EDGE_CLASS_OUTLIER_MIN_DOMINANT_COUNT", "2"))
)
MAX_IMAGES_PER_JOB = max(1, int(os.getenv("MAX_IMAGES_PER_JOB", "50")))
MAX_FILE_SIZE_BYTES = int(
    os.getenv("MAX_IMAGE_SIZE_BYTES", str(50 * 1024 * 1024))
)
MAX_JOB_UPLOAD_SIZE_BYTES = int(
    os.getenv("MAX_JOB_UPLOAD_SIZE_BYTES", str(200 * 1024 * 1024))
)
# Keep R2 transfers serial by default. The Windows Python runtime used by this
# workstation has crashed inside OpenSSL/libcrypto when several boto3 TLS
# transfers share a process. This remains configurable for a future runtime
# that has been verified to handle concurrent transfers safely.
R2_TRANSFER_WORKERS = max(1, int(os.getenv("R2_TRANSFER_WORKERS", "1")))

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
KIOTVIET_CREATE_AS_DRAFT = env_bool("KIOTVIET_CREATE_AS_DRAFT", False)
KIOTVIET_MERGE_DAILY_DRAFTS = env_bool("KIOTVIET_MERGE_DAILY_DRAFTS", True)
KIOTVIET_REPLACE_COMPLETED_ON_UPDATE_FAILURE = env_bool(
    "KIOTVIET_REPLACE_COMPLETED_ON_UPDATE_FAILURE", True
)
# Explicit confirmation is mandatory in the single-product intake flow.
KIOTVIET_AUTO_CREATE_DRAFT = False
KIOTVIET_DEFAULT_PURCHASE_PRICE = float(
    os.getenv("KIOTVIET_DEFAULT_PURCHASE_PRICE", "0")
)

MANUFACTURING_RPA_BASE_URL = os.getenv(
    "MANUFACTURING_RPA_BASE_URL", "http://127.0.0.1:8000"
).strip().rstrip("/")
MANUFACTURING_RPA_INTERNAL_TOKEN = os.getenv(
    "MANUFACTURING_RPA_INTERNAL_TOKEN", ""
).strip()
MANUFACTURING_RPA_TIMEOUT_SECONDS = max(
    10.0, float(os.getenv("MANUFACTURING_RPA_TIMEOUT_SECONDS", "180"))
)

RUNTIME_PATH.mkdir(parents=True, exist_ok=True)
