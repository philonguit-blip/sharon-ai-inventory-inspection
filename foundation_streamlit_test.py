from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch


# =========================================================
# 1. PROJECT PATH DISCOVERY
# =========================================================
THIS_FILE = Path(__file__).resolve()
SEARCH_ROOTS = [
    THIS_FILE.parent,
    THIS_FILE.parent.parent,
    Path.cwd(),
    Path.cwd().parent,
]


def find_backend_root() -> Path:
    for base in SEARCH_ROOTS:
        direct = base / "backend"
        if (direct / "app" / "services" / "foundation_inference_service.py").is_file():
            return direct.resolve()
        if (base / "app" / "services" / "foundation_inference_service.py").is_file():
            return base.resolve()

    raise RuntimeError(
        "Cannot find backend/app/services/foundation_inference_service.py. "
        "Place this Streamlit file in the project root or one folder below it."
    )


BACKEND_ROOT = find_backend_root()
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import (  # noqa: E402
    FOUNDATION_DINO_MODEL,
    FOUNDATION_EDGE_MARGIN_RATIO,
    FOUNDATION_MASK_NMS_IOU,
    FOUNDATION_MASK_QUALITY,
    FOUNDATION_MAX_BOX_AREA_RATIO,
    FOUNDATION_MAX_MASK_AREA_RATIO,
    FOUNDATION_MIN_MASK_AREA_RATIO,
    FOUNDATION_POINTS_STRIDE,
    FOUNDATION_REFERENCE_PATH,
    FOUNDATION_REGISTRY_PATH,
    FOUNDATION_SAM_MODEL_PATH,
    FOUNDATION_SIMILARITY_MARGIN,
    FOUNDATION_SIMILARITY_THRESHOLD,
)
from app.services.foundation_inference_service import (  # noqa: E402
    DinoEmbeddingEncoder,
    FoundationInferenceService,
)
from app.services.reference_autocrop_service import (  # noqa: E402
    ReferenceCropResult,
    auto_crop_reference,
    composite_reference_on_white,
    segment_tight_reference_foreground,
)
from app.services.video_frame_extraction_service import (  # noqa: E402
    SUPPORTED_VIDEO_EXTENSIONS,
    VideoFrameExtractionConfig,
    extract_video_frames,
)


# =========================================================
# 2. STREAMLIT PAGE
# =========================================================
st.set_page_config(
    page_title="Foundation Test Lab",
    page_icon="🧪",
    layout="wide",
)

st.title("Foundation Test Lab")
st.caption(
    "Standalone FOUNDATION-only test UI: SAM2 segmentation + DINOv2 reference matching. "
    "No YOLO, n8n, R2 or KiotViet is used. Local reference artifacts are changed "
    "only after an explicit Build & activate action in the reference manager."
)


# =========================================================
# 3. GENERAL HELPERS
# =========================================================
def fmt_pct(value: float | int | None) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "—"


def fmt_ms(value: float | int | None) -> str:
    try:
        return f"{float(value):,.0f} ms"
    except Exception:
        return "—"


def decode_bgr(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image.")
    return image


def resize_preview(image_bgr: np.ndarray, max_side: int = 1100) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    longest = max(height, width)

    if longest <= max_side:
        return image_bgr

    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def crop_from_box(
    image_bgr: np.ndarray,
    box: list[float],
    pad_ratio: float = 0.06,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]

    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio

    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))

    return image_bgr[y1:y2, x1:x2].copy()


def draw_numbered_boxes(
    image_bgr: np.ndarray,
    objects: list[dict[str, Any]],
) -> np.ndarray:
    canvas = image_bgr.copy()

    for index, item in enumerate(objects, start=1):
        x1, y1, x2, y2 = [
            int(round(float(value)))
            for value in item.get("box_xyxy", [0, 0, 0, 0])
        ]

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 140, 90),
            2,
        )

        cv2.putText(
            canvas,
            str(index),
            (x1, max(22, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 90, 58),
            2,
            cv2.LINE_AA,
        )

    return canvas


def load_registry() -> dict[str, Any]:
    path = Path(FOUNDATION_REGISTRY_PATH)

    if not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        products = payload.get("products")
        return products if isinstance(products, dict) else {}
    except Exception:
        return {}


def reference_distribution() -> pd.DataFrame:
    path = Path(FOUNDATION_REFERENCE_PATH)

    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "reference_key",
                "reference_count",
            ]
        )

    payload = np.load(path, allow_pickle=False)
    keys = np.asarray(payload["reference_keys"]).astype(str)

    unique, counts = np.unique(
        keys,
        return_counts=True,
    )

    return (
        pd.DataFrame(
            {
                "reference_key": unique,
                "reference_count": counts,
            }
        )
        .sort_values(
            ["reference_count", "reference_key"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


# =========================================================
# 3B. REFERENCE MANAGER HELPERS
# =========================================================
REFERENCE_LIBRARY_ROOT = (
    BACKEND_ROOT
    / "hybrid_data"
    / "references"
)
REFERENCE_BACKUP_ROOT = (
    BACKEND_ROOT
    / "hybrid_data"
    / "reference_backups"
)
PRODUCT_MAPPING_PATH = (
    BACKEND_ROOT
    / "config"
    / "product_mapping.json"
)
FOUNDATION_MANIFEST_PATH = (
    BACKEND_ROOT
    / "models"
    / "FOUNDATION_MANIFEST.json"
)
REFERENCE_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def _reference_safe_code(value: str) -> str:
    code = str(value or "").strip()
    if (
        not code
        or any(char in code for char in '\\/:*?"<>|')
        or code in {".", ".."}
    ):
        raise ValueError(
            "Product/reference code is empty or unsafe for a folder name."
        )
    return code


def _reference_images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in REFERENCE_IMAGE_EXTENSIONS
    )


def _read_registry_payload() -> dict[str, Any]:
    path = Path(FOUNDATION_REGISTRY_PATH)
    if not path.is_file():
        return {
            "schema_version": 1,
            "description": (
                "Foundation references managed by Foundation Test Lab."
            ),
            "products": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", 1)
    payload.setdefault("products", {})
    if not isinstance(payload["products"], dict):
        raise ValueError("registry.products must be an object")
    return payload


def _read_product_mapping_payload() -> dict[str, Any]:
    if not PRODUCT_MAPPING_PATH.is_file():
        raise FileNotFoundError(
            f"Product mapping not found: {PRODUCT_MAPPING_PATH}"
        )
    payload = json.loads(
        PRODUCT_MAPPING_PATH.read_text(encoding="utf-8")
    )
    payload.setdefault("catalog_products", [])
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sync_foundation_catalog_in_memory(
    mapping: dict[str, Any],
    *,
    product_code: str,
    product_name: str,
    display_name: str,
) -> None:
    """Expose a new Foundation-only direct SKU in the operator catalog."""
    existing_codes: set[str] = set()
    for item in (mapping.get("classes") or {}).values():
        code = str(item.get("product_code") or "").strip()
        if code:
            existing_codes.add(code.casefold())
        for member in item.get("members") or []:
            member_code = str(
                member.get("product_code") or ""
            ).strip()
            if member_code:
                existing_codes.add(member_code.casefold())

    catalog = mapping.setdefault("catalog_products", [])
    if not isinstance(catalog, list):
        raise ValueError(
            "product_mapping.catalog_products must be a list"
        )

    target = product_code.casefold()
    if target not in existing_codes:
        replacement = {
            "visual_class": product_code,
            "source_type": "foundation_reference",
            "product_code": product_code,
            "product_name": product_name,
            "display_name": display_name,
        }
        for index, item in enumerate(catalog):
            if (
                str(item.get("product_code") or "").casefold()
                == target
            ):
                catalog[index] = replacement
                break
        else:
            catalog.append(replacement)

    mapping["supported_product_count"] = int(
        mapping.get("business_sku_count")
        or len(existing_codes)
    ) + len(catalog)


def _decode_reference_upload(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("Image cannot be decoded.")
    return image


@st.cache_resource(show_spinner=False)
def _load_reference_crop_sam(device: str):
    from ultralytics import SAM

    model = SAM(str(Path(FOUNDATION_SAM_MODEL_PATH)))
    return model, device


@st.cache_resource(show_spinner=False)
def _load_reference_encoder(device: str) -> DinoEmbeddingEncoder:
    return DinoEmbeddingEncoder(
        FOUNDATION_DINO_MODEL,
        device,
    )


def _resolve_reference_device(choice: str) -> str:
    normalized = str(choice or "AUTO").strip().upper()
    if normalized == "CUDA":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was selected but torch.cuda.is_available() is False."
            )
        return "cuda:0"
    if normalized == "AUTO":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _tight_crop_reference(
    image: np.ndarray,
    *,
    max_side: int,
    padding: float,
    device: str,
    encoder: DinoEmbeddingEncoder | None = None,
    target_embeddings: np.ndarray | None = None,
    batch_size: int = 16,
) -> ReferenceCropResult:
    model, _ = _load_reference_crop_sam(device)
    result = auto_crop_reference(
        image,
        sam_model=model,
        max_side=max_side,
        padding=padding,
        device=device,
        encoder=encoder,
        target_embeddings=target_embeddings,
        batch_size=batch_size,
    )
    return result


def _encode_reference_image(
    image_bgr: np.ndarray,
    *,
    extension: str,
) -> tuple[bytes, np.ndarray]:
    normalized_extension = str(extension or ".jpg").strip().lower()
    if normalized_extension not in {".jpg", ".png"}:
        raise ValueError(f"Unsupported reference output extension: {extension}")
    encode_options = (
        [cv2.IMWRITE_JPEG_QUALITY, 94]
        if normalized_extension == ".jpg"
        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    ok, encoded = cv2.imencode(
        normalized_extension,
        image_bgr,
        encode_options,
    )
    if not ok:
        raise RuntimeError(
            f"Cannot encode reference image as {normalized_extension}."
        )

    raw = encoded.tobytes()
    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )
    return raw, rgb


def _prepare_reference_variants(
    image_bgr: np.ndarray,
    *,
    input_mode: str,
    background_mode: str,
    max_side: int,
    padding: float,
    feather_radius: float,
    device: str,
    encoder: DinoEmbeddingEncoder | None = None,
    target_embeddings: np.ndarray | None = None,
    batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prepare original/white reference variants without writing artifacts."""
    normalized_input = str(input_mode or "tight").strip().lower()
    normalized_background = str(background_mode or "original").strip().lower()
    if normalized_input not in {"tight", "auto_crop"}:
        raise ValueError(f"Unsupported reference input mode: {input_mode}")
    if normalized_background not in {"original", "white", "both"}:
        raise ValueError(f"Unsupported reference background mode: {background_mode}")

    crop_box: list[int] | None = None
    mask_ratio: float | None = None
    crop_metadata: dict[str, Any] = {}
    foreground_mask: np.ndarray | None = None

    if normalized_input == "auto_crop":
        crop_result = _tight_crop_reference(
            image_bgr,
            max_side=max(640, int(max_side)),
            padding=max(0.0, float(padding)),
            device=device,
            encoder=encoder,
            target_embeddings=target_embeddings,
            batch_size=max(1, int(batch_size)),
        )
        prepared_bgr = crop_result.crop_bgr
        crop_box = crop_result.box_xyxy
        mask_ratio = crop_result.mask_ratio
        crop_metadata = dict(crop_result.metadata)
        foreground_mask = crop_result.foreground_mask
    else:
        prepared_bgr = image_bgr.copy()
        height, width = prepared_bgr.shape[:2]
        crop_box = [0, 0, int(width), int(height)]
        crop_metadata = {"method": "already_tight_reference"}
        if normalized_background in {"white", "both"}:
            sam_model, _ = _load_reference_crop_sam(device)
            foreground_mask, segmentation_metadata = (
                segment_tight_reference_foreground(
                    prepared_bgr,
                    sam_model=sam_model,
                    max_side=max(640, int(max_side)),
                    device=device,
                )
            )
            crop_metadata.update(segmentation_metadata)
            mask_ratio = float(foreground_mask.mean())

    variants: list[dict[str, Any]] = []
    if normalized_background in {"original", "both"}:
        raw, rgb = _encode_reference_image(
            prepared_bgr,
            extension=".jpg",
        )
        variants.append(
            {
                "variant": "original",
                "extension": ".jpg",
                "image_bytes": raw,
                "rgb": rgb,
                "background_metadata": {"mode": "original"},
            }
        )

    if normalized_background in {"white", "both"}:
        if foreground_mask is None:
            raise RuntimeError(
                "Cannot create a safe white-background reference because the "
                "selected auto-crop has no matching SAM foreground mask. Review "
                "the crop or use an already-tight input image."
            )
        white_bgr, white_metadata = composite_reference_on_white(
            prepared_bgr,
            foreground_mask,
            feather_radius=max(0.0, float(feather_radius)),
        )
        raw, rgb = _encode_reference_image(
            white_bgr,
            extension=".png",
        )
        variants.append(
            {
                "variant": "white",
                "extension": ".png",
                "image_bytes": raw,
                "rgb": rgb,
                "background_metadata": white_metadata,
            }
        )

    return variants, {
        "input_mode": normalized_input,
        "background_mode": normalized_background,
        "crop_box": crop_box,
        "mask_ratio": mask_ratio,
        "crop_metadata": crop_metadata,
    }


def _existing_reference_hashes(
    reference_key: str,
) -> set[str]:
    target = REFERENCE_LIBRARY_ROOT / reference_key
    hashes: set[str] = set()
    for path in _reference_images_in(target):
        try:
            hashes.add(
                hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            )
        except OSError:
            continue
    return hashes


def _backup_reference_assets() -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    backup_dir = (
        REFERENCE_BACKUP_ROOT
        / timestamp
    )
    backup_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for source in [
        Path(FOUNDATION_REFERENCE_PATH),
        Path(FOUNDATION_REGISTRY_PATH),
        PRODUCT_MAPPING_PATH,
        FOUNDATION_MANIFEST_PATH,
    ]:
        if source.is_file():
            shutil.copy2(
                source,
                backup_dir / source.name,
            )

    return backup_dir


def _restore_reference_assets(
    backup_dir: Path,
) -> None:
    targets = {
        "reference_embeddings.npz": Path(
            FOUNDATION_REFERENCE_PATH
        ),
        Path(FOUNDATION_REGISTRY_PATH).name: Path(
            FOUNDATION_REGISTRY_PATH
        ),
        PRODUCT_MAPPING_PATH.name: PRODUCT_MAPPING_PATH,
        FOUNDATION_MANIFEST_PATH.name: FOUNDATION_MANIFEST_PATH,
    }

    for name, target in targets.items():
        source = backup_dir / name
        if source.is_file():
            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(source, target)


def _write_reference_npz_atomic(
    embeddings: np.ndarray,
    keys: np.ndarray,
) -> None:
    path = Path(FOUNDATION_REFERENCE_PATH)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with NamedTemporaryFile(
        mode="wb",
        suffix=".npz",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            embeddings=np.asarray(
                embeddings,
                dtype=np.float32,
            ),
            reference_keys=np.asarray(
                keys,
                dtype="U256",
            ),
        )

    temporary.replace(path)


def _load_reference_artifact() -> tuple[np.ndarray, np.ndarray]:
    path = Path(FOUNDATION_REFERENCE_PATH)
    if not path.is_file():
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype="U256"),
        )

    payload = np.load(
        path,
        allow_pickle=False,
    )
    embeddings = np.asarray(
        payload["embeddings"],
        dtype=np.float32,
    )
    keys = np.asarray(
        payload["reference_keys"]
    ).astype(str)

    if embeddings.ndim != 2:
        raise ValueError(
            "Existing reference embeddings must be a 2D array."
        )
    if len(keys) != len(embeddings):
        raise ValueError(
            "Existing NPZ has mismatched embeddings/reference_keys lengths."
        )

    return embeddings, keys


def _update_reference_manifest(
    embeddings: np.ndarray,
    keys: np.ndarray,
) -> None:
    if not FOUNDATION_MANIFEST_PATH.is_file():
        return

    payload = json.loads(
        FOUNDATION_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    path = Path(FOUNDATION_REFERENCE_PATH)
    payload.update(
        {
            "reference_sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest().upper(),
            "reference_count": int(len(keys)),
            "embedding_dimensions": (
                int(embeddings.shape[1])
                if embeddings.ndim == 2
                and embeddings.size
                else 0
            ),
            "visual_class_count": int(
                len(set(keys.tolist()))
            ),
            "built_at": datetime.now(
                timezone.utc
            ).date().isoformat(),
        }
    )
    _write_json_atomic(
        FOUNDATION_MANIFEST_PATH,
        payload,
    )


def _commit_reference_queue(
    queue: list[dict[str, Any]],
    *,
    device_choice: str,
    batch_size: int,
) -> dict[str, Any]:
    """Validate, encode and atomically activate all queued reference batches."""
    if not queue:
        raise ValueError("Reference queue is empty.")

    registry_payload = _read_registry_payload()
    registry_products = registry_payload["products"]
    mapping_payload = _read_product_mapping_payload()

    device = _resolve_reference_device(
        device_choice
    )
    old_embeddings, old_keys = (
        _load_reference_artifact()
    )
    crop_encoder: DinoEmbeddingEncoder | None = None

    prepared: list[dict[str, Any]] = []
    class_report: dict[str, dict[str, Any]] = {}
    seen_new_class_metadata: dict[
        str,
        tuple[str, str],
    ] = {}

    for batch in queue:
        reference_key = _reference_safe_code(
            batch["reference_key"]
        )
        is_new = bool(batch.get("is_new"))
        product_name = str(
            batch.get("product_name") or ""
        ).strip()
        display_name = str(
            batch.get("display_name")
            or product_name
            or reference_key
        ).strip()

        if is_new:
            if not product_name:
                raise ValueError(
                    f"New class {reference_key} requires a product name."
                )

            previous = seen_new_class_metadata.get(
                reference_key.casefold()
            )
            current = (
                product_name,
                display_name,
            )
            if previous is not None and previous != current:
                raise ValueError(
                    f"Conflicting metadata was queued for new class {reference_key}."
                )
            seen_new_class_metadata[
                reference_key.casefold()
            ] = current

            if (
                reference_key in registry_products
                and not batch.get(
                    "allow_existing_as_new",
                    False,
                )
            ):
                raise ValueError(
                    f"{reference_key} already exists. Queue it as an existing class."
                )
        elif reference_key not in registry_products:
            raise ValueError(
                f"Existing class not found in registry: {reference_key}"
            )

        report = class_report.setdefault(
            reference_key,
            {
                "reference_key": reference_key,
                "is_new": is_new,
                "uploaded": 0,
                "prepared": 0,
                "duplicates": 0,
                "failed": 0,
                "errors": [],
                "crop_methods": {},
                "background_variants": {},
                "crop_target_similarities": [],
                "before_library_count": len(
                    _reference_images_in(
                        REFERENCE_LIBRARY_ROOT
                        / reference_key
                    )
                ),
            },
        )

        existing_hashes = _existing_reference_hashes(
            reference_key
        )
        batch_hashes = {
            item["digest"]
            for item in prepared
            if item["reference_key"]
            == reference_key
        }

        for upload in batch.get("files") or []:
            report["uploaded"] += 1
            file_name = str(
                upload.get("name") or "image"
            )
            raw = upload.get("raw") or b""

            try:
                image_bgr = _decode_reference_upload(
                    raw
                )

                input_mode = str(
                    batch.get("input_mode")
                    or ("auto_crop" if batch.get("auto_crop") else "tight")
                )
                background_mode = str(
                    batch.get("background_mode") or "original"
                )
                target_positions = np.where(
                    old_keys == reference_key
                )[0]
                target_embeddings = (
                    old_embeddings[target_positions]
                    if len(target_positions)
                    else None
                )
                if input_mode == "auto_crop" and target_embeddings is not None:
                    if crop_encoder is None:
                        crop_encoder = _load_reference_encoder(device)

                variants, preparation_metadata = _prepare_reference_variants(
                    image_bgr,
                    input_mode=input_mode,
                    background_mode=background_mode,
                    max_side=max(
                        640,
                        int(batch.get("crop_max_side", 1280)),
                    ),
                    padding=max(
                        0.0,
                        float(batch.get("crop_padding", 0.08)),
                    ),
                    feather_radius=max(
                        0.0,
                        float(batch.get("feather_radius", 1.5)),
                    ),
                    device=device,
                    encoder=crop_encoder,
                    target_embeddings=target_embeddings,
                    batch_size=max(1, int(batch_size)),
                )
                crop_metadata = preparation_metadata["crop_metadata"]
                crop_method = str(
                    crop_metadata.get("method") or "sam_geometry"
                )
                report["crop_methods"][crop_method] = (
                    int(report["crop_methods"].get(crop_method, 0)) + 1
                )
                if "target_similarity" in crop_metadata:
                    report["crop_target_similarities"].append(
                        float(crop_metadata["target_similarity"])
                    )

                for variant in variants:
                    image_bytes = variant["image_bytes"]
                    digest = hashlib.sha256(image_bytes).hexdigest()
                    if digest in existing_hashes or digest in batch_hashes:
                        report["duplicates"] += 1
                        continue

                    prepared.append(
                        {
                            "reference_key": reference_key,
                            "product_name": product_name,
                            "display_name": display_name,
                            "is_new": is_new,
                            "source_name": file_name,
                            "image_bytes": image_bytes,
                            "extension": variant["extension"],
                            "variant": variant["variant"],
                            "rgb": variant["rgb"],
                            "digest": digest,
                            "crop_box": preparation_metadata["crop_box"],
                            "mask_ratio": preparation_metadata["mask_ratio"],
                            "crop_metadata": crop_metadata,
                            "background_metadata": variant[
                                "background_metadata"
                            ],
                        }
                    )
                    batch_hashes.add(digest)
                    report["prepared"] += 1
                    variant_name = str(variant["variant"])
                    report["background_variants"][variant_name] = (
                        int(
                            report["background_variants"].get(
                                variant_name,
                                0,
                            )
                        )
                        + 1
                    )

            except Exception as exc:
                report["failed"] += 1
                report["errors"].append(
                    f"{file_name}: {exc}"
                )

    if not prepared:
        raise ValueError(
            "No new valid reference images remained after validation/deduplication."
        )

    # New classes are registered only when at least one valid image survived.
    valid_new_keys = {
        item["reference_key"]
        for item in prepared
        if item["is_new"]
    }
    for reference_key in sorted(valid_new_keys):
        sample = next(
            item
            for item in prepared
            if item["reference_key"]
            == reference_key
        )
        registry_products[reference_key] = {
            "type": "direct",
            "product_code": reference_key,
            "product_name": sample["product_name"],
            "display_name": sample["display_name"],
        }
        _sync_foundation_catalog_in_memory(
            mapping_payload,
            product_code=reference_key,
            product_name=sample["product_name"],
            display_name=sample["display_name"],
        )

    encoder = crop_encoder or _load_reference_encoder(device)
    new_embeddings = encoder.encode_rgb(
        [item["rgb"] for item in prepared],
        batch_size=max(1, int(batch_size)),
    )

    if len(new_embeddings) != len(prepared):
        raise RuntimeError(
            "DINO returned an unexpected number of embeddings."
        )

    if old_embeddings.size:
        if (
            new_embeddings.ndim != 2
            or old_embeddings.shape[1]
            != new_embeddings.shape[1]
        ):
            raise ValueError(
                "New DINO embedding dimensions do not match the existing NPZ artifact."
            )
        merged_embeddings = np.concatenate(
            [old_embeddings, new_embeddings],
            axis=0,
        )
        merged_keys = np.concatenate(
            [
                old_keys.astype("U256"),
                np.asarray(
                    [
                        item["reference_key"]
                        for item in prepared
                    ],
                    dtype="U256",
                ),
            ],
            axis=0,
        )
    else:
        merged_embeddings = new_embeddings.astype(
            np.float32
        )
        merged_keys = np.asarray(
            [
                item["reference_key"]
                for item in prepared
            ],
            dtype="U256",
        )

    backup_dir = _backup_reference_assets()
    created_paths: list[Path] = []

    try:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        per_key_counter: dict[str, int] = {}

        for item in prepared:
            key = item["reference_key"]
            target_dir = (
                REFERENCE_LIBRARY_ROOT
                / key
            )
            target_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            counter = per_key_counter.get(
                key,
                0,
            ) + 1
            per_key_counter[key] = counter

            extension = str(item.get("extension") or ".jpg")
            variant_suffix = (
                "_white" if item.get("variant") == "white" else ""
            )
            destination = target_dir / (
                f"web_{timestamp}_{counter:04d}{variant_suffix}{extension}"
            )
            while destination.exists():
                counter += 1
                per_key_counter[key] = counter
                destination = target_dir / (
                    f"web_{timestamp}_{counter:04d}{variant_suffix}{extension}"
                )

            destination.write_bytes(
                item["image_bytes"]
            )
            created_paths.append(destination)

        _write_reference_npz_atomic(
            merged_embeddings,
            merged_keys,
        )
        _write_json_atomic(
            Path(FOUNDATION_REGISTRY_PATH),
            registry_payload,
        )
        _write_json_atomic(
            PRODUCT_MAPPING_PATH,
            mapping_payload,
        )
        _update_reference_manifest(
            merged_embeddings,
            merged_keys,
        )

    except Exception:
        for path in created_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        _restore_reference_assets(backup_dir)
        raise

    for key, report in class_report.items():
        report["added"] = sum(
            item["reference_key"] == key
            for item in prepared
        )
        report["after_library_count"] = (
            report["before_library_count"]
            + report["added"]
        )
        similarities = report.pop(
            "crop_target_similarities",
            [],
        )
        report["crop_target_similarity_min"] = (
            min(similarities) if similarities else None
        )
        report["crop_target_similarity_max"] = (
            max(similarities) if similarities else None
        )

    return {
        "device": device,
        "backup_dir": str(backup_dir),
        "added_total": len(prepared),
        "reference_rows_before": int(
            len(old_keys)
        ),
        "reference_rows_after": int(
            len(merged_keys)
        ),
        "class_count_after": int(
            len(set(merged_keys.tolist()))
        ),
        "classes": [
            class_report[key]
            for key in sorted(class_report)
        ],
    }


def _reference_image_metadata(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        size_bytes = int(stat.st_size)
        modified_at = datetime.fromtimestamp(
            stat.st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        size_bytes = 0
        modified_at = ""

    width = 0
    height = 0
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            height, width = image.shape[:2]
    except Exception:
        pass

    return {
        "path": path,
        "name": path.name,
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024 * 1024),
        "width": int(width),
        "height": int(height),
        "modified_at": modified_at,
    }


def _rebuild_reference_artifact_from_library(
    *,
    device_choice: str,
    batch_size: int,
) -> dict[str, Any]:
    """Rebuild the NPZ from the reference image library.

    Used after deleting references in the browser so the image library,
    `reference_keys` and DINO embeddings stay consistent.
    """
    registry_payload = _read_registry_payload()
    products = registry_payload.get("products") or {}
    if not products:
        raise ValueError("Foundation registry has no products.")

    image_paths: list[Path] = []
    reference_keys: list[str] = []

    for key in sorted(products):
        paths = _reference_images_in(
            REFERENCE_LIBRARY_ROOT / key
        )
        image_paths.extend(paths)
        reference_keys.extend([key] * len(paths))

    if not image_paths:
        raise ValueError(
            "Reference library is empty after deletion. "
            "At least one reference image is required."
        )

    images_rgb: list[np.ndarray] = []
    valid_keys: list[str] = []
    unreadable: list[str] = []

    for path, key in zip(
        image_paths,
        reference_keys,
    ):
        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            unreadable.append(str(path))
            continue
        images_rgb.append(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )
        )
        valid_keys.append(key)

    if not images_rgb:
        raise ValueError(
            "All remaining reference images are unreadable."
        )

    device = _resolve_reference_device(
        device_choice
    )
    encoder = _load_reference_encoder(device)
    embeddings = encoder.encode_rgb(
        images_rgb,
        batch_size=max(1, int(batch_size)),
    )

    if len(embeddings) != len(valid_keys):
        raise RuntimeError(
            "DINO returned an unexpected number of embeddings."
        )

    keys = np.asarray(
        valid_keys,
        dtype="U256",
    )
    _write_reference_npz_atomic(
        embeddings,
        keys,
    )
    _update_reference_manifest(
        embeddings,
        keys,
    )

    return {
        "device": device,
        "reference_rows_after": int(len(keys)),
        "class_count_after": int(
            len(set(keys.tolist()))
        ),
        "unreadable": unreadable,
    }


def _delete_reference_images(
    paths: list[Path],
    *,
    device_choice: str,
    batch_size: int,
) -> dict[str, Any]:
    """Delete selected references safely and rebuild the DINO artifact."""
    unique_paths: list[Path] = []
    seen: set[str] = set()

    library_root = REFERENCE_LIBRARY_ROOT.resolve()

    for raw_path in paths:
        path = Path(raw_path).resolve()
        if str(path) in seen:
            continue
        seen.add(str(path))

        try:
            path.relative_to(library_root)
        except ValueError as exc:
            raise ValueError(
                f"Refusing to delete a path outside the reference library: {path}"
            ) from exc

        if not path.is_file():
            continue

        unique_paths.append(path)

    if not unique_paths:
        raise ValueError(
            "No valid reference images were selected."
        )

    # Prevent deleting the last image of a class because a registry class with
    # zero embeddings can never be matched by Foundation.
    selected_by_class: dict[str, int] = {}
    for path in unique_paths:
        class_key = path.parent.name
        selected_by_class[class_key] = (
            selected_by_class.get(class_key, 0)
            + 1
        )

    for class_key, selected_count in selected_by_class.items():
        current_count = len(
            _reference_images_in(
                REFERENCE_LIBRARY_ROOT
                / class_key
            )
        )
        if selected_count >= current_count:
            raise ValueError(
                f"Cannot delete all references for {class_key}. "
                "Add replacement references first, or remove the class "
                "from the registry/catalog separately."
            )

    backup_dir = _backup_reference_assets()
    removed_root = (
        backup_dir
        / "removed_reference_images"
    )
    removed_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    moved: list[tuple[Path, Path]] = []

    try:
        for path in unique_paths:
            class_key = path.parent.name
            target_dir = (
                removed_root
                / class_key
            )
            target_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            backup_path = target_dir / path.name
            counter = 1
            while backup_path.exists():
                backup_path = (
                    target_dir
                    / f"{path.stem}_{counter}{path.suffix}"
                )
                counter += 1

            shutil.move(
                str(path),
                str(backup_path),
            )
            moved.append(
                (path, backup_path)
            )

        rebuild = (
            _rebuild_reference_artifact_from_library(
                device_choice=device_choice,
                batch_size=batch_size,
            )
        )

    except Exception:
        # Put the images back first, then restore artifacts.
        for original, backup_path in reversed(moved):
            try:
                original.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                if backup_path.is_file():
                    shutil.move(
                        str(backup_path),
                        str(original),
                    )
            except Exception:
                pass
        _restore_reference_assets(
            backup_dir
        )
        raise

    return {
        "deleted": len(moved),
        "backup_dir": str(backup_dir),
        **rebuild,
    }




# =========================================================
# 3C. FOUNDATION RESULT -> ROBOFLOW PRE-ANNOTATION EXPORT
# =========================================================
def _preannotation_safe_component(value: str) -> str:
    """Return a conservative filename component for dataset export."""
    cleaned = "".join(
        char
        if char.isalnum() or char in {"-", "_", "."}
        else "_"
        for char in str(value or "").strip()
    )
    cleaned = cleaned.strip("._")
    return cleaned or "image"


def _extract_uploaded_video(
    raw_bytes: bytes,
    *,
    source_name: str,
    config: VideoFrameExtractionConfig,
):
    """Bridge a Streamlit in-memory upload to OpenCV's file-based decoder."""
    suffix = Path(source_name).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        suffix = ".mp4"
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="foundation_video_",
            delete=False,
        ) as temporary:
            temporary.write(raw_bytes)
            temporary_path = Path(temporary.name)
        return extract_video_frames(
            temporary_path,
            source_name=source_name,
            config=config,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _preannotation_class_name(
    obj: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    """Resolve the Foundation visual/reference class assigned to one box."""
    return str(
        obj.get("raw_class")
        or obj.get("reference_key")
        or decision.get("dominant_class")
        or obj.get("display_name")
        or decision.get("display_name")
        or ""
    ).strip()


def _clamp_xyxy(
    box: list[float] | tuple[float, ...],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if len(box) < 4 or width <= 0 or height <= 0:
        return None

    x1, y1, x2, y2 = [float(value) for value in box[:4]]

    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _build_foundation_preannotation_zip(
    selected_results: list[dict[str, Any]],
    *,
    export_format: str,
    include_empty_images: bool,
    min_object_confidence: float,
    video_extraction_summary: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a Roboflow-ready pre-annotation dataset from Foundation results.

    Only Foundation's final retained `objects` are exported as bounding boxes.
    Raw SAM candidates rejected by semantic/geometry/duplicate filtering are
    intentionally not exported.

    Supported formats:
      - `yolo`: standard YOLO object-detection layout with data.yaml.
      - `coco`: COCO JSON layout with train/_annotations.coco.json.

    The ZIP also includes image-level and object-level CSV manifests so every
    pre-annotation can be audited before it becomes training ground truth.
    """
    export_format = str(export_format or "yolo").strip().lower()
    if export_format not in {"yolo", "coco"}:
        raise ValueError(f"Unsupported pre-annotation format: {export_format}")

    min_conf = float(np.clip(min_object_confidence, 0.0, 1.0))

    prepared_images: list[dict[str, Any]] = []
    observed_classes: set[str] = set()

    # ---------------------------------------------------------
    # First pass: validate images, boxes and collect class names.
    # ---------------------------------------------------------
    for source_index, item in enumerate(selected_results, start=1):
        if item.get("error"):
            continue

        result = item.get("result") or {}
        decision = result.get("decision") or {}
        raw_bytes = item.get("raw_bytes") or b""

        if not raw_bytes:
            continue

        width = int(result.get("width") or 0)
        height = int(result.get("height") or 0)

        if width <= 0 or height <= 0:
            try:
                decoded = decode_bgr(raw_bytes)
                height, width = decoded.shape[:2]
            except Exception:
                continue

        source_name = str(item.get("file_name") or f"image_{source_index:04d}.jpg")
        source_suffix = Path(source_name).suffix.lower()
        if source_suffix not in REFERENCE_IMAGE_EXTENSIONS:
            source_suffix = ".jpg"

        safe_stem = _preannotation_safe_component(Path(source_name).stem)
        export_stem = f"{source_index:04d}_{safe_stem}"
        export_image_name = export_stem + source_suffix

        exported_objects: list[dict[str, Any]] = []

        for obj_index, obj in enumerate(result.get("objects") or [], start=1):
            confidence = float(obj.get("confidence") or 0.0)
            if confidence < min_conf:
                continue

            class_name = _preannotation_class_name(obj, decision)
            if not class_name:
                continue

            clamped = _clamp_xyxy(
                obj.get("box_xyxy") or [],
                width,
                height,
            )
            if clamped is None:
                continue

            x1, y1, x2, y2 = clamped
            observed_classes.add(class_name)

            exported_objects.append(
                {
                    "object_index": obj_index,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box_xyxy": [x1, y1, x2, y2],
                    "display_name": str(
                        obj.get("display_name")
                        or decision.get("display_name")
                        or class_name
                    ),
                }
            )

        if not exported_objects and not include_empty_images:
            continue

        prepared_images.append(
            {
                "source_index": source_index,
                "source_name": source_name,
                "export_stem": export_stem,
                "export_image_name": export_image_name,
                "raw_bytes": raw_bytes,
                "width": width,
                "height": height,
                "decision": decision,
                "objects": exported_objects,
                "source_metadata": dict(item.get("source_metadata") or {}),
            }
        )

    if not prepared_images:
        raise ValueError(
            "No Foundation test images are eligible for export with the current filters."
        )

    class_names = sorted(observed_classes)
    class_to_yolo_id = {
        name: index
        for index, name in enumerate(class_names)
    }
    class_to_coco_id = {
        name: index + 1
        for index, name in enumerate(class_names)
    }

    image_manifest_rows: list[dict[str, Any]] = []
    object_manifest_rows: list[dict[str, Any]] = []

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        # -----------------------------------------------------
        # YOLO format
        # -----------------------------------------------------
        if export_format == "yolo":
            for image_item in prepared_images:
                archive.writestr(
                    f"train/images/{image_item['export_image_name']}",
                    image_item["raw_bytes"],
                )

                label_lines: list[str] = []

                for obj in image_item["objects"]:
                    x1, y1, x2, y2 = obj["box_xyxy"]
                    width = float(image_item["width"])
                    height = float(image_item["height"])

                    x_center = ((x1 + x2) / 2.0) / width
                    y_center = ((y1 + y2) / 2.0) / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height

                    class_id = class_to_yolo_id[obj["class_name"]]

                    label_lines.append(
                        f"{class_id} "
                        f"{x_center:.8f} "
                        f"{y_center:.8f} "
                        f"{box_width:.8f} "
                        f"{box_height:.8f}"
                    )

                archive.writestr(
                    f"train/labels/{image_item['export_stem']}.txt",
                    (
                        "\n".join(label_lines) + ("\n" if label_lines else "")
                    ).encode("utf-8"),
                )

            yaml_lines = [
                "path: .",
                "train: train/images",
                "val: train/images",
                f"nc: {len(class_names)}",
                "names:",
            ]

            for class_id, class_name in enumerate(class_names):
                yaml_lines.append(
                    f"  {class_id}: {json.dumps(class_name, ensure_ascii=False)}"
                )

            archive.writestr(
                "data.yaml",
                ("\n".join(yaml_lines) + "\n").encode("utf-8"),
            )

            archive.writestr(
                "classes.txt",
                ("\n".join(class_names) + ("\n" if class_names else "")).encode("utf-8"),
            )

        # -----------------------------------------------------
        # COCO format
        # -----------------------------------------------------
        else:
            coco_images: list[dict[str, Any]] = []
            coco_annotations: list[dict[str, Any]] = []
            annotation_id = 1

            for image_id, image_item in enumerate(prepared_images, start=1):
                archive.writestr(
                    f"train/{image_item['export_image_name']}",
                    image_item["raw_bytes"],
                )

                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": image_item["export_image_name"],
                        "width": int(image_item["width"]),
                        "height": int(image_item["height"]),
                    }
                )

                for obj in image_item["objects"]:
                    x1, y1, x2, y2 = obj["box_xyxy"]
                    box_width = x2 - x1
                    box_height = y2 - y1

                    coco_annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_to_coco_id[obj["class_name"]],
                            "bbox": [
                                round(x1, 4),
                                round(y1, 4),
                                round(box_width, 4),
                                round(box_height, 4),
                            ],
                            "area": round(box_width * box_height, 4),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

            coco_payload = {
                "info": {
                    "description": (
                        "Sharon Bakery Foundation Test Lab pre-annotations. "
                        "Predictions require manual review before use as ground truth."
                    ),
                    "date_created": datetime.now().isoformat(timespec="seconds"),
                },
                "licenses": [],
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [
                    {
                        "id": class_to_coco_id[class_name],
                        "name": class_name,
                        "supercategory": "bakery",
                    }
                    for class_name in class_names
                ],
            }

            archive.writestr(
                "train/_annotations.coco.json",
                json.dumps(
                    coco_payload,
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
            )

        # -----------------------------------------------------
        # Auditing manifests
        # -----------------------------------------------------
        for image_item in prepared_images:
            decision = image_item["decision"] or {}
            source_metadata = image_item.get("source_metadata") or {}

            image_manifest_rows.append(
                {
                    "exported_image": image_item["export_image_name"],
                    "source_image": image_item["source_name"],
                    "decision": str(decision.get("decision") or ""),
                    "dominant_class": str(
                        decision.get("dominant_class")
                        or decision.get("display_name")
                        or ""
                    ),
                    "foundation_count": int(decision.get("count") or 0),
                    "exported_box_count": len(image_item["objects"]),
                    "tray_similarity": float(
                        decision.get("tray_similarity")
                        or decision.get("avg_confidence")
                        or 0.0
                    ),
                    "similarity_margin": float(
                        decision.get("similarity_margin") or 0.0
                    ),
                    "width": int(image_item["width"]),
                    "height": int(image_item["height"]),
                    "source_type": str(
                        source_metadata.get("source_type") or "image"
                    ),
                    "source_video": str(
                        source_metadata.get("source_video") or ""
                    ),
                    "source_frame_idx": source_metadata.get(
                        "source_frame_idx", ""
                    ),
                    "timestamp_ms": source_metadata.get("timestamp_ms", ""),
                    "blur_score": source_metadata.get("blur_score", ""),
                    "brightness": source_metadata.get("brightness", ""),
                    "phash": str(source_metadata.get("phash") or ""),
                    "nearest_saved_phash_distance": source_metadata.get(
                        "nearest_saved_phash_distance", ""
                    ),
                    "source_width": source_metadata.get("source_width", ""),
                    "source_height": source_metadata.get("source_height", ""),
                    "saved_width": source_metadata.get("saved_width", ""),
                    "saved_height": source_metadata.get("saved_height", ""),
                }
            )

            for obj in image_item["objects"]:
                x1, y1, x2, y2 = obj["box_xyxy"]
                object_manifest_rows.append(
                    {
                        "exported_image": image_item["export_image_name"],
                        "source_image": image_item["source_name"],
                        "object_index": int(obj["object_index"]),
                        "class_name": obj["class_name"],
                        "display_name": obj["display_name"],
                        "confidence": float(obj["confidence"]),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "source_type": str(
                            source_metadata.get("source_type") or "image"
                        ),
                        "source_video": str(
                            source_metadata.get("source_video") or ""
                        ),
                        "source_frame_idx": source_metadata.get(
                            "source_frame_idx", ""
                        ),
                        "timestamp_ms": source_metadata.get("timestamp_ms", ""),
                    }
                )

        image_manifest_df = pd.DataFrame(image_manifest_rows)
        object_manifest_df = pd.DataFrame(
            object_manifest_rows,
            columns=[
                "exported_image",
                "source_image",
                "object_index",
                "class_name",
                "display_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "source_type",
                "source_video",
                "source_frame_idx",
                "timestamp_ms",
            ],
        )

        archive.writestr(
            "foundation_preannotation_images.csv",
            image_manifest_df.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            "foundation_preannotation_objects.csv",
            object_manifest_df.to_csv(index=False).encode("utf-8-sig"),
        )
        video_frame_manifest_df = image_manifest_df[
            image_manifest_df["source_type"] == "video_frame"
        ].copy()
        if not video_frame_manifest_df.empty:
            archive.writestr(
                "foundation_video_frames.csv",
                video_frame_manifest_df.to_csv(index=False).encode("utf-8-sig"),
            )
        video_summary_rows = list(video_extraction_summary or [])
        if video_summary_rows:
            archive.writestr(
                "foundation_video_summary.csv",
                pd.DataFrame(video_summary_rows)
                .to_csv(index=False)
                .encode("utf-8-sig"),
            )

        readme = (
            "Sharon Bakery - FOUNDATION Pre-annotation Export\n"
            "================================================\n\n"
            "Purpose\n"
            "-------\n"
            "This dataset was generated from the latest FOUNDATION Test Lab results.\n"
            "The bounding boxes are AI predictions, NOT reviewed ground truth.\n"
            "Upload/import the dataset into Roboflow and manually check every image,\n"
            "box, class label, missed object and false positive before training.\n\n"
            f"Format: {'YOLO Object Detection' if export_format == 'yolo' else 'COCO JSON'}\n"
            f"Images exported: {len(prepared_images)}\n"
            f"Boxes exported: {sum(len(item['objects']) for item in prepared_images)}\n"
            f"Classes present: {len(class_names)}\n"
            f"Minimum exported object confidence: {min_conf:.3f}\n"
            f"Include images with zero exported boxes: {include_empty_images}\n"
            f"Created: {datetime.now().isoformat(timespec='seconds')}\n\n"
            "Audit files\n"
            "-----------\n"
            "- foundation_preannotation_images.csv: one row per source image.\n"
            "- foundation_preannotation_objects.csv: one row per exported box,\n"
            "  including Foundation instance similarity and original pixel box.\n\n"
            "- foundation_video_frames.csv: video/frame/timestamp/quality lineage for\n"
            "  extracted video frames (included only when videos were processed).\n\n"
            "- foundation_video_summary.csv: sampling and rejection totals per source\n"
            "  video (included only when videos were processed).\n\n"
            "Important\n"
            "---------\n"
            "- Only final Foundation `objects` are exported.\n"
            "- Rejected raw SAM masks are NOT exported.\n"
            "- Empty label files/images can represent Foundation misses and are useful\n"
            "  for manual correction when enabled.\n"
        )

        archive.writestr(
            "README_FOUNDATION_PREANNOTATION.txt",
            readme.encode("utf-8"),
        )

    buffer.seek(0)
    zip_bytes = buffer.getvalue()

    return {
        "zip_bytes": zip_bytes,
        "image_count": len(prepared_images),
        "box_count": sum(len(item["objects"]) for item in prepared_images),
        "class_count": len(class_names),
        "class_names": class_names,
        "size_bytes": len(zip_bytes),
        "format": export_format,
        "video_count": len(
            {
                str(row.get("video") or "")
                for row in (video_extraction_summary or [])
                if row.get("video")
            }
        ),
        "video_frame_count": sum(
            1
            for item in prepared_images
            if (item.get("source_metadata") or {}).get("source_type")
            == "video_frame"
        ),
    }


# =========================================================
# 4. FOUNDATION ENGINE CONFIG / CACHE
# =========================================================
def default_config() -> dict[str, Any]:
    return {
        "points_stride": int(FOUNDATION_POINTS_STRIDE),
        "min_area_ratio": float(FOUNDATION_MIN_MASK_AREA_RATIO),
        "max_area_ratio": float(FOUNDATION_MAX_MASK_AREA_RATIO),
        "max_box_area_ratio": float(FOUNDATION_MAX_BOX_AREA_RATIO),
        "edge_margin_ratio": float(FOUNDATION_EDGE_MARGIN_RATIO),
        "mask_nms_iou": float(FOUNDATION_MASK_NMS_IOU),
        "mask_quality": float(FOUNDATION_MASK_QUALITY),
        "similarity_threshold": float(FOUNDATION_SIMILARITY_THRESHOLD),
        "similarity_margin": float(FOUNDATION_SIMILARITY_MARGIN),
        "device": "AUTO",
    }


if "active_foundation_config" not in st.session_state:
    st.session_state.active_foundation_config = default_config()

if "foundation_results" not in st.session_state:
    st.session_state.foundation_results = []

if "foundation_run_id" not in st.session_state:
    st.session_state.foundation_run_id = 0

if "reference_manager_queue" not in st.session_state:
    st.session_state.reference_manager_queue = []

if "reference_manager_upload_nonce" not in st.session_state:
    st.session_state.reference_manager_upload_nonce = 0

if "reference_manager_last_result" not in st.session_state:
    st.session_state.reference_manager_last_result = None

if "reference_browser_delete_result" not in st.session_state:
    st.session_state.reference_browser_delete_result = None

if "reference_browser_page" not in st.session_state:
    st.session_state.reference_browser_page = 0


if "foundation_preannotation_zip" not in st.session_state:
    st.session_state.foundation_preannotation_zip = None

if "foundation_preannotation_filename" not in st.session_state:
    st.session_state.foundation_preannotation_filename = ""

if "foundation_preannotation_summary" not in st.session_state:
    st.session_state.foundation_preannotation_summary = None

if "foundation_video_extraction_summary" not in st.session_state:
    st.session_state.foundation_video_extraction_summary = []



def active_config_tuple() -> tuple:
    config = st.session_state.active_foundation_config

    return (
        config["points_stride"],
        config["min_area_ratio"],
        config["max_area_ratio"],
        config["max_box_area_ratio"],
        config["edge_margin_ratio"],
        config["mask_nms_iou"],
        config["mask_quality"],
        config["similarity_threshold"],
        config["similarity_margin"],
        config["device"],
    )


@st.cache_resource(show_spinner=False)
def load_foundation_service(
    config_key: tuple,
) -> FoundationInferenceService:
    (
        points_stride,
        min_area_ratio,
        max_area_ratio,
        max_box_area_ratio,
        edge_margin_ratio,
        mask_nms_iou,
        mask_quality,
        similarity_threshold,
        similarity_margin,
        device,
    ) = config_key

    resolved_device = (
        None
        if str(device).upper() == "AUTO"
        else str(device)
    )

    return FoundationInferenceService(
        sam_model_path=FOUNDATION_SAM_MODEL_PATH,
        reference_path=FOUNDATION_REFERENCE_PATH,
        registry_path=FOUNDATION_REGISTRY_PATH,
        dino_model=FOUNDATION_DINO_MODEL,
        points_stride=int(points_stride),
        min_area_ratio=float(min_area_ratio),
        max_area_ratio=float(max_area_ratio),
        max_box_area_ratio=float(max_box_area_ratio),
        edge_margin_ratio=float(edge_margin_ratio),
        mask_nms_iou=float(mask_nms_iou),
        mask_quality=float(mask_quality),
        similarity_threshold=float(similarity_threshold),
        similarity_margin=float(similarity_margin),
        device=resolved_device,
    )


# =========================================================
# 5. DIAGNOSTIC FOUNDATION RUN
#    Mirrors the current production service logic, but records
#    detailed timings and top-reference scores in the test app.
# =========================================================
def build_reference_ranking(
    service: FoundationInferenceService,
    embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    if (
        embeddings.size == 0
        or service.reference_embeddings.size == 0
    ):
        return []

    similarities = (
        embeddings
        @ service.reference_embeddings.T
    )

    rows: list[dict[str, Any]] = []

    for key in sorted(
        set(
            service.reference_keys.tolist()
        )
    ):
        positions = np.where(
            service.reference_keys == key
        )[0]

        per_instance = (
            similarities[:, positions]
            .max(axis=1)
        )

        score = float(
            np.median(per_instance)
        )

        registry_item = (
            service.registry.get(key)
            or {}
        )

        rows.append(
            {
                "reference_key": key,
                "display_name": (
                    registry_item.get(
                        "display_name"
                    )
                    or registry_item.get(
                        "product_name"
                    )
                    or key
                ),
                "median_similarity": score,
            }
        )

    rows.sort(
        key=lambda item:
            item["median_similarity"],
        reverse=True,
    )

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        row["rank"] = rank

    return rows


def run_foundation_diagnostic(
    service: FoundationInferenceService,
    raw_bytes: bytes,
    *,
    image_name: str,
) -> dict[str, Any]:
    total_started = time.perf_counter()

    decode_started = time.perf_counter()
    image_bgr = service._decode(raw_bytes)
    decode_ms = (
        time.perf_counter()
        - decode_started
    ) * 1000.0

    was_loaded = bool(
        service._loaded
    )

    print(
        f"[FOUNDATION-TEST] START "
        f"name={image_name} "
        f"bytes={len(raw_bytes)} "
        f"engine_loaded={was_loaded}",
        flush=True,
    )

    load_ms = 0.0
    sam_ms = 0.0
    crop_ms = 0.0
    dino_encode_ms = 0.0
    classify_ms = 0.0

    segments: list[dict[str, Any]] = []
    embeddings = np.empty(
        (0, 0),
        dtype=np.float32,
    )
    reference_key = ""
    similarity = 0.0
    margin = 0.0
    instance_scores = np.empty((0,), dtype=np.float32)
    instance_margins = np.empty((0,), dtype=np.float32)
    semantic_keep = np.empty((0,), dtype=bool)
    area_keep = np.empty((0,), dtype=bool)
    final_keep = np.empty((0,), dtype=bool)
    median_area = None
    semantic_duplicates_removed = 0

    with service._lock:
        load_started = (
            time.perf_counter()
        )
        service._ensure_loaded()
        load_ms = (
            time.perf_counter()
            - load_started
        ) * 1000.0

        sam_started = (
            time.perf_counter()
        )
        segments = service._segment(
            image_bgr
        )
        sam_ms = (
            time.perf_counter()
            - sam_started
        ) * 1000.0

        if segments:
            assert (
                service.encoder
                is not None
            )

            crop_started = (
                time.perf_counter()
            )
            crops_rgb = (
                service._masked_crops(
                    image_bgr,
                    segments,
                )
            )
            crop_ms = (
                time.perf_counter()
                - crop_started
            ) * 1000.0

            encode_started = (
                time.perf_counter()
            )
            embeddings = (
                service.encoder.encode_rgb(
                    crops_rgb
                )
            )
            dino_encode_ms = (
                time.perf_counter()
                - encode_started
            ) * 1000.0

            classify_started = (
                time.perf_counter()
            )
            similarities = service._similarities(
                embeddings
            )
            (
                reference_key,
                similarity,
                margin,
            ) = service._classify_tray_from_similarities(similarities)
            (
                instance_scores,
                instance_margins,
                semantic_keep,
            ) = service._instance_semantic_scores(similarities, reference_key)
            area_keep, median_area = service._robust_area_keep(
                segments, semantic_keep
            )
            final_keep, semantic_duplicates_removed = service._semantic_duplicate_keep(
                segments, instance_scores, area_keep
            )
            classify_ms = (
                time.perf_counter()
                - classify_started
            ) * 1000.0

    ranking = (
        build_reference_ranking(
            service,
            embeddings,
        )
        if len(segments)
        else []
    )

    registry_item = dict(
        service.registry.get(
            reference_key
        )
        or {}
    )

    display_name = str(
        registry_item.get(
            "display_name"
        )
        or registry_item.get(
            "product_name"
        )
        or reference_key
        or "Unknown"
    )

    confident = bool(
        segments
        and similarity
        >= service.similarity_threshold
        and margin
        >= service.similarity_margin
        and registry_item
    )

    objects: list[
        dict[str, Any]
    ] = []

    for index in np.flatnonzero(final_keep).tolist():
        segment = segments[index]
        item = {
            "class_id": -1,
            "raw_class": reference_key,
            "mapping_type": str(
                registry_item.get("type")
                or "direct"
            ),
            "display_name": (
                display_name
            ),
            "confidence_threshold": (
                service.instance_similarity_threshold
            ),
            "confidence": float(
                np.clip(instance_scores[index], 0.0, 1.0)
            ),
            "box_xyxy": (
                segment["box_xyxy"]
            ),
            "foundation_mask_index": (
                index
            ),
            "foundation_instance_margin": float(instance_margins[index]),
        }

        if registry_item.get(
            "product_code"
        ):
            item.update(
                {
                    "product_code": (
                        registry_item[
                            "product_code"
                        ]
                    ),
                    "product_name": (
                        registry_item.get(
                            "product_name",
                            display_name,
                        )
                    ),
                    "purchase_price": 0.0,
                }
            )

        objects.append(item)

    if not segments:
        decision = {
            "decision": (
                "NO_DETECTION"
            ),
            "count": 0,
            "purity": 0.0,
            "avg_confidence": 0.0,
            "requires_confirmation": (
                False
            ),
            "message": (
                "Foundation engine "
                "could not isolate "
                "bakery items."
            ),
        }

    elif not confident:
        decision = {
            "decision": "AMBIGUOUS",
            "dominant_class": (
                reference_key
            ),
            "display_name": (
                display_name
            ),
            "count": len(objects),
            "total_detections": len(objects),
            "purity": 1.0,
            "avg_confidence": float(
                max(
                    0.0,
                    min(
                        1.0,
                        similarity,
                    ),
                )
            ),
            "similarity_margin": (
                margin
            ),
            "requires_confirmation": (
                False
            ),
            "message": (
                "Foundation similarity "
                "is below the safe "
                "threshold; manual "
                "review is required."
            ),
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
            "similarity_margin": margin,
            "requires_confirmation": False,
            "message": "Per-instance DINO validation rejected every SAM mask.",
        }

    else:
        mapping_type = str(
            registry_item.get("type")
            or "direct"
        ).lower()

        decision = {
            "decision": (
                "FAMILY"
                if mapping_type
                == "family"
                else "DIRECT"
            ),
            "dominant_class": (
                reference_key
            ),
            "display_name": (
                display_name
            ),
            "product_code": (
                registry_item.get(
                    "product_code"
                )
            ),
            "product_name": (
                registry_item.get(
                    "product_name",
                    display_name,
                )
            ),
            "members": list(
                registry_item.get(
                    "members"
                )
                or []
            ),
            "count": len(objects),
            "total_detections": len(objects),
            "purity": 1.0,
            "avg_confidence": float(np.mean([item["confidence"] for item in objects])),
            "min_confidence": float(np.min([item["confidence"] for item in objects])),
            "max_confidence": float(np.max([item["confidence"] for item in objects])),
            "confidence_threshold": (
                service.instance_similarity_threshold
            ),
            "similarity_margin": (
                margin
            ),
            "requires_user_selection": (
                mapping_type
                == "family"
            ),
            "requires_confirmation": (
                True
            ),
            "message": (
                "Foundation reference "
                "match is ready for "
                "operator confirmation."
            ),
        }

    products = []

    if (
        decision["decision"]
        == "DIRECT"
    ):
        products = [
            {
                "product_code": (
                    decision[
                        "product_code"
                    ]
                ),
                "product_name": (
                    decision[
                        "product_name"
                    ]
                ),
                "purchase_price": 0,
                "quantity": (
                    decision["count"]
                ),
            }
        ]

    height, width = (
        image_bgr.shape[:2]
    )

    total_ms = (
        time.perf_counter()
        - total_started
    ) * 1000.0

    timing = {
        "decode_ms": decode_ms,
        "model_load_ms": load_ms,
        "sam_segmentation_ms": sam_ms,
        "crop_prepare_ms": crop_ms,
        "dino_encode_ms": (
            dino_encode_ms
        ),
        "dino_classify_ms": (
            classify_ms
        ),
        "total_inference_ms": (
            total_ms
        ),
        "engine_was_loaded": (
            was_loaded
        ),
    }

    result = {
        "image_name": image_name,
        "sha256": hashlib.sha256(
            raw_bytes
        ).hexdigest(),
        "width": int(width),
        "height": int(height),
        "raw_detections_before_class_filter": (
            len(segments)
        ),
        "detections_removed_by_class_filter": (
            len(segments) - len(objects)
        ),
        "detections_removed_as_duplicates": (
            int(service._last_segmentation_stats.get("duplicates_removed", 0))
            + int(semantic_duplicates_removed)
        ),
        "total_detections": (
            len(objects)
        ),
        "confidence_sum": float(
            sum(
                item["confidence"]
                for item in objects
            )
        ),
        "avg_confidence": float(
            decision.get(
                "avg_confidence"
            )
            or 0.0
        ),
        "inference_ms": (
            total_ms
        ),
        "decision": decision,
        "products": products,
        "objects": objects,
        "annotated_path": None,
        "engine": "FOUNDATION",
        "status": "SUCCESS",
        "error": "",
        "foundation_debug": {
            "timing": timing,
            "reference_ranking": (
                ranking
            ),
            "filtering": {
                "semantic_kept": int(np.count_nonzero(semantic_keep)),
                "geometry_kept": int(np.count_nonzero(area_keep)),
                "semantic_duplicates_removed": int(semantic_duplicates_removed),
                "final_kept": int(np.count_nonzero(final_keep)),
                "median_instance_area": median_area,
                "segmentation": dict(service._last_segmentation_stats),
            },
        },
    }

    print(
        "[FOUNDATION-TEST][TIMING] "
        f"name={image_name} "
        f"load={load_ms:.1f}ms "
        f"sam={sam_ms:.1f}ms "
        f"crop={crop_ms:.1f}ms "
        f"dino_encode={dino_encode_ms:.1f}ms "
        f"dino_classify={classify_ms:.1f}ms "
        f"total={total_ms:.1f}ms",
        flush=True,
    )

    print(
        "[FOUNDATION-TEST] DONE "
        f"name={image_name} "
        f"decision={decision.get('decision')} "
        f"class={decision.get('display_name') or decision.get('dominant_class')} "
        f"count={decision.get('count', 0)} "
        f"similarity={float(decision.get('avg_confidence') or 0):.4f} "
        f"margin={float(decision.get('similarity_margin') or 0):.4f}",
        flush=True,
    )

    return result


# =========================================================
# 6. SIDEBAR
# =========================================================
with st.sidebar:
    st.header(
        "Foundation configuration"
    )
    st.caption(
        "Changes apply only to this "
        "test app. Production config "
        "is not modified."
    )

    current = (
        st.session_state
        .active_foundation_config
    )

    points_stride = st.number_input(
        "SAM points stride",
        min_value=16,
        max_value=256,
        value=int(
            current[
                "points_stride"
            ]
        ),
        step=8,
        help=(
            "Lower = denser SAM prompt "
            "grid, usually more recall "
            "but slower."
        ),
    )

    mask_quality = st.slider(
        "SAM mask quality/conf",
        min_value=0.0,
        max_value=1.0,
        value=float(
            current[
                "mask_quality"
            ]
        ),
        step=0.01,
    )

    mask_nms_iou = st.slider(
        "Mask NMS IoU",
        min_value=0.0,
        max_value=1.0,
        value=float(
            current[
                "mask_nms_iou"
            ]
        ),
        step=0.01,
    )

    with st.expander(
        "Mask geometry filters",
        expanded=False,
    ):
        min_area_ratio = (
            st.number_input(
                "Min mask area ratio",
                min_value=0.0001,
                max_value=0.10,
                value=float(
                    current[
                        "min_area_ratio"
                    ]
                ),
                step=0.0001,
                format="%.4f",
            )
        )

        max_area_ratio = (
            st.number_input(
                "Max mask area ratio",
                min_value=0.01,
                max_value=1.0,
                value=float(
                    current[
                        "max_area_ratio"
                    ]
                ),
                step=0.01,
                format="%.3f",
            )
        )

        max_box_area_ratio = (
            st.number_input(
                "Max box area ratio",
                min_value=0.01,
                max_value=1.0,
                value=float(
                    current[
                        "max_box_area_ratio"
                    ]
                ),
                step=0.01,
                format="%.3f",
            )
        )

        edge_margin_ratio = (
            st.number_input(
                "Edge margin ratio",
                min_value=0.0,
                max_value=0.10,
                value=float(
                    current[
                        "edge_margin_ratio"
                    ]
                ),
                step=0.001,
                format="%.4f",
            )
        )

    st.divider()

    similarity_threshold = (
        st.slider(
            "DINO similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(
                current[
                    "similarity_threshold"
                ]
            ),
            step=0.01,
        )
    )

    similarity_margin = (
        st.slider(
            "DINO similarity margin",
            min_value=0.0,
            max_value=0.50,
            value=float(
                current[
                    "similarity_margin"
                ]
            ),
            step=0.01,
        )
    )

    device = st.selectbox(
        "Device",
        options=[
            "AUTO",
            "cuda:0",
            "cpu",
        ],
        index=[
            "AUTO",
            "cuda:0",
            "cpu",
        ].index(
            current.get(
                "device",
                "AUTO",
            )
        ),
    )

    if st.button(
        "Apply & reload engine",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.active_foundation_config = {
            "points_stride": int(
                points_stride
            ),
            "min_area_ratio": float(
                min_area_ratio
            ),
            "max_area_ratio": float(
                max_area_ratio
            ),
            "max_box_area_ratio": float(
                max_box_area_ratio
            ),
            "edge_margin_ratio": float(
                edge_margin_ratio
            ),
            "mask_nms_iou": float(
                mask_nms_iou
            ),
            "mask_quality": float(
                mask_quality
            ),
            "similarity_threshold": float(
                similarity_threshold
            ),
            "similarity_margin": float(
                similarity_margin
            ),
            "device": str(
                device
            ),
        }

        load_foundation_service.clear()

        st.session_state.foundation_results = []

        st.success(
            "Applied. The test engine "
            "will reload on next use."
        )

    if st.button(
        "Reset to backend defaults",
        use_container_width=True,
    ):
        st.session_state.active_foundation_config = (
            default_config()
        )

        load_foundation_service.clear()

        st.session_state.foundation_results = []

        st.rerun()


# =========================================================
# 7. ENGINE HEALTH
# =========================================================
service = load_foundation_service(
    active_config_tuple()
)

health = service.health()

st.subheader(
    "Foundation engine status"
)

health_cols = st.columns(6)

health_cols[0].metric(
    "Ready",
    "YES"
    if health.get("ready")
    else "NO",
)

health_cols[1].metric(
    "Loaded",
    "YES"
    if health.get("loaded")
    else "NO",
)

health_cols[2].metric(
    "Device",
    str(
        health.get("device")
        or "—"
    ),
)

health_cols[3].metric(
    "SAM stride",
    int(
        health.get(
            "points_stride"
        )
        or 0
    ),
)

health_cols[4].metric(
    "Similarity",
    f"{float(health.get('similarity_threshold') or 0):.2f}",
)

health_cols[5].metric(
    "Margin",
    f"{float(health.get('similarity_margin') or 0):.2f}",
)

asset_rows = [
    {
        "asset": key,
        "available": bool(
            value
        ),
    }
    for key, value in (
        health.get("assets")
        or {}
    ).items()
]

with st.expander(
    "Assets & paths",
    expanded=not bool(
        health.get("ready")
    ),
):
    st.dataframe(
        pd.DataFrame(
            asset_rows
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.code(
        "\n".join(
            [
                f"backend: {BACKEND_ROOT}",
                f"SAM: {health.get('sam_model_path')}",
                f"references: {health.get('reference_path')}",
                f"registry: {health.get('registry_path')}",
                f"DINO: {health.get('dino_model')}",
            ]
        ),
        language="text",
    )

    if health.get(
        "load_error"
    ):
        st.error(
            str(
                health[
                    "load_error"
                ]
            )
        )

if not health.get("ready"):
    st.error(
        "Foundation is not ready. "
        "Fix the missing assets or "
        "dependencies above."
    )


# =========================================================
# 8. OPTIONAL ENGINE WARM-UP
# =========================================================
warm_col, warm_info = st.columns(
    [0.28, 0.72]
)

with warm_col:
    warm_clicked = st.button(
        "Load engine only",
        disabled=not bool(
            health.get("ready")
        ),
        use_container_width=True,
        help=(
            "Load SAM2, DINOv2, "
            "reference embeddings and "
            "registry without running "
            "an image. Useful to separate "
            "startup time from inference."
        ),
    )

with warm_info:
    st.caption(
        "For clean timing, load the engine once first. "
        "Then subsequent image tests report SAM/DINO runtime without model startup."
    )

if warm_clicked:
    warm_status = st.empty()
    warm_status.info(
        "Loading SAM2 + DINOv2 + references..."
    )

    started = time.perf_counter()

    try:
        with service._lock:
            service._ensure_loaded()

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000.0

        warm_status.success(
            f"Engine loaded in {elapsed / 1000:.2f} s."
        )

        print(
            "[FOUNDATION-TEST][LOAD] "
            f"engine_loaded elapsed_ms={elapsed:.1f}",
            flush=True,
        )

    except Exception as exc:
        warm_status.error(
            f"Cannot load Foundation engine: {exc}"
        )

        print(
            "[FOUNDATION-TEST][LOAD][ERROR] "
            f"{exc}",
            flush=True,
        )


# =========================================================
# 9. REFERENCE INSPECTOR
# =========================================================
registry = load_registry()

with st.expander(
    "Reference registry",
    expanded=False,
):
    registry_rows = []

    for key, item in registry.items():
        registry_rows.append(
            {
                "reference_key": key,
                "type": item.get(
                    "type",
                    "direct",
                ),
                "display_name": (
                    item.get(
                        "display_name"
                    )
                    or item.get(
                        "product_name"
                    )
                    or key
                ),
                "product_code": (
                    item.get(
                        "product_code",
                        "",
                    )
                ),
                "family_members": len(
                    item.get(
                        "members"
                    )
                    or []
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            registry_rows
        ),
        hide_index=True,
        use_container_width=True,
    )

    try:
        distribution = (
            reference_distribution()
        )

        st.caption(
            "Reference embedding rows: "
            f"{int(distribution['reference_count'].sum()) if not distribution.empty else 0}"
        )

        st.dataframe(
            distribution,
            hide_index=True,
            use_container_width=True,
        )

    except Exception as exc:
        st.warning(
            "Could not inspect reference "
            f"NPZ: {exc}"
        )


# =========================================================
# 10. REFERENCE MANAGER
# =========================================================
st.subheader("Reference Manager")
st.caption(
    "Add approved references for one or many classes without PowerShell. "
    "Existing classes append references; new classes are registered as "
    "Foundation-only direct products in the confirmation catalog."
)

reference_registry_payload = _read_registry_payload()
reference_registry = reference_registry_payload.get("products") or {}

try:
    reference_dist = reference_distribution()
    total_reference_rows = int(
        reference_dist["reference_count"].sum()
    ) if not reference_dist.empty else 0
except Exception:
    reference_dist = pd.DataFrame()
    total_reference_rows = 0

rm_metric_a, rm_metric_b, rm_metric_c = st.columns(3)
rm_metric_a.metric(
    "Reference classes",
    len(reference_registry),
)
rm_metric_b.metric(
    "Embedding rows",
    total_reference_rows,
)
rm_metric_c.metric(
    "Queued class batches",
    len(st.session_state.reference_manager_queue),
)


st.markdown("### Reference Browser")
st.caption(
    "Inspect the real reference images stored under "
    "`backend/hybrid_data/references/<reference_key>/`. "
    "You can preview, open details and safely delete selected references."
)

browser_keys = sorted(reference_registry)

if browser_keys:
    browser_class_key = st.selectbox(
        "Browse reference class",
        browser_keys,
        format_func=lambda key: (
            (
                reference_registry[key].get("display_name")
                or reference_registry[key].get("product_name")
                or key
            )
            + f"  |  {key}"
        ),
        key="reference_browser_class",
    )

    browser_paths = _reference_images_in(
        REFERENCE_LIBRARY_ROOT
        / browser_class_key
    )

    browser_info_a, browser_info_b, browser_info_c = st.columns(3)
    browser_info_a.metric(
        "Images in class",
        len(browser_paths),
    )
    browser_info_b.metric(
        "NPZ rows for class",
        (
            int(
                reference_dist.loc[
                    reference_dist["reference_key"]
                    == browser_class_key,
                    "reference_count",
                ].sum()
            )
            if not reference_dist.empty
            else 0
        ),
    )
    browser_info_c.metric(
        "Library path",
        browser_class_key,
    )

    if browser_paths:
        browser_controls_a, browser_controls_b = st.columns(2)
        with browser_controls_a:
            browser_page_size = st.selectbox(
                "Images per page",
                [8, 12, 20, 32],
                index=1,
                key="reference_browser_page_size",
            )
        with browser_controls_b:
            browser_sort = st.selectbox(
                "Sort",
                [
                    "Newest first",
                    "Oldest first",
                    "Filename A → Z",
                    "Filename Z → A",
                ],
                key="reference_browser_sort",
            )

        metadata_rows = [
            _reference_image_metadata(path)
            for path in browser_paths
        ]

        if browser_sort == "Newest first":
            metadata_rows.sort(
                key=lambda row: (
                    row["path"].stat().st_mtime
                    if row["path"].exists()
                    else 0
                ),
                reverse=True,
            )
        elif browser_sort == "Oldest first":
            metadata_rows.sort(
                key=lambda row: (
                    row["path"].stat().st_mtime
                    if row["path"].exists()
                    else 0
                ),
            )
        elif browser_sort == "Filename Z → A":
            metadata_rows.sort(
                key=lambda row: row["name"].casefold(),
                reverse=True,
            )
        else:
            metadata_rows.sort(
                key=lambda row: row["name"].casefold(),
            )

        total_pages = max(
            1,
            int(
                np.ceil(
                    len(metadata_rows)
                    / float(browser_page_size)
                )
            ),
        )

        current_page = min(
            int(st.session_state.reference_browser_page),
            total_pages - 1,
        )
        st.session_state.reference_browser_page = current_page

        page_nav_a, page_nav_b, page_nav_c = st.columns([1, 2, 1])
        with page_nav_a:
            if st.button(
                "← Previous",
                disabled=current_page <= 0,
                use_container_width=True,
                key="reference_browser_previous",
            ):
                st.session_state.reference_browser_page = (
                    current_page - 1
                )
                st.rerun()
        with page_nav_b:
            st.markdown(
                f"<div style='text-align:center; padding-top:8px;'>"
                f"Page <b>{current_page + 1}</b> / {total_pages}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with page_nav_c:
            if st.button(
                "Next →",
                disabled=current_page >= total_pages - 1,
                use_container_width=True,
                key="reference_browser_next",
            ):
                st.session_state.reference_browser_page = (
                    current_page + 1
                )
                st.rerun()

        start_index = (
            current_page
            * int(browser_page_size)
        )
        page_rows = metadata_rows[
            start_index:
            start_index + int(browser_page_size)
        ]

        selected_delete_paths: list[Path] = []
        thumbnail_columns = st.columns(4)

        for item_index, item in enumerate(page_rows):
            with thumbnail_columns[item_index % 4]:
                try:
                    st.image(
                        str(item["path"]),
                        caption=item["name"],
                        use_container_width=True,
                    )
                except Exception:
                    st.warning(
                        f"Cannot preview {item['name']}"
                    )

                st.caption(
                    f"{item['width']}×{item['height']} · "
                    f"{item['size_mb']:.2f} MB"
                )
                st.caption(
                    item["modified_at"]
                    or "Unknown modified time"
                )

                selected = st.checkbox(
                    "Select for deletion",
                    key=(
                        "reference_browser_delete_"
                        + hashlib.sha1(
                            str(item["path"]).encode(
                                "utf-8"
                            )
                        ).hexdigest()
                    ),
                )
                if selected:
                    selected_delete_paths.append(
                        item["path"]
                    )

                with st.expander(
                    "Details",
                    expanded=False,
                ):
                    st.code(
                        str(item["path"]),
                        language=None,
                    )
                    st.write(
                        {
                            "filename": item["name"],
                            "width": item["width"],
                            "height": item["height"],
                            "size_bytes": item["size_bytes"],
                            "modified_at": item["modified_at"],
                        }
                    )

        st.caption(
            "Deletion rebuilds `reference_embeddings.npz` from all remaining "
            "reference images so the image library and DINO artifact remain "
            "consistent. This can take time on CPU."
        )

        delete_controls_a, delete_controls_b = st.columns(2)
        with delete_controls_a:
            delete_device = st.selectbox(
                "Delete/rebuild device",
                ["AUTO", "CPU", "CUDA"],
                key="reference_browser_delete_device",
            )
        with delete_controls_b:
            delete_batch_size = st.number_input(
                "Delete/rebuild DINO batch size",
                min_value=1,
                max_value=64,
                value=16,
                step=1,
                key="reference_browser_delete_batch_size",
            )

        confirm_delete = st.checkbox(
            "I understand these selected reference images will be removed "
            "from the active reference library.",
            key="reference_browser_delete_confirm",
        )

        if st.button(
            f"Delete selected references ({len(selected_delete_paths)})",
            type="secondary",
            disabled=(
                not selected_delete_paths
                or not confirm_delete
            ),
            use_container_width=True,
            key="reference_browser_delete_button",
        ):
            with st.spinner(
                "Backing up selected images and artifacts, rebuilding DINO references..."
            ):
                try:
                    delete_result = (
                        _delete_reference_images(
                            selected_delete_paths,
                            device_choice=delete_device,
                            batch_size=int(
                                delete_batch_size
                            ),
                        )
                    )
                    st.session_state.reference_browser_delete_result = (
                        delete_result
                    )
                    load_foundation_service.clear()
                    _load_reference_encoder.clear()
                    st.session_state.reference_browser_page = 0
                    st.rerun()
                except Exception as exc:
                    st.error(
                        f"Reference deletion failed: {exc}"
                    )

    else:
        st.warning(
            "This registry class currently has no reference image files."
        )

    browser_delete_result = (
        st.session_state.reference_browser_delete_result
    )
    if browser_delete_result:
        st.success(
            "Reference deletion completed: "
            f"{browser_delete_result['deleted']} image(s) removed. "
            f"Artifact now contains "
            f"{browser_delete_result['reference_rows_after']} rows across "
            f"{browser_delete_result['class_count_after']} classes."
        )
        st.caption(
            "Backup: "
            f"{browser_delete_result['backup_dir']}"
        )
        if browser_delete_result.get("unreadable"):
            st.warning(
                "Unreadable remaining reference files were skipped during rebuild: "
                + ", ".join(
                    browser_delete_result["unreadable"][:10]
                )
            )
        if st.button(
            "Dismiss deletion result",
            key="reference_browser_delete_result_dismiss",
        ):
            st.session_state.reference_browser_delete_result = None
            st.rerun()

else:
    st.info(
        "No reference classes are registered yet."
    )

st.divider()

last_reference_result = st.session_state.reference_manager_last_result
if last_reference_result:
    st.success(
        "Reference update completed: "
        f"+{last_reference_result['added_total']} images, "
        f"{last_reference_result['reference_rows_before']} → "
        f"{last_reference_result['reference_rows_after']} embedding rows, "
        f"{last_reference_result['class_count_after']} classes."
    )
    st.caption(
        "Backup: "
        f"{last_reference_result['backup_dir']}"
    )
    result_rows = []
    for row in last_reference_result.get("classes") or []:
        result_rows.append(
            {
                "reference_key": row.get("reference_key"),
                "uploaded": row.get("uploaded", 0),
                "added": row.get("added", 0),
                "duplicates": row.get("duplicates", 0),
                "failed": row.get("failed", 0),
                "background_variants": ", ".join(
                    f"{name}:{count}"
                    for name, count in sorted(
                        (row.get("background_variants") or {}).items()
                    )
                ),
                "crop_method": ", ".join(
                    sorted((row.get("crop_methods") or {}).keys())
                ),
                "min_target_similarity": row.get(
                    "crop_target_similarity_min"
                ),
                "references_before": row.get("before_library_count", 0),
                "references_after": row.get("after_library_count", 0),
            }
        )
    if result_rows:
        st.dataframe(
            pd.DataFrame(result_rows),
            hide_index=True,
            use_container_width=True,
        )
    st.info(
        "The Test Lab cache was reloaded. If the production FastAPI backend "
        "already loaded Foundation earlier, restart the backend before using "
        "the new references in production."
    )
    if st.button(
        "Dismiss result",
        key="reference_manager_dismiss_result",
    ):
        st.session_state.reference_manager_last_result = None
        st.rerun()

with st.expander(
    "Add a class batch to the queue",
    expanded=True,
):
    rm_mode = st.radio(
        "Class mode",
        [
            "Existing class",
            "New Foundation-only class",
        ],
        horizontal=True,
        key="reference_manager_class_mode",
    )

    selected_reference_key = ""
    new_product_name = ""
    new_display_name = ""

    if rm_mode == "Existing class":
        existing_keys = sorted(reference_registry)
        if existing_keys:
            selected_reference_key = st.selectbox(
                "Reference class",
                existing_keys,
                format_func=lambda key: (
                    f"{reference_registry[key].get('display_name') or reference_registry[key].get('product_name') or key}  |  {key}"
                ),
                key="reference_manager_existing_class",
            )
            selected_item = reference_registry.get(
                selected_reference_key
            ) or {}
            st.caption(
                "Type: "
                f"{selected_item.get('type', 'direct')} · "
                "Current reference images in library: "
                f"{len(_reference_images_in(REFERENCE_LIBRARY_ROOT / selected_reference_key))}"
            )
        else:
            st.warning("The Foundation registry has no classes yet.")
    else:
        new_col_a, new_col_b = st.columns(2)
        with new_col_a:
            selected_reference_key = st.text_input(
                "Product code / reference key",
                placeholder="BR-NEW-0001",
                key="reference_manager_new_code",
                help=(
                    "For a new Foundation-only class the current production "
                    "pipeline uses product_code as reference_key."
                ),
            ).strip()
            new_product_name = st.text_input(
                "Product name",
                placeholder="New bakery product",
                key="reference_manager_new_product_name",
            ).strip()
        with new_col_b:
            new_display_name = st.text_input(
                "Display name",
                placeholder="Defaults to product name",
                key="reference_manager_new_display_name",
            ).strip()
            st.text_input(
                "Reference type",
                value="direct",
                disabled=True,
                help=(
                    "New classes created here are Foundation-only direct "
                    "products. Existing family visual classes can still receive "
                    "more references through Existing class mode."
                ),
            )

    rm_input_mode = st.radio(
        "Input image type",
        [
            "Already-tight product crops",
            "Full one-product photos — auto-crop with SAM",
        ],
        horizontal=True,
        key="reference_manager_input_mode",
    )
    rm_auto_crop = rm_input_mode.startswith("Full")

    background_label = st.radio(
        "Reference background output",
        [
            "Original background (JPEG)",
            "White background only (PNG)",
            "Both original + white (recommended)",
        ],
        horizontal=True,
        key="reference_manager_background_mode",
        help=(
            "White output uses the SAM product mask, fills internal texture holes, "
            "and saves an opaque PNG. Both variants are embedded independently."
        ),
    )
    background_mode = {
        "Original background (JPEG)": "original",
        "White background only (PNG)": "white",
        "Both original + white (recommended)": "both",
    }[background_label]
    needs_foreground_mask = background_mode in {"white", "both"}

    crop_max_side = 1280
    crop_padding = 0.08
    feather_radius = 1.5
    if rm_auto_crop or needs_foreground_mask:
        crop_col_a, crop_col_b = st.columns(2)
        with crop_col_a:
            crop_max_side = st.number_input(
                "SAM processing max side",
                min_value=640,
                max_value=2048,
                value=1280,
                step=64,
                key="reference_manager_crop_max_side",
            )
        with crop_col_b:
            if rm_auto_crop:
                crop_padding = st.slider(
                    "Crop padding",
                    min_value=0.00,
                    max_value=0.20,
                    value=0.08,
                    step=0.01,
                    key="reference_manager_crop_padding",
                )
            else:
                feather_radius = st.slider(
                    "White-edge feather radius",
                    min_value=0.0,
                    max_value=4.0,
                    value=1.5,
                    step=0.25,
                    key="reference_manager_feather_radius_tight",
                )
        if rm_auto_crop and needs_foreground_mask:
            feather_radius = st.slider(
                "White-edge feather radius",
                min_value=0.0,
                max_value=4.0,
                value=1.5,
                step=0.25,
                key="reference_manager_feather_radius_autocrop",
            )
    if rm_auto_crop:
        st.caption(
            "Auto-crop searches the whole frame with SAM2. For an existing "
            "class it also uses that SKU's current DINO references to choose "
            "the correct product mask; broad container/background masks are "
            "rejected. For a brand-new class, start with at least one tight "
            "crop whenever possible."
        )
    elif needs_foreground_mask:
        st.caption(
            "The crop frame is preserved. SAM is used only to isolate the product "
            "silhouette before compositing it onto white; unsafe empty or whole-frame "
            "masks are rejected instead of being saved."
        )

    upload_key = (
        "reference_manager_upload_"
        f"{st.session_state.reference_manager_upload_nonce}"
    )
    rm_uploaded_files = st.file_uploader(
        "Reference images for this class",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
        ],
        accept_multiple_files=True,
        key=upload_key,
    )

    if rm_uploaded_files:
        st.caption(
            f"Selected {len(rm_uploaded_files)} image(s). "
            "The build step validates, normalizes to JPEG and deduplicates them."
        )
        preview_columns = st.columns(
            min(4, len(rm_uploaded_files))
        )
        for preview_index, uploaded in enumerate(
            rm_uploaded_files[:8]
        ):
            with preview_columns[
                preview_index % len(preview_columns)
            ]:
                st.image(
                    uploaded.getvalue(),
                    caption=uploaded.name,
                    use_container_width=True,
                )

        if needs_foreground_mask and st.button(
            "Preview white-background result for first image",
            key="reference_manager_preview_white",
        ):
            try:
                preview_image = _decode_reference_upload(
                    rm_uploaded_files[0].getvalue()
                )
                preview_device = _resolve_reference_device("AUTO")
                preview_target_embeddings = None
                preview_encoder = None
                preview_key = str(selected_reference_key or "").strip()
                if rm_auto_crop and preview_key:
                    old_embeddings, old_keys = _load_reference_artifact()
                    positions = np.where(old_keys == preview_key)[0]
                    if len(positions):
                        preview_target_embeddings = old_embeddings[positions]
                        preview_encoder = _load_reference_encoder(preview_device)
                preview_variants, preview_metadata = (
                    _prepare_reference_variants(
                        preview_image,
                        input_mode="auto_crop" if rm_auto_crop else "tight",
                        background_mode=background_mode,
                        max_side=int(crop_max_side),
                        padding=float(crop_padding),
                        feather_radius=float(feather_radius),
                        device=preview_device,
                        encoder=preview_encoder,
                        target_embeddings=preview_target_embeddings,
                        batch_size=16,
                    )
                )
                variant_columns = st.columns(len(preview_variants))
                source_stem = Path(rm_uploaded_files[0].name).stem
                for variant_index, variant in enumerate(preview_variants):
                    with variant_columns[variant_index]:
                        st.image(
                            variant["image_bytes"],
                            caption=variant["variant"],
                            use_container_width=True,
                        )
                        st.download_button(
                            f"Download {variant['variant']}",
                            data=variant["image_bytes"],
                            file_name=(
                                f"{source_stem}_{variant['variant']}"
                                f"{variant['extension']}"
                            ),
                            mime=(
                                "image/png"
                                if variant["extension"] == ".png"
                                else "image/jpeg"
                            ),
                            key=(
                                "reference_manager_preview_download_"
                                f"{variant['variant']}"
                            ),
                            on_click="ignore",
                        )
                segmentation_quality = dict(
                    preview_metadata.get("crop_metadata") or {}
                )
                if "removed_mask_ratio" in segmentation_quality:
                    removed_ratio = max(
                        0.0,
                        float(segmentation_quality["removed_mask_ratio"]),
                    )
                    proposal_source = str(
                        segmentation_quality.get("proposal_source") or "SAM"
                    )
                    st.caption(
                        "Foreground cleanup: "
                        f"{removed_ratio:.1%} tray/shadow pixels removed "
                        f"from the {proposal_source} proposal."
                    )
                    if removed_ratio >= 0.30:
                        st.warning(
                            "The foreground mask required a strong correction. "
                            "Inspect the white preview before adding this batch."
                        )
                    else:
                        st.success(
                            "Foreground boundary passed guarded SAM + GrabCut "
                            "cleanup."
                        )
                with st.expander("Preview segmentation metadata"):
                    st.json(preview_metadata)
            except Exception as exc:
                st.error(f"White-background preview failed safely: {exc}")

    add_batch_clicked = st.button(
        "Add class batch to queue",
        type="primary",
        disabled=not bool(rm_uploaded_files),
        key="reference_manager_add_batch",
    )

    if add_batch_clicked:
        try:
            reference_key = _reference_safe_code(
                selected_reference_key
            )
            is_new = rm_mode != "Existing class"

            if is_new:
                if reference_key in reference_registry:
                    raise ValueError(
                        "This reference key already exists. Select Existing class instead."
                    )
                if not new_product_name:
                    raise ValueError(
                        "Product name is required for a new class."
                    )
                product_name = new_product_name
                display_name = (
                    new_display_name
                    or new_product_name
                )
            else:
                item = reference_registry.get(
                    reference_key
                ) or {}
                product_name = str(
                    item.get("product_name") or ""
                )
                display_name = str(
                    item.get("display_name")
                    or product_name
                    or reference_key
                )

            batch_payload = {
                "batch_id": hashlib.sha1(
                    (
                        reference_key
                        + str(time.time_ns())
                    ).encode("utf-8")
                ).hexdigest()[:12],
                "reference_key": reference_key,
                "is_new": is_new,
                "product_name": product_name,
                "display_name": display_name,
                "auto_crop": bool(rm_auto_crop),
                "input_mode": "auto_crop" if rm_auto_crop else "tight",
                "background_mode": background_mode,
                "feather_radius": float(feather_radius),
                "crop_max_side": int(crop_max_side),
                "crop_padding": float(crop_padding),
                "files": [
                    {
                        "name": uploaded.name,
                        "raw": uploaded.getvalue(),
                    }
                    for uploaded in rm_uploaded_files
                ],
            }

            st.session_state.reference_manager_queue.append(
                batch_payload
            )
            st.session_state.reference_manager_upload_nonce += 1
            st.success(
                f"Queued {len(rm_uploaded_files)} image(s) for {reference_key}."
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

queue = st.session_state.reference_manager_queue

if queue:
    st.markdown("#### Pending reference queue")
    queue_rows = []
    for batch in queue:
        queue_rows.append(
            {
                "batch_id": batch["batch_id"],
                "reference_key": batch["reference_key"],
                "mode": "new" if batch["is_new"] else "existing",
                "images": len(batch.get("files") or []),
                "input": "SAM auto-crop" if batch.get("auto_crop") else "tight crops",
                "background": batch.get("background_mode", "original"),
                "display_name": batch.get("display_name", ""),
            }
        )
    st.dataframe(
        pd.DataFrame(queue_rows),
        hide_index=True,
        use_container_width=True,
    )

    queue_labels = {
        (
            f"{row['reference_key']} · {row['images']} images · {row['batch_id']}"
        ): row["batch_id"]
        for row in queue_rows
    }
    remove_labels = st.multiselect(
        "Remove queued batches",
        list(queue_labels),
        key="reference_manager_remove_selection",
    )

    queue_action_a, queue_action_b = st.columns(2)
    with queue_action_a:
        if st.button(
            "Remove selected",
            disabled=not bool(remove_labels),
            use_container_width=True,
            key="reference_manager_remove_selected",
        ):
            remove_ids = {
                queue_labels[label]
                for label in remove_labels
            }
            st.session_state.reference_manager_queue = [
                batch
                for batch in queue
                if batch["batch_id"] not in remove_ids
            ]
            st.rerun()
    with queue_action_b:
        if st.button(
            "Clear queue",
            use_container_width=True,
            key="reference_manager_clear_queue",
        ):
            st.session_state.reference_manager_queue = []
            st.rerun()

    build_col_a, build_col_b = st.columns(2)
    with build_col_a:
        rm_device = st.selectbox(
            "Reference build device",
            ["AUTO", "CPU", "CUDA"],
            index=0,
            key="reference_manager_build_device",
            help=(
                "AUTO uses CUDA only when PyTorch reports it available. "
                "Otherwise it uses CPU."
            ),
        )
    with build_col_b:
        rm_batch_size = st.number_input(
            "DINO batch size",
            min_value=1,
            max_value=64,
            value=16,
            step=1,
            key="reference_manager_batch_size",
        )

    queued_file_total = sum(
        len(batch.get("files") or [])
        for batch in queue
    )
    st.warning(
        "Build & activate writes to the real local Foundation reference library, "
        "registry, product_mapping.json and reference_embeddings.npz. A timestamped "
        "backup is created first, and artifacts are rolled back if activation fails."
    )

    build_all_clicked = st.button(
        f"Build & activate all queued references ({queued_file_total} images)",
        type="primary",
        use_container_width=True,
        key="reference_manager_build_all",
    )

    if build_all_clicked:
        progress_box = st.empty()
        with st.spinner(
            "Validating images, cropping when requested, encoding DINO references and activating artifact..."
        ):
            try:
                result = _commit_reference_queue(
                    queue,
                    device_choice=rm_device,
                    batch_size=int(rm_batch_size),
                )
                st.session_state.reference_manager_last_result = result
                st.session_state.reference_manager_queue = []

                # Force the test app to reload the newly written NPZ/registry on rerun.
                load_foundation_service.clear()

                progress_box.success(
                    "Reference artifact activated successfully."
                )
                st.rerun()
            except Exception as exc:
                progress_box.error(
                    f"Reference update failed: {exc}"
                )
else:
    st.info(
        "The reference queue is empty. Add one class batch above; repeat for as "
        "many classes as needed, then build once."
    )


# =========================================================
# 10. UPLOAD + RUN
# =========================================================
st.subheader(
    "Run FOUNDATION test"
)

uploaded_files = st.file_uploader(
    "Upload bakery images and/or videos",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "mp4",
        "mov",
        "avi",
        "mkv",
        "m4v",
    ],
    accept_multiple_files=True,
    help=(
        "Images are inferred directly. Videos are first sampled into sharp, "
        "visually diverse JPEG frames; all retained images and frames then run "
        "through the same Foundation inference and pre-annotation export."
    ),
)

video_target_fps = 2.0
video_blur_threshold = 60.0
video_similarity_threshold = 6
video_max_frames = 200
video_max_total_frames = 500
video_max_dimension = 1920
video_jpeg_quality = 92
has_video_uploads = any(
    Path(uploaded.name).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    for uploaded in (uploaded_files or [])
)

if has_video_uploads:
    with st.expander("Video frame extraction settings", expanded=True):
        st.caption(
            "Sampling is capped and filtered before Foundation inference. pHash "
            "deduplication compares each candidate with every retained frame from "
            "the same video; original uploaded videos are not included in the ZIP."
        )
        video_row_a = st.columns(3)
        with video_row_a[0]:
            video_target_fps = st.number_input(
                "Candidate sampling FPS",
                min_value=0.1,
                max_value=30.0,
                value=2.0,
                step=0.1,
                key="foundation_video_target_fps",
            )
        with video_row_a[1]:
            video_blur_threshold = st.number_input(
                "Minimum sharpness (Laplacian)",
                min_value=0.0,
                max_value=1000.0,
                value=60.0,
                step=5.0,
                key="foundation_video_blur_threshold",
            )
        with video_row_a[2]:
            video_similarity_threshold = st.slider(
                "Duplicate pHash distance ≤",
                min_value=0,
                max_value=24,
                value=6,
                step=1,
                key="foundation_video_similarity_threshold",
            )

        video_row_b = st.columns(4)
        with video_row_b[0]:
            video_max_frames = st.number_input(
                "Maximum frames / video",
                min_value=1,
                max_value=2000,
                value=200,
                step=10,
                key="foundation_video_max_frames",
            )
        with video_row_b[1]:
            video_max_total_frames = st.number_input(
                "Maximum video frames / run",
                min_value=1,
                max_value=5000,
                value=500,
                step=25,
                key="foundation_video_max_total_frames",
            )
        with video_row_b[2]:
            video_max_dimension = st.number_input(
                "Saved frame long side (0 = original)",
                min_value=0,
                max_value=4096,
                value=1920,
                step=128,
                key="foundation_video_max_dimension",
            )
        with video_row_b[3]:
            video_jpeg_quality = st.slider(
                "Frame JPEG quality",
                min_value=70,
                max_value=100,
                value=92,
                step=1,
                key="foundation_video_jpeg_quality",
            )

run_clicked = st.button(
    "Run FOUNDATION",
    type="primary",
    disabled=(
        not bool(uploaded_files)
        or not bool(
            health.get("ready")
        )
    ),
)

if run_clicked and uploaded_files:
    st.session_state.foundation_run_id += 1

    run_id = (
        st.session_state
        .foundation_run_id
    )

    progress = st.progress(
        0,
        text="Preparing Foundation test...",
    )

    live_status = st.empty()

    st.session_state.foundation_preannotation_zip = None
    st.session_state.foundation_preannotation_filename = ""
    st.session_state.foundation_preannotation_summary = None

    results: list[
        dict[str, Any]
    ] = []

    inference_inputs: list[dict[str, Any]] = []
    extraction_summaries: list[dict[str, Any]] = []
    total_video_frames = 0

    for media_index, uploaded in enumerate(uploaded_files, start=1):
        source_name = uploaded.name
        suffix = Path(source_name).suffix.lower()
        raw = uploaded.getvalue()
        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            inference_inputs.append(
                {
                    "file_name": source_name,
                    "raw_bytes": raw,
                    "source_metadata": {
                        "source_type": "image",
                        "source_image": source_name,
                    },
                }
            )
            continue

        remaining_total = int(video_max_total_frames) - total_video_frames
        if remaining_total <= 0:
            extraction_summaries.append(
                {
                    "video": source_name,
                    "status": "skipped_total_limit",
                    "saved": 0,
                    "error": "Maximum video frames per run was already reached.",
                }
            )
            continue
        live_status.info(
            f"Extracting video {media_index}/{len(uploaded_files)}: {source_name}"
        )
        try:
            extraction = _extract_uploaded_video(
                raw,
                source_name=source_name,
                config=VideoFrameExtractionConfig(
                    target_fps=float(video_target_fps),
                    blur_threshold=float(video_blur_threshold),
                    similarity_threshold=int(video_similarity_threshold),
                    max_dimension=int(video_max_dimension),
                    jpeg_quality=int(video_jpeg_quality),
                    max_frames=min(int(video_max_frames), remaining_total),
                ),
            )
            extraction_summaries.append(
                {"status": "success", "error": "", **extraction.stats}
            )
            for frame in extraction.frames:
                inference_inputs.append(
                    {
                        "file_name": frame.file_name,
                        "raw_bytes": frame.jpeg_bytes,
                        "source_metadata": frame.metadata,
                    }
                )
            total_video_frames += len(extraction.frames)
        except Exception as exc:
            extraction_summaries.append(
                {
                    "video": source_name,
                    "status": "failed",
                    "saved": 0,
                    "error": str(exc),
                }
            )

    st.session_state.foundation_video_extraction_summary = extraction_summaries

    batch_started = (
        time.perf_counter()
    )

    for index, inference_item in enumerate(
        inference_inputs,
        start=1,
    ):
        raw = inference_item["raw_bytes"]
        file_name = inference_item["file_name"]

        live_status.info(
            f"Processing {index}/{len(inference_inputs)}: {file_name}"
        )

        try:
            result = (
                run_foundation_diagnostic(
                    service,
                    raw,
                    image_name=file_name,
                )
            )

            results.append(
                {
                    "file_name": file_name,
                    "raw_bytes": raw,
                    "source_metadata": inference_item["source_metadata"],
                    "result": result,
                    "error": "",
                }
            )

        except Exception as exc:
            results.append(
                {
                    "file_name": file_name,
                    "raw_bytes": raw,
                    "source_metadata": inference_item["source_metadata"],
                    "result": {},
                    "error": str(exc),
                }
            )

            print(
                "[FOUNDATION-TEST][ERROR] "
                f"name={file_name} "
                f"error={exc}",
                flush=True,
            )

        percent = int(
            index / max(1, len(inference_inputs)) * 100
        )

        progress.progress(
            percent,
            text=(
                f"Inference complete "
                f"{index}/{len(inference_inputs)}"
            ),
        )

    batch_ms = (
        time.perf_counter()
        - batch_started
    ) * 1000.0

    # Critical UI fix:
    # the inference stage ends HERE.
    # No full-resolution image/crop rendering
    # happens inside the processing loop.
    progress.progress(
        100,
        text="FOUNDATION inference finished.",
    )

    live_status.success(
        f"FOUNDATION finished {len(results)} image(s) "
        f"in {batch_ms / 1000:.2f} s. "
        "Rendering diagnostics is now separate from inference."
    )

    st.session_state.foundation_results = results

    print(
        "[FOUNDATION-TEST][BATCH] "
        f"run_id={run_id} "
        f"images={len(results)} "
        f"elapsed_ms={batch_ms:.1f}",
        flush=True,
    )

if st.session_state.foundation_video_extraction_summary:
    with st.expander("Video extraction summary", expanded=True):
        extraction_df = pd.DataFrame(
            st.session_state.foundation_video_extraction_summary
        )
        st.dataframe(
            extraction_df,
            hide_index=True,
            use_container_width=True,
        )
        saved_frames = int(
            extraction_df.get("saved", pd.Series(dtype=int)).fillna(0).sum()
        )
        st.caption(
            f"Retained {saved_frames} video frame(s). Every retained frame is "
            "processed and exported like a normal uploaded image."
        )


# =========================================================
# 11. SUMMARY
# =========================================================
results = (
    st.session_state
    .foundation_results
)

if results:
    st.divider()
    st.subheader(
        "Batch results"
    )

    summary_rows = []

    for item in results:
        result = (
            item.get("result")
            or {}
        )

        decision = (
            result.get("decision")
            or {}
        )

        timing = (
            result.get(
                "foundation_debug",
                {},
            )
            .get(
                "timing",
                {},
            )
        )

        summary_rows.append(
            {
                "image": item.get(
                    "file_name"
                ),
                "decision": (
                    decision.get(
                        "decision"
                    )
                    or (
                        "ERROR"
                        if item.get(
                            "error"
                        )
                        else "—"
                    )
                ),
                "reference": (
                    decision.get(
                        "display_name"
                    )
                    or decision.get(
                        "dominant_class"
                    )
                    or ""
                ),
                "count": int(
                    decision.get(
                        "count"
                    )
                    or 0
                ),
                "similarity": float(
                    decision.get(
                        "avg_confidence"
                    )
                    or 0.0
                ),
                "margin": float(
                    decision.get(
                        "similarity_margin"
                    )
                    or 0.0
                ),
                "load_ms": float(
                    timing.get(
                        "model_load_ms"
                    )
                    or 0.0
                ),
                "sam_ms": float(
                    timing.get(
                        "sam_segmentation_ms"
                    )
                    or 0.0
                ),
                "dino_ms": float(
                    (
                        timing.get(
                            "dino_encode_ms"
                        )
                        or 0.0
                    )
                    + (
                        timing.get(
                            "dino_classify_ms"
                        )
                        or 0.0
                    )
                ),
                "total_ms": float(
                    timing.get(
                        "total_inference_ms"
                    )
                    or 0.0
                ),
                "error": (
                    item.get(
                        "error",
                        "",
                    )
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    st.dataframe(
        summary_df,
        hide_index=True,
        use_container_width=True,
    )

    metric_cols = st.columns(5)

    metric_cols[0].metric(
        "Images",
        len(
            summary_df
        ),
    )

    metric_cols[1].metric(
        "Total count",
        int(
            summary_df[
                "count"
            ].sum()
        ),
    )

    metric_cols[2].metric(
        "Mean SAM",
        fmt_ms(
            summary_df[
                "sam_ms"
            ].mean()
        ),
    )

    metric_cols[3].metric(
        "Mean DINO",
        fmt_ms(
            summary_df[
                "dino_ms"
            ].mean()
        ),
    )

    metric_cols[4].metric(
        "Mean total",
        fmt_ms(
            summary_df[
                "total_ms"
            ].mean()
        ),
    )

    csv_bytes = (
        summary_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )

    export_payload = []

    for item in results:
        export_payload.append(
            {
                "file_name": (
                    item.get(
                        "file_name"
                    )
                ),
                "result": (
                    item.get(
                        "result"
                    )
                ),
                "error": (
                    item.get(
                        "error"
                    )
                ),
            }
        )

    json_bytes = (
        json.dumps(
            export_payload,
            ensure_ascii=False,
            indent=2,
        )
        .encode(
            "utf-8"
        )
    )

    dl1, dl2 = st.columns(2)

    with dl1:
        st.download_button(
            "Download summary CSV",
            data=csv_bytes,
            file_name=(
                "foundation_test_summary.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with dl2:
        st.download_button(
            "Download full JSON",
            data=json_bytes,
            file_name=(
                "foundation_test_results.json"
            ),
            mime=(
                "application/json"
            ),
            use_container_width=True,
        )


    st.markdown("### Roboflow Pre-annotation ZIP")
    st.caption(
        "Export the current FOUNDATION test results as an object-detection "
        "dataset with Foundation's final bounding boxes already attached. "
        "Upload/import it into Roboflow, then manually review and correct "
        "the pre-annotations before using them as ground truth."
    )

    eligible_result_labels: list[str] = []
    eligible_result_lookup: dict[str, int] = {}

    for result_index, item in enumerate(results):
        if item.get("error"):
            continue

        result_payload = item.get("result") or {}
        decision_payload = result_payload.get("decision") or {}
        object_count = len(result_payload.get("objects") or [])

        label = (
            f"{result_index + 1}. {item.get('file_name')} "
            f"| {decision_payload.get('decision') or '—'} "
            f"| boxes={object_count}"
        )
        eligible_result_labels.append(label)
        eligible_result_lookup[label] = result_index

    if eligible_result_labels:
        pa_selected_labels = st.multiselect(
            "Images to export",
            eligible_result_labels,
            default=eligible_result_labels,
            key="foundation_preannotation_selected_images",
            help=(
                "By default all successful Foundation test images are exported. "
                "Failed inference results are excluded."
            ),
        )

        pa_format_label = st.selectbox(
            "Annotation format",
            [
                "YOLOv8 / Ultralytics — recommended",
                "COCO JSON",
            ],
            index=0,
            key="foundation_preannotation_format",
        )
        pa_format = (
            "coco"
            if pa_format_label.startswith("COCO")
            else "yolo"
        )

        pa_options_a, pa_options_b = st.columns(2)

        with pa_options_a:
            pa_include_empty = st.checkbox(
                "Include images with zero boxes",
                value=True,
                key="foundation_preannotation_include_empty",
                help=(
                    "Recommended for manual checking: Foundation misses remain "
                    "in the exported dataset so you can draw the missing boxes "
                    "yourself in Roboflow."
                ),
            )

        with pa_options_b:
            pa_min_confidence = st.slider(
                "Minimum object similarity to export",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                key="foundation_preannotation_min_confidence",
                help=(
                    "0.00 exports every final Foundation object. These objects "
                    "have already passed the Test Lab's semantic/geometry filters."
                ),
            )

        selected_result_indices = [
            eligible_result_lookup[label]
            for label in pa_selected_labels
            if label in eligible_result_lookup
        ]

        selected_result_items = [
            results[index]
            for index in selected_result_indices
        ]

        pa_preview_images = len(selected_result_items)
        pa_preview_boxes = 0
        pa_preview_classes: set[str] = set()

        for item in selected_result_items:
            result_payload = item.get("result") or {}
            decision_payload = result_payload.get("decision") or {}

            filtered_box_count = 0
            for obj in result_payload.get("objects") or []:
                confidence = float(obj.get("confidence") or 0.0)
                if confidence < float(pa_min_confidence):
                    continue

                class_name = _preannotation_class_name(
                    obj,
                    decision_payload,
                )
                if not class_name:
                    continue

                filtered_box_count += 1
                pa_preview_classes.add(class_name)

            if not pa_include_empty and filtered_box_count == 0:
                pa_preview_images -= 1

            pa_preview_boxes += filtered_box_count

        pa_metrics = st.columns(3)
        pa_metrics[0].metric(
            "Images in ZIP",
            max(0, pa_preview_images),
        )
        pa_metrics[1].metric(
            "Pre-annotation boxes",
            pa_preview_boxes,
        )
        pa_metrics[2].metric(
            "Classes present",
            len(pa_preview_classes),
        )

        st.info(
            "The ZIP is generated from the current Test Lab results, not from "
            "the Foundation reference-image library. Only the final `objects` "
            "shown by the Test Lab are exported as pre-annotation boxes."
        )

        prepare_pa_zip = st.button(
            "Prepare Roboflow pre-annotation ZIP",
            type="primary",
            use_container_width=True,
            disabled=not bool(selected_result_items),
            key="foundation_preannotation_prepare",
        )

        if prepare_pa_zip:
            with st.spinner(
                "Creating images + Foundation bounding-box annotations..."
            ):
                try:
                    pa_result = _build_foundation_preannotation_zip(
                        selected_result_items,
                        export_format=pa_format,
                        include_empty_images=bool(pa_include_empty),
                        min_object_confidence=float(pa_min_confidence),
                        video_extraction_summary=list(
                            st.session_state.foundation_video_extraction_summary
                        ),
                    )

                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    st.session_state.foundation_preannotation_zip = (
                        pa_result["zip_bytes"]
                    )
                    st.session_state.foundation_preannotation_filename = (
                        "foundation_preannotated_"
                        f"{pa_result['format']}_"
                        f"{timestamp}.zip"
                    )
                    st.session_state.foundation_preannotation_summary = {
                        key: value
                        for key, value in pa_result.items()
                        if key != "zip_bytes"
                    }

                except Exception as exc:
                    st.session_state.foundation_preannotation_zip = None
                    st.session_state.foundation_preannotation_filename = ""
                    st.session_state.foundation_preannotation_summary = None
                    st.error(
                        f"Could not create pre-annotation ZIP: {exc}"
                    )

        prepared_pa_zip = (
            st.session_state.foundation_preannotation_zip
        )
        prepared_pa_summary = (
            st.session_state.foundation_preannotation_summary
        )

        if prepared_pa_zip and prepared_pa_summary:
            pa_size_mb = (
                float(prepared_pa_summary["size_bytes"])
                / (1024 * 1024)
            )

            st.success(
                "Pre-annotation ZIP ready: "
                f"{prepared_pa_summary['image_count']} images, "
                f"{prepared_pa_summary['box_count']} boxes, "
                f"{prepared_pa_summary['class_count']} classes, "
                f"{pa_size_mb:.2f} MB."
            )
            if prepared_pa_summary.get("video_frame_count"):
                st.caption(
                    f"Includes {prepared_pa_summary['video_frame_count']} retained "
                    f"frame(s) from {prepared_pa_summary.get('video_count', 0)} "
                    "source video(s), with frame/timestamp/quality manifests."
                )

            st.download_button(
                "Download pre-annotation ZIP for Roboflow",
                data=prepared_pa_zip,
                file_name=(
                    st.session_state
                    .foundation_preannotation_filename
                ),
                mime="application/zip",
                use_container_width=True,
                key="foundation_preannotation_download",
            )

            if prepared_pa_summary.get("class_names"):
                with st.expander(
                    "Classes included in this ZIP",
                    expanded=False,
                ):
                    st.code(
                        "\n".join(
                            prepared_pa_summary[
                                "class_names"
                            ]
                        ),
                        language=None,
                    )

            st.warning(
                "Treat these labels as AI pre-annotations only. Review false "
                "positives, missed products, box boundaries and class names in "
                "Roboflow before adding them to a training version."
            )

            if st.button(
                "Clear prepared pre-annotation ZIP",
                key="foundation_preannotation_clear",
            ):
                st.session_state.foundation_preannotation_zip = None
                st.session_state.foundation_preannotation_filename = ""
                st.session_state.foundation_preannotation_summary = None
                st.rerun()

    else:
        st.info(
            "Run a successful FOUNDATION test first. The pre-annotation ZIP "
            "will be generated from those test results."
        )



# =========================================================
# 12. ON-DEMAND DETAIL VIEW
#     Heavy image/crop rendering happens only for ONE selected image.
# =========================================================
if results:
    st.divider()
    st.subheader(
        "Inspect one result"
    )

    labels = [
        f"{index + 1}. {item['file_name']}"
        for index, item in enumerate(
            results
        )
    ]

    selected_label = st.selectbox(
        "Select image",
        labels,
    )

    selected_index = labels.index(
        selected_label
    )

    selected = results[
        selected_index
    ]

    if selected.get("error"):
        st.error(
            selected[
                "error"
            ]
        )

    else:
        result = (
            selected["result"]
        )

        decision = (
            result.get("decision")
            or {}
        )

        objects = list(
            result.get("objects")
            or []
        )

        debug = (
            result.get(
                "foundation_debug",
                {},
            )
        )

        timing = (
            debug.get(
                "timing",
                {},
            )
        )

        detail_metrics = (
            st.columns(6)
        )

        detail_metrics[0].metric(
            "Decision",
            str(
                decision.get(
                    "decision"
                )
                or "—"
            ),
        )

        detail_metrics[1].metric(
            "Count",
            int(
                decision.get(
                    "count"
                )
                or 0
            ),
        )

        detail_metrics[2].metric(
            "Similarity",
            fmt_pct(
                decision.get(
                    "avg_confidence"
                )
            ),
        )

        detail_metrics[3].metric(
            "Margin",
            f"{float(decision.get('similarity_margin') or 0):.4f}",
        )

        detail_metrics[4].metric(
            "SAM",
            fmt_ms(
                timing.get(
                    "sam_segmentation_ms"
                )
            ),
        )

        detail_metrics[5].metric(
            "DINO",
            fmt_ms(
                (
                    timing.get(
                        "dino_encode_ms"
                    )
                    or 0.0
                )
                + (
                    timing.get(
                        "dino_classify_ms"
                    )
                    or 0.0
                )
            ),
        )

        st.write(
            "**Reference/class:**",
            (
                decision.get(
                    "display_name"
                )
                or decision.get(
                    "dominant_class"
                )
                or "—"
            ),
        )

        st.write(
            "**Message:**",
            decision.get(
                "message"
            )
            or "—",
        )

        if (
            decision.get(
                "decision"
            )
            == "FAMILY"
            and decision.get(
                "members"
            )
        ):
            st.info(
                "Foundation identified a "
                "shared visual family. "
                "Exact business SKU still "
                "requires operator selection."
            )

            st.dataframe(
                pd.DataFrame(
                    decision[
                        "members"
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

        # Detailed timing is cheap to render.
        st.caption(
            "Phase timing"
        )

        timing_rows = [
            {
                "phase": "Decode",
                "ms": timing.get(
                    "decode_ms",
                    0.0,
                ),
            },
            {
                "phase": (
                    "Model load"
                ),
                "ms": timing.get(
                    "model_load_ms",
                    0.0,
                ),
            },
            {
                "phase": (
                    "SAM segmentation"
                ),
                "ms": timing.get(
                    "sam_segmentation_ms",
                    0.0,
                ),
            },
            {
                "phase": (
                    "Crop preparation"
                ),
                "ms": timing.get(
                    "crop_prepare_ms",
                    0.0,
                ),
            },
            {
                "phase": (
                    "DINO encoding"
                ),
                "ms": timing.get(
                    "dino_encode_ms",
                    0.0,
                ),
            },
            {
                "phase": (
                    "DINO classification"
                ),
                "ms": timing.get(
                    "dino_classify_ms",
                    0.0,
                ),
            },
            {
                "phase": (
                    "Total inference"
                ),
                "ms": timing.get(
                    "total_inference_ms",
                    0.0,
                ),
            },
        ]

        timing_df = pd.DataFrame(
            timing_rows
        )

        timing_df["ms"] = (
            timing_df["ms"]
            .astype(float)
            .round(1)
        )

        st.dataframe(
            timing_df,
            hide_index=True,
            use_container_width=True,
        )

        # Decode only selected image now.
        preview_decode_started = (
            time.perf_counter()
        )

        image_bgr = decode_bgr(
            selected[
                "raw_bytes"
            ]
        )

        numbered = (
            draw_numbered_boxes(
                image_bgr,
                objects,
            )
        )

        original_preview = (
            resize_preview(
                image_bgr,
                max_side=1100,
            )
        )

        numbered_preview = (
            resize_preview(
                numbered,
                max_side=1100,
            )
        )

        render_prepare_ms = (
            time.perf_counter()
            - preview_decode_started
        ) * 1000.0

        print(
            "[FOUNDATION-TEST][RENDER] "
            f"name={selected['file_name']} "
            f"preview_prepare_ms={render_prepare_ms:.1f}",
            flush=True,
        )

        preview_left, preview_right = (
            st.columns(2)
        )

        with preview_left:
            st.caption(
                "Original preview "
                "(resized for UI)"
            )
            st.image(
                original_preview,
                channels="BGR",
                use_container_width=True,
            )

        with preview_right:
            st.caption(
                "Foundation boxes "
                "(resized for UI)"
            )
            st.image(
                numbered_preview,
                channels="BGR",
                use_container_width=True,
            )

        if objects:
            object_rows = []

            for object_index, obj in enumerate(
                objects,
                start=1,
            ):
                object_rows.append(
                    {
                        "object": (
                            object_index
                        ),
                        "reference_key": (
                            obj.get(
                                "raw_class",
                                "",
                            )
                        ),
                        "display_name": (
                            obj.get(
                                "display_name",
                                "",
                            )
                        ),
                        "similarity": float(
                            obj.get(
                                "confidence"
                            )
                            or 0.0
                        ),
                        "box_xyxy": [
                            round(
                                float(
                                    value
                                ),
                                1,
                            )
                            for value in (
                                obj.get(
                                    "box_xyxy",
                                    [],
                                )
                            )
                        ],
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    object_rows
                ),
                hide_index=True,
                use_container_width=True,
            )

        # Ranking was computed from the SAME DINO pass.
        # Showing it now does not re-run DINO.
        ranking = list(
            debug.get(
                "reference_ranking",
                []
            )
        )

        with st.expander(
            "Top reference ranking",
            expanded=False,
        ):
            if ranking:
                top_k = st.slider(
                    "Top K",
                    min_value=3,
                    max_value=min(
                        15,
                        len(ranking),
                    ),
                    value=min(
                        8,
                        len(ranking),
                    ),
                    key=(
                        f"ranking_top_k_{selected_index}"
                    ),
                )

                ranking_df = (
                    pd.DataFrame(
                        ranking[
                            :top_k
                        ]
                    )
                )

                ranking_df[
                    "median_similarity"
                ] = (
                    ranking_df[
                        "median_similarity"
                    ]
                    .astype(float)
                    .round(5)
                )

                st.dataframe(
                    ranking_df,
                    hide_index=True,
                    use_container_width=True,
                )

            else:
                st.info(
                    "No DINO ranking "
                    "is available because "
                    "no valid segment was "
                    "classified."
                )

        # Crops are truly on-demand and only for selected image.
        with st.expander(
            f"Detected object crops ({len(objects)})",
            expanded=False,
        ):
            if objects:
                max_crops = min(
                    len(objects),
                    40,
                )

                if max_crops == 1:
                    # Streamlit sliders require min_value < max_value.
                    # A single detected object needs no selector.
                    crop_limit = 1
                    st.caption("Rendering the only detected crop.")
                else:
                    crop_limit = st.slider(
                        "Number of crops to render",
                        min_value=1,
                        max_value=max_crops,
                        value=min(
                            12,
                            max_crops,
                        ),
                        key=(
                            f"crop_limit_{selected_index}"
                        ),
                    )

                crop_started = (
                    time.perf_counter()
                )

                crops = []

                for obj in objects[
                    :crop_limit
                ]:
                    crop = crop_from_box(
                        image_bgr,
                        obj[
                            "box_xyxy"
                        ],
                    )

                    crop = (
                        resize_preview(
                            crop,
                            max_side=320,
                        )
                    )

                    crops.append(
                        crop
                    )

                crop_render_prepare_ms = (
                    time.perf_counter()
                    - crop_started
                ) * 1000.0

                print(
                    "[FOUNDATION-TEST][CROPS] "
                    f"name={selected['file_name']} "
                    f"count={crop_limit} "
                    f"prepare_ms={crop_render_prepare_ms:.1f}",
                    flush=True,
                )

                st.image(
                    crops,
                    caption=[
                        f"#{index}"
                        for index in range(
                            1,
                            len(crops)
                            + 1
                        )
                    ],
                    channels="BGR",
                    width=160,
                )

            else:
                st.info(
                    "No object crops."
                )

        with st.expander(
            "Raw JSON result",
            expanded=False,
        ):
            # Raw image bytes are not part
            # of result JSON.
            st.json(
                result
            )
