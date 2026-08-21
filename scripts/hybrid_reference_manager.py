r"""Manage private Foundation references for the hybrid bakery detector.

Examples (run from the repository root):
  backend\.venv\Scripts\python.exe scripts\hybrid_reference_manager.py status
  backend\.venv\Scripts\python.exe scripts\hybrid_reference_manager.py sync-mapping
  backend\.venv\Scripts\python.exe scripts\hybrid_reference_manager.py add \
      --product-code BR-NEW-001 --product-name "New loaf" --source D:\approved-crops
  backend\.venv\Scripts\python.exe scripts\hybrid_reference_manager.py build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import (  # noqa: E402
    FOUNDATION_DINO_MODEL,
    FOUNDATION_REFERENCE_PATH,
    FOUNDATION_REGISTRY_PATH,
)
from app.services.foundation_inference_service import DinoEmbeddingEncoder  # noqa: E402


REFERENCES_ROOT = BACKEND_ROOT / "hybrid_data" / "references"
MAPPING_PATH = BACKEND_ROOT / "config" / "product_mapping.json"
FOUNDATION_MANIFEST_PATH = BACKEND_ROOT / "models" / "FOUNDATION_MANIFEST.json"
EXISTING_REVIEW_ROOT = (
    REPO_ROOT
    / "pre-annotation"
    / "products_preannotation"
    / "pre-annotated_imgs"
)
EXISTING_ORIGINAL_ROOT = (
    REPO_ROOT
    / "pre-annotation"
    / "products_preannotation"
    / "original_imgs"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_registry() -> dict[str, Any]:
    if not FOUNDATION_REGISTRY_PATH.is_file():
        return {"schema_version": 1, "products": {}}
    payload = json.loads(FOUNDATION_REGISTRY_PATH.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", 1)
    payload.setdefault("products", {})
    return payload


def write_registry(payload: dict[str, Any]) -> None:
    FOUNDATION_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FOUNDATION_REGISTRY_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(FOUNDATION_REGISTRY_PATH)


def read_mapping() -> dict[str, Any]:
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


def write_mapping(payload: dict[str, Any]) -> None:
    temporary = MAPPING_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(MAPPING_PATH)


def sync_operator_catalog_product(
    *,
    product_code: str,
    product_name: str,
    display_name: str,
) -> None:
    """Expose a Foundation-only SKU to the closed confirmation catalog."""
    mapping = read_mapping()
    existing_codes: set[str] = set()
    for item in (mapping.get("classes") or {}).values():
        code = str(item.get("product_code") or "").strip()
        if code:
            existing_codes.add(code.casefold())
        for member in item.get("members") or []:
            member_code = str(member.get("product_code") or "").strip()
            if member_code:
                existing_codes.add(member_code.casefold())

    catalog = mapping.setdefault("catalog_products", [])
    if not isinstance(catalog, list):
        raise ValueError("product_mapping.catalog_products must be a list")

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
            if str(item.get("product_code") or "").casefold() == target:
                catalog[index] = replacement
                break
        else:
            catalog.append(replacement)

    mapping["supported_product_count"] = int(
        mapping.get("business_sku_count") or len(existing_codes)
    ) + len(catalog)
    write_mapping(mapping)


def safe_code(value: str) -> str:
    code = str(value).strip()
    if not code or any(char in code for char in "\\/:*?\"<>|") or code in {".", ".."}:
        raise ValueError("Product code is empty or unsafe for a folder name.")
    return code


def images_in(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def command_status(_: argparse.Namespace) -> int:
    registry = read_registry()
    print(f"Registry: {FOUNDATION_REGISTRY_PATH}")
    print(f"Artifact: {FOUNDATION_REFERENCE_PATH}")
    print(f"Products registered: {len(registry['products'])}")
    total = 0
    for key, item in sorted(registry["products"].items()):
        count = len(images_in(REFERENCES_ROOT / key))
        total += count
        print(f"  {key}: {count} reference image(s) - {item.get('display_name', key)}")
    print(f"Reference images: {total}")
    print(f"Embedding artifact ready: {FOUNDATION_REFERENCE_PATH.is_file()}")
    return 0


def command_sync_mapping(_: argparse.Namespace) -> int:
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    registry = read_registry()
    products = registry["products"]
    for visual_key, item in mapping["classes"].items():
        if item.get("type") == "direct":
            entry = {
                "type": "direct",
                "product_code": item["product_code"],
                "product_name": item["product_name"],
                "display_name": item.get("display_name") or item["product_name"],
            }
        else:
            entry = {
                "type": "family",
                "display_name": item["display_name"],
                "members": item["members"],
            }
        products.setdefault(visual_key, entry)
        (REFERENCES_ROOT / visual_key).mkdir(parents=True, exist_ok=True)
    write_registry(registry)
    print(f"Synced {len(mapping['classes'])} visual classes. Add approved crops, then run build.")
    return 0


def command_add(arguments: argparse.Namespace) -> int:
    code = safe_code(arguments.product_code)
    source = Path(arguments.source).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Reference source folder not found: {source}")
    sources = images_in(source)
    if not sources:
        raise ValueError(f"No supported reference images found in: {source}")

    registry = read_registry()
    target = REFERENCES_ROOT / code
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for index, path in enumerate(sources, start=1):
        destination = target / f"ref_{index:04d}{path.suffix.lower()}"
        if destination.exists() and destination.read_bytes() == path.read_bytes():
            continue
        while destination.exists():
            index += 1
            destination = target / f"ref_{index:04d}{path.suffix.lower()}"
        shutil.copy2(path, destination)
        copied += 1

    registry["products"][code] = {
        "type": "direct",
        "product_code": code,
        "product_name": arguments.product_name,
        "display_name": arguments.display_name or arguments.product_name,
    }
    write_registry(registry)
    sync_operator_catalog_product(
        product_code=code,
        product_name=arguments.product_name,
        display_name=arguments.display_name or arguments.product_name,
    )
    print(f"Added {copied} reference image(s) for {code}. Run build to activate them.")
    return 0


def command_bootstrap_existing(arguments: argparse.Namespace) -> int:
    """Seed references from the already-generated per-object review crops."""
    command_sync_mapping(arguments)
    registry = read_registry()
    if not EXISTING_REVIEW_ROOT.is_dir():
        raise FileNotFoundError(f"Existing review-crop root not found: {EXISTING_REVIEW_ROOT}")
    candidates = [
        path for path in EXISTING_REVIEW_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "review_crops" in {part.lower() for part in path.parts}
    ]
    if not candidates:
        raise ValueError("No existing review crops were found.")

    total = 0
    for visual_key, item in sorted(registry["products"].items()):
        tokens = [str(item.get("product_code") or "")]
        tokens.extend(str(member.get("product_code") or "") for member in item.get("members", []))
        tokens = [token.casefold() for token in tokens if token]
        matched = [path for path in candidates if any(token in str(path).casefold() for token in tokens)]
        target = REFERENCES_ROOT / visual_key
        target.mkdir(parents=True, exist_ok=True)
        existing_hashes = {
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in images_in(target)
        }
        added = 0
        for path in matched:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in existing_hashes:
                continue
            destination = target / f"existing_{len(existing_hashes) + 1:04d}{path.suffix.lower()}"
            shutil.copy2(path, destination)
            existing_hashes.add(digest)
            added += 1
            total += 1
            if len(existing_hashes) >= arguments.max_per_class:
                break
        print(f"  {visual_key}: +{added}, total {len(existing_hashes)}")
    print(f"Bootstrapped {total} distinct approved-crop candidates. Review folders before build.")
    return 0


def command_bootstrap_yolo(arguments: argparse.Namespace) -> int:
    """Create missing reference crops from trusted production YOLO boxes."""
    command_sync_mapping(arguments)
    from app.services.bakery_inference_service import BakeryInferenceService

    registry = read_registry()
    candidates = images_in(EXISTING_ORIGINAL_ROOT)
    if not candidates:
        raise ValueError(f"No original product images found in {EXISTING_ORIGINAL_ROOT}")
    print("Loading production YOLO once for reference extraction...")
    model_path = Path(arguments.model).expanduser().resolve() if arguments.model else None
    detector = BakeryInferenceService(model_path=model_path) if model_path else BakeryInferenceService()
    total = 0
    for visual_key, item in sorted(registry["products"].items()):
        target = REFERENCES_ROOT / visual_key
        target.mkdir(parents=True, exist_ok=True)
        existing_hashes = {
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in images_in(target)
        }
        if len(existing_hashes) >= arguments.max_per_class:
            continue
        tokens = [str(item.get("product_code") or "")]
        tokens.extend(str(member.get("product_code") or "") for member in item.get("members", []))
        tokens = [token.casefold() for token in tokens if token]
        matched = [path for path in candidates if any(token in str(path).casefold() for token in tokens)]
        processed = 0
        added = 0
        for path in matched:
            if processed >= arguments.max_images_per_class:
                break
            processed += 1
            try:
                output = detector.infer_path(path)
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                height, width = image.shape[:2]
                for obj in output.get("objects", []):
                    if str(obj.get("raw_class") or "") != visual_key:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in obj["box_xyxy"]]
                    pad_x = (x2 - x1) * 0.06
                    pad_y = (y2 - y1) * 0.06
                    xa = max(0, int(x1 - pad_x))
                    ya = max(0, int(y1 - pad_y))
                    xb = min(width, int(x2 + pad_x))
                    yb = min(height, int(y2 + pad_y))
                    crop = image[ya:yb, xa:xb]
                    if crop.size == 0:
                        continue
                    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
                    if not ok:
                        continue
                    raw = encoded.tobytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest in existing_hashes:
                        continue
                    destination = target / f"yolo_{len(existing_hashes) + 1:04d}.jpg"
                    destination.write_bytes(raw)
                    existing_hashes.add(digest)
                    added += 1
                    total += 1
                    if len(existing_hashes) >= arguments.max_per_class:
                        break
            except Exception as exc:
                print(f"WARNING: {path.name}: {exc}")
            if len(existing_hashes) >= arguments.max_per_class:
                break
        print(f"  {visual_key}: YOLO +{added}, total {len(existing_hashes)}")
    print(f"Added {total} production-YOLO reference crops.")
    return 0


def command_bootstrap_yolo_folder(arguments: argparse.Namespace) -> int:
    """Create tight reference crops for one registry key from any local folder."""
    command_sync_mapping(arguments)
    from app.services.bakery_inference_service import BakeryInferenceService

    visual_key = safe_code(arguments.visual_class)
    source = Path(arguments.source).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Reference source folder not found: {source}")
    registry = read_registry()
    if visual_key not in registry["products"]:
        raise ValueError(f"Visual class is not registered: {visual_key}")
    candidates = images_in(source)
    if not candidates:
        raise ValueError(f"No supported images found in: {source}")

    target = REFERENCES_ROOT / visual_key
    target.mkdir(parents=True, exist_ok=True)
    existing_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in images_in(target)
    }
    print(f"Loading production YOLO for {visual_key}...")
    model_path = Path(arguments.model).expanduser().resolve() if arguments.model else None
    detector = BakeryInferenceService(model_path=model_path) if model_path else BakeryInferenceService()
    detected_class = str(arguments.detected_class or visual_key).strip()
    added = 0
    processed = 0
    for path in candidates:
        if processed >= arguments.max_images or len(existing_hashes) >= arguments.max_references:
            break
        processed += 1
        try:
            output = detector.infer_path(path)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            object_pool = output.get("yolo_candidate_objects") or output.get("objects") or []
            for obj in object_pool:
                if str(obj.get("raw_class") or "") != detected_class:
                    continue
                if float(obj.get("confidence") or 0.0) < arguments.min_confidence:
                    continue
                x1, y1, x2, y2 = [float(v) for v in obj["box_xyxy"]]
                pad_x = (x2 - x1) * arguments.padding_ratio
                pad_y = (y2 - y1) * arguments.padding_ratio
                xa, ya = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
                xb, yb = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
                crop = image[ya:yb, xa:xb]
                if crop.size == 0 or min(crop.shape[:2]) < 64:
                    continue
                ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
                if not ok:
                    continue
                raw = encoded.tobytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest in existing_hashes:
                    continue
                destination = target / f"yolo_folder_{len(existing_hashes) + 1:04d}.jpg"
                destination.write_bytes(raw)
                existing_hashes.add(digest)
                added += 1
                if len(existing_hashes) >= arguments.max_references:
                    break
        except Exception as exc:
            print(f"WARNING: {path.name}: {exc}")
    print(
        f"{visual_key}: processed {processed} image(s), added {added} tight crop(s), "
        f"total {len(existing_hashes)} reference(s)."
    )
    if not existing_hashes:
        raise ValueError(f"YOLO produced no accepted crop for {visual_key}")
    return 0


def command_build(arguments: argparse.Namespace) -> int:
    registry = read_registry()
    image_paths: list[Path] = []
    reference_keys: list[str] = []
    for key in sorted(registry["products"]):
        paths = images_in(REFERENCES_ROOT / key)
        image_paths.extend(paths)
        reference_keys.extend([key] * len(paths))
    if not image_paths:
        raise ValueError("No approved reference images found. Run sync-mapping/add first.")

    images_rgb: list[np.ndarray] = []
    valid_keys: list[str] = []
    large_reference_warnings: list[Path] = []
    for path, key in zip(image_paths, reference_keys):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"WARNING: skipped unreadable image: {path}")
            continue
        if max(image.shape[:2]) > 2048:
            large_reference_warnings.append(path)
        images_rgb.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        valid_keys.append(key)
    if not images_rgb:
        raise ValueError("All reference images were unreadable.")
    if large_reference_warnings:
        print(
            "WARNING: large reference images may contain too much tray/background. "
            "Use scripts/prepare_foundation_references.py to create tight product "
            "crops before building:"
        )
        for path in large_reference_warnings[:20]:
            print(f"  - {path}")
        if len(large_reference_warnings) > 20:
            print(f"  ... and {len(large_reference_warnings) - 20} more")

    device = arguments.device or "cpu"
    print(f"Loading {FOUNDATION_DINO_MODEL} on {device}...")
    encoder = DinoEmbeddingEncoder(FOUNDATION_DINO_MODEL, device)
    embeddings = encoder.encode_rgb(images_rgb, batch_size=arguments.batch_size)
    FOUNDATION_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb",
        suffix=".npz",
        dir=FOUNDATION_REFERENCE_PATH.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            embeddings=embeddings.astype(np.float32),
            reference_keys=np.asarray(valid_keys, dtype="U256"),
        )
    temporary.replace(FOUNDATION_REFERENCE_PATH)
    if FOUNDATION_MANIFEST_PATH.is_file():
        manifest = json.loads(FOUNDATION_MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest.update(
            {
                "reference_sha256": hashlib.sha256(
                    FOUNDATION_REFERENCE_PATH.read_bytes()
                ).hexdigest().upper(),
                "reference_count": len(valid_keys),
                "embedding_dimensions": int(embeddings.shape[1]),
                "visual_class_count": len(set(valid_keys)),
                "built_at": datetime.now(timezone.utc).date().isoformat(),
            }
        )
        FOUNDATION_MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"Built {len(valid_keys)} embeddings ({embeddings.shape[1]} dimensions) "
        f"for {len(set(valid_keys))} visual classes: {FOUNDATION_REFERENCE_PATH}"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Sharon Bakery hybrid reference manager")
    sub = root.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="Show Foundation reference readiness")
    status.set_defaults(handler=command_status)
    sync = sub.add_parser("sync-mapping", help="Register all current YOLO visual classes")
    sync.set_defaults(handler=command_sync_mapping)
    add = sub.add_parser("add", help="Add approved reference crops for one new SKU")
    add.add_argument("--product-code", required=True)
    add.add_argument("--product-name", required=True)
    add.add_argument("--display-name")
    add.add_argument("--source", required=True)
    add.set_defaults(handler=command_add)
    bootstrap = sub.add_parser(
        "bootstrap-existing",
        help="Seed current visual classes from existing per-object review crops",
    )
    bootstrap.add_argument("--max-per-class", type=int, default=24)
    bootstrap.set_defaults(handler=command_bootstrap_existing)
    bootstrap_yolo = sub.add_parser(
        "bootstrap-yolo",
        help="Fill missing references by cropping existing originals with production YOLO",
    )
    bootstrap_yolo.add_argument("--max-per-class", type=int, default=24)
    bootstrap_yolo.add_argument("--max-images-per-class", type=int, default=40)
    bootstrap_yolo.add_argument("--model")
    bootstrap_yolo.set_defaults(handler=command_bootstrap_yolo)
    bootstrap_folder = sub.add_parser(
        "bootstrap-yolo-folder",
        help="Create tight references for one visual class from a local image folder",
    )
    bootstrap_folder.add_argument("--visual-class", required=True)
    bootstrap_folder.add_argument("--source", required=True)
    bootstrap_folder.add_argument("--max-references", type=int, default=24)
    bootstrap_folder.add_argument("--max-images", type=int, default=40)
    bootstrap_folder.add_argument("--min-confidence", type=float, default=0.15)
    bootstrap_folder.add_argument("--padding-ratio", type=float, default=0.06)
    bootstrap_folder.add_argument(
        "--detected-class",
        help="YOLO class used only for crop localization; defaults to --visual-class",
    )
    bootstrap_folder.add_argument(
        "--model",
        help="Detector checkpoint; defaults to the backend environment setting",
    )
    bootstrap_folder.set_defaults(handler=command_bootstrap_yolo_folder)
    build = sub.add_parser("build", help="Generate the local DINOv2 NPZ artifact")
    build.add_argument("--device", default="cpu")
    build.add_argument("--batch-size", type=int, default=16)
    build.set_defaults(handler=command_build)
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        return int(arguments.handler(arguments))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
