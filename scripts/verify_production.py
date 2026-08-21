from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-missing-env",
        action="store_true",
        help="Validate repository artifacts without requiring local secrets.",
    )
    parser.add_argument(
        "--require-foundation",
        action="store_true",
        help="Require external SAM2 weight in addition to tracked hybrid artifacts.",
    )
    args = parser.parse_args()
    errors: list[str] = []
    manifest_path = BACKEND / "models" / "MODEL_MANIFEST.json"
    foundation_manifest_path = BACKEND / "models" / "FOUNDATION_MANIFEST.json"
    foundation_registry_path = BACKEND / "config" / "hybrid_reference_registry.json"
    mapping_path = BACKEND / "config" / "product_mapping.json"
    workflow_path = ROOT / "n8n" / "Workflow 4_ Sharon Bakery Outbound Worker.json"
    config_path = BACKEND / "app" / "config.py"
    env_example_path = BACKEND / ".env.example"
    frontend_config_path = BACKEND / "frontend" / "assets" / "config.js"
    frontend_app_path = BACKEND / "frontend" / "assets" / "app.js"

    require(manifest_path.is_file(), f"Missing {manifest_path}", errors)
    require(foundation_manifest_path.is_file(), f"Missing {foundation_manifest_path}", errors)
    require(foundation_registry_path.is_file(), f"Missing {foundation_registry_path}", errors)
    require(mapping_path.is_file(), f"Missing {mapping_path}", errors)
    require(workflow_path.is_file(), f"Missing {workflow_path}", errors)
    require(frontend_config_path.is_file(), "Missing frontend runtime config.js", errors)
    require(frontend_app_path.is_file(), "Missing frontend app.js", errors)
    require(config_path.is_file(), "Missing backend app/config.py", errors)
    require(env_example_path.is_file(), "Missing backend/.env.example", errors)
    if not args.allow_missing_env:
        require((BACKEND / ".env").is_file(), "Missing backend/.env (copy .env.example and fill credentials)", errors)

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    foundation_manifest = json.loads(foundation_manifest_path.read_text(encoding="utf-8"))
    model_path = BACKEND / "models" / str(manifest["production_model"])
    require(model_path.is_file(), f"Missing production model: {model_path}", errors)
    if model_path.is_file():
        require(model_path.stat().st_size == int(manifest["size_bytes"]), "Production model size does not match manifest", errors)
        require(sha256(model_path) == str(manifest["sha256"]).upper(), "Production model SHA256 does not match manifest", errors)
    sam_path = BACKEND / "models" / str(foundation_manifest["sam_model"])
    reference_path = (BACKEND / "models" / str(foundation_manifest["reference_artifact"])).resolve()
    require(reference_path.is_file(), f"Missing Foundation reference artifact: {reference_path}", errors)
    if reference_path.is_file():
        require(
            sha256(reference_path) == str(foundation_manifest["reference_sha256"]).upper(),
            "Foundation reference SHA256 does not match manifest",
            errors,
        )
    if args.require_foundation:
        require(sam_path.is_file(), f"Missing external SAM2 model: {sam_path}", errors)
    if sam_path.is_file():
        require(
            sha256(sam_path) == str(foundation_manifest["sam_sha256"]).upper(),
            "SAM2 SHA256 does not match manifest",
            errors,
        )

    production_checkpoints = list((BACKEND / "models").glob("*.pt")) + list(
        (BACKEND / "models").glob("*.onnx")
    )
    allowed_checkpoints = {model_path.resolve()}
    selectable_models = manifest.get("selectable_models") or []
    require(
        isinstance(selectable_models, list),
        "MODEL_MANIFEST.selectable_models must be a list",
        errors,
    )
    if isinstance(selectable_models, list):
        for entry in selectable_models:
            require(
                isinstance(entry, dict),
                "Every selectable model manifest entry must be an object",
                errors,
            )
            if not isinstance(entry, dict):
                continue
            selectable_path = BACKEND / "models" / str(entry.get("file") or "")
            require(
                selectable_path.suffix.casefold() in {".pt", ".onnx"},
                f"Invalid selectable model suffix: {selectable_path.name}",
                errors,
            )
            require(
                selectable_path.is_file(),
                f"Missing selectable model: {selectable_path}",
                errors,
            )
            if selectable_path.is_file():
                require(
                    selectable_path.stat().st_size == int(entry["size_bytes"]),
                    f"Selectable model size does not match manifest: {selectable_path.name}",
                    errors,
                )
                require(
                    sha256(selectable_path) == str(entry["sha256"]).upper(),
                    f"Selectable model SHA256 does not match manifest: {selectable_path.name}",
                    errors,
                )
                if entry.get("compatible_with_current_mapping", True):
                    require(
                        int(entry.get("visual_class_count") or 0)
                        == int(manifest.get("visual_class_count") or 0),
                        f"Selectable model class count does not match production: {selectable_path.name}",
                        errors,
                    )
            allowed_checkpoints.add(selectable_path.resolve())
    if sam_path.is_file():
        allowed_checkpoints.add(sam_path.resolve())
    require(
        {path.resolve() for path in production_checkpoints} == allowed_checkpoints,
        "backend/models contains a checkpoint not declared by a manifest",
        errors,
    )

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    foundation_registry = json.loads(foundation_registry_path.read_text(encoding="utf-8"))
    classes = mapping.get("classes") or {}
    visual_class_count = int(mapping.get("visual_class_count") or 0)
    require(len(classes) == visual_class_count == int(manifest.get("visual_class_count") or 0), "Model manifest and product mapping visual-class counts must match", errors)
    business_skus = sum(1 if value.get("type") == "direct" else len(value.get("members") or []) for value in classes.values())
    require(business_skus == int(mapping.get("business_sku_count") or 0) == int(manifest.get("business_sku_count") or 0), "Model manifest and product mapping business-SKU counts must match", errors)
    catalog_products = mapping.get("catalog_products") or []
    supported_products = business_skus + len(catalog_products)
    require(
        supported_products == int(mapping.get("supported_product_count") or 0) == int(manifest.get("supported_product_count") or 0),
        "Model manifest and operator catalog supported-product counts must match",
        errors,
    )
    require(all(0 < float(value.get("confidence_threshold", 0)) <= 1 for value in classes.values()), "Every class must have a valid confidence threshold", errors)
    foundation_class_count = len(foundation_registry.get("products") or {})
    declared_foundation_classes = int(
        foundation_manifest.get("visual_class_count") or 0
    )
    require(foundation_class_count >= declared_foundation_classes, "Foundation registry must contain every embedded reference key", errors)
    if reference_path.is_file():
        with np.load(reference_path, allow_pickle=False) as artifact:
            reference_keys = [str(value) for value in artifact["reference_keys"].tolist()]
            embeddings = artifact["embeddings"]
        require(len(reference_keys) == int(foundation_manifest.get("reference_count") or 0), "Foundation reference count does not match its artifact", errors)
        require(embeddings.ndim == 2 and embeddings.shape[0] == len(reference_keys), "Foundation embedding matrix shape is invalid", errors)
        require(embeddings.shape[1] == int(foundation_manifest.get("embedding_dimensions") or 0), "Foundation embedding dimension does not match its manifest", errors)
        embedded_keys = set(reference_keys)
        registry_keys = set((foundation_registry.get("products") or {}).keys())
        require(len(embedded_keys) == declared_foundation_classes, "Foundation embedded-class count does not match its manifest", errors)
        require(embedded_keys <= registry_keys, "Foundation artifact contains an unregistered key", errors)
        require(set(foundation_manifest.get("required_embedding_keys") or []) <= embedded_keys, "Foundation artifact is missing a required production embedding key", errors)

    config_text = config_path.read_text(encoding="utf-8")
    env_example = env_example_path.read_text(encoding="utf-8")
    frontend_config = frontend_config_path.read_text(encoding="utf-8")
    frontend_app = frontend_app_path.read_text(encoding="utf-8")
    require(str(manifest["production_model"]) in config_text, "Backend default model does not match the manifest", errors)
    require(f"YOLO_MODEL_PATH=models/{manifest['production_model']}" in env_example, ".env.example model does not match the manifest", errors)
    require(int(manifest.get("image_size") or 0) == 1024, "Production YOLO image size must be 1024", errors)
    require("BAKERY_IMAGE_SIZE=1024" in env_example, ".env.example must use the 1024 YOLO image size", errors)
    require("MAX_IMAGES_PER_JOB=50" in env_example, ".env.example must allow 50-image same-SKU batches", errors)
    require("MAX_JOB_UPLOAD_SIZE_BYTES=209715200" in env_example, ".env.example must use the 200 MB batch limit", errors)
    require("defaultMaxImagesPerJob = 50" in frontend_app, "Frontend must allow 50-image batches", errors)
    require("defaultMaxJobUploadSizeMb = 200" in frontend_app, "Frontend must use the 200 MB batch limit", errors)
    require("n8n.sharon-finefoods.com/webhook" in frontend_config, "Frontend remote webhook base is missing", errors)

    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    node_names = {str(node.get("name") or "") for node in workflow.get("nodes") or []}
    require("00a - Web UI" in node_names, "n8n workflow has no embedded Web UI", errors)
    require("03b - Confirm Job" in node_names, "n8n workflow has no confirmation endpoint", errors)
    require(len(workflow.get("nodes") or []) == 36, "n8n workflow must contain 36 generated nodes", errors)
    workflow_text = workflow_path.read_text(encoding="utf-8")
    require("max_images_per_job || 50" in workflow_text, "n8n workflow must inherit the 50-image worker limit", errors)
    require("max_job_upload_size_mb || 200" in workflow_text, "n8n workflow must use the 200 MB fallback batch-size limit", errors)
    require("'AUTO', 'YOLO', 'FOUNDATION', 'COMPARE'" in workflow_text, "n8n workflow must preserve hybrid inference modes", errors)

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1

    print("[OK] Production structure is consistent.")
    print(f"[OK] Model: {model_path.name} ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"[OK] Taxonomy: {visual_class_count} YOLO classes -> {business_skus} YOLO SKUs; {supported_products} supported products")
    print(f"[OK] Foundation: {foundation_manifest['reference_count']} references; SAM2 " + ("present" if sam_path.is_file() else "external/not provisioned"))
    print("[OK] n8n: embedded Web UI + outbound queue + explicit confirmation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
