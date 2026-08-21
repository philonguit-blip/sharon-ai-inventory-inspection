"""Create tight single-product Foundation references from full tray photos.

This utility never overwrites source images. Review the generated crops, then
use ``hybrid_reference_manager.py add`` or copy the approved folder into the
reference library before rebuilding the DINO NPZ artifact. Background output
can be original, white, or both; ``both`` is recommended for domain robustness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
os.environ.setdefault("YOLO_CONFIG_DIR", str(BACKEND_ROOT / "hybrid_data"))
os.environ.setdefault("MPLCONFIGDIR", str(BACKEND_ROOT / "hybrid_data"))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ultralytics import SAM  # noqa: E402
from app.config import FOUNDATION_DINO_MODEL  # noqa: E402
from app.services.foundation_inference_service import (  # noqa: E402
    DinoEmbeddingEncoder,
)
from app.services.reference_autocrop_service import (  # noqa: E402
    auto_crop_reference,
    composite_reference_on_white,
    segment_tight_reference_foreground,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", required=True, help="Folder of one-product photos")
    value.add_argument("--output", required=True, help="New folder for reviewable crops")
    value.add_argument(
        "--input-mode",
        choices=("auto-crop", "tight"),
        default="auto-crop",
        help=(
            "Use 'auto-crop' for raw photos or 'tight' for already-approved "
            "reference crops whose frame must remain unchanged."
        ),
    )
    value.add_argument(
        "--sam-model",
        default=str(BACKEND_ROOT / "models" / "sam2.1_s.pt"),
    )
    value.add_argument("--max-side", type=int, default=1280)
    value.add_argument("--padding", type=float, default=0.08)
    value.add_argument(
        "--background",
        choices=("original", "white", "both"),
        default="original",
        help=(
            "Reference background mode. 'both' writes the real crop and a "
            "white-background PNG without changing the source image."
        ),
    )
    value.add_argument(
        "--feather-radius",
        type=float,
        default=1.5,
        help="White-background edge feather radius in pixels (0 disables it).",
    )
    value.add_argument(
        "--reference-key",
        default="",
        help=(
            "Existing reference key. When supplied, DINO uses that SKU's "
            "active vectors to choose among SAM crop candidates."
        ),
    )
    value.add_argument(
        "--reference-path",
        default=str(BACKEND_ROOT / "hybrid_data" / "reference_embeddings.npz"),
    )
    value.add_argument("--dino-model", default=FOUNDATION_DINO_MODEL)
    value.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    value.add_argument("--batch-size", type=int, default=16)
    return value


def main() -> int:
    arguments = parser().parse_args()
    source = Path(arguments.source).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = SAM(str(Path(arguments.sam_model).expanduser().resolve()))
    encoder = None
    target_embeddings = None
    reference_key = str(arguments.reference_key or "").strip()
    if reference_key and arguments.input_mode == "auto-crop":
        reference_path = Path(arguments.reference_path).expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        payload = np.load(reference_path, allow_pickle=False)
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        keys = np.asarray(payload["reference_keys"]).astype(str)
        positions = np.where(keys == reference_key)[0]
        if not len(positions):
            raise ValueError(
                f"Reference key has no active embeddings: {reference_key}"
            )
        target_embeddings = embeddings[positions]
        encoder = DinoEmbeddingEncoder(arguments.dino_model, arguments.device)
    report = []
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            report.append({"source": path.name, "status": "SKIPPED_UNREADABLE"})
            continue
        try:
            if arguments.input_mode == "tight":
                crop = image.copy()
                box = [0, 0, int(image.shape[1]), int(image.shape[0])]
                foreground_mask, selection_metadata = (
                    segment_tight_reference_foreground(
                        crop,
                        sam_model=model,
                        max_side=max(640, arguments.max_side),
                        device=arguments.device,
                    )
                )
                mask_ratio = float(foreground_mask.mean())
            else:
                crop_result = auto_crop_reference(
                    image,
                    sam_model=model,
                    max_side=max(640, arguments.max_side),
                    padding=max(0.0, arguments.padding),
                    device=arguments.device,
                    encoder=encoder,
                    target_embeddings=target_embeddings,
                    batch_size=max(1, arguments.batch_size),
                )
                crop = crop_result.crop_bgr
                box = crop_result.box_xyxy
                mask_ratio = crop_result.mask_ratio
                foreground_mask = crop_result.foreground_mask
                selection_metadata = crop_result.metadata
            outputs: dict[str, str] = {}
            background_metadata: dict[str, object] = {
                "mode": arguments.background,
            }
            white_crop = None
            if arguments.background in {"white", "both"}:
                if foreground_mask is None:
                    raise RuntimeError(
                        "Selected crop has no trustworthy aligned foreground mask; "
                        "keep it for manual review instead of forcing a white background."
                    )
                white_crop, white_metadata = composite_reference_on_white(
                    crop,
                    foreground_mask,
                    feather_radius=max(0.0, arguments.feather_radius),
                )
                background_metadata.update(white_metadata)
            if arguments.background in {"original", "both"}:
                destination = output / f"{path.stem}.jpg"
                if not cv2.imwrite(
                    str(destination),
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 94],
                ):
                    raise RuntimeError("cannot write original-background crop")
                outputs["original"] = destination.name
            if arguments.background in {"white", "both"}:
                assert white_crop is not None
                white_destination = output / f"{path.stem}_white.png"
                if not cv2.imwrite(str(white_destination), white_crop):
                    raise RuntimeError("cannot write white-background crop")
                outputs["white"] = white_destination.name
            report.append(
                {
                    "source": path.name,
                    "output": next(iter(outputs.values())),
                    "outputs": outputs,
                    "status": "OK",
                    "box_xyxy": box,
                    "mask_area_ratio": mask_ratio,
                    "crop_width": int(crop.shape[1]),
                    "crop_height": int(crop.shape[0]),
                    "input_mode": arguments.input_mode,
                    "selection": selection_metadata,
                    "background": background_metadata,
                }
            )
            print(
                f"OK {path.name}: box={box}, "
                f"background={arguments.background}, outputs={len(outputs)}"
            )
        except Exception as exc:
            report.append({"source": path.name, "status": "REVIEW", "error": str(exc)})
            print(f"REVIEW {path.name}: {exc}")
    (output / "crop_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    succeeded = sum(item["status"] == "OK" for item in report)
    print(f"Generated {succeeded}/{len(report)} tight references in {output}")
    return 0 if succeeded == len(report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
