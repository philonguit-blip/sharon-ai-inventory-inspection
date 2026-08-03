"""
YOLO26s batch pre-annotation exporter for Roboflow Object Detection (class-order safe).

Output:
KetQua4_Roboflow/
├── roboflow_upload/
│   ├── images/
│   ├── labels/
│   ├── data.yaml
│   ├── classes.txt
│   └── label_map.json
├── previews/
├── predictions.csv
├── summary.csv
├── errors.csv
└── roboflow_upload.zip
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm.auto import tqdm
from ultralytics import YOLO


# Cho phép Pillow đọc HEIC/HEIF
register_heif_opener()


# =========================================================
# 1. CONFIGURATION
# =========================================================
MODEL_PATH = Path(
    "/content/drive/MyDrive/YOLO26_Models/best_YOLO26s_107_SKU_v2.pt"
)
INPUT_FOLDER = Path(
    "/content/drive/MyDrive/YOLO26_Models/Test_107SKUs/retrain5"
)
OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/YOLO26_Models/"
    "Preannotation/KetQua_retrain5_Roboflow_YOLO26s_107_SKUs"
)

# Pre-annotation should favor recall, then be reviewed in Roboflow.
GLOBAL_CONFIDENCE = 0.10
IOU_THRESHOLD = 0.65
IMAGE_SIZE = 768
MAX_DETECTIONS = 100
DEVICE = 0  # 0 = first GPU, "cpu" = CPU
RECURSIVE = True
CLEAN_OUTPUT = True

# Optional per-class thresholds, calibrated from validation data.
# Example:
# CLASS_THRESHOLDS = {
#     "FL_0004_BotAtta": 0.15,
#     "ISO_OF_0012_BaoRac": 0.35,
# }
CLASS_THRESHOLDS: dict[str, float] = {}

# Conservative post-NMS duplicate cleanup.
SAME_CLASS_DUPLICATE_IOU = 0.85
CROSS_CLASS_CONFLICT_IOU = 0.92

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}



# Canonical 30-SKU mapping used by Roboflow, FastAPI, n8n,
# Supabase, Excel reports, and Teams callbacks.
#
# IMPORTANT:
# MODEL_PATH must point to a YOLO26s checkpoint fine-tuned on these
# 30 classes. The generic pretrained "yolo26s.pt" uses COCO classes
# and must not be used for SKU pre-annotation.
CANONICAL_CLASSES: list[str] = [
    "YE_0003_MenTrang",
    "ISO_SN_0010_BanhCracker",
    "ISO_PA_0029_CuonDayThungNho",
    "ISO_PA_0020_BaoBanhCuonMem",
    "OF_0013_GangTay",
    "PA_0014_NapBanhKemTron",
    "OF_0011_GiayNen",
    "ISO_OF_0012_BaoRac",
    "OF_0009_GiayVS",
    "OF_0010_GiayBep",
    "OF_0008_NuocLavie",
    "OF_0004_NuocRuaTay",
    "OF_0003_NuocRuaChen",
    "OF_0004_MangBocThucPham",
    "FL_0011_BotCustard",
    "OF_0001_VimVeSinh",
    "FV_0003_Chuoi_Trai",
    "CH_0006_SocolaCompoundTrang",
    "FL_0002_BotCake",
    "SN_0019_Oreo",
    "PW_0015_BotNoi",
    "JM_0007_MutPhuBanh",
    "FV_0002_CaRot_Cu",
    "FV_0002_CaRot_Bich",
    "BK_0007_BoLatPilot",
    "FL_0004_BotAtta",
    "FL_0013_BotXanhLa",
    "FL_0014_BotCam",
    "PW_0010_CafeBot_Bich",
    "PW_0010_CafeBot_Hu",

    # =====================================================
    # BỘ NHÃN MỚI — 30 SKU
    # =====================================================
    "SY_0009_TinhChatChanh",
    "FL_0012_BotDo",
    "FL_0003_BotNguyenCam",
    "SY_0008_TinhChatBo",
    "SY_0007_TinhChatCam",
    "SY_0006_TinhChatHanhNhan",
    "JM_0008_PureAnhDao",
    "JM_0005_XoaiPureAndros",
    "JM_0004_PureVietQuat",
    "JM_0003_PureDau",
    "GV_0003_LaKinhGioi",
    "YE_0006_Gelatin",
    "BK_0010_BoVegan",
    "YE_0005_RauCauDeo",
    "YE_0004_BakingSoda",
    "SY_0002_TraEarlGrey",
    "SY_0001_SyrupVani",
    "SU_0005_DuongNau",
    "SU_0004_DuongMachNha",
    "SU_0003_DuongBot",
    "SU_0002_DuongVang",
    "SU_0001_DuongTrang",
    "SN_0011_HanhNhanLat",
    "SN_0006_NamVietQuat",
    "PW_0009_BotPhoMai",
    "PW_0008_BotLaDua",
    "PW_0006_BotQue",
    "PW_0004_BotCacao",
    "PW_0005_BoCacao",
    "PW_0003_BotMatcha",
    "PW_0001_BotGung",

    # =====================================================
    # BỔ SUNG — 46 CLASS MỚI
    # =====================================================
    "OD_0005_SuaDauNanh_Thung",
    "OD_0001_DauAn_Thung",
    "SN_0018_HatDieu",
    "PW_0014_BotCustardHieuSuTu",
    "SN_0001_YenMach",
    "FC_0005_MauNau",
    "SN_0004_OcCho",
    "OD_0006_Trung_Qua",
    "FC_0001_MauXanhLa",
    "FL_0005_BotLuaMachDen",
    "SN_0003_HuongDuong",
    "BK_0003_KemSuaTuoi",
    "FC_0004_MauVang",
    "FV_0001_CaChuaBi",
    "BK_0002_KemPhoMai",
    "OD_0010_BoTuongAn",
    "CH_0007_ChocoStickPuratos",
    "FC_0002_MauDen",
    "FC_0003_MauHong",
    "FC_0001_MauXanhLa_ChaiLon",
    "FL_0008_BotBap",
    "CN_0002_CaChuaLon",
    "FL_0009_BotNang",
    "CH_0001_ChocoCompoundDen",
    "SN_0002_HatBi",
    "FC_0006_MauSieuDo",
    "CH_0002_ChocoCompoundChip",
    "FL_0007_BotKieuMach",
    "GV_0005_PsylliumHusk",
    "FL_0010_BotGao",
    "BK_0009_KemTopping",
    "OD_0001_DauAn",
    "JM_0006_MutMo",
    "OD_0003_DauOliu",
    "CN_0001_OliuNgam",
    "JM_0001_SinhToChanhDay",
    "OD_0005_SuaDauNanh",
    "JM_0002_SinhToXoai",
    "PW_0007_BotNghe",
    "GV_0002_GiamTao",
    "OD_0002_Daudua",
    "OD_0004_SuaTuoi",
    "YE_0001_MenVang",
    "OD_0006_Trung_Khay",
    "YE_002_MenDo",
    "GV_0001_Muoi",
]

EXPECTED_CLASS_COUNT = 107

if len(CANONICAL_CLASSES) != EXPECTED_CLASS_COUNT:
    raise RuntimeError(
        f"Expected {EXPECTED_CLASS_COUNT} classes, "
        f"got {len(CANONICAL_CLASSES)}"
    )

duplicate_classes = sorted(
    {
        class_name
        for class_name in CANONICAL_CLASSES
        if CANONICAL_CLASSES.count(class_name) > 1
    }
)

if duplicate_classes:
    raise RuntimeError(
        f"Duplicate canonical classes: {duplicate_classes}"
    )

# =========================================================
# 2. DATA STRUCTURES
# =========================================================
@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    x_center: float
    y_center: float
    width: float
    height: float


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("yolo26s-roboflow-preannotation")


# =========================================================
# 3. HELPERS
# =========================================================
def prepare_output_directories() -> dict[str, Path]:
    if CLEAN_OUTPUT and OUTPUT_ROOT.exists():
        resolved = OUTPUT_ROOT.resolve()
        if len(resolved.parts) < 4:
            raise RuntimeError(f"Unsafe OUTPUT_ROOT: {resolved}")
        shutil.rmtree(resolved)

    upload_dir = OUTPUT_ROOT / "roboflow_upload"
    images_dir = upload_dir / "images"
    labels_dir = upload_dir / "labels"
    previews_dir = OUTPUT_ROOT / "previews"

    for directory in (images_dir, labels_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "upload": upload_dir,
        "images": images_dir,
        "labels": labels_dir,
        "previews": previews_dir,
    }


def discover_images(root: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        [
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda item: str(item).lower(),
    )


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "image"


def make_sample_id(image_path: Path, input_root: Path) -> str:
    relative = image_path.relative_to(input_root)
    parts = [sanitize_name(part) for part in relative.with_suffix("").parts]
    readable = "__".join(parts)

    if len(relative.parts) > 1:
        digest = hashlib.sha1(
            relative.as_posix().encode("utf-8")
        ).hexdigest()[:8]
        return f"{readable}__{digest}"

    return readable


def load_and_normalize_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def get_class_names(model: YOLO) -> list[str]:
    names = model.names

    if isinstance(names, dict):
        ordered_ids = sorted(int(class_id) for class_id in names)
        if ordered_ids != list(range(len(ordered_ids))):
            raise RuntimeError(f"Non-contiguous class IDs: {ordered_ids}")
        return [str(names[class_id]) for class_id in ordered_ids]

    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]

    raise TypeError(f"Unsupported model.names type: {type(names)}")


def confidence_threshold_for(class_name: str) -> float:
    return float(CLASS_THRESHOLDS.get(class_name, GLOBAL_CONFIDENCE))


def box_iou(a: Detection, b: Detection) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def resolve_duplicates(detections: list[Detection]) -> list[Detection]:
    kept: list[Detection] = []

    for candidate in sorted(
        detections,
        key=lambda item: item.confidence,
        reverse=True,
    ):
        drop = False

        for accepted in kept:
            overlap = box_iou(candidate, accepted)

            if (
                candidate.class_id == accepted.class_id
                and overlap >= SAME_CLASS_DUPLICATE_IOU
            ):
                drop = True
                break

            if (
                candidate.class_id != accepted.class_id
                and overlap >= CROSS_CLASS_CONFLICT_IOU
            ):
                drop = True
                break

        if not drop:
            kept.append(candidate)

    return kept


def extract_detections(result, class_names: list[str]) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxyn = boxes.xyxyn.detach().cpu().numpy()
    xywhn = boxes.xywhn.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()

    detections: list[Detection] = []

    for xyxy, xywh, class_id, confidence in zip(
        xyxyn, xywhn, class_ids, confidences
    ):
        if not 0 <= class_id < len(class_names):
            continue

        class_name = class_names[class_id]
        if float(confidence) < confidence_threshold_for(class_name):
            continue

        x1, y1, x2, y2 = np.clip(xyxy.astype(float), 0.0, 1.0)
        x_center, y_center, width, height = np.clip(
            xywh.astype(float), 0.0, 1.0
        )

        if width <= 0.0 or height <= 0.0:
            continue

        detections.append(
            Detection(
                class_id=int(class_id),
                class_name=class_name,
                confidence=float(confidence),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                x_center=float(x_center),
                y_center=float(y_center),
                width=float(width),
                height=float(height),
            )
        )

    return resolve_duplicates(detections)


def validate_yolo_label(path: Path, class_count: int | None = None) -> None:
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{path.name}:{line_number}: expected 5 fields, got {len(fields)}"
            )

        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]

        if class_id < 0:
            raise ValueError(f"{path.name}:{line_number}: negative class ID")

        if class_count is not None and not 0 <= class_id < class_count:
            raise ValueError(
                f"{path.name}:{line_number}: class ID {class_id} "
                f"outside 0..{class_count - 1}"
            )

        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(
                f"{path.name}:{line_number}: coordinates outside [0, 1]"
            )

        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise ValueError(
                f"{path.name}:{line_number}: width/height must be positive"
            )


def write_yolo_label(
    path: Path,
    detections: list[Detection],
    class_count: int,
) -> None:
    lines = [
        (
            f"{det.class_id} "
            f"{det.x_center:.6f} "
            f"{det.y_center:.6f} "
            f"{det.width:.6f} "
            f"{det.height:.6f}"
        )
        for det in detections
    ]

    content = "\n".join(lines)
    if lines:
        content += "\n"

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    validate_yolo_label(temporary, class_count)
    temporary.replace(path)


def draw_preview(
    image: Image.Image,
    detections: list[Detection],
    destination: Path,
) -> None:
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_height, image_width = frame.shape[:2]

    for det in detections:
        x1 = int(round(det.x1 * image_width))
        y1 = int(round(det.y1 * image_height))
        x2 = int(round(det.x2 * image_width))
        y2 = int(round(det.y2 * image_height))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        caption = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            frame,
            caption,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(destination), frame):
        raise IOError(f"Could not write preview: {destination}")


def write_catalog_files(upload_dir: Path, class_names: list[str]) -> None:
    (upload_dir / "classes.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )

    label_map = {
        str(class_id): class_name
        for class_id, class_name in enumerate(class_names)
    }
    (upload_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    yaml_lines = [
        "path: .",
        "train: images",
        "val: images",
        f"nc: {len(class_names)}",
        "names:",
    ]
    yaml_lines.extend(
        f"  {class_id}: {json.dumps(class_name, ensure_ascii=False)}"
        for class_id, class_name in enumerate(class_names)
    )
    (upload_dir / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_export(
    images_dir: Path,
    labels_dir: Path,
    class_count: int,
) -> None:
    image_stems = {path.stem for path in images_dir.glob("*.jpg")}
    label_stems = {path.stem for path in labels_dir.glob("*.txt")}

    if image_stems != label_stems:
        raise RuntimeError(
            "Image/label mismatch. "
            f"Missing labels={sorted(image_stems - label_stems)[:10]}, "
            f"missing images={sorted(label_stems - image_stems)[:10]}"
        )

    for label_path in labels_dir.glob("*.txt"):
        validate_yolo_label(label_path, class_count)


def create_upload_zip(upload_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for file_path in sorted(upload_dir.rglob("*")):
            if file_path.is_file():
                archive.write(
                    file_path,
                    arcname=file_path.relative_to(upload_dir),
                )


# =========================================================
# 4. MAIN
# =========================================================
def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not INPUT_FOLDER.is_dir():
        raise NotADirectoryError(f"Input folder not found: {INPUT_FOLDER}")

    directories = prepare_output_directories()
    image_paths = discover_images(INPUT_FOLDER, RECURSIVE)

    if not image_paths:
        raise RuntimeError(f"No images found in: {INPUT_FOLDER}")

    LOGGER.info("Found %d images", len(image_paths))
    LOGGER.info("Loading fine-tuned YOLO26s model: %s", MODEL_PATH)

    model = YOLO(str(MODEL_PATH))
    class_names = get_class_names(model)

    # The checkpoint was trained from the current Roboflow data.yaml.
    # Roboflow may assign a different numeric order (often alphabetical)
    # than an older hand-maintained catalog. Numeric IDs are valid only
    # together with the class-name list used during training.
    #
    # Therefore:
    # - require the exact same set of 30 SKU names;
    # - do NOT require the legacy numeric order;
    # - export labels/classes/data.yaml in the checkpoint's own order.
    model_class_set = set(class_names)
    canonical_class_set = set(CANONICAL_CLASSES)

    if len(class_names) != len(CANONICAL_CLASSES) or model_class_set != canonical_class_set:
        missing_classes = sorted(canonical_class_set - model_class_set)
        unexpected_classes = sorted(model_class_set - canonical_class_set)

        raise RuntimeError(
            "YOLO26s checkpoint không chứa đúng bộ 30 SKU.\n"
            f"Missing classes: {missing_classes}\n"
            f"Unexpected classes: {unexpected_classes}\n"
            "Hãy kiểm tra checkpoint hoặc data.yaml dùng để train."
        )

    if class_names != CANONICAL_CLASSES:
        LOGGER.warning(
            "Thứ tự class trong YOLO26s khác catalog cũ. "
            "Script sẽ dùng chính model.names để giữ đúng class ID của checkpoint."
        )
        LOGGER.warning(
            "Không được diễn giải class_id bằng thứ tự cũ; luôn dùng classes.txt, "
            "data.yaml hoặc label_map.json được xuất cùng batch này."
        )

    write_catalog_files(directories["upload"], class_names)

    LOGGER.info("Model classes: %d", len(class_names))
    for class_id, class_name in enumerate(class_names):
        LOGGER.info("  %02d: %s", class_id, class_name)

    inference_floor = min(
        [GLOBAL_CONFIDENCE, *CLASS_THRESHOLDS.values()]
    )

    prediction_rows: list[dict] = []
    summary_rows: list[dict] = []
    error_rows: list[dict] = []

    succeeded = 0
    failed = 0
    total_detections = 0

    for image_path in tqdm(image_paths, desc="Pre-annotating"):
        sample_id = make_sample_id(image_path, INPUT_FOLDER)

        export_image_path = directories["images"] / f"{sample_id}.jpg"
        label_path = directories["labels"] / f"{sample_id}.txt"
        preview_path = directories["previews"] / f"{sample_id}.jpg"

        try:
            normalized_image = load_and_normalize_image(image_path)

            # Save exactly the same normalized pixels used for inference.
            normalized_image.save(
                export_image_path,
                format="JPEG",
                quality=95,
                subsampling=0,
            )

            results = model.predict(
                source=normalized_image,
                conf=inference_floor,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                max_det=MAX_DETECTIONS,
                device=DEVICE,
                save=False,
                verbose=False,
            )

            if len(results) != 1:
                raise RuntimeError(f"Expected 1 result, got {len(results)}")

            detections = extract_detections(results[0], class_names)

            write_yolo_label(
                label_path,
                detections,
                len(class_names),
            )
            draw_preview(
                normalized_image,
                detections,
                preview_path,
            )

            for det in detections:
                prediction_rows.append(
                    {
                        "sample_id": sample_id,
                        "source_image": str(image_path),
                        "class_id": det.class_id,
                        "class_name": det.class_name,
                        "confidence": round(det.confidence, 6),
                        "x_center": round(det.x_center, 6),
                        "y_center": round(det.y_center, 6),
                        "width": round(det.width, 6),
                        "height": round(det.height, 6),
                    }
                )

            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "source_image": str(image_path),
                    "export_image": str(export_image_path),
                    "label_file": str(label_path),
                    "detection_count": len(detections),
                    "status": "SUCCESS",
                }
            )

            succeeded += 1
            total_detections += len(detections)

        except Exception as exc:
            failed += 1
            LOGGER.exception("Failed: %s", image_path)
            error_rows.append(
                {
                    "sample_id": sample_id,
                    "source_image": str(image_path),
                    "error": str(exc),
                }
            )
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "source_image": str(image_path),
                    "export_image": "",
                    "label_file": "",
                    "detection_count": 0,
                    "status": "ERROR",
                }
            )

    write_csv(
        OUTPUT_ROOT / "predictions.csv",
        [
            "sample_id", "source_image", "class_id", "class_name",
            "confidence", "x_center", "y_center", "width", "height",
        ],
        prediction_rows,
    )

    write_csv(
        OUTPUT_ROOT / "summary.csv",
        [
            "sample_id", "source_image", "export_image",
            "label_file", "detection_count", "status",
        ],
        summary_rows,
    )

    write_csv(
        OUTPUT_ROOT / "errors.csv",
        ["sample_id", "source_image", "error"],
        error_rows,
    )

    validate_export(
        directories["images"],
        directories["labels"],
        len(class_names),
    )

    zip_path = OUTPUT_ROOT / "roboflow_upload.zip"
    create_upload_zip(directories["upload"], zip_path)

    LOGGER.info("=" * 70)
    LOGGER.info(
        "Completed: succeeded=%d | failed=%d | detections=%d",
        succeeded,
        failed,
        total_detections,
    )
    LOGGER.info("Roboflow ZIP: %s", zip_path)
    LOGGER.info("Previews: %s", directories["previews"])
    LOGGER.info("Audit CSV: %s", OUTPUT_ROOT / "predictions.csv")

    if failed:
        raise RuntimeError(
            f"{failed} image(s) failed. Review {OUTPUT_ROOT / 'errors.csv'}"
        )


if __name__ == "__main__":
    main()
