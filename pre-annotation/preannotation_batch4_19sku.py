from __future__ import annotations

"""
YOLO-World pre-annotation pipeline for 19 newly analyzed Sharon Bakery SKUs.

Key properties
--------------
- YOLO-World only. No Gemini, no external API, no secret key.
- Uses multiple English prompts per SKU and maps them back to one numeric YOLO class ID.
- Applies per-SKU confidence thresholds.
- Suppresses duplicate boxes produced by prompt ensembles.
- Resolves overlapping predictions between visually similar SKUs.
- Sends low-confidence or ambiguous predictions to a review queue instead of writing noisy labels.
- Writes standard Ultralytics YOLO labels:
      class_id x_center y_center width height
  where class_id is zero-based and all coordinates are normalized.

Recommended Python: 3.10 or 3.11.
"""

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from tqdm import tqdm
from importlib.metadata import PackageNotFoundError, version

try:
    ULTRALYTICS_VERSION = version("ultralytics")
except PackageNotFoundError:  # pragma: no cover - dependency validated at runtime
    ULTRALYTICS_VERSION = "not-installed"


SCRIPT_VERSION = "3.1.0"
EXPECTED_SKU_COUNT = 19
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class PromptSpec:
    text: str
    role: str
    auto_accept: bool


@dataclass(frozen=True)
class SKUConfig:
    class_name: str
    display_name: str
    prompts: tuple[PromptSpec, ...]
    candidate_conf: float
    keep_conf: float
    conflict_group: str
    same_sku_iou: float = 0.70
    nested_containment: float = 0.90
    expected_aspect_ratio: tuple[float, float] | None = None


@dataclass(frozen=True)
class GroupRule:
    iou_threshold: float
    containment_threshold: float
    score_margin: float


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: int
    sku_id: int
    class_name: str
    text: str
    role: str
    auto_accept: bool


@dataclass
class Detection:
    sku_id: int
    class_name: str
    score: float
    xyxy: np.ndarray
    prompt_texts: set[str] = field(default_factory=set)
    prompt_roles: set[str] = field(default_factory=set)
    auto_supported: bool = False
    status: str = "candidate"
    reasons: list[str] = field(default_factory=list)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy.tolist()
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# -----------------------------------------------------------------------------
# 19-SKU vocabulary
# Class order below is the final numeric YOLO class-ID order.
# This is a replacement set containing only the 19 newly analyzed SKUs.
# -----------------------------------------------------------------------------
SKU_CONFIGS: tuple[SKUConfig, ...] = (
    SKUConfig(
        class_name="OD_0005_SuaDauNanh",
        display_name="SuaDauNanh_VinamilkUnsweetenedSoyMilk",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="rectangular_beverage_cartons",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.35, 1.25),
        prompts=(
            PromptSpec(
                "single light green Vinamilk unsweetened soy milk carton with white screw cap",
                "primary",
                True,
            ),
            PromptSpec(
                "upright one liter soybean milk box labeled Sua Dau Nanh Khong Duong with green soybean graphics",
                "secondary",
                True,
            ),
            PromptSpec(
                "side or rear view of a pale green Vinamilk soy milk carton with nutrition label and barcode",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a light green beverage carton with a large white plastic screw cap",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="OD_0004_SuaTuoi",
        display_name="SuaTuoi_THTrueMilkWholeMilk",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="rectangular_beverage_cartons",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.35, 1.25),
        prompts=(
            PromptSpec(
                "single white and light blue TH true MILK whole milk carton with white screw cap",
                "primary",
                True,
            ),
            PromptSpec(
                "upright one liter fresh milk box labeled TH true MILK and Nguyen Chat with blue sky design",
                "secondary",
                True,
            ),
            PromptSpec(
                "side or rear view of a white and pale blue TH milk carton with circular logo and barcode",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a light blue and white beverage carton with a large white plastic screw cap",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="OD_0001_DauAn",
        display_name="DauAn_OitaCookingOil5L",
        candidate_conf=0.10,
        keep_conf=0.22,
        conflict_group="large_cooking_oil_jugs",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.40, 1.15),
        prompts=(
            PromptSpec(
                "single large transparent five-liter Oita cooking oil jug with red cap white handle and yellow front label",
                "primary",
                True,
            ),
            PromptSpec(
                "large rectangular clear plastic container filled with golden cooking oil and labeled Oita Plus",
                "secondary",
                True,
            ),
            PromptSpec(
                "rear or side view of a large ribbed transparent cooking oil jug filled with amber yellow oil",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a large golden cooking oil container with red screw cap and white carry handle",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="JM_0006_MutMo",
        display_name="MutMo_PrimaFruttaApricotJam",
        candidate_conf=0.10,
        keep_conf=0.20,
        conflict_group="small_fruit_jam_jars",
        same_sku_iou=0.68,
        nested_containment=0.90,
        expected_aspect_ratio=(0.55, 1.20),
        prompts=(
            PromptSpec(
                "single small glass jar of Prima Frutta apricot jam with white fruit-patterned metal lid",
                "primary",
                True,
            ),
            PromptSpec(
                "short round Menz and Gasser apricot preserve jar with yellow apricot graphics and dark red jam",
                "secondary",
                True,
            ),
            PromptSpec(
                "rear or side view of a small glass fruit jam jar with white label apricot image and white printed lid",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a small round jam jar with white metal lid covered in colorful fruit illustrations",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="JM_0002_SinhToXoai",
        display_name="SinhToXoai_GoldenFarmMangoSmoothie",
        candidate_conf=0.10,
        keep_conf=0.22,
        conflict_group="fruit_smoothie_bottles",
        same_sku_iou=0.68,
        nested_containment=0.90,
        expected_aspect_ratio=(0.28, 0.80),
        prompts=(
            PromptSpec(
                "single tall clear Golden Farm mango smoothie bottle with dark teal ribbed cap and teal front label",
                "primary",
                True,
            ),
            PromptSpec(
                "one-liter transparent fruit smoothie bottle filled with yellow mango chunks and labeled Sinh To Xoai Mango",
                "secondary",
                True,
            ),
            PromptSpec(
                "rear or side view of a tall clear bottle containing thick yellow mango smoothie with visible fruit pieces",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a tall yellow fruit smoothie bottle with a large dark teal polygonal plastic cap",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="JM_0001_SinhToChanhDay",
        display_name="SinhToChanhDay_GoldenFarmPassionFruitSmoothie",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="fruit_smoothie_bottles",
        same_sku_iou=0.68,
        nested_containment=0.90,
        expected_aspect_ratio=(0.28, 0.80),
        prompts=(
            PromptSpec(
                "single tall clear Golden Farm passion fruit smoothie bottle with dark teal ribbed cap and teal front label",
                "primary",
                True,
            ),
            PromptSpec(
                "one-liter transparent bottle of orange passion fruit smoothie labeled Sinh To Chanh Day and Seedless Passion Fruit",
                "secondary",
                True,
            ),
            PromptSpec(
                "rear or side view of a tall clear smoothie bottle containing thick orange-brown passion fruit mixture with teal label",
                "view",
                False,
            ),
            PromptSpec(
                "top view of an orange-brown fruit smoothie bottle with a large dark teal polygonal plastic cap",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="FV_0001_CaChuaBi",
        display_name="CaChuaBi_BaggedCherryTomatoes",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="mesh_bagged_small_produce",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.65, 1.60),
        prompts=(
            PromptSpec(
                "single yellow mesh bag filled with many red and green cherry tomatoes",
                "primary",
                True,
            ),
            PromptSpec(
                "round net produce bag containing small oval cherry tomatoes with a yellow tied top",
                "secondary",
                True,
            ),
            PromptSpec(
                "side view of a bulging yellow mesh sack packed with mixed red orange and green cherry tomatoes",
                "view",
                False,
            ),
            PromptSpec(
                "top view of many small cherry tomatoes enclosed together inside a yellow net bag",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="FL_0005_BotLuaMachDen",
        display_name="BotLuaMachDen_RyeBreadMix",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="large_white_baking_mix_sacks",
        same_sku_iou=0.72,
        nested_containment=0.90,
        expected_aspect_ratio=(0.35, 1.15),
        prompts=(
            PromptSpec(
                "single large white Grand-Place Puratos bag of rye bread mix with magenta bottom panel",
                "primary",
                True,
            ),
            PromptSpec(
                "large flexible five-kilogram white baking mix sack labeled Bot Tron Banh Mi Lua Mach Den with bread photograph",
                "secondary",
                True,
            ),
            PromptSpec(
                "side or rear view of a bulky white Puratos ingredient bag with black nutrition text barcode and small magenta areas",
                "view",
                False,
            ),
            PromptSpec(
                "large wrinkled white bakery powder sack with folded open top and bright magenta lower corner",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="CH_0007_ChocoStickPuratos",
        display_name="ChocoStickPuratos_GrandPlaceDarkCompoundSticks",
        candidate_conf=0.08,
        keep_conf=0.24,
        conflict_group="rectangular_chocolate_ingredient_boxes",
        same_sku_iou=0.70,
        nested_containment=0.88,
        expected_aspect_ratio=(0.18, 3.20),
        prompts=(
            PromptSpec(
                "single wide silver Grand-Place box of dark chocolate compound sticks with purple cocoa artwork",
                "primary",
                True,
            ),
            PromptSpec(
                "rectangular one-kilogram chocolate baking sticks carton labeled Dark Compound Sticks D632",
                "secondary",
                True,
            ),
            PromptSpec(
                "opened Grand-Place chocolate sticks box with brown cardboard lid and three white compartments filled with dark rectangular chocolate sticks",
                "secondary",
                True,
            ),
            PromptSpec(
                "top view of an open chocolate ingredient carton containing three trays of dark compound chocolate sticks",
                "view",
                True,
            ),
            PromptSpec(
                "rear or narrow side view of a flat silver chocolate ingredient box with barcode and black product text",
                "view",
                False,
            ),
            PromptSpec(
                "long thin reflective silver carton of compound chocolate sticks viewed from the top edge",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="CH_0002_ChocoCompoundChip",
        display_name="ChocoCompoundChip_GrandPlaceDarkChocolateChips",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="rectangular_chocolate_ingredient_boxes",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.30, 2.80),
        prompts=(
            PromptSpec(
                "single brown Grand-Place carton of dark compound chocolate chips with maroon cocoa artwork",
                "primary",
                True,
            ),
            PromptSpec(
                "rectangular 2.5 kilogram chocolate ingredient box labeled Dark Compound Chips D07C",
                "secondary",
                True,
            ),
            PromptSpec(
                "side view of a tall brown kraft chocolate carton with red product instructions and cocoa graphics",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a brown cardboard box with white label reading dark compound chips and picture of chocolate chips",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="CH_0001_ChocoCompoundDen",
        display_name="ChocoCompoundDen_GrandPlaceDarkCompoundChocolate",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="grand_place_chocolate_pouches",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.25, 1.35),
        prompts=(
            PromptSpec(
                "single dark magenta Grand-Place pouch of dark compound chocolate with white cocoa farm illustrations",
                "primary",
                True,
            ),
            PromptSpec(
                "upright one-kilogram red wine colored chocolate ingredient bag labeled Dark Compound D03B",
                "secondary",
                True,
            ),
            PromptSpec(
                "rear view of a dark magenta flexible chocolate pouch with large white product label barcode and chocolate pieces",
                "view",
                False,
            ),
            PromptSpec(
                "thin side or top view of a sealed dark red flexible chocolate compound bag",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="BK_0009_KemTopping",
        display_name="KemTopping_PuratosAmbianteWhippableTopping",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="rectangular_whipping_topping_cartons",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.35, 1.25),
        prompts=(
            PromptSpec(
                "single white and blue Puratos Ambiante plant-based whipping topping carton with whipped cream cake image",
                "primary",
                True,
            ),
            PromptSpec(
                "upright one-liter bakery topping box labeled Ambiante Plant-Based Whippable Topping and For Decoration",
                "secondary",
                True,
            ),
            PromptSpec(
                "side or rear view of a white rectangular Ambiante topping carton with blue top ingredient text and barcode",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a rectangular white bakery cream carton with dark blue sealed top labeled Ambiante",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="BK_0003_KemSuaTuoi",
        display_name="KemSuaTuoi_TatuaWhippingCream",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="flexible_whipping_cream_pouches",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.25, 1.20),
        prompts=(
            PromptSpec(
                "single lavender Tatua whipping cream pouch with strawberry cream cake image",
                "primary",
                True,
            ),
            PromptSpec(
                "upright one-liter purple and silver dairy cream bag labeled Tatua Whipping Cream",
                "secondary",
                True,
            ),
            PromptSpec(
                "rear view of a lavender flexible whipping cream pouch with nutrition panel Vietnamese text and barcode",
                "view",
                False,
            ),
            PromptSpec(
                "crumpled tall purple dairy cream pouch with sealed silver edges and folded top",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="BK_0002_KemPhoMai",
        display_name="KemPhoMai_DairymontClassicCreamCheese",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="rectangular_dairy_product_boxes",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.35, 2.80),
        prompts=(
            PromptSpec(
                "single silver Dairymont Classic cream cheese box with dark red horizontal band",
                "primary",
                True,
            ),
            PromptSpec(
                "wide rectangular two-kilogram dairy carton labeled Dairymont Cream Cheese Classic",
                "secondary",
                True,
            ),
            PromptSpec(
                "side view of a silver cream cheese box with Vietnamese Pho Mai Kem label 2 to 4 degree storage icon and red Made in Australia panel",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a long flat silver dairy product carton with black manufacturing text",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="YE_002_MenDo",
        display_name="MenDo_SafInstantRedDryYeast",
        candidate_conf=0.10,
        keep_conf=0.26,
        conflict_group="saf_instant_yeast_packs",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.40, 1.20),
        prompts=(
            PromptSpec(
                "single white red and blue Saf-instant instant dry yeast vacuum pack with chef illustration",
                "primary",
                True,
            ),
            PromptSpec(
                "rectangular 500 gram Saf-instant The Original yeast package with dark red center panel and blue bottom strip",
                "secondary",
                True,
            ),
            PromptSpec(
                "rear or side view of a compact vacuum-packed yeast bag with white upper section red ingredient panel and blue lower edge",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a small red Saf-instant yeast pack with folded white sealed flap and blue lower strip",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="YE_0001_MenVang",
        display_name="MenVang_SafInstantGoldDryYeast",
        candidate_conf=0.10,
        keep_conf=0.26,
        conflict_group="saf_instant_yeast_packs",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.38, 1.25),
        prompts=(
            PromptSpec(
                "single white gold and blue Saf-instant dry yeast vacuum pack with chef illustration",
                "primary",
                True,
            ),
            PromptSpec(
                "rectangular 500 gram Saf-instant Gold yeast package with metallic golden center panel and blue bottom strip",
                "secondary",
                True,
            ),
            PromptSpec(
                "rear or side view of a compact gold Saf-instant yeast pack with Vietnamese label QR code and blue lower edge",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a small gold Saf-instant yeast pack with folded white sealed flap and blue lower strip",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SN_0018_HatDieu",
        display_name="HatDieu_NgocChauCashewNuts",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="clear_nut_pouches",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.35, 1.30),
        prompts=(
            PromptSpec(
                "single clear resealable pouch filled with pale whole cashew nuts and green Ngoc Chau printing",
                "primary",
                True,
            ),
            PromptSpec(
                "upright transparent stand-up bag of curved cream-colored cashew nuts with silver side edges",
                "secondary",
                True,
            ),
            PromptSpec(
                "side view of a narrow clear and silver cashew nut pouch with visible pale curved nuts inside",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a transparent zipper bag containing many whole cashew nuts",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SN_0004_OcCho",
        display_name="OcCho_KingdeliWalnutKernels",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="clear_nut_pouches",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.35, 1.45),
        prompts=(
            PromptSpec(
                "single clear plastic bag of brown walnut kernels with white Kingdeli Hat Oc Cho label",
                "primary",
                True,
            ),
            PromptSpec(
                "large transparent food bag filled with wrinkled brown walnut halves and pieces",
                "secondary",
                True,
            ),
            PromptSpec(
                "side or rear view of a clear flexible bag containing many ridged brown walnut kernels",
                "view",
                False,
            ),
            PromptSpec(
                "top view of loose walnut halves and pieces enclosed together inside one transparent plastic bag",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SN_0003_HuongDuong",
        display_name="HuongDuong_ShelledSunflowerSeeds",
        candidate_conf=0.10,
        keep_conf=0.24,
        conflict_group="clear_seed_pouches",
        same_sku_iou=0.70,
        nested_containment=0.90,
        expected_aspect_ratio=(0.40, 1.45),
        prompts=(
            PromptSpec(
                "single clear plastic bag filled with pale shelled sunflower seeds and a white label reading Hat Huong Duong",
                "primary",
                True,
            ),
            PromptSpec(
                "transparent food pouch containing many small elongated cream-colored sunflower seed kernels",
                "secondary",
                True,
            ),
            PromptSpec(
                "rear or side view of a clear flexible bag densely packed with pale gray shelled sunflower seeds",
                "view",
                False,
            ),
            PromptSpec(
                "top view of many small pointed sunflower seed kernels enclosed together inside one transparent plastic bag",
                "recall",
                False,
            ),
        ),
    ),
)


GROUP_RULES: dict[str, GroupRule] = {
    "rectangular_beverage_cartons": GroupRule(0.84, 0.93, 0.07),
    "large_cooking_oil_jugs": GroupRule(0.84, 0.93, 0.07),
    "small_fruit_jam_jars": GroupRule(0.85, 0.94, 0.07),
    "fruit_smoothie_bottles": GroupRule(0.85, 0.94, 0.07),
    "mesh_bagged_small_produce": GroupRule(0.84, 0.93, 0.07),
    "large_white_baking_mix_sacks": GroupRule(0.84, 0.93, 0.07),
    "rectangular_chocolate_ingredient_boxes": GroupRule(0.85, 0.94, 0.07),
    "grand_place_chocolate_pouches": GroupRule(0.85, 0.94, 0.08),
    "rectangular_whipping_topping_cartons": GroupRule(0.85, 0.94, 0.07),
    "flexible_whipping_cream_pouches": GroupRule(0.85, 0.94, 0.08),
    "rectangular_dairy_product_boxes": GroupRule(0.85, 0.94, 0.07),
    "saf_instant_yeast_packs": GroupRule(0.86, 0.94, 0.08),
    "clear_nut_pouches": GroupRule(0.85, 0.94, 0.07),
    "clear_seed_pouches": GroupRule(0.85, 0.94, 0.08),
}


CLASS_NAMES: tuple[str, ...] = tuple(config.class_name for config in SKU_CONFIGS)
CLASS_NAME_TO_ID: dict[str, int] = {
    class_name: class_id for class_id, class_name in enumerate(CLASS_NAMES)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-annotate 19 newly analyzed Sharon Bakery SKUs using YOLO-World only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Input image file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root. Default: <image-directory>_preannotations.",
    )
    parser.add_argument(
        "--model",
        default="yolov8x-worldv2.pt",
        help="YOLO-World checkpoint name or local .pt path.",
    )
    parser.add_argument(
        "--profile",
        choices=("strict", "balanced", "recall"),
        default="balanced",
        help=(
            "strict=1 prompt/SKU; balanced=2 strong prompts/SKU; "
            "recall=all prompts, with view/recall-only detections sent to review."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--device",
        default=None,
        help="Examples: cpu, 0, cuda:0. Omit for Ultralytics auto selection.",
    )
    parser.add_argument(
        "--model-conf",
        type=float,
        default=None,
        help=(
            "Candidate confidence passed into YOLO-World. By default the script "
            "uses the minimum candidate threshold across all 19 SKUs."
        ),
    )
    parser.add_argument(
        "--model-iou",
        type=float,
        default=0.65,
        help="Ultralytics NMS IoU before custom same-SKU deduplication.",
    )
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--min-box-side", type=int, default=10)
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-area-ratio", type=float, default=0.98)
    parser.add_argument(
        "--nested-score-tolerance",
        type=float,
        default=0.08,
        help=(
            "Prefer a larger whole-object box over a nested part box when its "
            "confidence is no more than this amount lower."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search image subdirectories recursively.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files. Without this flag, existing labels are protected.",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Do not write annotated preview images.",
    )
    parser.add_argument(
        "--save-review-crops",
        action="store_true",
        help="Save crops for review and ambiguous detections.",
    )
    parser.add_argument(
        "--include-review-labels",
        action="store_true",
        help=(
            "Include review-status boxes in YOLO label files so they appear "
            "when imported into Roboflow. The YOLO format does not preserve "
            "accepted/review status; use review_queue.jsonl to identify them."
        ),
    )
    parser.add_argument(
        "--no-empty-labels",
        action="store_true",
        help="Do not create empty .txt files for images with no exported box.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the 19-SKU configuration and exit without loading the model.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("preannotation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def validate_configuration() -> None:
    errors: list[str] = []

    if len(SKU_CONFIGS) != EXPECTED_SKU_COUNT:
        errors.append(
            f"Expected {EXPECTED_SKU_COUNT} SKU configs, found {len(SKU_CONFIGS)}."
        )

    if len(set(CLASS_NAMES)) != len(CLASS_NAMES):
        errors.append("Duplicate canonical class names exist.")

    prompt_texts: list[str] = []
    for class_id, config in enumerate(SKU_CONFIGS):
        if not config.class_name:
            errors.append(f"Class {class_id} has an empty class_name.")
        if not 0.0 < config.candidate_conf <= config.keep_conf <= 1.0:
            errors.append(
                f"Invalid thresholds for {config.class_name}: "
                f"candidate={config.candidate_conf}, keep={config.keep_conf}."
            )
        if config.conflict_group not in GROUP_RULES:
            errors.append(
                f"Missing GroupRule for {config.class_name}: {config.conflict_group}."
            )
        if len(config.prompts) < 2:
            errors.append(f"{config.class_name} must have at least 2 prompts.")
        for prompt in config.prompts:
            normalized = " ".join(prompt.text.strip().lower().split())
            if not normalized:
                errors.append(f"{config.class_name} contains an empty prompt.")
            prompt_texts.append(normalized)

    duplicates = sorted(
        text for text in set(prompt_texts) if prompt_texts.count(text) > 1
    )
    if duplicates:
        errors.append(f"Duplicate prompt texts found: {duplicates}")

    if errors:
        raise ValueError("Configuration validation failed:\n- " + "\n- ".join(errors))


def build_prompt_records(profile: str) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for sku_id, config in enumerate(SKU_CONFIGS):
        if profile == "strict":
            selected = config.prompts[:1]
        elif profile == "balanced":
            selected = config.prompts[:2]
        else:
            selected = config.prompts

        for prompt in selected:
            records.append(
                PromptRecord(
                    prompt_id=len(records),
                    sku_id=sku_id,
                    class_name=config.class_name,
                    text=prompt.text,
                    role=prompt.role,
                    auto_accept=prompt.auto_accept,
                )
            )
    return records


def collect_images(source: Path, recursive: bool) -> tuple[list[Path], Path]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source], source.parent

    iterator: Iterable[Path]
    iterator = source.rglob("*") if recursive else source.glob("*")
    images = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images, source


def default_output_path(source: Path) -> Path:
    source = source.expanduser().resolve()
    if source.is_file():
        return source.parent / f"{source.stem}_preannotations"
    return source.parent / f"{source.name}_preannotations"


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box.tolist()
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def calculate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter = intersection_area(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0.0 else 0.0


def calculate_max_containment(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter = intersection_area(box_a, box_b)
    minimum_area = min(box_area(box_a), box_area(box_b))
    return inter / minimum_area if minimum_area > 0.0 else 0.0


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = np.asarray(box, dtype=np.float32).copy()
    clipped[0] = np.clip(clipped[0], 0, max(0, width - 1))
    clipped[1] = np.clip(clipped[1], 0, max(0, height - 1))
    clipped[2] = np.clip(clipped[2], 0, max(0, width - 1))
    clipped[3] = np.clip(clipped[3], 0, max(0, height - 1))
    if clipped[2] < clipped[0]:
        clipped[0], clipped[2] = clipped[2], clipped[0]
    if clipped[3] < clipped[1]:
        clipped[1], clipped[3] = clipped[3], clipped[1]
    return clipped


def is_valid_geometry(
    box: np.ndarray,
    image_width: int,
    image_height: int,
    min_box_side: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[bool, str | None]:
    if not np.isfinite(box).all():
        return False, "NON_FINITE_BOX"

    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    if width < min_box_side or height < min_box_side:
        return False, "BOX_TOO_SMALL"

    image_area = float(image_width * image_height)
    area_ratio = box_area(box) / image_area if image_area > 0 else 0.0
    if area_ratio < min_area_ratio:
        return False, "AREA_RATIO_TOO_SMALL"
    if area_ratio > max_area_ratio:
        return False, "AREA_RATIO_TOO_LARGE"

    aspect_ratio = width / height if height > 0 else math.inf
    if aspect_ratio < 0.03 or aspect_ratio > 30.0:
        return False, "EXTREME_ASPECT_RATIO"
    return True, None


def merge_detection_metadata(target: Detection, source: Detection) -> None:
    target.prompt_texts.update(source.prompt_texts)
    target.prompt_roles.update(source.prompt_roles)
    target.auto_supported = target.auto_supported or source.auto_supported
    target.score = max(target.score, source.score)


def choose_duplicate_winner(
    first: Detection,
    second: Detection,
    nested_score_tolerance: float,
) -> tuple[Detection, Detection, str]:
    area_first = first.area
    area_second = second.area
    containment = calculate_max_containment(first.xyxy, second.xyxy)

    if containment >= 0.90:
        if area_first >= area_second:
            larger, smaller = first, second
        else:
            larger, smaller = second, first

        if larger.score + nested_score_tolerance >= smaller.score:
            return larger, smaller, "NESTED_PART_SUPPRESSED"

    if first.score >= second.score:
        return first, second, "DUPLICATE_PROMPT_BOX_SUPPRESSED"
    return second, first, "DUPLICATE_PROMPT_BOX_SUPPRESSED"


def deduplicate_same_sku(
    detections: Sequence[Detection],
    config: SKUConfig,
    nested_score_tolerance: float,
) -> tuple[list[Detection], list[Detection]]:
    kept: list[Detection] = []
    suppressed: list[Detection] = []

    for candidate in sorted(detections, key=lambda item: item.score, reverse=True):
        merged = False
        for index, existing in enumerate(kept):
            iou = calculate_iou(candidate.xyxy, existing.xyxy)
            containment = calculate_max_containment(candidate.xyxy, existing.xyxy)
            if iou < config.same_sku_iou and containment < config.nested_containment:
                continue

            winner, loser, reason = choose_duplicate_winner(
                existing,
                candidate,
                nested_score_tolerance,
            )
            merge_detection_metadata(winner, loser)
            loser.status = "rejected"
            loser.reasons.append(reason)
            suppressed.append(loser)
            kept[index] = winner
            merged = True
            break

        if not merged:
            kept.append(candidate)

    return kept, suppressed


def assign_initial_status(detection: Detection, config: SKUConfig) -> None:
    if detection.score < config.keep_conf:
        detection.status = "review"
        detection.reasons.append("BELOW_AUTO_KEEP_THRESHOLD")
    elif not detection.auto_supported:
        detection.status = "review"
        detection.reasons.append("REVIEW_ONLY_PROMPT")
    else:
        detection.status = "accepted"

    if config.expected_aspect_ratio is not None:
        width = float(detection.xyxy[2] - detection.xyxy[0])
        height = float(detection.xyxy[3] - detection.xyxy[1])
        ratio = width / height if height > 0 else math.inf
        minimum, maximum = config.expected_aspect_ratio
        if ratio < minimum or ratio > maximum:
            detection.reasons.append("ASPECT_RATIO_WARNING")


def resolve_cross_class_conflicts(detections: list[Detection]) -> None:
    for index_a in range(len(detections)):
        first = detections[index_a]
        if first.status == "rejected":
            continue

        first_config = SKU_CONFIGS[first.sku_id]
        rule = GROUP_RULES[first_config.conflict_group]

        for index_b in range(index_a + 1, len(detections)):
            second = detections[index_b]
            if second.status == "rejected" or first.sku_id == second.sku_id:
                continue

            second_config = SKU_CONFIGS[second.sku_id]
            if first_config.conflict_group != second_config.conflict_group:
                continue

            iou = calculate_iou(first.xyxy, second.xyxy)
            containment = calculate_max_containment(first.xyxy, second.xyxy)
            if iou < rule.iou_threshold and containment < rule.containment_threshold:
                continue

            score_difference = abs(first.score - second.score)
            if score_difference >= rule.score_margin:
                winner, loser = (
                    (first, second) if first.score > second.score else (second, first)
                )
                loser.status = "rejected"
                loser.reasons.append(
                    f"CROSS_CLASS_SUPPRESSED_BY:{winner.class_name}"
                )
                if loser is first:
                    break
                continue

            first.status = "review"
            second.status = "review"
            first.reasons.append(f"AMBIGUOUS_WITH:{second.class_name}")
            second.reasons.append(f"AMBIGUOUS_WITH:{first.class_name}")


def xyxy_to_yolo(box: np.ndarray, image_width: int, image_height: int) -> tuple[float, ...]:
    x1, y1, x2, y2 = [float(value) for value in box]
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    values = (x_center, y_center, width, height)
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in values)


def write_yolo_labels(
    path: Path,
    detections: Sequence[Detection],
    image_width: int,
    image_height: int,
    create_empty: bool,
    include_review: bool = False,
) -> None:
    allowed_statuses = {"accepted"}
    if include_review:
        allowed_statuses.add("review")

    exported = sorted(
        (item for item in detections if item.status in allowed_statuses),
        key=lambda item: (item.sku_id, float(item.xyxy[1]), float(item.xyxy[0])),
    )

    if not exported and not create_empty:
        if path.exists():
            path.unlink()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_handle:
        for detection in exported:
            x_center, y_center, width, height = xyxy_to_yolo(
                detection.xyxy,
                image_width,
                image_height,
            )
            file_handle.write(
                f"{detection.sku_id} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}\n"
            )


def draw_preview(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    preview = image.copy()
    colors = {
        "accepted": (40, 180, 40),
        "review": (0, 165, 255),
        "rejected": (50, 50, 220),
    }
    line_width = max(2, round(min(image.shape[:2]) / 450))
    font_scale = max(0.45, min(image.shape[:2]) / 1600)
    text_thickness = max(1, line_width // 2)

    for detection in detections:
        if detection.status == "rejected":
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in detection.xyxy]
        color = colors[detection.status]
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, line_width)

        label = (
            f"{detection.sku_id}:{detection.class_name} "
            f"{detection.score:.2f} [{detection.status}]"
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        text_y = max(text_height + baseline + 4, y1)
        cv2.rectangle(
            preview,
            (x1, text_y - text_height - baseline - 6),
            (min(preview.shape[1] - 1, x1 + text_width + 6), text_y + 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            preview,
            label,
            (x1 + 3, text_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return preview


def save_review_crop(
    image: np.ndarray,
    detection: Detection,
    crop_path: Path,
    padding_ratio: float = 0.05,
) -> None:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in detection.xyxy]
    padding_x = (x2 - x1) * padding_ratio
    padding_y = (y2 - y1) * padding_ratio
    x1 = max(0, int(math.floor(x1 - padding_x)))
    y1 = max(0, int(math.floor(y1 - padding_y)))
    x2 = min(image_width, int(math.ceil(x2 + padding_x)))
    y2 = min(image_height, int(math.ceil(y2 + padding_y)))

    if x2 <= x1 or y2 <= y1:
        return
    crop = image[y1:y2, x1:x2]
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_path), crop)


def relative_output_path(image_path: Path, image_root: Path) -> Path:
    try:
        relative = image_path.relative_to(image_root)
    except ValueError:
        relative = Path(image_path.name)
    return relative


def write_class_files(output_root: Path) -> None:
    classes_path = output_root / "classes.txt"
    classes_path.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    mapping = {
        str(class_id): {
            "class_name": config.class_name,
            "display_name": config.display_name,
            "candidate_conf": config.candidate_conf,
            "keep_conf": config.keep_conf,
            "conflict_group": config.conflict_group,
        }
        for class_id, config in enumerate(SKU_CONFIGS)
    }
    (output_root / "class_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    yaml_lines = [
        "# Replace path/train/val before training.",
        "path: CHANGE_ME",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend(
        f"  {class_id}: {class_name}"
        for class_id, class_name in enumerate(CLASS_NAMES)
    )
    (output_root / "data_template.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )


def serialize_detection(
    image_name: str,
    detection: Detection,
) -> dict[str, object]:
    x1, y1, x2, y2 = [round(float(value), 2) for value in detection.xyxy]
    return {
        "image": image_name,
        "class_id": detection.sku_id,
        "class_name": detection.class_name,
        "status": detection.status,
        "confidence": round(float(detection.score), 6),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "reasons": sorted(set(detection.reasons)),
        "prompt_roles": sorted(detection.prompt_roles),
        "prompts": sorted(detection.prompt_texts),
    }


def process_result(
    result: object,
    prompt_records: Sequence[PromptRecord],
    image_width: int,
    image_height: int,
    min_box_side: int,
    min_area_ratio: float,
    max_area_ratio: float,
    nested_score_tolerance: float,
) -> tuple[list[Detection], list[Detection]]:
    raw_candidates: list[Detection] = []
    geometry_rejected: list[Detection] = []

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], []

    xyxy_array = boxes.xyxy.detach().cpu().numpy()
    confidence_array = boxes.conf.detach().cpu().numpy()
    class_array = boxes.cls.detach().cpu().numpy().astype(int)

    for raw_box, score, prompt_id in zip(
        xyxy_array,
        confidence_array,
        class_array,
        strict=True,
    ):
        if prompt_id < 0 or prompt_id >= len(prompt_records):
            continue

        prompt_record = prompt_records[prompt_id]
        sku_config = SKU_CONFIGS[prompt_record.sku_id]
        score_value = float(score)
        if score_value < sku_config.candidate_conf:
            continue

        clipped_box = clip_box(raw_box, image_width, image_height)
        geometry_ok, geometry_reason = is_valid_geometry(
            clipped_box,
            image_width,
            image_height,
            min_box_side,
            min_area_ratio,
            max_area_ratio,
        )

        detection = Detection(
            sku_id=prompt_record.sku_id,
            class_name=prompt_record.class_name,
            score=score_value,
            xyxy=clipped_box,
            prompt_texts={prompt_record.text},
            prompt_roles={prompt_record.role},
            auto_supported=prompt_record.auto_accept,
        )
        if not geometry_ok:
            detection.status = "rejected"
            detection.reasons.append(geometry_reason or "INVALID_GEOMETRY")
            geometry_rejected.append(detection)
            continue
        raw_candidates.append(detection)

    final_detections: list[Detection] = []
    duplicate_rejected: list[Detection] = []
    for sku_id, sku_config in enumerate(SKU_CONFIGS):
        same_sku = [item for item in raw_candidates if item.sku_id == sku_id]
        deduplicated, suppressed = deduplicate_same_sku(
            same_sku,
            sku_config,
            nested_score_tolerance,
        )
        for detection in deduplicated:
            assign_initial_status(detection, sku_config)
        final_detections.extend(deduplicated)
        duplicate_rejected.extend(suppressed)

    resolve_cross_class_conflicts(final_detections)
    final_detections.sort(
        key=lambda item: (
            {"accepted": 0, "review": 1, "rejected": 2}.get(item.status, 3),
            -item.score,
            item.sku_id,
        )
    )
    return final_detections, geometry_rejected + duplicate_rejected


def run() -> int:
    args = parse_args()
    validate_configuration()

    if args.validate_only:
        print(f"Configuration OK: {len(SKU_CONFIGS)} classes.")
        for class_id, config in enumerate(SKU_CONFIGS):
            print(
                f"{class_id:02d} | {config.class_name} | "
                f"candidate={config.candidate_conf:.2f} | keep={config.keep_conf:.2f}"
            )
        return 0

    images, image_root = collect_images(args.images, args.recursive)
    if not images:
        raise RuntimeError(f"No supported images found in: {args.images}")

    output_root = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(args.images)
    )
    labels_root = output_root / "labels"
    previews_root = output_root / "previews"
    review_crops_root = output_root / "review_crops"
    output_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_root / "preannotation.log")
    write_class_files(output_root)

    prompt_records = build_prompt_records(args.profile)
    prompt_texts = [record.text for record in prompt_records]
    minimum_candidate = min(config.candidate_conf for config in SKU_CONFIGS)
    model_conf = (
        float(args.model_conf)
        if args.model_conf is not None
        else minimum_candidate
    )
    if not 0.0 < model_conf <= 1.0:
        raise ValueError("--model-conf must be in the interval (0, 1].")
    if model_conf > minimum_candidate:
        logger.warning(
            "model-conf %.3f is above the minimum SKU candidate threshold %.3f; "
            "some configured candidates can never reach post-processing.",
            model_conf,
            minimum_candidate,
        )

    run_config = {
        "script_version": SCRIPT_VERSION,
        "ultralytics_version": ULTRALYTICS_VERSION,
        "model": args.model,
        "images": str(args.images.expanduser().resolve()),
        "output": str(output_root),
        "profile": args.profile,
        "prompt_count": len(prompt_records),
        "class_count": len(SKU_CONFIGS),
        "imgsz": args.imgsz,
        "device": args.device,
        "model_conf": model_conf,
        "model_iou": args.model_iou,
        "max_det": args.max_det,
        "min_box_side": args.min_box_side,
        "min_area_ratio": args.min_area_ratio,
        "max_area_ratio": args.max_area_ratio,
        "nested_score_tolerance": args.nested_score_tolerance,
        "overwrite": args.overwrite,
        "include_review_labels": args.include_review_labels,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Script version: %s", SCRIPT_VERSION)
    logger.info("Ultralytics version: %s", ULTRALYTICS_VERSION)
    logger.info("Images: %d", len(images))
    logger.info("Prompt profile: %s (%d prompts)", args.profile, len(prompt_records))
    logger.info("Loading YOLO-World model: %s", args.model)

    try:
        from ultralytics import YOLOWorld
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'ultralytics'. Install it with: "
            "python -m pip install -U ultralytics opencv-python tqdm"
        ) from exc

    model = YOLOWorld(args.model)
    model.set_classes(prompt_texts)

    detail_csv_path = output_root / "detections.csv"
    summary_csv_path = output_root / "image_summary.csv"
    review_jsonl_path = output_root / "review_queue.jsonl"

    detail_fieldnames = [
        "image",
        "class_id",
        "class_name",
        "status",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "reasons",
        "prompt_roles",
        "prompts",
    ]
    summary_fieldnames = [
        "image",
        "raw_model_boxes",
        "accepted",
        "review",
        "rejected",
        "label_path",
        "skipped_existing_label",
    ]

    total_accepted = 0
    total_review = 0
    total_rejected = 0
    processed_images = 0
    failed_images = 0
    skipped_images = 0

    with (
        detail_csv_path.open("w", encoding="utf-8-sig", newline="") as detail_file,
        summary_csv_path.open("w", encoding="utf-8-sig", newline="") as summary_file,
        review_jsonl_path.open("w", encoding="utf-8") as review_file,
    ):
        detail_writer = csv.DictWriter(detail_file, fieldnames=detail_fieldnames)
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fieldnames)
        detail_writer.writeheader()
        summary_writer.writeheader()

        for image_path in tqdm(images, desc="YOLO-World pre-annotation"):
            relative_image = relative_output_path(image_path, image_root)
            label_relative = relative_image.with_suffix(".txt")
            preview_relative = relative_image.with_suffix(".jpg")
            label_path = labels_root / label_relative
            preview_path = previews_root / preview_relative

            if label_path.exists() and not args.overwrite:
                skipped_images += 1
                logger.info("Skip protected label: %s", label_path)
                summary_writer.writerow(
                    {
                        "image": str(relative_image),
                        "raw_model_boxes": 0,
                        "accepted": 0,
                        "review": 0,
                        "rejected": 0,
                        "label_path": str(label_path),
                        "skipped_existing_label": True,
                    }
                )
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                failed_images += 1
                logger.error("OpenCV cannot read image: %s", image_path)
                continue

            image_height, image_width = image.shape[:2]
            predict_kwargs: dict[str, object] = {
                "source": str(image_path),
                "conf": model_conf,
                "iou": args.model_iou,
                "imgsz": args.imgsz,
                "max_det": args.max_det,
                "agnostic_nms": False,
                "rect": False,
                "augment": False,
                "verbose": False,
                "save": False,
            }
            if args.device:
                predict_kwargs["device"] = args.device

            try:
                results = model.predict(**predict_kwargs)
                if not results:
                    raise RuntimeError("Ultralytics returned an empty result list.")
                result = results[0]
                raw_model_boxes = len(result.boxes) if result.boxes is not None else 0
                detections, rejected_detections = process_result(
                    result,
                    prompt_records,
                    image_width,
                    image_height,
                    args.min_box_side,
                    args.min_area_ratio,
                    args.max_area_ratio,
                    args.nested_score_tolerance,
                )
            except Exception as exc:
                failed_images += 1
                logger.exception("Inference failed for %s: %s", image_path, exc)
                continue

            all_detections = detections + rejected_detections
            accepted_count = sum(item.status == "accepted" for item in detections)
            review_count = sum(item.status == "review" for item in detections)
            rejected_count = sum(item.status == "rejected" for item in all_detections)

            write_yolo_labels(
                label_path,
                detections,
                image_width,
                image_height,
                create_empty=not args.no_empty_labels,
                include_review=args.include_review_labels,
            )

            if not args.no_previews:
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                preview = draw_preview(image, detections)
                cv2.imwrite(str(preview_path), preview)

            crop_index = 0
            for detection in all_detections:
                serialized = serialize_detection(str(relative_image), detection)
                csv_row = serialized.copy()
                csv_row["reasons"] = " | ".join(serialized["reasons"])
                csv_row["prompt_roles"] = " | ".join(serialized["prompt_roles"])
                csv_row["prompts"] = " | ".join(serialized["prompts"])
                detail_writer.writerow(csv_row)

                if detection.status == "review":
                    review_file.write(
                        json.dumps(serialized, ensure_ascii=False) + "\n"
                    )
                    if args.save_review_crops:
                        crop_index += 1
                        crop_path = (
                            review_crops_root
                            / relative_image.parent
                            / relative_image.stem
                            / (
                                f"{crop_index:03d}_"
                                f"{detection.class_name}_"
                                f"{detection.score:.3f}.jpg"
                            )
                        )
                        save_review_crop(image, detection, crop_path)

            summary_writer.writerow(
                {
                    "image": str(relative_image),
                    "raw_model_boxes": raw_model_boxes,
                    "accepted": accepted_count,
                    "review": review_count,
                    "rejected": rejected_count,
                    "label_path": str(label_path),
                    "skipped_existing_label": False,
                }
            )

            processed_images += 1
            total_accepted += accepted_count
            total_review += review_count
            total_rejected += rejected_count
            logger.info(
                "%s | raw=%d accepted=%d review=%d rejected=%d",
                relative_image,
                raw_model_boxes,
                accepted_count,
                review_count,
                rejected_count,
            )

    final_summary = {
        "images_found": len(images),
        "images_processed": processed_images,
        "images_skipped_existing_labels": skipped_images,
        "images_failed": failed_images,
        "accepted_boxes": total_accepted,
        "review_boxes": total_review,
        "rejected_boxes": total_rejected,
        "labels_directory": str(labels_root),
        "previews_directory": None if args.no_previews else str(previews_root),
        "review_queue": str(review_jsonl_path),
        "include_review_labels": args.include_review_labels,
    }
    (output_root / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Completed: %s", json.dumps(final_summary, ensure_ascii=False))
    return 0 if failed_images == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise SystemExit(1)
