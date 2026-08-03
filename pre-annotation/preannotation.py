from __future__ import annotations

"""
YOLO-World pre-annotation pipeline for 30 Sharon Bakery SKUs.

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


SCRIPT_VERSION = "3.0.0"
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
# 30-SKU vocabulary
# Class order below is the final numeric YOLO class-ID order.
# -----------------------------------------------------------------------------
SKU_CONFIGS: tuple[SKUConfig, ...] = (
    SKUConfig(
        class_name="SY_0009_TinhChatChanh",
        display_name="TinhChatChanh_LemonFlavoringEssence",
        candidate_conf=0.08,
        keep_conf=0.14,
        conflict_group="small_flavoring_bottles",
        same_sku_iou=0.65,
        expected_aspect_ratio=(0.28, 0.85),
        prompts=(
            PromptSpec(
                "small clear glass lemon flavoring bottle with silver screw cap and yellow black lemon label",
                "primary",
                True,
            ),
            PromptSpec(
                "miniature Rayner lemon essence bottle with lemon fruit graphic and pale clear liquid",
                "secondary",
                True,
            ),
            PromptSpec(
                "small cylindrical clear flavoring bottle with ribbed silver cap and yellow black lemon wraparound label",
                "view",
                False,
            ),
            PromptSpec(
                "small food flavoring bottle containing clear liquid with a yellow lemon themed label",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="FL_0012_BotDo",
        display_name="BotDo_ChiaKhoaDoWheatFlour",
        candidate_conf=0.10,
        keep_conf=0.22,
        conflict_group="large_woven_ingredient_sacks",
        expected_aspect_ratio=(0.45, 1.20),
        prompts=(
            PromptSpec(
                "large white woven wheat flour sack with red pink Chia Khoa Do printing",
                "primary",
                True,
            ),
            PromptSpec(
                "industrial white polypropylene flour bag with bold red key logo and Vietnamese text",
                "secondary",
                True,
            ),
            PromptSpec(
                "large stacked white flour sacks with horizontal red bands and red printed branding",
                "view",
                False,
            ),
            PromptSpec(
                "bulky white woven ingredient sack with prominent red printing",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="FL_0003_BotNguyenCam",
        display_name="BotNguyenCam_WholeMealFlour",
        candidate_conf=0.10,
        keep_conf=0.22,
        conflict_group="large_woven_ingredient_sacks",
        expected_aspect_ratio=(0.45, 1.25),
        prompts=(
            PromptSpec(
                "large beige whole meal wheat flour sack with dark brown WHOLE MEAL band",
                "primary",
                True,
            ),
            PromptSpec(
                "industrial tan paper flour bag labeled whole meal whole wheat flour with brown globe graphics",
                "secondary",
                True,
            ),
            PromptSpec(
                "large horizontal beige whole wheat flour sacks stacked in a warehouse",
                "view",
                False,
            ),
            PromptSpec(
                "bulky tan industrial flour sack with wide dark brown printed stripe",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SY_0008_TinhChatBo",
        display_name="TinhChatBo_ButterFlavoringEssence",
        candidate_conf=0.08,
        keep_conf=0.14,
        conflict_group="small_flavoring_bottles",
        same_sku_iou=0.65,
        expected_aspect_ratio=(0.28, 0.85),
        prompts=(
            PromptSpec(
                "small clear glass butter flavoring bottle with silver screw cap and bright yellow label",
                "primary",
                True,
            ),
            PromptSpec(
                "miniature glass butter essence bottle with black oval panel on a yellow label",
                "secondary",
                True,
            ),
            PromptSpec(
                "small cylindrical clear flavoring bottle with ribbed silver cap and yellow wraparound label",
                "view",
                False,
            ),
            PromptSpec(
                "small food flavoring bottle containing clear liquid with a bright yellow label",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SY_0007_TinhChatCam",
        display_name="TinhChatCam_OrangeFlavoringEssence",
        candidate_conf=0.08,
        keep_conf=0.14,
        conflict_group="small_flavoring_bottles",
        same_sku_iou=0.65,
        expected_aspect_ratio=(0.28, 0.85),
        prompts=(
            PromptSpec(
                "small clear glass orange flavoring bottle with silver screw cap and orange label",
                "primary",
                True,
            ),
            PromptSpec(
                "miniature glass orange essence bottle with black oval panel and orange fruit graphic",
                "secondary",
                True,
            ),
            PromptSpec(
                "small cylindrical clear flavoring bottle with ribbed silver cap and orange wraparound label",
                "view",
                False,
            ),
            PromptSpec(
                "small food flavoring bottle containing pale yellow liquid with an orange label",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SY_0006_TinhChatHanhNhan",
        display_name="TinhChatHanhNhan_AlmondFlavoringEssence",
        candidate_conf=0.08,
        keep_conf=0.14,
        conflict_group="small_flavoring_bottles",
        same_sku_iou=0.65,
        expected_aspect_ratio=(0.28, 0.85),
        prompts=(
            PromptSpec(
                "small clear glass almond flavoring bottle with silver screw cap and almond label",
                "primary",
                True,
            ),
            PromptSpec(
                "miniature glass almond essence bottle with black top label and almond nut graphic",
                "secondary",
                True,
            ),
            PromptSpec(
                "small cylindrical clear flavoring bottle with ribbed silver cap and brown beige wraparound label",
                "view",
                False,
            ),
            PromptSpec(
                "small food flavoring bottle containing pale clear liquid with an almond themed label",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="JM_0008_PureAnhDao",
        display_name="PureAnhDao_MorelloCherryPuree",
        candidate_conf=0.10,
        keep_conf=0.18,
        conflict_group="rectangular_fruit_puree_tubs",
        expected_aspect_ratio=(1.20, 4.00),
        prompts=(
            PromptSpec(
                "single white rectangular plastic tub of Morello cherry puree with black and dark red lid label",
                "primary",
                True,
            ),
            PromptSpec(
                "large white food container with snap-on lid and red cherry fruit graphics",
                "secondary",
                True,
            ),
            PromptSpec(
                "top view of a long white rectangular puree tub with centered black and dark red cherry label",
                "view",
                False,
            ),
            PromptSpec(
                "deep white rectangular plastic food tub with rounded corners and cherry label on the lid",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="JM_0005_XoaiPureAndros",
        display_name="XoaiPureAndros_MangoPuree",
        candidate_conf=0.10,
        keep_conf=0.18,
        conflict_group="rectangular_fruit_puree_tubs",
        expected_aspect_ratio=(1.20, 4.00),
        prompts=(
            PromptSpec(
                "single white rectangular plastic tub of mango puree with yellow orange lid label",
                "primary",
                True,
            ),
            PromptSpec(
                "large white food container with snap-on lid and yellow mango fruit graphics",
                "secondary",
                True,
            ),
            PromptSpec(
                "top view of a long white rectangular puree tub with centered yellow orange mango label",
                "view",
                False,
            ),
            PromptSpec(
                "deep white rectangular plastic puree tub with rounded corners and mango label",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="JM_0004_PureVietQuat",
        display_name="PureVietQuat_WildBlueberryPuree",
        candidate_conf=0.10,
        keep_conf=0.18,
        conflict_group="rectangular_fruit_puree_tubs",
        expected_aspect_ratio=(1.20, 4.00),
        prompts=(
            PromptSpec(
                "single white rectangular plastic tub of wild blueberry puree with dark blue lid label",
                "primary",
                True,
            ),
            PromptSpec(
                "large white food container with snap-on lid and blue purple blueberry graphics",
                "secondary",
                True,
            ),
            PromptSpec(
                "top view of a long white rectangular puree tub with centered dark blue blueberry label",
                "view",
                False,
            ),
            PromptSpec(
                "deep white rectangular plastic puree container with blue berry label on the lid",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="JM_0003_PureDau",
        display_name="PureDau_StrawberryPuree",
        candidate_conf=0.10,
        keep_conf=0.18,
        conflict_group="rectangular_fruit_puree_tubs",
        expected_aspect_ratio=(1.20, 4.00),
        prompts=(
            PromptSpec(
                "single white rectangular plastic tub of strawberry puree with bright red pink lid label",
                "primary",
                True,
            ),
            PromptSpec(
                "large white puree container with snap-on lid and multiple red strawberry graphics",
                "secondary",
                True,
            ),
            PromptSpec(
                "top view of a long white rectangular puree tub with centered red strawberry label",
                "view",
                False,
            ),
            PromptSpec(
                "deep white rectangular plastic food tub with rounded corners and red strawberry lid label",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="GV_0003_LaKinhGioi",
        display_name="LaKinhGioi_DriedVietnameseBalmLeaves",
        candidate_conf=0.10,
        keep_conf=0.18,
        conflict_group="dried_herb_pouches",
        expected_aspect_ratio=(0.45, 1.60),
        prompts=(
            PromptSpec(
                "single large bright green plastic pouch of dried Vietnamese balm leaves with transparent lower window",
                "primary",
                True,
            ),
            PromptSpec(
                "bright green flexible dried herb bag with visible brown green leaves through a clear window",
                "secondary",
                True,
            ),
            PromptSpec(
                "large green resealable herb pouch with round white bird logo and transparent bottom",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a folded bright green dried herb pouch with a circular white logo",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="YE_0006_Gelatin",
        display_name="Gelatin_GelatinSheets",
        candidate_conf=0.07,
        keep_conf=0.13,
        conflict_group="transparent_ingredient_packets",
        same_sku_iou=0.65,
        expected_aspect_ratio=(1.10, 4.00),
        prompts=(
            PromptSpec(
                "single clear plastic packet of pale amber gelatin sheets with crosshatched texture",
                "primary",
                True,
            ),
            PromptSpec(
                "flat transparent vacuum-sealed rectangular package containing gelatin leaves",
                "secondary",
                True,
            ),
            PromptSpec(
                "clear rectangular gelatin sheet packet with a large white paper label and barcode",
                "view",
                False,
            ),
            PromptSpec(
                "transparent food ingredient pouch containing stacked translucent yellow gelatin sheets",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="BK_0010_BoVegan",
        display_name="BoVegan_VeganButterBlock",
        candidate_conf=0.08,
        keep_conf=0.16,
        conflict_group="bulk_fats_in_blue_liners",
        expected_aspect_ratio=(0.45, 1.80),
        prompts=(
            PromptSpec(
                "single large open blue plastic liner bag containing a pale yellow vegan butter block",
                "primary",
                True,
            ),
            PromptSpec(
                "bulk yellow plant based butter inside a crumpled blue plastic bag",
                "secondary",
                True,
            ),
            PromptSpec(
                "large upright blue food-grade plastic liner bag filled with solid yellow vegan margarine",
                "view",
                False,
            ),
            PromptSpec(
                "opened blue bulk ingredient bag containing a soft pale yellow fat block",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="YE_0005_RauCauDeo",
        display_name="RauCauDeo_3DFlexibleJellyPowder",
        candidate_conf=0.10,
        keep_conf=0.18,
        conflict_group="small_baking_ingredient_cartons",
        expected_aspect_ratio=(0.45, 1.55),
        prompts=(
            PromptSpec(
                "single upright white cardboard box of 3D flexible jelly powder with colorful tropical fruit graphics",
                "primary",
                True,
            ),
            PromptSpec(
                "small white and blue agar jelly powder carton with yellow fish logo and red text",
                "secondary",
                True,
            ),
            PromptSpec(
                "rectangular jelly powder box with pineapple coconut kiwi graphics and blue bottom panel",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a small white jelly powder carton with colorful fruit illustrations",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="YE_0004_BakingSoda",
        display_name="BakingSoda_CasterDaily",
        candidate_conf=0.10,
        keep_conf=0.20,
        conflict_group="small_baking_ingredient_cartons",
        expected_aspect_ratio=(0.45, 1.45),
        prompts=(
            PromptSpec(
                "single upright orange cardboard box of baking soda with a large yellow circle",
                "primary",
                True,
            ),
            PromptSpec(
                "small Caster baking soda carton with orange packaging and dark blue lettering",
                "secondary",
                True,
            ),
            PromptSpec(
                "rectangular orange baking ingredient box with yellow front panel and cookie graphics",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a small orange baking soda carton with a white nutrition label",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SY_0002_TraEarlGrey",
        display_name="TraEarlGrey_DilmahEarlGreyTea",
        candidate_conf=0.10,
        keep_conf=0.20,
        conflict_group="large_tea_cartons",
        expected_aspect_ratio=(0.45, 1.35),
        prompts=(
            PromptSpec(
                "single upright dark green Dilmah Earl Grey tea box covered with green leaf graphics",
                "primary",
                True,
            ),
            PromptSpec(
                "large rectangular tea carton with Dilmah logo beige Earl Grey label and number 100",
                "secondary",
                True,
            ),
            PromptSpec(
                "tall beige tea box with brewing instructions tea cup icons and green side panels",
                "view",
                False,
            ),
            PromptSpec(
                "large upright cardboard package containing one hundred individually wrapped Earl Grey tea bags",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SY_0001_SyrupVani",
        display_name="SyrupVani_VanillaFlavoringEssence",
        candidate_conf=0.10,
        keep_conf=0.20,
        conflict_group="large_dark_flavoring_bottles",
        same_sku_iou=0.68,
        expected_aspect_ratio=(0.30, 0.95),
        prompts=(
            PromptSpec(
                "single large dark vanilla flavoring bottle with black screw cap and cream label",
                "primary",
                True,
            ),
            PromptSpec(
                "500 ml dark brown vanilla essence bottle with white vanilla flower and brown vanilla pods",
                "secondary",
                True,
            ),
            PromptSpec(
                "large cylindrical black flavoring bottle with rounded shoulders and beige vanilla label",
                "view",
                False,
            ),
            PromptSpec(
                "dark food flavoring bottle with black ribbed cap and pale yellow wraparound label",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SU_0005_DuongNau",
        display_name="DuongNau_BienHoaBrownSugar",
        candidate_conf=0.10,
        keep_conf=0.19,
        conflict_group="flexible_sugar_pouches",
        expected_aspect_ratio=(0.40, 1.35),
        prompts=(
            PromptSpec(
                "single upright brown plastic pouch of coarse brown sugar with green Bien Hoa label",
                "primary",
                True,
            ),
            PromptSpec(
                "brown sugar bag with green curved band checkerboard top and transparent side window",
                "secondary",
                True,
            ),
            PromptSpec(
                "flexible dark brown food pouch showing coarse brown sugar crystals through a clear window",
                "view",
                False,
            ),
            PromptSpec(
                "crumpled upright brown sugar package with green logo and beverage image on the front",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SU_0004_DuongMachNha",
        display_name="DuongMachNha_MaltoseSyrup",
        candidate_conf=0.10,
        keep_conf=0.18,
        conflict_group="clear_food_syrup_jars",
        same_sku_iou=0.68,
        expected_aspect_ratio=(0.50, 1.20),
        prompts=(
            PromptSpec(
                "single clear cylindrical plastic jar of maltose syrup with a wide white screw lid",
                "primary",
                True,
            ),
            PromptSpec(
                "transparent one kilogram syrup jar with purple checkered and blue wraparound label",
                "secondary",
                True,
            ),
            PromptSpec(
                "clear plastic malt syrup container with brown circular label and colorful Mama Choice logo",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a large clear food syrup jar with a round white plastic lid",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SU_0003_DuongBot",
        display_name="DuongBot_BienHoaPowderedSugar",
        candidate_conf=0.10,
        keep_conf=0.19,
        conflict_group="flexible_sugar_pouches",
        expected_aspect_ratio=(0.45, 1.40),
        prompts=(
            PromptSpec(
                "single upright pink plastic pouch of powdered baking sugar with green Bien Hoa Pro logo",
                "primary",
                True,
            ),
            PromptSpec(
                "large flexible icing sugar bag with pink front green curved band and cupcake graphic",
                "secondary",
                True,
            ),
            PromptSpec(
                "powdered sugar pouch with pale turquoise back panel and pink checkered sealed top",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a folded pink baking sugar pouch with green Bien Hoa Pro logo",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SU_0002_DuongVang",
        display_name="DuongVang_CoBaYellowSugar",
        candidate_conf=0.10,
        keep_conf=0.20,
        conflict_group="large_woven_ingredient_sacks",
        expected_aspect_ratio=(0.40, 1.20),
        prompts=(
            PromptSpec(
                "large white woven sack of yellow sugar with big CO BA text and woman logo",
                "primary",
                True,
            ),
            PromptSpec(
                "industrial white sugar bag with golden brown Vietnamese text Duong Vang CO BA",
                "secondary",
                True,
            ),
            PromptSpec(
                "large white woven sugar sack with vertical Bien Hoa Duong Vang Co Ba text",
                "view",
                False,
            ),
            PromptSpec(
                "bulky white ingredient sack with gold print used for yellow sugar",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SU_0001_DuongTrang",
        display_name="DuongTrang_CoBaWhiteSugar",
        candidate_conf=0.10,
        keep_conf=0.22,
        conflict_group="large_woven_ingredient_sacks",
        expected_aspect_ratio=(0.40, 1.15),
        prompts=(
            PromptSpec(
                "large white woven sack of white sugar with green Bien Hoa and Co Ba print",
                "primary",
                True,
            ),
            PromptSpec(
                "industrial white sugar bag with green Vietnamese text Bien Hoa Duong Sach Co Ba",
                "secondary",
                True,
            ),
            PromptSpec(
                "large white woven sugar sack with vertical green Bien Hoa Duong Sach Co Ba text",
                "view",
                False,
            ),
            PromptSpec(
                "bulky white ingredient sack with green print used for white sugar",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SN_0011_HanhNhanLat",
        display_name="HanhNhanLat_SlicedAlmond",
        candidate_conf=0.10,
        keep_conf=0.20,
        conflict_group="transparent_ingredient_packets",
        expected_aspect_ratio=(0.45, 1.35),
        prompts=(
            PromptSpec(
                "clear plastic bag of sliced almonds with pale cream almond flakes inside",
                "primary",
                True,
            ),
            PromptSpec(
                "transparent food ingredient bag filled with thin sliced almond flakes",
                "secondary",
                True,
            ),
            PromptSpec(
                "clear bag of sliced almonds with Vietnamese Hanh Nhan label and almond image",
                "view",
                False,
            ),
            PromptSpec(
                "transparent pouch containing thin light beige almond slices",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="SN_0006_NamVietQuat",
        display_name="NamVietQuat_OceanSprayDriedCranberries",
        candidate_conf=0.10,
        keep_conf=0.20,
        conflict_group="flexible_dried_fruit_pouches",
        expected_aspect_ratio=(0.45, 1.40),
        prompts=(
            PromptSpec(
                "single large white Ocean Spray Craisins pouch of dried cranberries with magenta top band",
                "primary",
                True,
            ),
            PromptSpec(
                "upright resealable dried cranberry bag with blue oval logo and red cranberry graphics",
                "secondary",
                True,
            ),
            PromptSpec(
                "white and magenta flexible food pouch labeled Craisins Whole and Juicy dried cranberries",
                "view",
                False,
            ),
            PromptSpec(
                "partially folded white dried fruit pouch with magenta sealed top and Ocean Spray branding",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="PW_0009_BotPhoMai",
        display_name="BotPhoMai_CheeseSeasoningPowder",
        candidate_conf=0.10,
        keep_conf=0.20,
        conflict_group="commercial_powder_pouches",
        expected_aspect_ratio=(0.45, 1.30),
        prompts=(
            PromptSpec(
                "single light cream pouch of cheese seasoning powder with red O Sajang logo",
                "primary",
                True,
            ),
            PromptSpec(
                "upright Korean cheese seasoning bag with large black text and bowl of orange powder",
                "secondary",
                True,
            ),
            PromptSpec(
                "flexible pale beige food seasoning pouch with red top logo and golden cheese powder image",
                "view",
                False,
            ),
            PromptSpec(
                "partially folded cream colored cheese powder bag with red branding and Korean lettering",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="PW_0008_BotLaDua",
        display_name="BotLaDua_PandanLeafPowder",
        candidate_conf=0.08,
        keep_conf=0.18,
        conflict_group="commercial_powder_pouches",
        expected_aspect_ratio=(0.45, 1.25),
        prompts=(
            PromptSpec(
                "single silver resealable stand-up pouch of green pandan leaf powder with a clear front window",
                "primary",
                True,
            ),
            PromptSpec(
                "metallic silver zipper food pouch filled with pale green pandan powder and white green label",
                "secondary",
                True,
            ),
            PromptSpec(
                "silver pandan powder bag with Vietnamese Bot La Dua label and bowl of green powder",
                "view",
                False,
            ),
            PromptSpec(
                "plain matte silver resealable ingredient pouch with zipper and centered hanging hole",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="PW_0006_BotQue",
        display_name="BotQue_VipepCinnamonPowder",
        candidate_conf=0.09,
        keep_conf=0.19,
        conflict_group="small_clear_spice_jars",
        same_sku_iou=0.68,
        expected_aspect_ratio=(0.40, 1.05),
        prompts=(
            PromptSpec(
                "single small clear plastic jar of cinnamon powder with a wide black lid",
                "primary",
                True,
            ),
            PromptSpec(
                "small Vipep cinnamon spice jar containing fine brown powder with a white dark brown label",
                "secondary",
                True,
            ),
            PromptSpec(
                "clear cylindrical cinnamon powder container with cinnamon stick graphic and black screw cap",
                "view",
                False,
            ),
            PromptSpec(
                "top view of a small transparent spice jar with a round black lid and brown powder",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="PW_0005_BoCacao",
        display_name="BoCacao_GrandPlaceCocoaButter",
        candidate_conf=0.10,
        keep_conf=0.23,
        conflict_group="grand_place_cocoa_pouches",
        expected_aspect_ratio=(0.40, 1.35),
        prompts=(
            PromptSpec(
                "single large Grand Place Chocolante cocoa butter pouch with dark teal bottom and cocoa pod illustrations",
                "primary",
                True,
            ),
            PromptSpec(
                "upright flexible cocoa butter ingredient bag with brown cocoa beans and green tropical artwork",
                "secondary",
                True,
            ),
            PromptSpec(
                "large teal illustrated food ingredient pouch with white Cocoa Butter CB01 label",
                "view",
                False,
            ),
            PromptSpec(
                "thin side view of a large folded cocoa ingredient stand-up pouch with teal green base",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="PW_0003_BotMatcha",
        display_name="BotMatcha_RedCapBaker",
        candidate_conf=0.10,
        keep_conf=0.22,
        conflict_group="commercial_powder_pouches",
        expected_aspect_ratio=(0.40, 1.20),
        prompts=(
            PromptSpec(
                "single green and cream matcha powder pouch with teapot illustration and Bot Tra Xanh Matcha text",
                "primary",
                True,
            ),
            PromptSpec(
                "upright stand-up pouch of matcha tea powder with green border light cream center and tea cup graphic",
                "secondary",
                True,
            ),
            PromptSpec(
                "food ingredient pouch labeled Matcha Powder with Red Cap Baker branding and green tea design",
                "view",
                False,
            ),
            PromptSpec(
                "side view of a slim green and cream flexible matcha powder bag",
                "recall",
                False,
            ),
        ),
    ),
    SKUConfig(
        class_name="PW_0001_BotGung",
        display_name="BotGung_Vianco",
        candidate_conf=0.10,
        keep_conf=0.22,
        conflict_group="commercial_powder_pouches",
        expected_aspect_ratio=(0.40, 1.20),
        prompts=(
            PromptSpec(
                "single green and yellow ginger powder bag with GINGER Powder and Bot Gung text",
                "primary",
                True,
            ),
            PromptSpec(
                "upright flexible pouch of ginger powder with ginger root images and green packaging",
                "secondary",
                True,
            ),
            PromptSpec(
                "food ingredient package labeled Ginger Powder and Bot Gung with Vianco branding",
                "view",
                False,
            ),
            PromptSpec(
                "back view of a ginger powder bag with nutrition facts and Vietnamese product text",
                "recall",
                False,
            ),
        ),
    ),
)


GROUP_RULES: dict[str, GroupRule] = {
    "small_flavoring_bottles": GroupRule(0.80, 0.92, 0.05),
    "large_woven_ingredient_sacks": GroupRule(0.80, 0.92, 0.06),
    "rectangular_fruit_puree_tubs": GroupRule(0.85, 0.94, 0.06),
    "dried_herb_pouches": GroupRule(0.82, 0.92, 0.06),
    "transparent_ingredient_packets": GroupRule(0.80, 0.92, 0.06),
    "bulk_fats_in_blue_liners": GroupRule(0.82, 0.92, 0.07),
    "small_baking_ingredient_cartons": GroupRule(0.82, 0.92, 0.06),
    "large_tea_cartons": GroupRule(0.85, 0.94, 0.07),
    "large_dark_flavoring_bottles": GroupRule(0.84, 0.93, 0.07),
    "flexible_sugar_pouches": GroupRule(0.82, 0.92, 0.06),
    "clear_food_syrup_jars": GroupRule(0.84, 0.93, 0.07),
    "flexible_dried_fruit_pouches": GroupRule(0.82, 0.92, 0.06),
    "commercial_powder_pouches": GroupRule(0.82, 0.92, 0.07),
    "small_clear_spice_jars": GroupRule(0.84, 0.93, 0.07),
    "grand_place_cocoa_pouches": GroupRule(0.85, 0.94, 0.08),
}


CLASS_NAMES: tuple[str, ...] = tuple(config.class_name for config in SKU_CONFIGS)
CLASS_NAME_TO_ID: dict[str, int] = {
    class_name: class_id for class_id, class_name in enumerate(CLASS_NAMES)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-annotate 30 Sharon Bakery SKUs using YOLO-World only.",
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
            "uses the minimum candidate threshold across all 30 SKUs."
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
        "--no-empty-labels",
        action="store_true",
        help="Do not create empty .txt files for images with no accepted box.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the 30-SKU configuration and exit without loading the model.",
    )
    parser.add_argument(
    "--include-review-labels",
    action="store_true",
    help=(
        "Include review-status boxes in YOLO label files so they "
        "appear when imported into Roboflow."
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

    EXPECTED_SKU_COUNT = 40

    if len(SKU_CONFIGS) != EXPECTED_SKU_COUNT:
        errors.append(
            f"Expected {EXPECTED_SKU_COUNT} SKU configs, "
            f"found {len(SKU_CONFIGS)}."
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
        (
            item
            for item in detections
            if item.status in allowed_statuses
        ),
        key=lambda item: (
            item.sku_id,
            float(item.xyxy[1]),
            float(item.xyxy[0]),
        ),
    )

    if not exported and not create_empty:
        if path.exists():
            path.unlink()
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file_handle:
        for detection in exported:
            x_center, y_center, width, height = xyxy_to_yolo(
                detection.xyxy,
                image_width,
                image_height,
            )

            file_handle.write(
                f"{detection.sku_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
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