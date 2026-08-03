from __future__ import annotations

import gc
import hashlib
import html
import io
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from openpyxl.styles import Alignment, Font, PatternFill
from ultralytics import YOLO


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================
APP_TITLE = "AI Inventory Inspection"
APP_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    str(
        Path(__file__).resolve().parent
        / "models"
        / "best_YOLO26s_107_SKU_v2.pt"
    ),
)

CLASS_MAPPING_PATH = Path(
    os.getenv(
        "CLASS_MAPPING_PATH",
        str(APP_DIR / "class_display_mapping.json"),
    )
).expanduser()

CLASS_CONFIDENCE_PATH = Path(
    os.getenv(
        "CLASS_CONFIDENCE_PATH",
        str(APP_DIR / "class_confidence_thresholds_v9_precision_count.json"),
    )
).expanduser()

RUNTIME_ROOT = Path(
    os.getenv(
        "STREAMLIT_RUNTIME_DIR",
        str(APP_DIR / ".runtime"),
    )
).expanduser().resolve()

DEFAULT_IMAGE_SIZE = 768
DEFAULT_CONFIDENCE = 0.05
DEFAULT_IOU = 0.55
DEFAULT_MAX_DETECTIONS = 300

MAX_IMAGES_PER_RUN = int(
    os.getenv("MAX_IMAGES_PER_RUN", "500")
)
MAX_IMAGE_PIXELS = int(
    os.getenv("MAX_IMAGE_PIXELS", "60000000")
)
PREVIEW_MAX_SIDE = int(
    os.getenv("PREVIEW_MAX_SIDE", "1400")
)
PREVIEW_JPEG_QUALITY = int(
    os.getenv("PREVIEW_JPEG_QUALITY", "82")
)
ANNOTATED_JPEG_QUALITY = int(
    os.getenv("ANNOTATED_JPEG_QUALITY", "92")
)
RUNTIME_RETENTION_HOURS = int(
    os.getenv("RUNTIME_RETENTION_HOURS", "24")
)

MAX_GT_ZIP_FILES = int(
    os.getenv("MAX_GT_ZIP_FILES", "20000")
)
MAX_GT_ZIP_UNCOMPRESSED_BYTES = int(
    os.getenv(
        "MAX_GT_ZIP_UNCOMPRESSED_BYTES",
        str(4 * 1024 * 1024 * 1024),
    )
)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

LOGGER = logging.getLogger("sharon-streamlit-yolo26")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Prevent OpenCV from creating too many CPU worker threads.
cv2.setNumThreads(
    max(1, min(4, os.cpu_count() or 1))
)


# =========================================================
# 2. PAGE AND PROFESSIONAL STYLING
# =========================================================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="locked",
)

st.markdown(
    """
    <style>
        :root {
            --sb-navy: #15324A;
            --sb-blue: #2B5C7D;
            --sb-border: #DCE4E9;
            --sb-muted: #667784;
        }

        [data-testid="stAppViewContainer"] {
            background: #F7F9FA;
        }

        [data-testid="stSidebar"] {
            background: #EEF3F6;
            border-right: 1px solid var(--sb-border);
        }

        /* Hide only the menu and footer.
        Keep the header so the sidebar control remains available. */
        #MainMenu,
        footer {
            visibility: hidden;
        }

        /* Keep the header small and transparent */
        [data-testid="stHeader"] {
            background: transparent !important;
        }


        .block-container {
            max-width: 1480px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        .sb-header {
            background: linear-gradient(120deg, #15324A 0%, #2B5C7D 100%);
            border-radius: 16px;
            padding: 28px 32px;
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(21, 50, 74, 0.14);
        }

        .sb-eyebrow {
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.80;
            margin-bottom: 8px;
        }

        .sb-header h1 {
            color: white;
            font-size: 2rem;
            line-height: 1.2;
            margin: 0;
            padding: 0;
        }

        .sb-header p {
            margin: 10px 0 0 0;
            color: rgba(255, 255, 255, 0.86);
            font-size: 0.98rem;
        }

        .sb-section-title {
            color: var(--sb-navy);
            font-size: 1.18rem;
            font-weight: 700;
            margin: 6px 0 14px 0;
        }

        .sb-card {
            background: white;
            border: 1px solid var(--sb-border);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 3px 12px rgba(21, 50, 74, 0.06);
        }

        .sb-message {
            border-radius: 10px;
            padding: 12px 14px;
            margin: 8px 0 12px 0;
            font-size: 0.92rem;
            border: 1px solid;
        }

        .sb-message.info {
            background: #EAF2F8;
            border-color: #8FB5CF;
            color: #183A52;
        }

        .sb-message.success {
            background: #EAF5EF;
            border-color: #8DC5A6;
            color: #275B3D;
        }

        .sb-message.warning {
            background: #FFF7E6;
            border-color: #E6BF6A;
            color: #6C531F;
        }

        .sb-message.error {
            background: #FDEEEE;
            border-color: #D88B8B;
            color: #7A2929;
        }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--sb-border);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 2px 10px rgba(21, 50, 74, 0.05);
        }

        [data-testid="stFileUploader"] section {
            border: 1px dashed #8198A8;
            border-radius: 12px;
            background: white;
        }

        [data-testid="stFileUploaderDropzone"] svg,
        [data-testid="stFileUploaderDropzone"] button svg,
        [data-testid="stFileUploader"] svg {
            display: none !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            padding-left: 16px !important;
            padding-right: 16px !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 9px;
            border: 1px solid #2B5C7D;
            min-height: 42px;
            font-weight: 600;
        }

        .stButton > button[kind="primary"] {
            background: #1E4A68;
            color: white;
        }

        .stButton > button[kind="primary"]:hover {
            background: #153B55;
            border-color: #153B55;
        }

        div[data-testid="stImage"] img {
            border-radius: 10px;
            border: 1px solid var(--sb-border);
        }

        .sb-muted {
            color: var(--sb-muted);
            font-size: 0.88rem;
        }

        .sb-footer {
            color: #71808B;
            font-size: 0.78rem;
            text-align: center;
            margin-top: 30px;
            padding-top: 16px;
            border-top: 1px solid var(--sb-border);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. DATA STRUCTURES
# =========================================================
@dataclass
class ModelBundle:
    model: YOLO
    raw_names: dict[int, str]
    display_names: dict[int, str]
    lock: threading.Lock


@dataclass
class UploadRecord:
    display_name: str
    safe_name: str
    data: bytes
    content_type: str | None
    sha256: str


@dataclass
class RunDirectories:
    root: Path
    original: Path
    annotated: Path
    original_preview: Path
    annotated_preview: Path
    reports: Path


@dataclass
class GroundTruthBundle:
    data_yaml_path: Path
    split_name: str
    class_names: dict[int, str]
    image_paths: list[Path]
    objects_by_filename: dict[str, list[dict[str, Any]]]
    objects_by_stem: dict[str, list[dict[str, Any]]]
    objects_by_sha256: dict[str, list[dict[str, Any]]]
    class_scope: dict[str, Any]


# =========================================================
# 4. CLASS NAME MAPPING
# =========================================================
DEFAULT_CLASS_DISPLAY_MAPPING = {
    "BK_0007_BoLatPilot": "BoLatPilot_UnsaltedButter",
    "CH_0006_SocolaCompoundTrang": "SocolaCompoundTrang_WhiteChocolateCompound",
    "FL_0002_BotCake": "BotCake_CakeFlour",
    "FL_0004_BotAtta": "BotAtta_AttaFlour",
    "FL_0011_BotCustard": "BotCustard_CustardPowder",
    "FL_0013_BotXanhLa": "BotXanhLa_GreenFlour",
    "FL_0014_BotCam": "BotCam_OrangeFlour",
    "FV_0002_CaRot_Bich": "CaRot_Bich_Carrot",
    "FV_0002_CaRot_Cu": "CaRot_Cu_Carrot",
    "FV_0003_Chuoi_Trai": "Chuoi_Trai_Banana",
    "ISO_OF_0012_BaoRac": "BaoRac_TrashBags",
    "ISO_PA_0020_BaoBanhCuonMem": "BaoBanhCuonMem_SoftRollCakeBag",
    "ISO_PA_0029_CuonDayThungNho": "CuonDayThungNho_SmallTwineRoll",
    "ISO_SN_0010_BanhCracker": "BanhCracker_Cracker",
    "JM_0007_MutPhuBanh": "MutPhuBanh_Glaze",
    "OF_0001_VimVeSinh": "VimVeSinh_ToiletLiquid",
    "OF_0003_NuocRuaChen": "NuocRuaChen_DishwashingLiquid",
    "OF_0004_MangBocThucPham": "MangBocThucPham_WrapFood",
    "OF_0004_NuocRuaTay": "NuocRuaTay_HandWashLiquid",
    "OF_0008_NuocLavie": "NuocLavie_Water",
    "OF_0009_GiayVS": "GiayVS_ToiletPaper",
    "OF_0010_GiayBep": "GiayBep_KitchenPaper",
    "OF_0011_GiayNen": "GiayNen_BakingPaper",
    "OF_0013_GangTay": "GangTay_Gloves",
    "PA_0014_NapBanhKemTron": "NapBanhKemTron_RoundCakeLid",
    "PW_0010_CafeBot_Bich": "CafeBot_Bich_CoffeePowder",
    "PW_0010_CafeBot_Hu": "CafeBot_Hu_CoffeePowder",
    "PW_0015_BotNoi": "BotNoi_BakingPowder",
    "SN_0019_Oreo": "Oreo_Cookie",
    "YE_0003_MenTrang": "MenTrang_WhiteInstantYeast",
    "ISO_YE_0003_MenTrang": "MenTrang_WhiteInstantYeast",
        # =====================================================
    # BỘ NHÃN MỚI — 30 SKU
    # =====================================================
    "SY_0009_TinhChatChanh": "TinhChatChanh_LemonFlavor",
    "FL_0012_BotDo": "BotDo_RedRiceFlour",
    "FL_0003_BotNguyenCam": "BotNguyenCam_WholeMealFlour",
    "SY_0008_TinhChatBo": "TinhChatBo_ButterFlavor",
    "SY_0007_TinhChatCam": "TinhChatCam_OrangeFlavor",
    "SY_0006_TinhChatHanhNhan": "TinhChatHanhNhan_AlmondFlavor",
    "JM_0008_PureAnhDao": "PureAnhDao_CherryPureAndros",
    "JM_0005_XoaiPureAndros": "XoaiPureAndros_MangoPuree",
    "JM_0004_PureVietQuat": "PureVietQuat_BlueberryPureAndros",
    "JM_0003_PureDau": "PureDau_StrawberryPureeAndros",
    "GV_0003_LaKinhGioi": "LaKinhGioi_VietnameseBalm",
    "YE_0006_Gelatin": "Gelatin_Gelatin",
    "BK_0010_BoVegan": "BoVegan_VeganButter",
    "YE_0005_RauCauDeo": "RauCauDeo_JellyPowder",
    "YE_0004_BakingSoda": "BakingSoda_BakingSoda",
    "SY_0002_TraEarlGrey": "TraEarlGrey_EarlGreyTea",
    "SY_0001_SyrupVani": "SyrupVani_VanillaSyrup",
    "SU_0005_DuongNau": "DuongNau_BrownSugar",
    "SU_0004_DuongMachNha": "DuongMachNha_Maltose",
    "SU_0003_DuongBot": "DuongBot_IcingSugar",
    "SU_0002_DuongVang": "DuongVang_YellowSugar",
    "SU_0001_DuongTrang": "DuongTrang_WhiteSugar",
    "SN_0011_HanhNhanLat": "HanhNhanLat_SlicedAlmond",
    "SN_0006_NamVietQuat": "NamVietQuat_Cranberry",
    "PW_0009_BotPhoMai": "BotPhoMai_CheesePowder",
    "PW_0008_BotLaDua": "BotLaDua_PandanPowder",
    "PW_0006_BotQue": "BotQue_CinnamonPowder",
    "PW_0005_BoCacao": "BoCacao_CocoaPaste",
    "PW_0004_BotCacao": "BotCacao_CocoaPowder",
    "PW_0003_BotMatcha": "BotMatcha_MatchaPowder",
    "PW_0001_BotGung": "BotGung_GingerPowder",
}

DEFAULT_CLASS_DISPLAY_MAPPING.update(
    {
        # =====================================================
        # BAKERY / DAIRY
        # =====================================================
        "BK_0002_KemPhoMai": (
            "KemPhoMai_CreamCheese"
        ),
        "BK_0003_KemSuaTuoi": (
            "KemSuaTuoi_WhippingCream"
        ),
        "BK_0009_KemTopping": (
            "KemTopping_WhippableTopping"
        ),
        "BK_0010_BoVegan": (
            "BoVegan_VeganButter"
        ),

        # =====================================================
        # CHOCOLATE
        # =====================================================
        "CH_0001_ChocoCompoundDen": (
            "ChocoCompoundDen_DarkChocolateCompound"
        ),
        "CH_0002_ChocoCompoundChip": (
            "ChocoCompoundChip_DarkChocolateChips"
        ),
        "CH_0007_ChocoStickPuratos": (
            "ChocoStickPuratos_ChocolateSticks"
        ),

        # =====================================================
        # CANNED PRODUCTS
        # =====================================================
        "CN_0001_OliuNgam": (
            "OliuNgam_PickledOlives"
        ),
        "CN_0002_CaChuaLon": (
            "CaChuaLon_CannedTomatoes"
        ),

        # =====================================================
        # FOOD COLORS
        # =====================================================
        "FC_0001_MauXanhLa": (
            "MauXanhLa_GreenFoodColor"
        ),
        "FC_0001_MauXanhLa_ChaiLon": (
            "MauXanhLa_ChaiLon_LargeBottleGreenFoodColor"
        ),
        "FC_0002_MauDen": (
            "MauDen_BlackFoodColor"
        ),
        "FC_0003_MauHong": (
            "MauHong_PinkFoodColor"
        ),
        "FC_0004_MauVang": (
            "MauVang_YellowFoodColor"
        ),
        "FC_0005_MauNau": (
            "MauNau_BrownFoodColor"
        ),
        "FC_0006_MauSieuDo": (
            "MauSieuDo_SuperRedFoodColor"
        ),

        # =====================================================
        # FLOUR / STARCH
        # =====================================================
        "FL_0003_BotNguyenCam": (
            "BotNguyenCam_WholeWheatFlour"
        ),
        "FL_0005_BotLuaMachDen": (
            "BotLuaMachDen_RyeBreadMix"
        ),
        "FL_0007_BotKieuMach": (
            "BotKieuMach_BuckwheatFlour"
        ),
        "FL_0008_BotBap": (
            "BotBap_CornFlour"
        ),
        "FL_0009_BotNang": (
            "BotNang_TapiocaStarch"
        ),
        "FL_0010_BotGao": (
            "BotGao_RiceFlour"
        ),
        "FL_0012_BotDo": (
            "BotDo_RedKeyFlour"
        ),

        # =====================================================
        # FRUIT / VEGETABLE
        # =====================================================
        "FV_0001_CaChuaBi": (
            "CaChuaBi_CherryTomatoes"
        ),

        # =====================================================
        # SEASONING / OTHER INGREDIENTS
        # =====================================================
        "GV_0001_Muoi": (
            "Muoi_Salt"
        ),
        "GV_0002_GiamTao": (
            "GiamTao_AppleCiderVinegar"
        ),
        "GV_0003_LaKinhGioi": (
            "LaKinhGioi_VietnameseBalmLeaves"
        ),
        "GV_0005_PsylliumHusk": (
            "PsylliumHusk_PsylliumFiber"
        ),

        # =====================================================
        # JAM / PUREE / SMOOTHIE
        # =====================================================
        "JM_0001_SinhToChanhDay": (
            "SinhToChanhDay_PassionFruitSmoothie"
        ),
        "JM_0002_SinhToXoai": (
            "SinhToXoai_MangoSmoothie"
        ),
        "JM_0003_PureDau": (
            "PureDau_StrawberryPuree"
        ),
        "JM_0004_PureVietQuat": (
            "PureVietQuat_BlueberryPuree"
        ),
        "JM_0005_XoaiPureAndros": (
            "XoaiPureAndros_MangoPuree"
        ),
        "JM_0006_MutMo": (
            "MutMo_ApricotJam"
        ),
        "JM_0008_PureAnhDao": (
            "PureAnhDao_CherryPuree"
        ),

        # =====================================================
        # OIL / MILK / EGGS
        # =====================================================
        "OD_0001_DauAn": (
            "DauAn_CookingOil"
        ),
        "OD_0001_DauAn_Thung": (
            "DauAn_Thung_CookingOilCase"
        ),
        "OD_0002_Daudua": (
            "Daudua_CoconutOil"
        ),
        "OD_0003_DauOliu": (
            "DauOliu_OliveOil"
        ),
        "OD_0004_SuaTuoi": (
            "SuaTuoi_FreshMilk"
        ),
        "OD_0005_SuaDauNanh": (
            "SuaDauNanh_SoyMilk"
        ),
        "OD_0005_SuaDauNanh_Thung": (
            "SuaDauNanh_Thung_SoyMilkCase"
        ),
        "OD_0006_Trung_Khay": (
            "Trung_Khay_EggTray"
        ),
        "OD_0006_Trung_Qua": (
            "Trung_Qua_Egg"
        ),
        "OD_0010_BoTuongAn": (
            "BoTuongAn_Margarine"
        ),

        # =====================================================
        # POWDER
        # =====================================================
        "PW_0001_BotGung": (
            "BotGung_GingerPowder"
        ),
        "PW_0003_BotMatcha": (
            "BotMatcha_MatchaPowder"
        ),
        "PW_0004_BotCacao": (
            "BotCacao_CocoaPowder"
        ),
        "PW_0005_BoCacao": (
            "BoCacao_CocoaButter"
        ),
        "PW_0006_BotQue": (
            "BotQue_CinnamonPowder"
        ),
        "PW_0007_BotNghe": (
            "BotNghe_TurmericPowder"
        ),
        "PW_0008_BotLaDua": (
            "BotLaDua_PandanPowder"
        ),
        "PW_0009_BotPhoMai": (
            "BotPhoMai_CheesePowder"
        ),
        "PW_0014_BotCustardHieuSuTu": (
            "BotCustardHieuSuTu_LionCustardPowder"
        ),

        # =====================================================
        # NUTS / DRIED FRUIT
        # =====================================================
        "SN_0002_HatBi": (
            "HatBi_PumpkinSeeds"
        ),
        "SN_0003_HuongDuong": (
            "HuongDuong_SunflowerSeeds"
        ),
        "SN_0004_OcCho": (
            "OcCho_Walnuts"
        ),
        "SN_0006_NamVietQuat": (
            "NamVietQuat_DriedCranberries"
        ),
        "SN_0011_HanhNhanLat": (
            "HanhNhanLat_SlicedAlmonds"
        ),

        # =====================================================
        # SUGAR
        # =====================================================
        "SU_0001_DuongTrang": (
            "DuongTrang_WhiteSugar"
        ),
        "SU_0002_DuongVang": (
            "DuongVang_YellowSugar"
        ),
        "SU_0003_DuongBot": (
            "DuongBot_PowderedSugar"
        ),
        "SU_0004_DuongMachNha": (
            "DuongMachNha_MaltoseSyrup"
        ),
        "SU_0005_DuongNau": (
            "DuongNau_BrownSugar"
        ),

        # =====================================================
        # SYRUP / TEA / EXTRACT
        # =====================================================
        "SY_0001_SyrupVani": (
            "SyrupVani_VanillaSyrup"
        ),
        "SY_0002_TraEarlGrey": (
            "TraEarlGrey_EarlGreyTea"
        ),
        "SY_0006_TinhChatHanhNhan": (
            "TinhChatHanhNhan_AlmondExtract"
        ),
        "SY_0007_TinhChatCam": (
            "TinhChatCam_OrangeExtract"
        ),
        "SY_0008_TinhChatBo": (
            "TinhChatBo_ButterExtract"
        ),
        "SY_0009_TinhChatChanh": (
            "TinhChatChanh_LemonExtract"
        ),

        # =====================================================
        # YEAST / GELATIN / LEAVENING
        # =====================================================
        "YE_0001_MenVang": (
            "MenVang_GoldInstantYeast"
        ),
        "YE_0004_BakingSoda": (
            "BakingSoda_SodiumBicarbonate"
        ),
        "YE_0005_RauCauDeo": (
            "RauCauDeo_JellyPowder"
        ),
        "YE_0006_Gelatin": (
            "Gelatin_GelatinPowder"
        ),
        "YE_002_MenDo": (
            "MenDo_RedInstantYeast"
        ),
    }
)

@st.cache_data(show_spinner=False)
def load_class_display_mapping(
    mapping_path: str,
    modified_time_ns: int,
) -> dict[str, str]:
    del modified_time_ns

    resolved_path = Path(mapping_path)

    if not resolved_path.is_file():
        return dict(DEFAULT_CLASS_DISPLAY_MAPPING)

    try:
        payload = json.loads(
            resolved_path.read_text(encoding="utf-8")
        )
        classes = payload.get("classes", {})

        if not isinstance(classes, dict):
            raise TypeError(
                "The 'classes' field must be an object."
            )

        mapping = dict(DEFAULT_CLASS_DISPLAY_MAPPING)
        mapping.update(
            {
                str(raw_name): str(display_name)
                for raw_name, display_name in classes.items()
            }
        )
        return mapping

    except Exception as error:
        LOGGER.warning(
            "Cannot load class mapping from %s: %s",
            resolved_path,
            error,
        )
        return dict(DEFAULT_CLASS_DISPLAY_MAPPING)


def get_class_mapping_signature() -> tuple[str, int]:
    if CLASS_MAPPING_PATH.is_file():
        stat = CLASS_MAPPING_PATH.stat()
        return str(CLASS_MAPPING_PATH), stat.st_mtime_ns

    return str(CLASS_MAPPING_PATH), 0


def fallback_display_name(raw_name: str) -> str:
    label = raw_name[4:] if raw_name.startswith("ISO_") else raw_name
    match = re.match(
        r"^[A-Z]+_[0-9]{4}_(.+)$",
        label,
    )

    return match.group(1) if match else label


def resolve_display_names(
    raw_names: dict[int, str],
    class_mapping: dict[str, str],
) -> dict[int, str]:
    display_names: dict[int, str] = {}
    used_names: set[str] = set()

    for class_id, raw_name in raw_names.items():
        display_name = class_mapping.get(raw_name)

        if display_name is None and raw_name.startswith("ISO_"):
            display_name = class_mapping.get(raw_name[4:])

        if display_name is None:
            display_name = fallback_display_name(raw_name)

        if display_name in used_names:
            display_name = f"{display_name}_Class{class_id}"

        used_names.add(display_name)
        display_names[class_id] = display_name

    return display_names


def clamp_confidence_threshold(
    value: Any,
    fallback: float,
) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = float(fallback)

    return min(0.99, max(0.01, threshold))


def load_class_confidence_defaults(
    raw_names: dict[int, str],
    fallback: float,
) -> dict[int, float]:
    """
    Load optional defaults from class_confidence_thresholds.json.

    Format:
    {
        "default": 0.25,
        "classes": {
            "BK_0007_BoLatPilot": 0.30
        }
    }
    """
    configured_default = clamp_confidence_threshold(
        fallback,
        DEFAULT_CONFIDENCE,
    )
    configured_classes: dict[str, Any] = {}

    if CLASS_CONFIDENCE_PATH.is_file():
        try:
            payload = json.loads(
                CLASS_CONFIDENCE_PATH.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(payload, dict):
                raise TypeError(
                    "Threshold configuration must be a JSON object."
                )

            configured_default = clamp_confidence_threshold(
                payload.get(
                    "default",
                    configured_default,
                ),
                configured_default,
            )
            classes_payload = payload.get(
                "classes",
                {},
            )

            if not isinstance(classes_payload, dict):
                raise TypeError(
                    "The 'classes' field must be a JSON object."
                )

            configured_classes = classes_payload

        except Exception as error:
            LOGGER.warning(
                "Cannot load class confidence configuration from %s: %s",
                CLASS_CONFIDENCE_PATH,
                error,
            )

    return {
        class_id: clamp_confidence_threshold(
            configured_classes.get(
                raw_name,
                configured_default,
            ),
            configured_default,
        )
        for class_id, raw_name
        in raw_names.items()
    }


def build_class_confidence_payload(
    raw_names: dict[int, str],
    thresholds: dict[int, float],
    fallback: float,
) -> dict[str, Any]:
    return {
        "default": clamp_confidence_threshold(
            fallback,
            DEFAULT_CONFIDENCE,
        ),
        "classes": {
            raw_names[class_id]: float(
                thresholds.get(
                    class_id,
                    fallback,
                )
            )
            for class_id in sorted(raw_names)
        },
    }


# =========================================================
# 5. GENERIC HELPERS
# =========================================================
def render_message(
    message: str,
    message_type: str = "info",
) -> None:
    st.markdown(
        (
            f'<div class="sb-message {message_type}">'
            f"{html.escape(message)}</div>"
        ),
        unsafe_allow_html=True,
    )


def normalize_class_names(
    names: Any,
) -> dict[int, str]:
    if isinstance(names, dict):
        normalized = {
            int(class_id): str(class_name)
            for class_id, class_name in names.items()
        }
    elif isinstance(names, (list, tuple)):
        normalized = {
            class_id: str(class_name)
            for class_id, class_name in enumerate(names)
        }
    else:
        raise TypeError(
            "Unsupported model.names type: "
            f"{type(names).__name__}."
        )

    if sorted(normalized) != list(range(len(normalized))):
        raise ValueError(
            "Model class IDs must be continuous from zero."
        )

    return normalized


def sanitize_filename(
    name: str,
) -> str:
    normalized = name.replace("\\", "/").strip("/")
    flattened = normalized.replace("/", "__")
    flattened = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        flattened,
    ).strip("._")

    return flattened or "uploaded_image"


def canonical_image_stem(name: str) -> str:
    """
    Normalize local and Roboflow-exported filenames for Ground Truth matching.

    Examples:
        image.jpg -> image
        image_jpg.rf.abc123.jpg -> image
        folder/image_png.rf.abc123.png -> image
    """
    basename = Path(
        str(name).replace("\\", "/")
    ).name.lower()
    stem = Path(basename).stem

    # Roboflow commonly appends: .rf.<hash>
    stem = re.sub(
        r"\.rf\.[0-9a-f]+$",
        "",
        stem,
        flags=re.IGNORECASE,
    )

    # Roboflow may preserve the original extension as a suffix.
    stem = re.sub(
        r"_(jpg|jpeg|png|bmp|webp|tif|tiff)$",
        "",
        stem,
        flags=re.IGNORECASE,
    )

    return stem


def make_unique_name(
    proposed_name: str,
    used_names: set[str],
) -> str:
    candidate = proposed_name
    stem = Path(proposed_name).stem
    suffix = Path(proposed_name).suffix
    counter = 2

    while candidate.lower() in used_names:
        candidate = f"{stem}__{counter}{suffix}"
        counter += 1

    used_names.add(candidate.lower())
    return candidate


def build_upload_records(
    uploaded_files: list[Any],
) -> list[UploadRecord]:
    records: list[UploadRecord] = []
    used_names: set[str] = set()

    for uploaded_file in uploaded_files:
        raw_bytes = uploaded_file.getvalue()

        if not raw_bytes:
            continue

        original_name = (
            uploaded_file.name
            if uploaded_file.name
            else "uploaded_image"
        )

        safe_name = make_unique_name(
            sanitize_filename(original_name),
            used_names,
        )

        records.append(
            UploadRecord(
                display_name=original_name,
                safe_name=safe_name,
                data=raw_bytes,
                content_type=getattr(
                    uploaded_file,
                    "type",
                    None,
                ),
                sha256=hashlib.sha256(
                    raw_bytes
                ).hexdigest(),
            )
        )

    return records


def decode_image(
    raw_bytes: bytes,
) -> np.ndarray:
    image_array = np.frombuffer(
        raw_bytes,
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(
            "The image cannot be decoded. "
            "The file may be damaged or unsupported."
        )

    height, width = image_bgr.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            "The image size is not valid."
        )

    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError(
            "The image exceeds the pixel limit: "
            f"{width}x{height}."
        )

    return image_bgr


def resize_for_preview(
    image_bgr: np.ndarray,
    max_side: int = PREVIEW_MAX_SIDE,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_side:
        return image_bgr

    scale = max_side / longest_side
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))

    return cv2.resize(
        image_bgr,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def write_jpeg(
    path: Path,
    image_bgr: np.ndarray,
    quality: int,
) -> None:
    success, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )

    if not success:
        raise RuntimeError(
            f"Cannot encode image: {path.name}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def resolve_device(
    selection: str,
) -> int | str:
    if selection == "Auto":
        return 0 if torch.cuda.is_available() else "cpu"

    if selection == "GPU 0":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is not available."
            )
        return 0

    return "cpu"


def create_run_directories() -> RunDirectories:
    session_id = st.session_state["session_id"]
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )

    root = RUNTIME_ROOT / session_id / run_id
    directories = RunDirectories(
        root=root,
        original=root / "original",
        annotated=root / "annotated",
        original_preview=root / "preview" / "original",
        annotated_preview=root / "preview" / "annotated",
        reports=root / "reports",
    )

    for directory in directories.__dict__.values():
        Path(directory).mkdir(parents=True, exist_ok=True)

    return directories


def is_path_inside_runtime(
    path: Path,
) -> bool:
    try:
        path.resolve().relative_to(RUNTIME_ROOT)
        return True
    except ValueError:
        return False


def remove_run_directory(
    run_directory: str | Path | None,
) -> None:
    if not run_directory:
        return

    path = Path(run_directory)

    if path.exists() and is_path_inside_runtime(path):
        shutil.rmtree(path, ignore_errors=True)


def cleanup_old_runtime_directories() -> None:
    if not RUNTIME_ROOT.exists():
        return

    cutoff = time.time() - (
        RUNTIME_RETENTION_HOURS * 3600
    )

    for session_directory in RUNTIME_ROOT.iterdir():
        if not session_directory.is_dir():
            continue

        for run_directory in session_directory.iterdir():
            try:
                if (
                    run_directory.is_dir()
                    and run_directory.stat().st_mtime < cutoff
                ):
                    shutil.rmtree(
                        run_directory,
                        ignore_errors=True,
                    )
            except OSError:
                LOGGER.warning(
                    "Cannot inspect runtime directory: %s",
                    run_directory,
                )

        try:
            if not any(session_directory.iterdir()):
                session_directory.rmdir()
        except OSError:
            pass


@st.cache_resource(show_spinner=False)
def load_model(
    model_path: str,
    mapping_path: str,
    mapping_modified_time_ns: int,
) -> ModelBundle:
    resolved_path = Path(model_path).expanduser()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Model file was not found: {resolved_path}"
        )

    LOGGER.info(
        "Loading model from %s",
        resolved_path,
    )

    model = YOLO(str(resolved_path))
    raw_names = normalize_class_names(model.names)
    class_mapping = load_class_display_mapping(
        mapping_path,
        mapping_modified_time_ns,
    )
    display_names = resolve_display_names(
        raw_names,
        class_mapping,
    )

    return ModelBundle(
        model=model,
        raw_names=raw_names,
        display_names=display_names,
        lock=threading.Lock(),
    )


@st.cache_data(
    show_spinner=False,
    ttl=3600,
    max_entries=256,
)
def read_file_bytes_cached(
    path: str,
    modified_time_ns: int,
    file_size: int,
) -> bytes:
    del modified_time_ns, file_size
    return Path(path).read_bytes()


def read_file_bytes(
    path: str | Path,
) -> bytes:
    resolved_path = Path(path)

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"File was not found: {resolved_path}"
        )

    stat = resolved_path.stat()
    return read_file_bytes_cached(
        str(resolved_path),
        stat.st_mtime_ns,
        stat.st_size,
    )


def deferred_file_reader(
    path: str | Path,
) -> Callable[[], bytes]:
    resolved_path = str(Path(path))

    def reader() -> bytes:
        return Path(resolved_path).read_bytes()

    return reader


# =========================================================
# 6. DETECTION AND ANNOTATION
# =========================================================
BOX_COLORS = [
    (74, 50, 21),
    (125, 92, 43),
    (72, 111, 57),
    (128, 82, 68),
    (73, 89, 143),
    (104, 80, 150),
    (67, 130, 117),
    (52, 119, 166),
]


def draw_predictions(
    image_bgr: np.ndarray,
    result: Any,
    display_names: dict[int, str],
    show_confidence: bool,
    line_width: int,
) -> np.ndarray:
    """
    Draw bounding boxes and labels with a dynamic scale.

    The annotation size is calculated from the original image resolution,
    so labels remain readable after the image is resized for web preview.
    """
    canvas = image_bgr.copy()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return canvas

    xyxy_values = (
        boxes.xyxy
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    class_ids = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )
    confidence_values = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )

    image_height, image_width = canvas.shape[:2]
    longest_side = max(image_height, image_width)

    # Scale annotation based on the original image resolution.
    # Example:
    # 1200 px image -> scale 1.0
    # 2400 px image -> scale 2.0
    # 4000 px image -> scale 3.33
    annotation_scale = max(
        1.0,
        longest_side / 1200.0,
    )

    box_thickness = max(
        2,
        int(round(line_width * annotation_scale)),
    )

    base_font_scale = max(
        0.75,
        min(
            2.2,
            0.70 * annotation_scale,
        ),
    )

    text_thickness = max(
        1,
        int(round(box_thickness * 0.55)),
    )

    padding = max(
        5,
        int(round(5 * annotation_scale)),
    )

    for xyxy, class_id, confidence in zip(
        xyxy_values,
        class_ids,
        confidence_values,
    ):
        x1, y1, x2, y2 = [
            int(round(value))
            for value in xyxy
        ]

        x1 = min(max(x1, 0), image_width - 1)
        y1 = min(max(y1, 0), image_height - 1)
        x2 = min(max(x2, 0), image_width - 1)
        y2 = min(max(y2, 0), image_height - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        color = BOX_COLORS[
            int(class_id) % len(BOX_COLORS)
        ]

        class_name = display_names.get(
            int(class_id),
            f"UnknownClass{class_id}",
        )

        label = class_name

        if show_confidence:
            label = (
                f"{class_name} "
                f"{float(confidence):.2f}"
            )

        # Draw bounding box.
        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            thickness=box_thickness,
            lineType=cv2.LINE_AA,
        )

        # Reduce the font only when the label is wider
        # than the available image width.
        local_font_scale = base_font_scale

        while local_font_scale > 0.45:
            (text_width, text_height), baseline = (
                cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    local_font_scale,
                    text_thickness,
                )
            )

            if (
                text_width + padding * 2
                <= image_width
            ):
                break

            local_font_scale *= 0.90

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                local_font_scale,
                text_thickness,
            )
        )

        label_width = text_width + padding * 2
        label_height = (
            text_height
            + baseline
            + padding * 2
        )

        # Keep the label inside the image.
        label_x1 = min(
            max(x1, 0),
            max(0, image_width - label_width),
        )
        label_x2 = min(
            image_width - 1,
            label_x1 + label_width,
        )

        # Prefer displaying the label above the box.
        if y1 >= label_height:
            label_y1 = y1 - label_height
            label_y2 = y1
            text_y = (
                label_y1
                + padding
                + text_height
            )
        else:
            # Display below the top edge when there is
            # not enough space above the box.
            label_y1 = y1
            label_y2 = min(
                image_height - 1,
                y1 + label_height,
            )
            text_y = min(
                image_height - baseline - 1,
                label_y1
                + padding
                + text_height,
            )

        cv2.rectangle(
            canvas,
            (label_x1, label_y1),
            (label_x2, label_y2),
            color,
            thickness=-1,
        )

        cv2.putText(
            canvas,
            label,
            (
                label_x1 + padding,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            local_font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return canvas


def filter_result_by_class_confidence(
    result: Any,
    class_thresholds: dict[int, float],
    fallback_threshold: float,
) -> tuple[int, int]:
    """
    Filter YOLO detections with a threshold selected by predicted class.

    YOLO first runs with the lowest configured threshold. The returned
    detections are then filtered here before drawing, counting, reporting,
    and Ground Truth evaluation.

    Returns:
        raw detection count and kept detection count.
    """
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return 0, 0

    raw_count = len(boxes)
    class_ids = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )
    confidences = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )

    keep_numpy = np.asarray(
        [
            confidence
            >= class_thresholds.get(
                int(class_id),
                float(fallback_threshold),
            )
            for class_id, confidence in zip(
                class_ids,
                confidences,
            )
        ],
        dtype=bool,
    )

    box_data = boxes.data

    if isinstance(box_data, torch.Tensor):
        keep_mask = torch.as_tensor(
            keep_numpy,
            dtype=torch.bool,
            device=box_data.device,
        )
    else:
        keep_mask = keep_numpy

    # Ultralytics Results.update() rebuilds the Boxes object safely.
    result.update(
        boxes=box_data[keep_mask]
    )

    return raw_count, int(
        keep_numpy.sum()
    )


def extract_class_statistics(
    result: Any,
    display_names: dict[int, str],
    class_thresholds: dict[int, float],
    fallback_threshold: float,
) -> tuple[list[dict[str, Any]], int, float]:
    class_statistics: dict[str, dict[str, float]] = {}
    total_detections = 0
    total_confidence_sum = 0.0
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        class_ids = (
            boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )
        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

        for class_id, confidence in zip(
            class_ids,
            confidences,
        ):
            class_name = display_names.get(
                class_id,
                f"UnknownClass{class_id}",
            )
            confidence_value = float(confidence)
            confidence_threshold = float(
                class_thresholds.get(
                    class_id,
                    fallback_threshold,
                )
            )

            if class_name not in class_statistics:
                class_statistics[class_name] = {
                    "quantity": 0,
                    "confidence_sum": 0.0,
                    "min_confidence": 1.0,
                    "max_confidence": 0.0,
                    "confidence_threshold": (
                        confidence_threshold
                    ),
                }

            class_statistics[class_name]["quantity"] += 1
            class_statistics[class_name][
                "confidence_sum"
            ] += confidence_value
            class_statistics[class_name][
                "min_confidence"
            ] = min(
                class_statistics[class_name][
                    "min_confidence"
                ],
                confidence_value,
            )
            class_statistics[class_name][
                "max_confidence"
            ] = max(
                class_statistics[class_name][
                    "max_confidence"
                ],
                confidence_value,
            )

            total_detections += 1
            total_confidence_sum += confidence_value

    rows: list[dict[str, Any]] = []

    for class_name in sorted(class_statistics):
        values = class_statistics[class_name]
        quantity = int(values["quantity"])
        confidence_sum = float(values["confidence_sum"])

        rows.append(
            {
                "class_name": class_name,
                "quantity": quantity,
                "confidence_sum": confidence_sum,
                "avg_confidence": (
                    confidence_sum / quantity
                    if quantity > 0
                    else 0.0
                ),
                "min_confidence": float(
                    values["min_confidence"]
                ),
                "max_confidence": float(
                    values["max_confidence"]
                ),
                "confidence_threshold": float(
                    values[
                        "confidence_threshold"
                    ]
                ),
            }
        )

    return rows, total_detections, total_confidence_sum


def extract_predictions_for_evaluation(
    result: Any,
    raw_names: dict[int, str],
    display_names: dict[int, str],
) -> list[dict[str, Any]]:
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return []

    xyxyn_values = (
        boxes.xyxyn
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    class_ids = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )
    confidences = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )

    predictions: list[dict[str, Any]] = []

    for box, class_id, confidence in zip(
        xyxyn_values,
        class_ids,
        confidences,
    ):
        predictions.append(
            {
                "class_id": int(class_id),
                "raw_class_name": raw_names.get(
                    int(class_id),
                    f"unknown_{class_id}",
                ),
                "display_class_name": display_names.get(
                    int(class_id),
                    f"UnknownClass{class_id}",
                ),
                "confidence": float(confidence),
                "box": np.clip(
                    box,
                    0.0,
                    1.0,
                ).tolist(),
            }
        )

    return predictions


def run_single_inference(
    bundle: ModelBundle,
    upload: UploadRecord,
    run_directories: RunDirectories,
    image_size: int,
    candidate_confidence: float,
    class_thresholds: dict[int, float],
    fallback_confidence: float,
    iou: float,
    max_detections: int,
    device: int | str,
    show_confidence: bool,
    line_width: int,
) -> dict[str, Any]:
    started = time.perf_counter()

    original_path = (
        run_directories.original / upload.safe_name
    )
    original_path.write_bytes(upload.data)

    image_bgr = decode_image(upload.data)

    original_preview_path = (
        run_directories.original_preview
        / f"{Path(upload.safe_name).stem}.jpg"
    )
    write_jpeg(
        original_preview_path,
        resize_for_preview(image_bgr),
        PREVIEW_JPEG_QUALITY,
    )

    with bundle.lock:
        with torch.inference_mode():
            prediction_results = bundle.model.predict(
                source=str(original_path),
                imgsz=int(image_size),

                # Candidate floor: the lowest class threshold.
                conf=float(candidate_confidence),

                iou=float(iou),
                max_det=int(max_detections),

                end2end=False,
                agnostic_nms=False,

                rect=False,
                augment=False,
                quantize="fp32",
                compile=False,

                device=device,
                save=False,
                verbose=False,
            )

    if not prediction_results:
        raise RuntimeError(
            "The model returned no prediction result."
        )

    result = prediction_results[0]

    (
        raw_detection_count,
        kept_detection_count,
    ) = filter_result_by_class_confidence(
        result=result,
        class_thresholds=class_thresholds,
        fallback_threshold=fallback_confidence,
    )

    evaluation_predictions = (
        extract_predictions_for_evaluation(
            result=result,
            raw_names=bundle.raw_names,
            display_names=bundle.display_names,
        )
    )

    (
        class_rows,
        total_detections,
        total_confidence_sum,
    ) = extract_class_statistics(
        result=result,
        display_names=bundle.display_names,
        class_thresholds=class_thresholds,
        fallback_threshold=fallback_confidence,
    )

    annotated_bgr = draw_predictions(
        image_bgr=image_bgr,
        result=result,
        display_names=bundle.display_names,
        show_confidence=show_confidence,
        line_width=line_width,
    )

    annotated_name = (
        f"{Path(upload.safe_name).stem}_annotated.jpg"
    )
    annotated_path = (
        run_directories.annotated / annotated_name
    )
    annotated_preview_path = (
        run_directories.annotated_preview
        / annotated_name
    )

    write_jpeg(
        annotated_path,
        annotated_bgr,
        ANNOTATED_JPEG_QUALITY,
    )
    write_jpeg(
        annotated_preview_path,
        resize_for_preview(annotated_bgr),
        PREVIEW_JPEG_QUALITY,
    )

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000.0

    return {
        "display_name": upload.display_name,
        "safe_name": upload.safe_name,
        "original_path": str(original_path),
        "original_preview_path": str(
            original_preview_path
        ),
        "annotated_path": str(annotated_path),
        "annotated_preview_path": str(
            annotated_preview_path
        ),
        "annotated_download_name": annotated_name,
        "class_rows": class_rows,
        "predictions": evaluation_predictions,
        "raw_detections_before_class_filter": (
            raw_detection_count
        ),
        "detections_removed_by_class_filter": (
            raw_detection_count
            - kept_detection_count
        ),
        "total_detections": total_detections,
        "confidence_sum": total_confidence_sum,
        "avg_confidence": (
            total_confidence_sum / total_detections
            if total_detections > 0
            else 0.0
        ),
        "inference_ms": elapsed_ms,
        "sha256": upload.sha256,
        "status": "SUCCESS",
        "error": "",
    }


def build_error_result(
    upload: UploadRecord,
    error_message: str,
) -> dict[str, Any]:
    return {
        "display_name": upload.display_name,
        "safe_name": upload.safe_name,
        "original_path": "",
        "original_preview_path": "",
        "annotated_path": "",
        "annotated_preview_path": "",
        "annotated_download_name": "",
        "class_rows": [],
        "predictions": [],
        "raw_detections_before_class_filter": 0,
        "detections_removed_by_class_filter": 0,
        "total_detections": 0,
        "confidence_sum": 0.0,
        "avg_confidence": 0.0,
        "inference_ms": 0.0,
        "sha256": upload.sha256,
        "status": "ERROR",
        "error": error_message,
    }



# =========================================================
# 7. OPTIONAL GROUND TRUTH EVALUATION
# =========================================================
GROUNDTRUTH_PER_CLASS_COLUMNS = [
    "class_id",
    "raw_class_name",
    "class_name",
    "declared_in_groundtruth",
    "groundtruth_count",
    "prediction_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "count_groundtruth",
    "count_prediction",
    "count_matched",
    "count_coverage",
    "count_accuracy",
]

GROUNDTRUTH_PER_IMAGE_COLUMNS = [
    "image_name",
    "groundtruth_objects",
    "predicted_objects",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "count_groundtruth",
    "count_prediction",
    "count_matched",
    "count_coverage",
    "count_accuracy",
    "count_exact",
]

GROUNDTRUTH_COUNT_COMPARISON_COLUMNS = [
    "image_name",
    "class_id",
    "raw_class_name",
    "class_name",
    "predicted_quantity",
    "groundtruth_quantity",
    "difference",
    "absolute_error",
    "count_match",
]

GROUNDTRUTH_UNMATCHED_COLUMNS = [
    "image_name",
    "reason",
]


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    return (
        float(numerator) / float(denominator)
        if denominator > 0
        else 0.0
    )


def calculate_f1(
    precision: float,
    recall: float,
) -> float:
    return safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )


def calculate_iou(
    box_a: np.ndarray,
    box_b: np.ndarray,
) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0,
        float(box_a[3] - box_a[1]),
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0,
        float(box_b[3] - box_b[1]),
    )
    union = area_a + area_b - intersection

    return intersection / union if union > 0.0 else 0.0


def load_yaml_config(
    data_yaml_path: Path,
) -> dict[str, Any]:
    with data_yaml_path.open(
        "r",
        encoding="utf-8",
    ) as yaml_file:
        config = yaml.safe_load(yaml_file) or {}

    if not isinstance(config, dict):
        raise ValueError(
            "Ground Truth data.yaml must contain a YAML object."
        )

    return config


def resolve_dataset_root(
    data_yaml_path: Path,
    config: dict[str, Any],
) -> Path:
    configured_root = config.get("path")

    if configured_root in (None, ""):
        return data_yaml_path.parent.resolve()

    configured_root_path = Path(str(configured_root))

    if configured_root_path.is_absolute():
        return configured_root_path.resolve()

    return (
        data_yaml_path.parent
        / configured_root_path
    ).resolve()


def choose_groundtruth_split(
    config: dict[str, Any],
    requested_split: str,
) -> str:
    if requested_split != "auto":
        if not config.get(requested_split):
            raise KeyError(
                "Ground Truth data.yaml does not contain "
                f"the requested split: {requested_split}."
            )
        return requested_split

    for candidate in (
        "test",
        "val",
        "valid",
        "train",
    ):
        if config.get(candidate):
            return candidate

    raise KeyError(
        "Ground Truth data.yaml does not contain "
        "train, val, valid, or test."
    )


def resolve_split_image_paths(
    data_yaml_path: Path,
    config: dict[str, Any],
    split_name: str,
) -> list[Path]:
    split_value = config.get(split_name)

    if not split_value:
        return []

    data_yaml_directory = data_yaml_path.parent.resolve()
    dataset_root = resolve_dataset_root(
        data_yaml_path,
        config,
    )

    split_entries = (
        split_value
        if isinstance(split_value, list)
        else [split_value]
    )

    split_aliases = {
        "train": ["train"],
        "test": ["test"],
        "val": ["val", "valid"],
        "valid": ["valid", "val"],
    }
    aliases = split_aliases.get(
        split_name,
        [split_name],
    )

    image_paths: list[Path] = []

    for split_entry in split_entries:
        raw_split_path = Path(str(split_entry))
        candidates: list[Path] = []

        if raw_split_path.is_absolute():
            candidates.append(
                raw_split_path.resolve()
            )
        else:
            candidates.extend(
                [
                    (
                        dataset_root
                        / raw_split_path
                    ).resolve(),
                    (
                        data_yaml_directory
                        / raw_split_path
                    ).resolve(),
                ]
            )

            cleaned_parts = [
                part
                for part in raw_split_path.parts
                if part not in {
                    ".",
                    "..",
                    "",
                }
            ]

            if cleaned_parts:
                cleaned_path = Path(*cleaned_parts)
                candidates.extend(
                    [
                        (
                            data_yaml_directory
                            / cleaned_path
                        ).resolve(),
                        (
                            dataset_root
                            / cleaned_path
                        ).resolve(),
                    ]
                )

            for alias in aliases:
                candidates.extend(
                    [
                        (
                            data_yaml_directory
                            / alias
                            / "images"
                        ).resolve(),
                        (
                            dataset_root
                            / alias
                            / "images"
                        ).resolve(),
                    ]
                )

        unique_candidates: list[Path] = []
        seen_candidates: set[str] = set()

        for candidate in candidates:
            candidate_key = str(candidate)

            if candidate_key not in seen_candidates:
                seen_candidates.add(candidate_key)
                unique_candidates.append(candidate)

        resolved_split_path: Path | None = None

        for candidate in unique_candidates:
            if candidate.exists():
                resolved_split_path = candidate
                break

        if resolved_split_path is None:
            for images_directory in data_yaml_directory.rglob(
                "images"
            ):
                if (
                    images_directory.is_dir()
                    and images_directory.parent.name.lower()
                    in {
                        alias.lower()
                        for alias in aliases
                    }
                ):
                    resolved_split_path = (
                        images_directory.resolve()
                    )
                    break

        if resolved_split_path is None:
            continue

        if resolved_split_path.is_dir():
            image_paths.extend(
                path
                for path in resolved_split_path.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in VALID_EXTENSIONS
                )
            )

        elif (
            resolved_split_path.is_file()
            and resolved_split_path.suffix.lower()
            == ".txt"
        ):
            for raw_line in (
                resolved_split_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ):
                image_path_text = raw_line.strip()

                if not image_path_text:
                    continue

                image_path = Path(image_path_text)

                if not image_path.is_absolute():
                    possible_paths = [
                        (
                            resolved_split_path.parent
                            / image_path
                        ).resolve(),
                        (
                            data_yaml_directory
                            / image_path
                        ).resolve(),
                        (
                            dataset_root
                            / image_path
                        ).resolve(),
                    ]
                    image_path = next(
                        (
                            candidate
                            for candidate in possible_paths
                            if candidate.is_file()
                        ),
                        possible_paths[0],
                    )

                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in VALID_EXTENSIONS
                ):
                    image_paths.append(image_path)

    return sorted(
        set(image_paths),
        key=lambda path: str(path).lower(),
    )


def image_path_to_label_path(
    image_path: Path,
) -> Path:
    parts = list(image_path.parts)
    image_indexes = [
        index
        for index, part in enumerate(parts)
        if part.lower() == "images"
    ]

    if not image_indexes:
        raise ValueError(
            "The Ground Truth image path does not contain "
            f"an images directory: {image_path}"
        )

    parts[image_indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo_groundtruth(
    label_path: Path,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []

    if not label_path.exists():
        return objects

    for line_number, raw_line in enumerate(
        label_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        columns = line.split()

        if len(columns) != 5:
            raise ValueError(
                f"{label_path.name}:{line_number} must have "
                "five YOLO fields."
            )

        class_id = int(float(columns[0]))

        if class_id not in class_names:
            raise ValueError(
                f"{label_path.name}:{line_number} contains "
                f"unknown class ID {class_id}."
            )

        x_center, y_center, width, height = map(
            float,
            columns[1:],
        )

        if not all(
            0.0 <= value <= 1.0
            for value in (
                x_center,
                y_center,
                width,
                height,
            )
        ):
            raise ValueError(
                f"{label_path.name}:{line_number} has "
                "coordinates outside [0, 1]."
            )

        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"{label_path.name}:{line_number} has "
                "non-positive width or height."
            )

        objects.append(
            {
                "class_id": class_id,
                "raw_class_name": class_names[class_id],
                "box": [
                    max(
                        0.0,
                        x_center - width / 2.0,
                    ),
                    max(
                        0.0,
                        y_center - height / 2.0,
                    ),
                    min(
                        1.0,
                        x_center + width / 2.0,
                    ),
                    min(
                        1.0,
                        y_center + height / 2.0,
                    ),
                ],
            }
        )

    return objects


def validate_groundtruth_class_scope(
    model_names: dict[int, str],
    groundtruth_names: dict[int, str],
) -> dict[str, Any]:
    model_set = set(model_names.values())
    groundtruth_set = set(
        groundtruth_names.values()
    )

    missing_in_groundtruth = sorted(
        model_set - groundtruth_set
    )
    extra_in_groundtruth = sorted(
        groundtruth_set - model_set
    )

    if extra_in_groundtruth:
        raise RuntimeError(
            "Ground Truth contains classes that do not exist "
            "in the checkpoint: "
            + " | ".join(extra_in_groundtruth)
        )

    return {
        "model_class_count": len(model_set),
        "groundtruth_class_count": len(
            groundtruth_set
        ),
        "coverage_ratio": (
            len(groundtruth_set) / len(model_set)
            if model_set
            else 0.0
        ),
        "missing_in_groundtruth": (
            missing_in_groundtruth
        ),
        "extra_in_groundtruth": (
            extra_in_groundtruth
        ),
    }


def safe_extract_groundtruth_zip(
    zip_bytes: bytes,
    destination: Path,
) -> Path:
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(
        io.BytesIO(zip_bytes),
        mode="r",
    ) as archive:
        members = archive.infolist()

        if len(members) > MAX_GT_ZIP_FILES:
            raise ValueError(
                "Ground Truth ZIP contains too many files."
            )

        total_size = sum(
            member.file_size
            for member in members
        )

        if total_size > MAX_GT_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError(
                "Ground Truth ZIP is above the allowed "
                "uncompressed size."
            )

        for member in members:
            member_path = (
                destination
                / member.filename
            ).resolve()

            try:
                member_path.relative_to(
                    destination_resolved
                )
            except ValueError as error:
                raise ValueError(
                    "Ground Truth ZIP contains an unsafe path."
                ) from error

        archive.extractall(destination)

    yaml_candidates = sorted(
        destination.rglob("data.yaml"),
        key=lambda path: (
            len(path.parts),
            str(path).lower(),
        ),
    )

    if not yaml_candidates:
        raise FileNotFoundError(
            "The uploaded Ground Truth ZIP does not "
            "contain data.yaml."
        )

    return yaml_candidates[0].resolve()


def prepare_groundtruth_yaml(
    run_directory: Path,
    source_mode: str,
    local_yaml_path: str,
    uploaded_zip_bytes: bytes | None,
) -> Path:
    if source_mode == "Local data.yaml":
        resolved_path = Path(
            local_yaml_path
        ).expanduser().resolve()

        if not resolved_path.is_file():
            raise FileNotFoundError(
                "Ground Truth data.yaml was not found: "
                f"{resolved_path}"
            )

        return resolved_path

    if not uploaded_zip_bytes:
        raise ValueError(
            "Please upload a Ground Truth YOLO ZIP file."
        )

    return safe_extract_groundtruth_zip(
        uploaded_zip_bytes,
        run_directory / "groundtruth_source",
    )


def build_groundtruth_bundle(
    data_yaml_path: Path,
    requested_split: str,
    model_names: dict[int, str],
) -> GroundTruthBundle:
    config = load_yaml_config(
        data_yaml_path
    )
    raw_names = config.get("names")

    if raw_names is None:
        raise KeyError(
            "Ground Truth data.yaml is missing names."
        )

    class_names = normalize_class_names(
        raw_names
    )
    class_scope = validate_groundtruth_class_scope(
        model_names,
        class_names,
    )
    split_name = choose_groundtruth_split(
        config,
        requested_split,
    )
    image_paths = resolve_split_image_paths(
        data_yaml_path,
        config,
        split_name,
    )

    if not image_paths:
        raise FileNotFoundError(
            "No Ground Truth images were found "
            f"in split '{split_name}'."
        )

    by_filename: dict[
        str,
        list[dict[str, Any]],
    ] = {}
    by_stem: dict[
        str,
        list[dict[str, Any]],
    ] = {}
    by_sha256: dict[
        str,
        list[dict[str, Any]],
    ] = {}
    duplicate_filenames: set[str] = set()
    duplicate_stems: set[str] = set()
    duplicate_hashes: set[str] = set()

    for image_path in image_paths:
        label_path = image_path_to_label_path(
            image_path
        )
        objects = read_yolo_groundtruth(
            label_path,
            class_names,
        )

        filename_key = image_path.name.lower()
        stem_keys = {
            image_path.stem.lower(),
            canonical_image_stem(image_path.name),
        }
        image_hash = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        if filename_key in by_filename:
            duplicate_filenames.add(filename_key)
        else:
            by_filename[filename_key] = objects

        for stem_key in stem_keys:
            if not stem_key:
                continue

            if stem_key in by_stem:
                duplicate_stems.add(stem_key)
            else:
                by_stem[stem_key] = objects

        if image_hash in by_sha256:
            duplicate_hashes.add(image_hash)
        else:
            by_sha256[image_hash] = objects

    for key in duplicate_filenames:
        by_filename.pop(key, None)

    for key in duplicate_stems:
        by_stem.pop(key, None)

    for key in duplicate_hashes:
        by_sha256.pop(key, None)

    return GroundTruthBundle(
        data_yaml_path=data_yaml_path,
        split_name=split_name,
        class_names=class_names,
        image_paths=image_paths,
        objects_by_filename=by_filename,
        objects_by_stem=by_stem,
        objects_by_sha256=by_sha256,
        class_scope=class_scope,
    )


def find_groundtruth_for_result(
    result: dict[str, Any],
    groundtruth: GroundTruthBundle,
) -> list[dict[str, Any]] | None:
    image_hash = result.get("sha256")

    if (
        image_hash
        and image_hash
        in groundtruth.objects_by_sha256
    ):
        return groundtruth.objects_by_sha256[
            image_hash
        ]

    candidate_names = [
        str(result.get("display_name", "")),
        str(result.get("safe_name", "")),
    ]

    for candidate_name in candidate_names:
        if not candidate_name:
            continue

        basename = Path(
            candidate_name.replace("\\", "/")
        ).name.lower()

        if basename in groundtruth.objects_by_filename:
            return groundtruth.objects_by_filename[
                basename
            ]

        stem_candidates = {
            Path(basename).stem.lower(),
            canonical_image_stem(basename),
        }

        for stem in stem_candidates:
            if (
                stem
                and stem
                in groundtruth.objects_by_stem
            ):
                return groundtruth.objects_by_stem[
                    stem
                ]

    return None


def match_predictions_by_class(
    predictions: list[dict[str, Any]],
    groundtruths: list[dict[str, Any]],
    model_names: dict[int, str],
    match_iou: float,
) -> dict[int, dict[str, int]]:
    counts = {
        class_id: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "gt": 0,
            "pred": 0,
        }
        for class_id in model_names
    }

    for class_id, class_name in model_names.items():
        class_predictions = [
            prediction
            for prediction in predictions
            if prediction["raw_class_name"]
            == class_name
        ]
        class_groundtruths = [
            groundtruth
            for groundtruth in groundtruths
            if groundtruth["raw_class_name"]
            == class_name
        ]

        class_predictions.sort(
            key=lambda item: item[
                "confidence"
            ],
            reverse=True,
        )

        matched_gt_indexes: set[int] = set()
        counts[class_id]["gt"] = len(
            class_groundtruths
        )
        counts[class_id]["pred"] = len(
            class_predictions
        )

        for prediction in class_predictions:
            best_iou = 0.0
            best_gt_index: int | None = None
            prediction_box = np.asarray(
                prediction["box"],
                dtype=float,
            )

            for gt_index, groundtruth in enumerate(
                class_groundtruths
            ):
                if gt_index in matched_gt_indexes:
                    continue

                current_iou = calculate_iou(
                    prediction_box,
                    np.asarray(
                        groundtruth["box"],
                        dtype=float,
                    ),
                )

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_index = gt_index

            if (
                best_gt_index is not None
                and best_iou >= match_iou
            ):
                counts[class_id]["tp"] += 1
                matched_gt_indexes.add(
                    best_gt_index
                )
            else:
                counts[class_id]["fp"] += 1

        counts[class_id]["fn"] = (
            len(class_groundtruths)
            - len(matched_gt_indexes)
        )

    return counts


def evaluate_results_against_groundtruth(
    results: list[dict[str, Any]],
    groundtruth: GroundTruthBundle,
    model_names: dict[int, str],
    display_names: dict[int, str],
    match_iou: float,
) -> dict[str, Any]:
    model_id_by_name = {
        class_name: class_id
        for class_id, class_name
        in model_names.items()
    }
    display_name_by_raw = {
        model_names[class_id]: (
            display_names[class_id]
        )
        for class_id in model_names
    }

    total_class_counts = {
        class_id: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "gt": 0,
            "pred": 0,
        }
        for class_id in model_names
    }

    count_comparison_rows: list[
        dict[str, Any]
    ] = []
    per_image_rows: list[
        dict[str, Any]
    ] = []
    unmatched_rows: list[
        dict[str, Any]
    ] = []

    matched_image_count = 0

    for result in results:
        if result["status"] != "SUCCESS":
            continue

        groundtruth_objects = (
            find_groundtruth_for_result(
                result,
                groundtruth,
            )
        )

        if groundtruth_objects is None:
            unmatched_rows.append(
                {
                    "image_name": result[
                        "display_name"
                    ],
                    "reason": (
                        "No unique Ground Truth match "
                        "by SHA-256, filename, or stem."
                    ),
                }
            )
            continue

        matched_image_count += 1
        predictions = result.get(
            "predictions",
            [],
        )

        prediction_counts = Counter(
            prediction["raw_class_name"]
            for prediction in predictions
        )
        groundtruth_counts = Counter(
            obj["raw_class_name"]
            for obj in groundtruth_objects
        )

        count_class_names = (
            set(prediction_counts)
            | set(groundtruth_counts)
        )

        count_matched = sum(
            min(
                prediction_counts.get(
                    class_name,
                    0,
                ),
                groundtruth_counts.get(
                    class_name,
                    0,
                ),
            )
            for class_name in count_class_names
        )
        count_groundtruth = sum(
            groundtruth_counts.values()
        )
        count_prediction = sum(
            prediction_counts.values()
        )
        count_union = sum(
            max(
                prediction_counts.get(
                    class_name,
                    0,
                ),
                groundtruth_counts.get(
                    class_name,
                    0,
                ),
            )
            for class_name in count_class_names
        )

        count_coverage = (
            count_matched / count_groundtruth
            if count_groundtruth > 0
            else (
                1.0
                if count_prediction == 0
                else 0.0
            )
        )
        count_accuracy = (
            count_matched / count_union
            if count_union > 0
            else 1.0
        )

        for class_name in sorted(
            count_class_names
        ):
            predicted_quantity = (
                prediction_counts.get(
                    class_name,
                    0,
                )
            )
            groundtruth_quantity = (
                groundtruth_counts.get(
                    class_name,
                    0,
                )
            )
            class_id = model_id_by_name.get(
                class_name
            )

            count_comparison_rows.append(
                {
                    "image_name": result[
                        "display_name"
                    ],
                    "class_id": class_id,
                    "raw_class_name": class_name,
                    "class_name": (
                        display_name_by_raw.get(
                            class_name,
                            class_name,
                        )
                    ),
                    "predicted_quantity": (
                        predicted_quantity
                    ),
                    "groundtruth_quantity": (
                        groundtruth_quantity
                    ),
                    "difference": (
                        predicted_quantity
                        - groundtruth_quantity
                    ),
                    "absolute_error": abs(
                        predicted_quantity
                        - groundtruth_quantity
                    ),
                    "count_match": (
                        predicted_quantity
                        == groundtruth_quantity
                    ),
                }
            )

        image_class_counts = (
            match_predictions_by_class(
                predictions=predictions,
                groundtruths=(
                    groundtruth_objects
                ),
                model_names=model_names,
                match_iou=match_iou,
            )
        )

        for class_id, metrics in (
            image_class_counts.items()
        ):
            for metric_name, value in (
                metrics.items()
            ):
                total_class_counts[class_id][
                    metric_name
                ] += value

        image_tp = sum(
            metrics["tp"]
            for metrics
            in image_class_counts.values()
        )
        image_fp = sum(
            metrics["fp"]
            for metrics
            in image_class_counts.values()
        )
        image_fn = sum(
            metrics["fn"]
            for metrics
            in image_class_counts.values()
        )

        precision = safe_divide(
            image_tp,
            image_tp + image_fp,
        )
        recall = safe_divide(
            image_tp,
            image_tp + image_fn,
        )

        per_image_rows.append(
            {
                "image_name": result[
                    "display_name"
                ],
                "groundtruth_objects": (
                    len(groundtruth_objects)
                ),
                "predicted_objects": (
                    len(predictions)
                ),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "precision": precision,
                "recall": recall,
                "f1": calculate_f1(
                    precision,
                    recall,
                ),
                "count_groundtruth": (
                    count_groundtruth
                ),
                "count_prediction": (
                    count_prediction
                ),
                "count_matched": (
                    count_matched
                ),
                "count_coverage": (
                    count_coverage
                ),
                "count_accuracy": (
                    count_accuracy
                ),
                "count_exact": (
                    prediction_counts
                    == groundtruth_counts
                ),
            }
        )

    per_class_rows: list[
        dict[str, Any]
    ] = []

    for class_id, raw_class_name in (
        model_names.items()
    ):
        metrics = total_class_counts[
            class_id
        ]
        class_count_rows = [
            row
            for row in count_comparison_rows
            if row["raw_class_name"]
            == raw_class_name
        ]
        class_count_groundtruth = sum(
            int(row["groundtruth_quantity"])
            for row in class_count_rows
        )
        class_count_prediction = sum(
            int(row["predicted_quantity"])
            for row in class_count_rows
        )
        class_count_matched = sum(
            min(
                int(row[
                    "predicted_quantity"
                ]),
                int(row[
                    "groundtruth_quantity"
                ]),
            )
            for row in class_count_rows
        )
        class_count_union = sum(
            max(
                int(row[
                    "predicted_quantity"
                ]),
                int(row[
                    "groundtruth_quantity"
                ]),
            )
            for row in class_count_rows
        )

        precision = safe_divide(
            metrics["tp"],
            metrics["tp"] + metrics["fp"],
        )
        recall = safe_divide(
            metrics["tp"],
            metrics["tp"] + metrics["fn"],
        )

        per_class_rows.append(
            {
                "class_id": class_id,
                "raw_class_name": raw_class_name,
                "class_name": display_names[
                    class_id
                ],
                "declared_in_groundtruth": (
                    raw_class_name
                    in set(
                        groundtruth.class_names.values()
                    )
                ),
                "groundtruth_count": (
                    metrics["gt"]
                ),
                "prediction_count": (
                    metrics["pred"]
                ),
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": precision,
                "recall": recall,
                "f1": calculate_f1(
                    precision,
                    recall,
                ),
                "count_groundtruth": (
                    class_count_groundtruth
                ),
                "count_prediction": (
                    class_count_prediction
                ),
                "count_matched": (
                    class_count_matched
                ),
                "count_coverage": (
                    class_count_matched
                    / class_count_groundtruth
                    if class_count_groundtruth > 0
                    else np.nan
                ),
                "count_accuracy": (
                    class_count_matched
                    / class_count_union
                    if class_count_union > 0
                    else np.nan
                ),
            }
        )

    overall_tp = sum(
        metrics["tp"]
        for metrics
        in total_class_counts.values()
    )
    overall_fp = sum(
        metrics["fp"]
        for metrics
        in total_class_counts.values()
    )
    overall_fn = sum(
        metrics["fn"]
        for metrics
        in total_class_counts.values()
    )
    overall_gt = sum(
        metrics["gt"]
        for metrics
        in total_class_counts.values()
    )
    overall_pred = sum(
        metrics["pred"]
        for metrics
        in total_class_counts.values()
    )

    micro_precision = safe_divide(
        overall_tp,
        overall_tp + overall_fp,
    )
    micro_recall = safe_divide(
        overall_tp,
        overall_tp + overall_fn,
    )

    classes_with_gt = [
        row
        for row in per_class_rows
        if row["groundtruth_count"] > 0
    ]
    valid_count_rows = (
        count_comparison_rows
    )

    overall_count_groundtruth = sum(
        int(row["groundtruth_quantity"])
        for row in valid_count_rows
    )
    overall_count_prediction = sum(
        int(row["predicted_quantity"])
        for row in valid_count_rows
    )
    overall_count_matched = sum(
        min(
            int(row["predicted_quantity"]),
            int(row["groundtruth_quantity"]),
        )
        for row in valid_count_rows
    )
    overall_count_union = sum(
        max(
            int(row["predicted_quantity"]),
            int(row["groundtruth_quantity"]),
        )
        for row in valid_count_rows
    )

    overall = {
        "matched_images": matched_image_count,
        "unmatched_images": len(
            unmatched_rows
        ),
        "groundtruth_dataset_images": len(
            groundtruth.image_paths
        ),
        "model_class_count": (
            groundtruth.class_scope[
                "model_class_count"
            ]
        ),
        "groundtruth_class_count": (
            groundtruth.class_scope[
                "groundtruth_class_count"
            ]
        ),
        "groundtruth_class_coverage": (
            groundtruth.class_scope[
                "coverage_ratio"
            ]
        ),
        "missing_model_classes": (
            groundtruth.class_scope[
                "missing_in_groundtruth"
            ]
        ),
        "groundtruth_objects": overall_gt,
        "prediction_objects": overall_pred,
        "tp": overall_tp,
        "fp": overall_fp,
        "fn": overall_fn,
        "micro_precision": (
            micro_precision
        ),
        "micro_recall": (
            micro_recall
        ),
        "micro_f1": calculate_f1(
            micro_precision,
            micro_recall,
        ),
        "macro_precision": (
            sum(
                row["precision"]
                for row in classes_with_gt
            )
            / len(classes_with_gt)
            if classes_with_gt
            else 0.0
        ),
        "macro_recall": (
            sum(
                row["recall"]
                for row in classes_with_gt
            )
            / len(classes_with_gt)
            if classes_with_gt
            else 0.0
        ),
        "macro_f1": (
            sum(
                row["f1"]
                for row in classes_with_gt
            )
            / len(classes_with_gt)
            if classes_with_gt
            else 0.0
        ),
        "count_groundtruth": (
            overall_count_groundtruth
        ),
        "count_prediction": (
            overall_count_prediction
        ),
        "count_matched": (
            overall_count_matched
        ),
        "overall_count_coverage": (
            overall_count_matched
            / overall_count_groundtruth
            if overall_count_groundtruth > 0
            else 0.0
        ),
        "overall_count_accuracy": (
            overall_count_matched
            / overall_count_union
            if overall_count_union > 0
            else 1.0
        ),
        "mean_image_count_coverage": (
            sum(
                row["count_coverage"]
                for row in per_image_rows
            )
            / len(per_image_rows)
            if per_image_rows
            else 0.0
        ),
        "mean_image_count_accuracy": (
            sum(
                row["count_accuracy"]
                for row in per_image_rows
            )
            / len(per_image_rows)
            if per_image_rows
            else 0.0
        ),
        "image_exact_count_rate": (
            sum(
                1
                for row in per_image_rows
                if row["count_exact"]
            )
            / len(per_image_rows)
            if per_image_rows
            else 0.0
        ),
        "count_mae": (
            sum(
                row["absolute_error"]
                for row in valid_count_rows
            )
            / len(valid_count_rows)
            if valid_count_rows
            else 0.0
        ),
        "match_iou": match_iou,
        "split": groundtruth.split_name,
        "data_yaml": str(
            groundtruth.data_yaml_path
        ),
    }

    return {
        "overall": overall,
        "overall_df": pd.DataFrame(
            [overall]
        ),
        "per_class_df": pd.DataFrame(
            per_class_rows,
            columns=GROUNDTRUTH_PER_CLASS_COLUMNS,
        ),
        "per_image_df": pd.DataFrame(
            per_image_rows,
            columns=GROUNDTRUTH_PER_IMAGE_COLUMNS,
        ),
        "count_comparison_df": pd.DataFrame(
            count_comparison_rows,
            columns=(
                GROUNDTRUTH_COUNT_COMPARISON_COLUMNS
            ),
        ),
        "unmatched_df": pd.DataFrame(
            unmatched_rows,
            columns=GROUNDTRUTH_UNMATCHED_COLUMNS,
        ),
    }


def create_aligned_evaluation_dataset(
    run_directory: Path,
    results: list[dict[str, Any]],
    groundtruth: GroundTruthBundle,
    model_names: dict[int, str],
) -> tuple[Path, str]:
    aligned_root = (
        run_directory
        / "groundtruth_aligned"
    )
    images_dir = (
        aligned_root
        / "test"
        / "images"
    )
    labels_dir = (
        aligned_root
        / "test"
        / "labels"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_id_by_name = {
        class_name: class_id
        for class_id, class_name
        in model_names.items()
    }
    used_names: set[str] = set()
    copied_count = 0

    for result in results:
        if result["status"] != "SUCCESS":
            continue

        gt_objects = find_groundtruth_for_result(
            result,
            groundtruth,
        )

        if gt_objects is None:
            continue

        source_image = Path(
            result["original_path"]
        )

        if not source_image.is_file():
            continue

        destination_name = (
            source_image.name
        )
        destination_key = (
            destination_name.lower()
        )

        if destination_key in used_names:
            destination_name = (
                f"{source_image.stem}_"
                f"{result['sha256'][:8]}"
                f"{source_image.suffix.lower()}"
            )

        used_names.add(
            destination_name.lower()
        )
        destination_image = (
            images_dir
            / destination_name
        )
        shutil.copy2(
            source_image,
            destination_image,
        )

        label_lines: list[str] = []

        for obj in gt_objects:
            raw_class_name = obj[
                "raw_class_name"
            ]
            model_class_id = (
                model_id_by_name[
                    raw_class_name
                ]
            )
            x1, y1, x2, y2 = map(
                float,
                obj["box"],
            )
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width / 2.0
            y_center = y1 + height / 2.0

            label_lines.append(
                (
                    f"{model_class_id} "
                    f"{x_center:.10f} "
                    f"{y_center:.10f} "
                    f"{width:.10f} "
                    f"{height:.10f}"
                )
            )

        (
            labels_dir
            / Path(destination_name)
            .with_suffix(".txt")
            .name
        ).write_text(
            (
                "\n".join(label_lines)
                + ("\n" if label_lines else "")
            ),
            encoding="utf-8",
        )
        copied_count += 1

    if copied_count == 0:
        raise RuntimeError(
            "No uploaded image could be matched to "
            "Ground Truth for official evaluation."
        )

    aligned_yaml = aligned_root / "data.yaml"
    split_path = "test/images"
    aligned_config = {
        "path": str(aligned_root.resolve()),
        "train": split_path,
        "val": split_path,
        "test": split_path,
        "nc": len(model_names),
        "names": {
            class_id: class_name
            for class_id, class_name
            in model_names.items()
        },
    }
    aligned_yaml.write_text(
        yaml.safe_dump(
            aligned_config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return aligned_yaml, "test"


def run_official_groundtruth_metrics(
    bundle: ModelBundle,
    run_directory: Path,
    results: list[dict[str, Any]],
    groundtruth: GroundTruthBundle,
    image_size: int,
    iou: float,
    max_detections: int,
    device: int | str,
    batch_size: int,
) -> dict[str, Any]:
    aligned_yaml, aligned_split = (
        create_aligned_evaluation_dataset(
            run_directory=run_directory,
            results=results,
            groundtruth=groundtruth,
            model_names=bundle.raw_names,
        )
    )

    with bundle.lock:
        validation_metrics = (
            bundle.model.val(
                data=str(aligned_yaml),
                split=aligned_split,
                imgsz=int(image_size),
                batch=int(batch_size),
                device=device,
                workers=0,
                conf=0.001,
                end2end=False,
                iou=float(iou),
                max_det=int(max_detections),
                agnostic_nms=False,
                quantize="fp32",
                plots=True,
                verbose=False,
                project=str(
                    run_directory
                    / "official_evaluation"
                ),
                name="groundtruth_metrics",
                exist_ok=True,
            )
        )

    box_metrics = validation_metrics.box
    precision = float(
        getattr(box_metrics, "mp", 0.0)
    )
    recall = float(
        getattr(box_metrics, "mr", 0.0)
    )
    overall = {
        "precision": precision,
        "recall": recall,
        "f1": calculate_f1(
            precision,
            recall,
        ),
        "map50": float(
            getattr(box_metrics, "map50", 0.0)
        ),
        "map75": float(
            getattr(box_metrics, "map75", 0.0)
        ),
        "map50_95": float(
            getattr(box_metrics, "map", 0.0)
        ),
        "validation_confidence": 0.001,
        "split": aligned_split,
        "results_folder": str(
            validation_metrics.save_dir
        ),
    }

    ap_class_indexes = np.asarray(
        getattr(
            box_metrics,
            "ap_class_index",
            [],
        ),
        dtype=int,
    )
    precision_array = np.asarray(
        getattr(box_metrics, "p", []),
        dtype=float,
    )
    recall_array = np.asarray(
        getattr(box_metrics, "r", []),
        dtype=float,
    )
    ap50_array = np.asarray(
        getattr(box_metrics, "ap50", []),
        dtype=float,
    )
    ap_array = np.asarray(
        getattr(box_metrics, "ap", []),
        dtype=float,
    )

    metric_index_by_class_id = {
        int(class_id): metric_index
        for metric_index, class_id
        in enumerate(ap_class_indexes)
    }
    per_class_rows: list[
        dict[str, Any]
    ] = []

    for class_id, raw_class_name in (
        bundle.raw_names.items()
    ):
        metric_index = (
            metric_index_by_class_id.get(
                class_id
            )
        )

        if metric_index is None:
            per_class_rows.append(
                {
                    "class_id": class_id,
                    "raw_class_name": (
                        raw_class_name
                    ),
                    "class_name": (
                        bundle.display_names[
                            class_id
                        ]
                    ),
                    "evaluation_status": (
                        "NO_GROUNDTRUTH_INSTANCES"
                    ),
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "ap50": np.nan,
                    "ap50_95": np.nan,
                }
            )
            continue

        class_precision = float(
            precision_array[metric_index]
        )
        class_recall = float(
            recall_array[metric_index]
        )

        per_class_rows.append(
            {
                "class_id": class_id,
                "raw_class_name": (
                    raw_class_name
                ),
                "class_name": (
                    bundle.display_names[
                        class_id
                    ]
                ),
                "evaluation_status": (
                    "EVALUATED"
                ),
                "precision": class_precision,
                "recall": class_recall,
                "f1": calculate_f1(
                    class_precision,
                    class_recall,
                ),
                "ap50": float(
                    ap50_array[
                        metric_index
                    ]
                ),
                "ap50_95": float(
                    ap_array[
                        metric_index
                    ]
                ),
            }
        )

    return {
        "overall": overall,
        "overall_df": pd.DataFrame(
            [overall]
        ),
        "per_class_df": pd.DataFrame(
            per_class_rows
        ),
    }


def write_optional_evaluation_reports(
    reports_directory: Path,
    evaluation_payload: dict[str, Any],
) -> list[Path]:
    report_paths: list[Path] = []
    dataframe_files = {
        "groundtruth_overall.csv": (
            evaluation_payload[
                "overall_df"
            ]
        ),
        "groundtruth_per_class.csv": (
            evaluation_payload[
                "per_class_df"
            ]
        ),
        "groundtruth_per_image.csv": (
            evaluation_payload[
                "per_image_df"
            ]
        ),
        "groundtruth_count_comparison.csv": (
            evaluation_payload[
                "count_comparison_df"
            ]
        ),
        "groundtruth_unmatched_images.csv": (
            evaluation_payload[
                "unmatched_df"
            ]
        ),
    }

    official = evaluation_payload.get(
        "official"
    )

    if official:
        dataframe_files[
            "official_metrics_overall.csv"
        ] = official["overall_df"]
        dataframe_files[
            "official_metrics_per_class.csv"
        ] = official["per_class_df"]

    for filename, dataframe in (
        dataframe_files.items()
    ):
        output_path = (
            reports_directory
            / filename
        )
        dataframe.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        report_paths.append(output_path)

    return report_paths


def render_groundtruth_evaluation_dashboard(
    evaluation_payload: dict[str, Any],
) -> None:
    overall = evaluation_payload["overall"]

    if overall.get("matched_images", 0) == 0:
        render_message(
            (
                "No uploaded image could be matched to Ground Truth. "
                "Use the exact images from the Ground Truth dataset, "
                "or verify the filename and Roboflow export naming."
            ),
            "error",
        )

    st.markdown(
        '<div class="sb-section-title">'
        "Ground Truth evaluation"
        "</div>",
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    with metric_1:
        st.metric(
            "Matched images",
            overall["matched_images"],
        )

    with metric_2:
        st.metric(
            "Ground Truth objects",
            overall["groundtruth_objects"],
        )

    with metric_3:
        st.metric(
            "Predicted objects",
            overall["prediction_objects"],
        )

    with metric_4:
        st.metric(
            "Class coverage",
            (
                f"{overall['groundtruth_class_count']}/"
                f"{overall['model_class_count']} "
                f"({overall['groundtruth_class_coverage']:.2%})"
            ),
        )

    st.markdown(
        '<div class="sb-section-title" '
        'style="margin-top: 18px;">'
        "Fixed-threshold detection metrics"
        "</div>",
        unsafe_allow_html=True,
    )

    detection_1, detection_2, detection_3, detection_4 = (
        st.columns(4)
    )

    with detection_1:
        st.metric(
            "Micro precision",
            f"{overall['micro_precision']:.2%}",
        )

    with detection_2:
        st.metric(
            "Micro recall",
            f"{overall['micro_recall']:.2%}",
        )

    with detection_3:
        st.metric(
            "Micro F1",
            f"{overall['micro_f1']:.2%}",
        )

    with detection_4:
        st.metric(
            "TP / FP / FN",
            (
                f"{overall['tp']} / "
                f"{overall['fp']} / "
                f"{overall['fn']}"
            ),
        )

    st.markdown(
        '<div class="sb-section-title" '
        'style="margin-top: 18px;">'
        "Count-only metrics"
        "</div>",
        unsafe_allow_html=True,
    )

    count_1, count_2, count_3, count_4 = (
        st.columns(4)
    )

    with count_1:
        st.metric(
            "Overall count coverage",
            (
                f"{overall['overall_count_coverage']:.2%}"
            ),
        )

    with count_2:
        st.metric(
            "Overall count accuracy",
            (
                f"{overall['overall_count_accuracy']:.2%}"
            ),
        )

    with count_3:
        st.metric(
            "Exact image count rate",
            (
                f"{overall['image_exact_count_rate']:.2%}"
            ),
        )

    with count_4:
        st.metric(
            "Count MAE",
            f"{overall['count_mae']:.4f}",
        )

    if overall["missing_model_classes"]:
        render_message(
            (
                "Classes not represented in Ground Truth: "
                + " | ".join(
                    overall[
                        "missing_model_classes"
                    ]
                )
            ),
            "warning",
        )

    class_tab, image_tab, count_tab = st.tabs(
        [
            "Per-class metrics",
            "Per-image metrics",
            "Count comparison",
        ]
    )

    with class_tab:
        class_df = evaluation_payload[
            "per_class_df"
        ].copy()
        st.dataframe(
            class_df,
            width="stretch",
            hide_index=True,
            column_config={
                "precision": st.column_config.ProgressColumn(
                    "Precision",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
                "recall": st.column_config.ProgressColumn(
                    "Recall",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
                "f1": st.column_config.ProgressColumn(
                    "F1",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
                "count_coverage": (
                    st.column_config.ProgressColumn(
                        "Count coverage",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f",
                    )
                ),
                "count_accuracy": (
                    st.column_config.ProgressColumn(
                        "Count accuracy",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f",
                    )
                ),
            },
        )

    with image_tab:
        st.dataframe(
            evaluation_payload[
                "per_image_df"
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "precision": st.column_config.ProgressColumn(
                    "Precision",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
                "recall": st.column_config.ProgressColumn(
                    "Recall",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
                "f1": st.column_config.ProgressColumn(
                    "F1",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
                "count_coverage": (
                    st.column_config.ProgressColumn(
                        "Count coverage",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f",
                    )
                ),
                "count_accuracy": (
                    st.column_config.ProgressColumn(
                        "Count accuracy",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f",
                    )
                ),
            },
        )

    with count_tab:
        st.dataframe(
            evaluation_payload[
                "count_comparison_df"
            ],
            width="stretch",
            hide_index=True,
        )

    unmatched_df = evaluation_payload[
        "unmatched_df"
    ]

    if not unmatched_df.empty:
        st.markdown(
            '<div class="sb-section-title" '
            'style="margin-top: 18px;">'
            "Unmatched images"
            "</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            unmatched_df,
            width="stretch",
            hide_index=True,
        )

    official = evaluation_payload.get(
        "official"
    )

    if official:
        st.markdown(
            '<div class="sb-section-title" '
            'style="margin-top: 22px;">'
            "Official Ultralytics metrics"
            "</div>",
            unsafe_allow_html=True,
        )
        official_overall = official[
            "overall"
        ]
        official_1, official_2, official_3, official_4 = (
            st.columns(4)
        )

        with official_1:
            st.metric(
                "mAP50",
                f"{official_overall['map50']:.2%}",
            )

        with official_2:
            st.metric(
                "mAP75",
                f"{official_overall['map75']:.2%}",
            )

        with official_3:
            st.metric(
                "mAP50-95",
                (
                    f"{official_overall['map50_95']:.2%}"
                ),
            )

        with official_4:
            st.metric(
                "Official F1",
                f"{official_overall['f1']:.2%}",
            )

        with st.expander(
            "Official per-class metrics",
            expanded=False,
        ):
            st.dataframe(
                official["per_class_df"],
                width="stretch",
                hide_index=True,
            )


# =========================================================
# 8. REPORT BUILDERS
# =========================================================
OVERALL_COLUMNS = [
    "class_name",
    "confidence_threshold",
    "total_quantity",
    "avg_confidence",
    "min_confidence",
    "max_confidence",
    "images_detected",
]

PER_IMAGE_COLUMNS = [
    "image_name",
    "class_name",
    "confidence_threshold",
    "quantity",
    "avg_confidence",
    "min_confidence",
    "max_confidence",
]

IMAGE_SUMMARY_COLUMNS = [
    "image_name",
    "status",
    "raw_detections_before_class_filter",
    "detections_removed_by_class_filter",
    "total_detections",
    "avg_confidence",
    "inference_ms",
    "error",
]


def build_report_dataframes(
    results: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_aggregates: dict[str, dict[str, Any]] = {}
    per_image_rows: list[dict[str, Any]] = []
    image_summary_rows: list[dict[str, Any]] = []

    for result in results:
        image_summary_rows.append(
            {
                "image_name": result["display_name"],
                "status": result["status"],
                "raw_detections_before_class_filter": result[
                    "raw_detections_before_class_filter"
                ],
                "detections_removed_by_class_filter": result[
                    "detections_removed_by_class_filter"
                ],
                "total_detections": result[
                    "total_detections"
                ],
                "avg_confidence": result[
                    "avg_confidence"
                ],
                "inference_ms": result[
                    "inference_ms"
                ],
                "error": result["error"],
            }
        )

        if result["status"] != "SUCCESS":
            continue

        if not result["class_rows"]:
            per_image_rows.append(
                {
                    "image_name": result["display_name"],
                    "class_name": "NO_OBJECT_DETECTED",
                    "confidence_threshold": np.nan,
                    "quantity": 0,
                    "avg_confidence": 0.0,
                    "min_confidence": 0.0,
                    "max_confidence": 0.0,
                }
            )
            continue

        for row in result["class_rows"]:
            per_image_rows.append(
                {
                    "image_name": result["display_name"],
                    "class_name": row["class_name"],
                    "confidence_threshold": row[
                        "confidence_threshold"
                    ],
                    "quantity": row["quantity"],
                    "avg_confidence": row[
                        "avg_confidence"
                    ],
                    "min_confidence": row[
                        "min_confidence"
                    ],
                    "max_confidence": row[
                        "max_confidence"
                    ],
                }
            )

            class_name = row["class_name"]

            if class_name not in class_aggregates:
                class_aggregates[class_name] = {
                    "confidence_threshold": float(
                        row["confidence_threshold"]
                    ),
                    "quantity": 0,
                    "confidence_sum": 0.0,
                    "min_confidence": 1.0,
                    "max_confidence": 0.0,
                    "images": set(),
                }

            aggregate = class_aggregates[class_name]
            aggregate["quantity"] += int(row["quantity"])
            aggregate["confidence_sum"] += float(
                row["confidence_sum"]
            )
            aggregate["min_confidence"] = min(
                aggregate["min_confidence"],
                float(row["min_confidence"]),
            )
            aggregate["max_confidence"] = max(
                aggregate["max_confidence"],
                float(row["max_confidence"]),
            )
            aggregate["images"].add(
                result["display_name"]
            )

    overall_rows: list[dict[str, Any]] = []

    for class_name in sorted(class_aggregates):
        aggregate = class_aggregates[class_name]
        quantity = int(aggregate["quantity"])

        overall_rows.append(
            {
                "class_name": class_name,
                "confidence_threshold": aggregate[
                    "confidence_threshold"
                ],
                "total_quantity": quantity,
                "avg_confidence": (
                    aggregate["confidence_sum"] / quantity
                    if quantity > 0
                    else 0.0
                ),
                "min_confidence": aggregate[
                    "min_confidence"
                ],
                "max_confidence": aggregate[
                    "max_confidence"
                ],
                "images_detected": len(
                    aggregate["images"]
                ),
            }
        )

    return (
        pd.DataFrame(
            overall_rows,
            columns=OVERALL_COLUMNS,
        ),
        pd.DataFrame(
            per_image_rows,
            columns=PER_IMAGE_COLUMNS,
        ),
        pd.DataFrame(
            image_summary_rows,
            columns=IMAGE_SUMMARY_COLUMNS,
        ),
    )


def dataframe_to_csv_file(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def auto_size_excel_sheet(
    worksheet: Any,
) -> None:
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="15324A",
    )
    header_font = Font(
        color="FFFFFF",
        bold=True,
    )

    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

    for column_cells in worksheet.columns:
        max_length = max(
            len(
                str(cell.value)
                if cell.value is not None
                else ""
            )
            for cell in column_cells
        )

        worksheet.column_dimensions[
            column_cells[0].column_letter
        ].width = min(
            max(max_length + 2, 12),
            48,
        )


def build_excel_file(
    output_path: Path,
    overall_df: pd.DataFrame,
    per_image_df: pd.DataFrame,
    image_summary_df: pd.DataFrame,
    run_metadata: dict[str, Any],
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> None:
    metadata_df = pd.DataFrame(
        [
            {
                "key": key,
                "value": (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                    if isinstance(value, (dict, list))
                    else value
                ),
            }
            for key, value in run_metadata.items()
        ]
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        overall_df.to_excel(
            writer,
            sheet_name="Overall by Class",
            index=False,
        )
        per_image_df.to_excel(
            writer,
            sheet_name="Per Image by Class",
            index=False,
        )
        image_summary_df.to_excel(
            writer,
            sheet_name="Image Summary",
            index=False,
        )
        metadata_df.to_excel(
            writer,
            sheet_name="Run Metadata",
            index=False,
        )

        for sheet_name, dataframe in (
            extra_sheets or {}
        ).items():
            safe_sheet_name = str(
                sheet_name
            )[:31]
            dataframe.to_excel(
                writer,
                sheet_name=safe_sheet_name,
                index=False,
            )

        for worksheet in writer.book.worksheets:
            auto_size_excel_sheet(worksheet)


def build_zip_file(
    output_path: Path,
    results: list[dict[str, Any]],
    report_files: list[Path],
    run_metadata: dict[str, Any],
) -> None:
    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for result in results:
            if result["status"] != "SUCCESS":
                continue

            original_path = Path(result["original_path"])
            annotated_path = Path(result["annotated_path"])

            if original_path.is_file():
                archive.write(
                    original_path,
                    f"original/{original_path.name}",
                )

            if annotated_path.is_file():
                archive.write(
                    annotated_path,
                    f"annotated/{annotated_path.name}",
                )

        for report_file in report_files:
            if report_file.is_file():
                archive.write(
                    report_file,
                    f"reports/{report_file.name}",
                )

        archive.writestr(
            "reports/run_metadata.json",
            json.dumps(
                run_metadata,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )

        if CLASS_MAPPING_PATH.is_file():
            archive.write(
                CLASS_MAPPING_PATH,
                "reports/class_display_mapping.json",
            )


# =========================================================
# 8. SESSION STATE
# =========================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex

if "inspection_results" not in st.session_state:
    st.session_state["inspection_results"] = []

if "report_payload" not in st.session_state:
    st.session_state["report_payload"] = None

if "selected_result_index" not in st.session_state:
    st.session_state["selected_result_index"] = 0

if "result_image_selector" not in st.session_state:
    st.session_state["result_image_selector"] = None


# =========================================================
# 9. PAGE HEADER
# =========================================================
st.markdown(
    f"""
    <div class="sb-header">
        <div class="sb-eyebrow">
            Sharon Bakery · Computer Vision
        </div>
        <h1>{APP_TITLE}</h1>
        <p>
            Upload images, run YOLO26s, compare original and labeled results,
            then download count and confidence reports.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 10. SIDEBAR SETTINGS
# =========================================================
with st.sidebar:
    st.markdown(
        '<div class="sb-section-title">'
        "Inference settings"
        "</div>",
        unsafe_allow_html=True,
    )

    model_path_input = st.text_input(
        "Model path",
        value=DEFAULT_MODEL_PATH,
        help=(
            "Path to the YOLO26s checkpoint file on this computer."
        ),
    )

    image_size = st.selectbox(
        "Image size",
        options=[640, 768, 896, 1024],
        index=1,
    )

    confidence_mode = st.radio(
        "Confidence mode",
        options=[
            "One threshold for all classes",
            "Different threshold per class",
        ],
        index=0,
        help=(
            "Per-class mode runs YOLO with the lowest configured "
            "threshold, then filters every prediction using the "
            "threshold of its predicted class."
        ),
    )

    confidence = st.slider(
        (
            "Confidence threshold"
            if confidence_mode
            == "One threshold for all classes"
            else "Default / fallback confidence"
        ),
        min_value=0.01,
        max_value=0.95,
        value=float(DEFAULT_CONFIDENCE),
        step=0.01,
    )

    class_confidence_thresholds: dict[int, float] = {}
    confidence_configuration_ready = True
    sidebar_bundle: ModelBundle | None = None

    if (
        confidence_mode
        == "Different threshold per class"
    ):
        sidebar_model_path = Path(
            model_path_input
        ).expanduser()

        if not sidebar_model_path.is_file():
            confidence_configuration_ready = False
            render_message(
                (
                    "A valid model path is required before "
                    "per-class thresholds can be edited."
                ),
                "warning",
            )
        else:
            try:
                mapping_path, mapping_mtime = (
                    get_class_mapping_signature()
                )
                sidebar_bundle = load_model(
                    model_path_input,
                    mapping_path,
                    mapping_mtime,
                )

                default_thresholds = (
                    load_class_confidence_defaults(
                        raw_names=(
                            sidebar_bundle.raw_names
                        ),
                        fallback=float(confidence),
                    )
                )

                threshold_rows = pd.DataFrame(
                    [
                        {
                            "class_id": class_id,
                            "raw_class_name": (
                                sidebar_bundle.raw_names[
                                    class_id
                                ]
                            ),
                            "display_name": (
                                sidebar_bundle.display_names[
                                    class_id
                                ]
                            ),
                            "confidence_threshold": (
                                default_thresholds[
                                    class_id
                                ]
                            ),
                        }
                        for class_id in sorted(
                            sidebar_bundle.raw_names
                        )
                    ]
                )

                signature_source = (
                    str(
                        sidebar_model_path.resolve()
                    )
                    + ":"
                    + str(
                        sidebar_model_path.stat().st_mtime_ns
                    )
                )
                editor_key = (
                    "class_confidence_editor_"
                    + hashlib.sha1(
                        signature_source.encode(
                            "utf-8"
                        )
                    ).hexdigest()[:12]
                )

                with st.expander(
                    "Per-class confidence thresholds",
                    expanded=True,
                ):
                    edited_threshold_rows = (
                        st.data_editor(
                            threshold_rows,
                            width="stretch",
                            height=520,
                            hide_index=True,
                            disabled=[
                                "class_id",
                                "raw_class_name",
                                "display_name",
                            ],
                            column_config={
                                "class_id": (
                                    st.column_config.NumberColumn(
                                        "ID",
                                        format="%d",
                                    )
                                ),
                                "raw_class_name": (
                                    "Raw class"
                                ),
                                "display_name": (
                                    "Display class"
                                ),
                                "confidence_threshold": (
                                    st.column_config.NumberColumn(
                                        "Threshold",
                                        min_value=0.01,
                                        max_value=0.99,
                                        step=0.01,
                                        format="%.2f",
                                        required=True,
                                    )
                                ),
                            },
                            key=editor_key,
                        )
                    )

                    for row in (
                        edited_threshold_rows
                        .to_dict("records")
                    ):
                        class_id = int(
                            row["class_id"]
                        )
                        class_confidence_thresholds[
                            class_id
                        ] = clamp_confidence_threshold(
                            row[
                                "confidence_threshold"
                            ],
                            confidence,
                        )

                    candidate_floor_preview = (
                        min(
                            class_confidence_thresholds.values()
                        )
                        if class_confidence_thresholds
                        else float(confidence)
                    )

                    st.caption(
                        (
                            "Candidate inference confidence: "
                            f"{candidate_floor_preview:.2f}. "
                            "The final result is filtered again "
                            "with each class threshold."
                        )
                    )

                    threshold_payload = (
                        build_class_confidence_payload(
                            raw_names=(
                                sidebar_bundle.raw_names
                            ),
                            thresholds=(
                                class_confidence_thresholds
                            ),
                            fallback=float(
                                confidence
                            ),
                        )
                    )

                    st.download_button(
                        "Download threshold JSON",
                        data=json.dumps(
                            threshold_payload,
                            ensure_ascii=False,
                            indent=2,
                        ).encode("utf-8"),
                        file_name=(
                            "class_confidence_thresholds.json"
                        ),
                        mime="application/json",
                        on_click="ignore",
                        width="stretch",
                    )

            except Exception as error:
                confidence_configuration_ready = False
                LOGGER.exception(
                    "Cannot prepare per-class thresholds"
                )
                render_message(
                    (
                        "Per-class thresholds could not be "
                        f"prepared: {error}"
                    ),
                    "error",
                )

    iou = st.slider(
        "NMS IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(DEFAULT_IOU),
        step=0.01,
    )

    max_detections = st.number_input(
        "Maximum detections per image",
        min_value=1,
        max_value=3000,
        value=DEFAULT_MAX_DETECTIONS,
        step=10,
    )

    device_selection = st.selectbox(
        "Processing device",
        options=["Auto", "GPU 0", "CPU"],
        index=0,
    )

    st.markdown(
        '<div class="sb-section-title" '
        'style="margin-top: 22px;">'
        "Result display"
        "</div>",
        unsafe_allow_html=True,
    )

    show_confidence = st.checkbox(
        "Show confidence on image",
        value=True,
    )

    line_width = st.slider(
        "Bounding box width",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
        help=(
            "Increase this value when bounding boxes "
            "or labels look too small."
        ),
    )

    st.markdown(
        '<div class="sb-muted">'
        "Default settings: imgsz=768, confidence=0.25, "
        "IoU=0.65, end2end=False. In per-class mode, "
        "every prediction must pass its class threshold."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sb-section-title" '
        'style="margin-top: 22px;">'
        "Ground Truth evaluation"
        "</div>",
        unsafe_allow_html=True,
    )

    enable_groundtruth_evaluation = st.checkbox(
        "Compare with Ground Truth",
        value=False,
        help=(
            "Optional. When disabled, the application "
            "works exactly like the standard inspection mode."
        ),
    )

    groundtruth_source_mode = "Local data.yaml"
    groundtruth_yaml_path_input = ""
    groundtruth_zip_upload = None
    groundtruth_split_selection = "auto"
    match_iou_threshold = 0.50
    run_official_metrics = False
    official_batch_size = 4

    if enable_groundtruth_evaluation:
        groundtruth_source_mode = st.radio(
            "Ground Truth source",
            options=[
                "Local data.yaml",
                "Upload YOLO dataset ZIP",
            ],
            horizontal=False,
        )

        if (
            groundtruth_source_mode
            == "Local data.yaml"
        ):
            groundtruth_yaml_path_input = (
                st.text_input(
                    "Ground Truth data.yaml path",
                    value="",
                    help=(
                        "Use a YOLO dataset exported with "
                        "images, labels, and data.yaml."
                    ),
                )
            )
        else:
            groundtruth_zip_upload = (
                st.file_uploader(
                    "Upload Ground Truth dataset ZIP",
                    type=["zip"],
                    accept_multiple_files=False,
                    key="groundtruth_zip_uploader",
                    help=(
                        "The ZIP must contain data.yaml and "
                        "the matching images and labels folders."
                    ),
                )
            )

        groundtruth_split_selection = st.selectbox(
            "Ground Truth split",
            options=[
                "auto",
                "test",
                "val",
                "valid",
                "train",
            ],
            index=0,
        )

        match_iou_threshold = st.slider(
            "Matching IoU threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help=(
                "A prediction is a TP only when the class "
                "matches and IoU is at least this value."
            ),
        )

        run_official_metrics = st.checkbox(
            "Run official mAP metrics",
            value=False,
            help=(
                "Optional and slower. Uses model.val() with "
                "confidence 0.001 to compute mAP."
            ),
        )

        if run_official_metrics:
            official_batch_size = st.number_input(
                "Official evaluation batch size",
                min_value=1,
                max_value=64,
                value=4,
                step=1,
            )


# =========================================================
# 11. UPLOAD AREA
# =========================================================
st.markdown(
    '<div class="sb-section-title">'
    "Input images"
    "</div>",
    unsafe_allow_html=True,
)

upload_mode = st.radio(
    "Upload method",
    options=[
        "One image",
        "Multiple images",
        "One folder",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

file_types = [
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp",
    "tif",
    "tiff",
]

uploaded_files: list[Any] = []

if upload_mode == "One image":
    uploaded_file = st.file_uploader(
        "Choose one image",
        type=file_types,
        accept_multiple_files=False,
        key="single_image_uploader",
    )

    if uploaded_file is not None:
        uploaded_files = [uploaded_file]

elif upload_mode == "Multiple images":
    uploaded_files = st.file_uploader(
        "Choose multiple images",
        type=file_types,
        accept_multiple_files=True,
        key="multiple_image_uploader",
    )

else:
    uploaded_files = st.file_uploader(
        "Choose one image folder",
        type=file_types,
        accept_multiple_files="directory",
        key="directory_image_uploader",
    )

upload_records = build_upload_records(
    list(uploaded_files or [])
)

total_upload_bytes = sum(
    len(record.data)
    for record in upload_records
)

upload_col_1, upload_col_2, upload_col_3 = st.columns(3)

with upload_col_1:
    st.metric(
        "Selected images",
        len(upload_records),
    )

with upload_col_2:
    st.metric(
        "Total size",
        f"{total_upload_bytes / (1024 ** 2):.2f} MB",
    )

with upload_col_3:
    st.metric(
        "Maximum per run",
        MAX_IMAGES_PER_RUN,
    )

if len(upload_records) > MAX_IMAGES_PER_RUN:
    render_message(
        (
            f"The image count is above {MAX_IMAGES_PER_RUN}. "
            "Please split the images into smaller runs."
        ),
        "error",
    )

model_path_valid = Path(
    model_path_input
).expanduser().is_file()

if not model_path_valid:
    render_message(
        (
            "The model file was not found. "
            "Analysis starts only when the model path is valid."
        ),
        "warning",
    )

groundtruth_source_ready = True

if enable_groundtruth_evaluation:
    if (
        groundtruth_source_mode
        == "Local data.yaml"
    ):
        groundtruth_source_ready = (
            bool(
                groundtruth_yaml_path_input.strip()
            )
            and Path(
                groundtruth_yaml_path_input
            ).expanduser().is_file()
        )

        if not groundtruth_source_ready:
            render_message(
                (
                    "Ground Truth evaluation is enabled, "
                    "but data.yaml was not found."
                ),
                "warning",
            )
    else:
        groundtruth_source_ready = (
            groundtruth_zip_upload is not None
        )

        if not groundtruth_source_ready:
            render_message(
                (
                    "Ground Truth evaluation is enabled, "
                    "but no YOLO dataset ZIP was uploaded."
                ),
                "warning",
            )

action_col_1, action_col_2 = st.columns(2)

with action_col_1:
    run_clicked = st.button(
        "Start analysis",
        type="primary",
        width="stretch",
        disabled=(
            not upload_records
            or not model_path_valid
            or not confidence_configuration_ready
            or not groundtruth_source_ready
            or len(upload_records) > MAX_IMAGES_PER_RUN
        ),
    )

with action_col_2:
    clear_clicked = st.button(
        "Clear current results",
        width="stretch",
    )

if clear_clicked:
    current_payload = st.session_state.get(
        "report_payload"
    )

    if current_payload:
        remove_run_directory(
            current_payload.get("run_directory")
        )

    st.session_state["inspection_results"] = []
    st.session_state["report_payload"] = None
    st.session_state["selected_result_index"] = 0
    st.session_state["result_image_selector"] = None
    read_file_bytes_cached.clear()
    gc.collect()
    st.rerun()


# =========================================================
# 12. RUN INFERENCE
# =========================================================
if run_clicked:
    new_run_directories: RunDirectories | None = None

    try:
        cleanup_old_runtime_directories()
        device = resolve_device(device_selection)
        mapping_path, mapping_mtime = (
            get_class_mapping_signature()
        )

        with st.spinner("Loading YOLO26s model..."):
            bundle = load_model(
                model_path_input,
                mapping_path,
                mapping_mtime,
            )

        if (
            confidence_mode
            == "Different threshold per class"
        ):
            missing_threshold_ids = (
                set(bundle.raw_names)
                - set(class_confidence_thresholds)
            )

            if missing_threshold_ids:
                raise RuntimeError(
                    "Missing confidence thresholds for class IDs: "
                    + ", ".join(
                        str(class_id)
                        for class_id
                        in sorted(missing_threshold_ids)
                    )
                )

            effective_class_thresholds = {
                class_id: clamp_confidence_threshold(
                    class_confidence_thresholds[
                        class_id
                    ],
                    confidence,
                )
                for class_id in bundle.raw_names
            }
        else:
            effective_class_thresholds = {
                class_id: float(confidence)
                for class_id in bundle.raw_names
            }

        candidate_confidence = min(
            effective_class_thresholds.values()
        )

        render_message(
            (
                f"Model is ready with {len(bundle.raw_names)} classes. "
                f"Device: {device}."
            ),
            "success",
        )

        new_run_directories = create_run_directories()

        groundtruth_bundle: GroundTruthBundle | None = None

        if enable_groundtruth_evaluation:
            uploaded_gt_zip_bytes = (
                groundtruth_zip_upload.getvalue()
                if groundtruth_zip_upload is not None
                else None
            )

            with st.spinner(
                "Loading Ground Truth dataset..."
            ):
                groundtruth_yaml_path = (
                    prepare_groundtruth_yaml(
                        run_directory=(
                            new_run_directories.root
                        ),
                        source_mode=(
                            groundtruth_source_mode
                        ),
                        local_yaml_path=(
                            groundtruth_yaml_path_input
                        ),
                        uploaded_zip_bytes=(
                            uploaded_gt_zip_bytes
                        ),
                    )
                )
                groundtruth_bundle = (
                    build_groundtruth_bundle(
                        data_yaml_path=(
                            groundtruth_yaml_path
                        ),
                        requested_split=(
                            groundtruth_split_selection
                        ),
                        model_names=(
                            bundle.raw_names
                        ),
                    )
                )

            render_message(
                (
                    "Ground Truth is ready: "
                    f"{len(groundtruth_bundle.image_paths)} "
                    f"images in split "
                    f"'{groundtruth_bundle.split_name}'."
                ),
                "success",
            )

        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        results: list[dict[str, Any]] = []

        for index, upload in enumerate(
            upload_records,
            start=1,
        ):
            progress_text.markdown(
                (
                    f"Processing {index}/{len(upload_records)}: "
                    f"`{html.escape(upload.display_name)}`"
                )
            )

            try:
                result = run_single_inference(
                    bundle=bundle,
                    upload=upload,
                    run_directories=new_run_directories,
                    image_size=int(image_size),
                    candidate_confidence=float(
                        candidate_confidence
                    ),
                    class_thresholds=(
                        effective_class_thresholds
                    ),
                    fallback_confidence=float(
                        confidence
                    ),
                    iou=float(iou),
                    max_detections=int(max_detections),
                    device=device,
                    show_confidence=show_confidence,
                    line_width=int(line_width),
                )

            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                result = build_error_result(
                    upload,
                    (
                        "CUDA ran out of memory. "
                        "Use a smaller image size or fewer images."
                    ),
                )

                LOGGER.exception(
                    "CUDA OOM on %s",
                    upload.display_name,
                )

            except Exception as error:
                result = build_error_result(
                    upload,
                    str(error),
                )

                LOGGER.exception(
                    "Inference failed for %s",
                    upload.display_name,
                )

            results.append(result)
            progress_bar.progress(
                index / len(upload_records)
            )

        progress_text.empty()

        (
            overall_df,
            per_image_df,
            image_summary_df,
        ) = build_report_dataframes(results)

        successful_results = [
            result
            for result in results
            if result["status"] == "SUCCESS"
        ]

        total_detections = sum(
            result["total_detections"]
            for result in successful_results
        )
        total_confidence_sum = sum(
            result["confidence_sum"]
            for result in successful_results
        )
        overall_average_confidence = (
            total_confidence_sum / total_detections
            if total_detections > 0
            else 0.0
        )

        evaluation_payload: dict[str, Any] | None = None

        if groundtruth_bundle is not None:
            with st.spinner(
                "Comparing predictions with Ground Truth..."
            ):
                evaluation_payload = (
                    evaluate_results_against_groundtruth(
                        results=results,
                        groundtruth=(
                            groundtruth_bundle
                        ),
                        model_names=(
                            bundle.raw_names
                        ),
                        display_names=(
                            bundle.display_names
                        ),
                        match_iou=float(
                            match_iou_threshold
                        ),
                    )
                )

                if run_official_metrics:
                    try:
                        official_payload = (
                            run_official_groundtruth_metrics(
                                bundle=bundle,
                                run_directory=(
                                    new_run_directories.root
                                ),
                                results=results,
                                groundtruth=(
                                    groundtruth_bundle
                                ),
                                image_size=int(
                                    image_size
                                ),
                                iou=float(iou),
                                max_detections=int(
                                    max_detections
                                ),
                                device=device,
                                batch_size=int(
                                    official_batch_size
                                ),
                            )
                        )
                        evaluation_payload[
                            "official"
                        ] = official_payload

                    except Exception as error:
                        LOGGER.exception(
                            "Official Ground Truth "
                            "evaluation failed"
                        )
                        evaluation_payload[
                            "official_error"
                        ] = str(error)

        run_metadata = {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "model_path": str(
                Path(model_path_input).expanduser()
            ),
            "model_file": Path(model_path_input).name,
            "class_count": len(bundle.raw_names),
            "class_mapping": {
                str(class_id): {
                    "raw_name": bundle.raw_names[class_id],
                    "display_name": bundle.display_names[class_id],
                }
                for class_id in bundle.raw_names
            },
            "input_image_count": len(upload_records),
            "successful_image_count": len(
                successful_results
            ),
            "failed_image_count": (
                len(results) - len(successful_results)
            ),
            "total_detections": total_detections,
            "overall_avg_confidence": (
                overall_average_confidence
            ),
            "groundtruth_evaluation": (
                evaluation_payload["overall"]
                if evaluation_payload is not None
                else None
            ),
            "settings": {
                "imgsz": int(image_size),
                "confidence_mode": confidence_mode,
                "default_confidence": float(
                    confidence
                ),
                "candidate_confidence": float(
                    candidate_confidence
                ),
                "class_confidence_thresholds": {
                    bundle.raw_names[class_id]: float(
                        effective_class_thresholds[
                            class_id
                        ]
                    )
                    for class_id in sorted(
                        bundle.raw_names
                    )
                },
                "iou": float(iou),
                "max_det": int(max_detections),
                "end2end": False,
                "agnostic_nms": False,
                "rect": False,
                "augment": False,
                "quantize": "fp32",
                "compile": False,
                "device": device,
                "preview_max_side": PREVIEW_MAX_SIDE,
            },
        }

        overall_csv_path = (
            new_run_directories.reports
            / "overall_by_class.csv"
        )
        per_image_csv_path = (
            new_run_directories.reports
            / "per_image_by_class.csv"
        )
        image_summary_csv_path = (
            new_run_directories.reports
            / "image_summary.csv"
        )
        excel_path = (
            new_run_directories.reports
            / "inventory_inspection_report.xlsx"
        )
        zip_path = (
            new_run_directories.root
            / "Sharon_Inventory_Inspection.zip"
        )

        dataframe_to_csv_file(
            overall_df,
            overall_csv_path,
        )
        dataframe_to_csv_file(
            per_image_df,
            per_image_csv_path,
        )
        dataframe_to_csv_file(
            image_summary_df,
            image_summary_csv_path,
        )

        extra_excel_sheets: dict[
            str,
            pd.DataFrame,
        ] = {}
        evaluation_report_paths: list[Path] = []

        if evaluation_payload is not None:
            extra_excel_sheets = {
                "GT Overall": (
                    evaluation_payload[
                        "overall_df"
                    ]
                ),
                "GT Per Class": (
                    evaluation_payload[
                        "per_class_df"
                    ]
                ),
                "GT Per Image": (
                    evaluation_payload[
                        "per_image_df"
                    ]
                ),
                "GT Count Comparison": (
                    evaluation_payload[
                        "count_comparison_df"
                    ]
                ),
                "GT Unmatched": (
                    evaluation_payload[
                        "unmatched_df"
                    ]
                ),
            }

            official_payload = (
                evaluation_payload.get(
                    "official"
                )
            )

            if official_payload:
                extra_excel_sheets[
                    "Official Overall"
                ] = official_payload[
                    "overall_df"
                ]
                extra_excel_sheets[
                    "Official Per Class"
                ] = official_payload[
                    "per_class_df"
                ]

            evaluation_report_paths = (
                write_optional_evaluation_reports(
                    reports_directory=(
                        new_run_directories.reports
                    ),
                    evaluation_payload=(
                        evaluation_payload
                    ),
                )
            )

        build_excel_file(
            excel_path,
            overall_df,
            per_image_df,
            image_summary_df,
            run_metadata,
            extra_sheets=(
                extra_excel_sheets
            ),
        )

        report_files_for_zip = [
            overall_csv_path,
            per_image_csv_path,
            image_summary_csv_path,
            excel_path,
            *evaluation_report_paths,
        ]

        build_zip_file(
            zip_path,
            results,
            report_files_for_zip,
            run_metadata,
        )

        previous_payload = st.session_state.get(
            "report_payload"
        )

        st.session_state["inspection_results"] = results
        st.session_state["report_payload"] = {
            "run_directory": str(
                new_run_directories.root
            ),
            "overall_df": overall_df,
            "per_image_df": per_image_df,
            "image_summary_df": image_summary_df,
            "overall_csv_path": str(
                overall_csv_path
            ),
            "per_image_csv_path": str(
                per_image_csv_path
            ),
            "image_summary_csv_path": str(
                image_summary_csv_path
            ),
            "excel_path": str(excel_path),
            "zip_path": str(zip_path),
            "metadata": run_metadata,
            "evaluation": evaluation_payload,
            "evaluation_report_paths": [
                str(path)
                for path in evaluation_report_paths
            ],
        }
        st.session_state["selected_result_index"] = 0
        st.session_state["result_image_selector"] = (
            successful_results[0]["display_name"]
            if successful_results
            else None
        )

        if previous_payload:
            previous_run_directory = previous_payload.get(
                "run_directory"
            )

            if previous_run_directory != str(
                new_run_directories.root
            ):
                remove_run_directory(
                    previous_run_directory
                )

        read_file_bytes_cached.clear()

        render_message(
            (
                "Analysis is complete. "
                f"Successful images: {len(successful_results)}/"
                f"{len(results)}."
            ),
            "success",
        )

        del upload_records
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as error:
        LOGGER.exception(
            "Application-level inference error"
        )

        if new_run_directories is not None:
            remove_run_directory(
                new_run_directories.root
            )

        render_message(
            f"Analysis could not start: {error}",
            "error",
        )


# =========================================================
# 13. FAST IMAGE VIEWER FRAGMENT
# =========================================================
def select_result_image(
    image_name: str,
) -> None:
    st.session_state["result_image_selector"] = image_name


@st.fragment
def render_image_detail_fragment() -> None:
    results = st.session_state.get(
        "inspection_results",
        [],
    )
    successful_results = [
        result
        for result in results
        if result["status"] == "SUCCESS"
    ]

    if not successful_results:
        render_message(
            "There are no successful images to display.",
            "error",
        )
        return

    result_names = [
        result["display_name"]
        for result in successful_results
    ]

    selected_name = st.session_state.get(
        "result_image_selector"
    )

    if selected_name not in result_names:
        selected_name = result_names[0]
        st.session_state[
            "result_image_selector"
        ] = selected_name

    selected_name = st.selectbox(
        "Choose an image",
        options=result_names,
        key="result_image_selector",
    )

    selected_index = result_names.index(selected_name)
    selected_result = successful_results[
        selected_index
    ]

    previous_name = (
        result_names[selected_index - 1]
        if selected_index > 0
        else selected_name
    )
    next_name = (
        result_names[selected_index + 1]
        if selected_index < len(result_names) - 1
        else selected_name
    )

    previous_col, position_col, next_col = st.columns(
        [1, 2, 1]
    )

    with previous_col:
        st.button(
            "Previous image",
            width="stretch",
            disabled=(selected_index == 0),
            on_click=select_result_image,
            args=(previous_name,),
            key="previous_result_image",
        )

    with position_col:
        st.markdown(
            (
                "<div style='text-align:center;"
                "padding-top:10px;color:#5C6F7C;'>"
                f"Image {selected_index + 1}/"
                f"{len(successful_results)}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    with next_col:
        st.button(
            "Next image",
            width="stretch",
            disabled=(
                selected_index
                >= len(successful_results) - 1
            ),
            on_click=select_result_image,
            args=(next_name,),
            key="next_result_image",
        )

    st.markdown(
        (
            '<div class="sb-section-title">'
            f"{html.escape(selected_name)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    try:
        original_preview = read_file_bytes(
            selected_result["original_preview_path"]
        )
        annotated_preview = read_file_bytes(
            selected_result["annotated_preview_path"]
        )

        original_col, annotated_col = st.columns(2)

        with original_col:
            st.markdown("**Original image**")
            st.image(
                original_preview,
                width="stretch",
            )

        with annotated_col:
            st.markdown("**Labeled image**")
            st.image(
                annotated_preview,
                width="stretch",
            )

    except Exception as error:
        render_message(
            f"The image preview cannot be loaded: {error}",
            "error",
        )
        return

    image_metric_1, image_metric_2 = st.columns(2)

    with image_metric_1:
        st.metric(
            "Objects in image",
            selected_result["total_detections"],
        )

    with image_metric_2:
        st.metric(
            "Average confidence",
            f"{selected_result['avg_confidence']:.2%}",
        )

    image_class_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key != "confidence_sum"
            }
            for row in selected_result["class_rows"]
        ],
        columns=[
            "class_name",
            "confidence_threshold",
            "quantity",
            "avg_confidence",
            "min_confidence",
            "max_confidence",
        ],
    )

    if image_class_df.empty:
        render_message(
            "No object was detected in this image.",
            "warning",
        )
    else:
        st.dataframe(
            image_class_df,
            width="stretch",
            hide_index=True,
            column_config={
                "class_name": "Class",
                "confidence_threshold": (
                    "Confidence threshold"
                ),
                "quantity": "Quantity",
                "avg_confidence": st.column_config.ProgressColumn(
                    "Average confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
                "min_confidence": "Minimum confidence",
                "max_confidence": "Maximum confidence",
            },
        )

    report_payload = st.session_state.get(
        "report_payload"
    )
    evaluation_payload = (
        report_payload.get("evaluation")
        if report_payload
        else None
    )

    if evaluation_payload is not None:
        per_image_evaluation_df = (
            evaluation_payload.get(
                "per_image_df",
                pd.DataFrame(
                    columns=(
                        GROUNDTRUTH_PER_IMAGE_COLUMNS
                    )
                ),
            )
        )

        selected_evaluation = pd.DataFrame(
            columns=GROUNDTRUTH_PER_IMAGE_COLUMNS
        )

        if (
            isinstance(
                per_image_evaluation_df,
                pd.DataFrame,
            )
            and not per_image_evaluation_df.empty
            and "image_name"
            in per_image_evaluation_df.columns
        ):
            selected_evaluation = (
                per_image_evaluation_df[
                    per_image_evaluation_df[
                        "image_name"
                    ]
                    == selected_result[
                        "display_name"
                    ]
                ]
            )

        if not selected_evaluation.empty:
            st.markdown(
                '<div class="sb-section-title" '
                'style="margin-top: 18px;">'
                "Ground Truth comparison for this image"
                "</div>",
                unsafe_allow_html=True,
            )

            evaluation_row = (
                selected_evaluation.iloc[0]
            )
            eval_metric_1, eval_metric_2, eval_metric_3, eval_metric_4 = (
                st.columns(4)
            )

            with eval_metric_1:
                st.metric(
                    "TP / FP / FN",
                    (
                        f"{int(evaluation_row['tp'])} / "
                        f"{int(evaluation_row['fp'])} / "
                        f"{int(evaluation_row['fn'])}"
                    ),
                )

            with eval_metric_2:
                st.metric(
                    "F1",
                    f"{float(evaluation_row['f1']):.2%}",
                )

            with eval_metric_3:
                st.metric(
                    "Count coverage",
                    (
                        f"{float(evaluation_row['count_coverage']):.2%}"
                    ),
                )

            with eval_metric_4:
                st.metric(
                    "Count accuracy",
                    (
                        f"{float(evaluation_row['count_accuracy']):.2%}"
                    ),
                )

            selected_count_df = (
                evaluation_payload.get(
                    "count_comparison_df",
                    pd.DataFrame(
                        columns=(
                            GROUNDTRUTH_COUNT_COMPARISON_COLUMNS
                        )
                    ),
                )
            )

            if (
                isinstance(
                    selected_count_df,
                    pd.DataFrame,
                )
                and not selected_count_df.empty
                and "image_name"
                in selected_count_df.columns
            ):
                selected_count_df = (
                    selected_count_df[
                        selected_count_df[
                            "image_name"
                        ]
                        == selected_result[
                            "display_name"
                        ]
                    ]
                )

                if not selected_count_df.empty:
                    st.dataframe(
                        selected_count_df,
                        width="stretch",
                        hide_index=True,
                    )

        else:
            unmatched_df = evaluation_payload.get(
                "unmatched_df",
                pd.DataFrame(
                    columns=(
                        GROUNDTRUTH_UNMATCHED_COLUMNS
                    )
                ),
            )

            selected_unmatched = pd.DataFrame(
                columns=GROUNDTRUTH_UNMATCHED_COLUMNS
            )

            if (
                isinstance(unmatched_df, pd.DataFrame)
                and not unmatched_df.empty
                and "image_name" in unmatched_df.columns
            ):
                selected_unmatched = unmatched_df[
                    unmatched_df["image_name"]
                    == selected_result["display_name"]
                ]

            if not selected_unmatched.empty:
                render_message(
                    (
                        "This image was not matched to Ground Truth. "
                        "The model result is still available, but "
                        "evaluation metrics cannot be calculated."
                    ),
                    "warning",
                )

    st.download_button(
        "Download labeled image",
        data=deferred_file_reader(
            selected_result["annotated_path"]
        ),
        file_name=selected_result[
            "annotated_download_name"
        ],
        mime="image/jpeg",
        on_click="ignore",
        width="stretch",
        key=f"download_annotated_{selected_result['sha256']}",
    )


# =========================================================
# 14. RESULTS DASHBOARD
# =========================================================
results = st.session_state["inspection_results"]
report_payload = st.session_state["report_payload"]

if results and report_payload:
    successful_results = [
        result
        for result in results
        if result["status"] == "SUCCESS"
    ]
    failed_results = [
        result
        for result in results
        if result["status"] != "SUCCESS"
    ]
    metadata = report_payload["metadata"]

    st.markdown(
        '<div class="sb-section-title" '
        'style="margin-top: 24px;">'
        "Analysis results"
        "</div>",
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        st.metric(
            "Successful images",
            (
                f"{len(successful_results)}/"
                f"{len(results)}"
            ),
        )

    with metric_2:
        st.metric(
            "Total objects",
            metadata["total_detections"],
        )

    with metric_3:
        st.metric(
            "Detected classes",
            len(report_payload["overall_df"]),
        )

    with metric_4:
        st.metric(
            "Average confidence",
            f"{metadata['overall_avg_confidence']:.2%}",
        )

    evaluation_payload = report_payload.get(
        "evaluation"
    )

    if evaluation_payload is not None:
        (
            overview_tab,
            image_tab,
            evaluation_tab,
            download_tab,
        ) = st.tabs(
            [
                "Overview",
                "Image details",
                "Ground Truth evaluation",
                "Reports and downloads",
            ]
        )
    else:
        (
            overview_tab,
            image_tab,
            download_tab,
        ) = st.tabs(
            [
                "Overview",
                "Image details",
                "Reports and downloads",
            ]
        )
        evaluation_tab = None

    with overview_tab:
        overall_df = report_payload["overall_df"]

        st.markdown(
            '<div class="sb-section-title">'
            "Total quantity by class"
            "</div>",
            unsafe_allow_html=True,
        )

        if overall_df.empty:
            render_message(
                "The model found no object in the uploaded images.",
                "warning",
            )
        else:
            chart_col, table_col = st.columns([0.9, 1.4])

            with chart_col:
                st.bar_chart(
                    overall_df[
                        ["class_name", "total_quantity"]
                    ].set_index("class_name"),
                    height=430,
                )

            with table_col:
                display_overall_df = overall_df.copy()

                for column in [
                    "avg_confidence",
                    "min_confidence",
                    "max_confidence",
                ]:
                    display_overall_df[column] = (
                        display_overall_df[column]
                        .astype(float)
                        .round(4)
                    )

                st.dataframe(
                    display_overall_df,
                    width="stretch",
                    hide_index=True,
                    height=430,
                    column_config={
                        "class_name": "Class",
                        "confidence_threshold": (
                            "Confidence threshold"
                        ),
                        "total_quantity": "Total quantity",
                        "avg_confidence": st.column_config.ProgressColumn(
                            "Average confidence",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.4f",
                        ),
                        "min_confidence": "Minimum confidence",
                        "max_confidence": "Maximum confidence",
                        "images_detected": "Images detected",
                    },
                )

        st.markdown(
            '<div class="sb-section-title" '
            'style="margin-top: 20px;">'
            "Processing status by image"
            "</div>",
            unsafe_allow_html=True,
        )

        display_summary_df = report_payload[
            "image_summary_df"
        ].copy()

        if not display_summary_df.empty:
            display_summary_df["avg_confidence"] = (
                display_summary_df["avg_confidence"]
                .astype(float)
                .round(4)
            )
            display_summary_df["inference_ms"] = (
                display_summary_df["inference_ms"]
                .astype(float)
                .round(2)
            )

        st.dataframe(
            display_summary_df,
            width="stretch",
            hide_index=True,
            column_config={
                "image_name": "Image",
                "status": "Status",
                "raw_detections_before_class_filter": (
                    "Raw candidates"
                ),
                "detections_removed_by_class_filter": (
                    "Removed by class threshold"
                ),
                "total_detections": "Objects",
                "avg_confidence": "Average confidence",
                "inference_ms": "Processing time (ms)",
                "error": "Error",
            },
        )

    with image_tab:
        render_image_detail_fragment()

    if (
        evaluation_tab is not None
        and evaluation_payload is not None
    ):
        with evaluation_tab:
            render_groundtruth_evaluation_dashboard(
                evaluation_payload
            )

            official_error = (
                evaluation_payload.get(
                    "official_error"
                )
            )

            if official_error:
                render_message(
                    (
                        "Official mAP evaluation failed: "
                        f"{official_error}"
                    ),
                    "error",
                )

    with download_tab:
        st.markdown(
            '<div class="sb-section-title">'
            "Result reports"
            "</div>",
            unsafe_allow_html=True,
        )

        render_message(
            (
                "The Excel report has four sheets: overall classes, "
                "per-image classes, image status, and run metadata."
            ),
            "info",
        )

        timestamp_name = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        download_col_1, download_col_2 = st.columns(2)

        with download_col_1:
            st.download_button(
                "Download Excel report",
                data=deferred_file_reader(
                    report_payload["excel_path"]
                ),
                file_name=(
                    "Sharon_Inventory_Inspection_"
                    f"{timestamp_name}.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                on_click="ignore",
                width="stretch",
            )

        with download_col_2:
            st.download_button(
                "Download all results as ZIP",
                data=deferred_file_reader(
                    report_payload["zip_path"]
                ),
                file_name=(
                    "Sharon_Inventory_Inspection_"
                    f"{timestamp_name}.zip"
                ),
                mime="application/zip",
                on_click="ignore",
                width="stretch",
            )

        csv_col_1, csv_col_2, csv_col_3 = st.columns(3)

        with csv_col_1:
            st.download_button(
                "Overall class CSV",
                data=deferred_file_reader(
                    report_payload["overall_csv_path"]
                ),
                file_name="overall_by_class.csv",
                mime="text/csv",
                on_click="ignore",
                width="stretch",
            )

        with csv_col_2:
            st.download_button(
                "Per-image CSV",
                data=deferred_file_reader(
                    report_payload["per_image_csv_path"]
                ),
                file_name="per_image_by_class.csv",
                mime="text/csv",
                on_click="ignore",
                width="stretch",
            )

        with csv_col_3:
            st.download_button(
                "Image status CSV",
                data=deferred_file_reader(
                    report_payload[
                        "image_summary_csv_path"
                    ]
                ),
                file_name="image_summary.csv",
                mime="text/csv",
                on_click="ignore",
                width="stretch",
            )

        st.markdown(
            '<div class="sb-section-title" '
            'style="margin-top: 22px;">'
            "Run settings"
            "</div>",
            unsafe_allow_html=True,
        )

        st.json(
            report_payload["metadata"],
            expanded=False,
        )

        if failed_results:
            st.markdown(
                '<div class="sb-section-title" '
                'style="margin-top: 22px;">'
                "Failed images"
                "</div>",
                unsafe_allow_html=True,
            )

            failure_df = pd.DataFrame(
                [
                    {
                        "image_name": result["display_name"],
                        "error": result["error"],
                    }
                    for result in failed_results
                ]
            )

            st.dataframe(
                failure_df,
                width="stretch",
                hide_index=True,
            )

else:
    st.markdown(
        """
        <div class="sb-card" style="margin-top: 24px;">
            <div class="sb-section-title">How to use</div>
            <div class="sb-muted">
                1. Enter the correct YOLO26s model path.<br>
                2. Upload one image, multiple images, or one folder.<br>
                3. Change confidence, IoU, or image size when needed.<br>
                4. Start analysis and compare original and labeled images.<br>
                5. Download Excel, CSV, or the full ZIP package.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="sb-footer">
        Sharon Bakery AI Inventory Inspection ·
        Static-image counts are based on detected bounding boxes.
    </div>
    """,
    unsafe_allow_html=True,
)
