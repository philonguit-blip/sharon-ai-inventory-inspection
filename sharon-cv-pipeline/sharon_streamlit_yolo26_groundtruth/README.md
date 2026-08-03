# Sharon Bakery AI Inventory Inspection

Streamlit application for YOLO26s image inference with optional Ground Truth evaluation.

## Main modes

### Standard inspection

Leave **Compare with Ground Truth** disabled. The application works as before:

- Upload one image, multiple images, or one folder.
- Run YOLO26s inference.
- Compare original and labeled images.
- Download Excel, CSV, and ZIP reports.

### Optional Ground Truth evaluation

Enable **Compare with Ground Truth** in the sidebar, then choose one source:

1. **Local data.yaml**: enter the full path to a local YOLO dataset `data.yaml`.
2. **Upload YOLO dataset ZIP**: upload an exported YOLO dataset ZIP containing `data.yaml`, `images`, and `labels`.

The uploaded inference images are matched to Ground Truth by:

1. SHA-256 hash.
2. Exact filename.
3. Filename stem.

The Ground Truth class IDs are mapped to checkpoint classes by class name, not by raw numeric ID.

## Evaluation metrics

The Ground Truth tab shows:

- Ground Truth class coverage.
- TP, FP, FN.
- Micro Precision, Recall, and F1.
- Macro Precision, Recall, and F1.
- Count coverage and count accuracy.
- Mean image count coverage and accuracy.
- Image exact count rate.
- Count MAE.
- Per-class metrics.
- Per-image metrics.
- Per-image and per-class count comparison.
- Unmatched images.

The optional **Run official mAP metrics** setting additionally runs `model.val()` with `conf=0.001` and reports:

- mAP50.
- mAP75.
- mAP50-95.
- Official Precision, Recall, and F1.
- Official per-class AP metrics.

Official mAP evaluation is slower and is disabled by default.

## Expected YOLO dataset structure

```text
dataset/
├── data.yaml
└── test/
    ├── images/
    │   ├── image_001.jpg
    │   └── image_002.jpg
    └── labels/
        ├── image_001.txt
        └── image_002.txt
```

Example `data.yaml`:

```yaml
path: .
train: train/images
val: valid/images
test: test/images
nc: 30
names:
  0: BK_0007_BoLatPilot
  1: CH_0006_SocolaCompoundTrang
```

Ground Truth may contain only a subset of the 30 checkpoint classes. All Ground Truth class names must exist in `model.names`.

## Install and run on Windows

```powershell
cd "C:\path\to\sharon_streamlit_yolo26_groundtruth"

py -3.11 -m venv .venv

Set-ExecutionPolicy `
  -Scope Process `
  -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Model location

Recommended structure:

```text
sharon_streamlit_yolo26_groundtruth/
├── app.py
├── models/
│   └── best_YOLO26s_30_SKU_v6.pt
├── class_display_mapping.json
├── requirements.txt
└── .streamlit/
    └── config.toml
```

The application does not download Ground Truth directly from Roboflow and does not request an API key. Export the dataset once and use the local `data.yaml` or ZIP option. This avoids storing credentials in the web application and makes evaluations reproducible.
