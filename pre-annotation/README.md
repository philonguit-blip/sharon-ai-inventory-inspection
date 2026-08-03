# YOLO-World pre-annotation — 30 Sharon Bakery SKUs

## 1. Environment

Recommended: Python 3.10 or 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_preannotation.txt
```

## 2. Validate the 30-class configuration

```powershell
python .\preannotation.py `
  --images "C:\path\to\images" `
  --validate-only
```

Expected first line:

```text
Configuration OK: 30 classes.
```

## 3. Recommended balanced run

```powershell
python .\preannotation.py `
  --images "C:\path\to\images" `
  --output "C:\path\to\preannotations_30sku" `
  --model "yolov8x-worldv2.pt" `
  --profile balanced `
  --imgsz 1280 `
  --model-iou 0.65 `
  --max-det 300 `
  --device 0 `
  --save-review-crops
```

Use `--device cpu` when CUDA is unavailable. Omit `--device` to let Ultralytics choose automatically.

The first run can download `yolov8x-worldv2.pt` automatically.

## 4. Prompt profiles

- `strict`: one strongest prompt per SKU; less noise, lower recall.
- `balanced`: two strong prompts per SKU; recommended starting point.
- `recall`: all prompts; view-only and broad prompts are sent to review rather than auto-accepted.

## 5. Output

```text
preannotations_30sku/
├── labels/                 # Standard YOLO numeric labels
├── previews/               # Green=accepted, orange=review
├── review_crops/           # Only when --save-review-crops is used
├── classes.txt
├── class_mapping.json
├── data_template.yaml
├── detections.csv
├── image_summary.csv
├── review_queue.jsonl
├── run_config.json
├── summary.json
└── preannotation.log
```

Review every orange box before training. The script intentionally excludes low-confidence and ambiguous cross-class predictions from `.txt` labels.

## 6. Re-run behavior

Existing `.txt` labels are protected by default. To regenerate them:

```powershell
python .\preannotation.py `
  --images "C:\path\to\images" `
  --output "C:\path\to\preannotations_30sku" `
  --profile balanced `
  --device 0 `
  --overwrite
```

## 7. Import into Roboflow or training dataset

The numeric class-ID order is fixed in `classes.txt`, `class_mapping.json`, and `data_template.yaml`. Keep this order unchanged when importing labels or creating `data.yaml`.
