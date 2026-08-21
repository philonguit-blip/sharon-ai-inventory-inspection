# Sharon Bakery FOUNDATION Test Lab v3

This is a standalone Streamlit lab for the current Foundation engine:

```text
SAM2 segmentation
→ mask filtering / duplicate suppression
→ DINOv2 embeddings
→ tray reference matching
→ DINO validation per candidate
→ median-area outlier filtering
→ DIRECT / FAMILY / AMBIGUOUS / NO_DETECTION
```

It does **not** call YOLO, n8n, R2, KiotViet, or any production write.

## What v3 fixes

v3 không còn gán class cấp khay cho mọi SAM mask. Mỗi candidate phải đạt đồng
thời similarity và margin với SKU đã chọn; sau đó mới qua lọc diện tích robust.
Count cuối là số object còn lại. SAM chỉ nhận ảnh cạnh dài tối đa 1280 px, còn
DINO crop và ảnh hiển thị vẫn lấy từ ảnh gốc.

Các mặc định production:

```text
SAM max side                 1280
SAM point stride             96
Tray similarity / margin     0.72 / 0.04
Instance similarity / margin 0.60 / 0.02
Instance area range          0.35x / 2.50x median
Semantic box coverage NMS    0.45
```

## What v2 fixed

The previous app performed inference and then immediately rendered:

- full-resolution original image,
- full-resolution annotated image,
- every detected crop,
- optional extra DINO ranking work.

That could make the browser still show `FOUNDATION processing...` even after the terminal printed `DONE`.

v2 separates the two phases:

```text
INFERENCE
→ immediately finish progress/status
→ cache compact inference result

then separately:

UI INSPECTION
→ select one image
→ render resized previews
→ open crops only when needed
→ show ranking without re-running DINO
```

## New timing/debug

For every image the terminal now prints:

```text
[FOUNDATION-TEST][TIMING]
load=...
sam=...
crop=...
dino_encode=...
dino_classify=...
total=...
```

The web also shows those values in a phase-timing table.

This makes it possible to determine whether the bottleneck is:

- first-time SAM/DINO model loading,
- SAM segmentation,
- DINO embedding,
- DINO classification,
- or UI rendering.

## Recommended placement

Place these files in the project root:

```text
sharon-AI-inventory-inspection/
├── foundation_streamlit_test.py
├── start_foundation_test.bat
└── backend/
    ├── .venv/
    ├── app/
    ├── models/
    │   └── sam2.1_s.pt
    ├── hybrid_data/
    │   └── reference_embeddings.npz
    └── config/
        └── hybrid_reference_registry.json
```

## Run

Double-click:

```text
start_foundation_test.bat
```

Then open:

```text
http://127.0.0.1:8502
```

## Recommended test sequence

1. Open the app.
2. Confirm `Ready = YES`.
3. Click **Load engine only** once.
4. Wait until engine loading completes.
5. Upload one image.
6. Click **Run FOUNDATION**.
7. Read:
   - SAM time,
   - DINO time,
   - total inference time,
   - count,
   - similarity,
   - margin.
8. Use **Inspect one result** only after inference has finished.
9. Expand **Top reference ranking** to inspect classification.
10. Expand **Detected object crops** only when needed.

`Load engine only` is important for benchmarking because it separates first-run model startup from actual image inference.

## Mixed image + video pre-annotation

The **Run FOUNDATION test** uploader accepts images and `.mp4`, `.mov`, `.avi`,
`.mkv`, or `.m4v` videos in the same run. For each video the app:

1. samples candidate frames at the configured FPS;
2. rejects blurred frames and optionally resizes the retained long side;
3. removes near-duplicates using pHash against every previously retained frame
   from that source video;
4. enforces per-video and per-run frame limits before inference;
5. runs every uploaded image and every retained frame through the same FOUNDATION
   pipeline; and
6. exports all selected results into one YOLO or COCO pre-annotation ZIP.

Defaults are 2 candidate FPS, sharpness 60, pHash distance 6, 200 retained
frames/video, 500 retained video frames/run and 1920 px long side. Increase the
limits deliberately: retained frame JPEG bytes and Foundation diagnostics remain
in the Streamlit session until a new run or page reset.

The ZIP contains `foundation_video_frames.csv` with source video, frame index,
timestamp, quality metrics and perceptual hash, plus
`foundation_video_summary.csv` with sampling/rejection totals. Videos themselves
are not copied into the dataset. Review all AI boxes and empty-label frames in
Roboflow before training.

## Sidebar configuration

The test lab reads defaults from the current backend `app/config.py`.

You may temporarily test:

- SAM points stride
- SAM mask quality
- Mask NMS IoU
- mask geometry filters
- DINO similarity threshold
- DINO similarity margin
- CPU / CUDA

Changes in the Streamlit sidebar do not modify production configuration.

Click **Apply & reload engine** after changing parameters.

## Preparing white-background references

The Streamlit Test Lab now exposes the same safe operation in **Foundation
reference manager → Add a class batch to the queue**:

1. Select **Already-tight product crops** to preserve the current frame, or
   **Full one-product photos** to crop with SAM first.
2. Select **Original background**, **White background only**, or **Both original
   + white**. The mixed `both` mode is recommended for runtime domain coverage.
3. For white output, click **Preview white-background result for first image**.
   The preview can be downloaded as an opaque PNG before anything is queued.
   The quality line reports which proposal supplied the final silhouette and
   how much tray/shadow cleanup was required.
4. Add the batch, inspect the pending queue's `background` column, then use
   **Build & activate**. The manager writes the image variants and matching DINO
   rows in one backed-up atomic transaction; an unsafe mask fails the source
   image instead of saving a damaged reference.

This manager writes to the real local reference library when **Build & activate**
is confirmed. Restart the FastAPI backend afterward if it had already loaded the
old Foundation artifact.

Use the offline reference preparation tool; it never overwrites source images:

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\prepare_foundation_references.py `
  --source "D:\raw-references\SKU" `
  --output "D:\approved-references\SKU" `
  --background both
```

`both` writes the original tight crop and an opaque white-background PNG. This
mixed reference set is safer than replacing every reference with white-only
images because production crops may retain a narrow tray/background border.
Review `crop_report.json` and every `_white.png` before adding the folder and
rebuilding `reference_embeddings.npz`.

For references that are already tightly cropped, add `--input-mode tight`.
This preserves their exact frame and uses SAM only to extract the foreground
silhouette. The default `--input-mode auto-crop` remains suitable for raw
one-product photos.

White-background masking uses a guarded hierarchy:

1. colour proposals may locate the bakery item and define the crop;
2. an overlapping high-quality SAM proposal is preferred for the final edge;
3. Lab-colour GrabCut runs only when no aligned SAM silhouette is available;
4. unsafe coverage or an excessive contraction fails for manual review.

This distinction prevents a warm/dark cast shadow from being kept merely
because it is connected to the product, while avoiding GrabCut damage to flour,
seeds, scoring marks or pale crust. Always inspect a strong-correction warning
before activating the batch.
