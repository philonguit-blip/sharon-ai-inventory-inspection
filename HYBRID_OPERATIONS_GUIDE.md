# Hướng dẫn vận hành hệ thống Hybrid

## 1. Hệ thống quyết định như thế nào

| Chế độ | Luồng | Dùng khi nào |
|---|---|---|
| `AUTO` | YOLO trước; Foundation xác minh theo batch các box dưới threshold rồi mới fallback toàn ảnh khi cần | Vận hành hằng ngày |
| `YOLO` | Chỉ YOLO26s | Cần tốc độ tối đa, SKU đã train |
| `FOUNDATION` | SAM2 tạo candidate; DINOv2 xác thực từng object | Thử SKU mới/chẩn đoán |
| `COMPARE` | Chạy hai engine và so SKU + count | Kiểm định trước khi promote |

Trong `AUTO`, mỗi ảnh đi qua ba tầng:

1. box YOLO vượt threshold được giữ nguyên;
2. box YOLO dưới threshold nhưng còn ít nhất `50%` threshold được gom lại, loại
   proposal trùng detection tốt, mở rộng `12%`, rồi gửi trong **một** lượt prompt
   SAM2 và một batch DINOv2;
3. chỉ box được Foundation xác nhận đúng cùng class mới được phục hồi. Foundation
   chọn class khác sẽ không tự thay SKU mà kích hoạt so sánh toàn ảnh. Ảnh không
   có proposal hoặc vẫn không chắc tiếp tục dùng fallback Foundation toàn ảnh.

Mặc định mỗi ảnh chỉ gửi tối đa 30 box nghi vấn. Các giá trị có thể điều chỉnh
qua `HYBRID_BOX_RESCUE_*` trong `.env`; không nên hạ rescue floor hoặc tăng giới
hạn khi chưa benchmark trên ground truth vì sẽ làm tăng false positive và thời
gian CPU.

Nếu hai engine đồng thuận, giao diện vẫn yêu cầu xác nhận. Nếu bất đồng, job
chuyển `REVIEW`; người dùng phải chọn một SKU trong các candidate và nhập count
đúng. Không có nhánh nào tự tạo phiếu KiotViet trước xác nhận.

Trên máy CPU hiện tại, ảnh kiểm thử thật cho kết quả:

- YOLO: đúng SKU/count 5, khoảng 3 giây;
- Foundation baseline cũ: khoảng 30 giây/ảnh; bản hiện tại resize riêng nhánh
  SAM và giảm prompt, cần đo lại trên bộ validation trước khi chốt SLA.

Vì vậy không dùng Foundation cho mọi ảnh. Nó giải quyết nút thắt onboarding và
ảnh khó, còn YOLO giữ throughput production.

## 2. Asset đang dùng

- YOLO: `backend/models/best_YOLO26s_2429_27SKUs.pt` (YOLO26s, 27 visual classes).
- SAM: `backend/models/sam2.1_s.pt` (tải ngoài Git, checksum trong
  `backend/models/FOUNDATION_MANIFEST.json`).
- DINOv2: `facebook/dinov2-small`, cache trong user profile.
- Reference: `backend/hybrid_data/reference_embeddings.npz`, 713 vector × 384
  chiều cho 40 reference class.
- Metadata: `backend/config/hybrid_reference_registry.json`.

Ảnh crop gốc nằm tại `backend/hybrid_data/references/` và không commit. Sao lưu
thư mục này vào R2/kho nội bộ có kiểm soát nếu cần rebuild artifact.
Ảnh toàn khung cũ của `BR-GF-00000155` đã được giữ tại
`backend/hybrid_data/reference_originals_archive/`; bộ đang hoạt động là crop
sát sản phẩm và artifact 713 vector/40 class đã được rebuild ngày 20/08/2026.

## 3. Khởi động hằng ngày

1. Chạy `start_backend.bat`; đợi `Application startup complete`.
2. Chạy `start_worker.bat`; đợi heartbeat không còn báo lỗi.
3. Mở URL webhook production.
4. Chọn `AUTO`, tải 1–50 ảnh hoặc chọn một thư mục ảnh của cùng một SKU.
5. Mỗi ảnh phải là một khay/lô cần cộng riêng; không tải nhiều góc của cùng
   một khay vì hệ thống sẽ cộng tất cả các ảnh.
6. Theo dõi tiến độ `Đã xử lý n/tổng ảnh`.
7. Kiểm tra SKU và tổng count, sửa tổng nếu cần, rồi xác nhận tạo phiếu nháp.

Khi các ảnh cùng SKU bị AI dự đoán thành nhiều class, backend chuyển job sang
`REVIEW`, giữ tổng count và yêu cầu người vận hành chọn lại sản phẩm trong catalog.
Ảnh không có detection vẫn được đánh dấu rõ để người dùng sửa tổng count. Không
có phiếu KiotViet nào được tạo trước bước xác nhận thủ công này.

Batch R2 được nhận ngay rồi tải nền. YOLO xử lý mặc định 4 ảnh/lượt
(`BAKERY_INFERENCE_BATCH_SIZE=4`) để giảm overhead. Nếu thiết bị không đủ RAM
hoặc runtime không hỗ trợ batch, service tự quay về xử lý từng ảnh mà không làm
hỏng cả job. FOUNDATION vẫn chạy tuần tự cho các ảnh mà AUTO đánh giá là khó.

Nếu worker từng báo `Task lease does not belong to this worker`, phải import
workflow JSON mới và khởi động lại worker. Worker mới có session ID riêng; n8n
tự trả các task đang bị lease bởi phiên cũ về hàng đợi và chỉ cho worker vừa
heartbeat nhận lại lease.

Foundation được lazy-load: nếu YOLO rõ và không có box dưới threshold đủ điều
kiện rescue thì SAM/DINO không chiếm thêm thời gian inference. Lần box rescue
hoặc fallback đầu tiên sau khi bật máy có thể chậm hơn do nạp model vào RAM.

## 4. Thêm SKU mới không phải train ngay

Chuẩn bị 12–24 ảnh crop sát một sản phẩm, padding 5–10%, gồm vài góc xoay, biến
thiên ánh sáng và kích thước nhưng vẫn cùng kiểu chụp production. Không embed
nguyên ảnh khay 4080×3060 vì nền inox/nylon sẽ lấn át đặc trưng sản phẩm.

Nếu nguồn hiện là ảnh toàn khung một sản phẩm đặt gần tâm, tạo thư mục crop để
review trước (lệnh không ghi đè ảnh nguồn):

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\prepare_foundation_references.py `
  --source "D:\raw-references\BR-NEW-0001" `
  --output "D:\approved-crops\BR-NEW-0001" `
  --background both
```

`--background both` tạo đồng thời crop nền thực tế `.jpg` và bản tách nền trắng
`_white.png`. Ảnh nguồn không bao giờ bị ghi đè. Chế độ này được khuyến nghị vì
giữ được đặc trưng sản phẩm sạch nhưng vẫn tránh lệch miền so với ảnh khay thật.
Có thể chọn riêng `original` hoặc `white`; không nên chỉ dùng toàn ảnh nền trắng
trong production nếu crop runtime vẫn còn nền khay.

Mở và kiểm tra toàn bộ crop; chỉ tiếp tục khi mỗi ảnh chứa đúng một sản phẩm,
không cắt mất mép bánh, nền trắng không ăn vào thân bánh và không giữ lại khay.
Các ảnh không có mask căn chỉnh đủ an toàn được ghi `REVIEW` trong
`crop_report.json`, không bị ép tách nền.

Nếu đầu vào đã là crop reference được duyệt và chỉ cần đổi nền, dùng chế độ
`tight` để giữ nguyên kích thước/khung ảnh và chỉ nhờ SAM lấy silhouette:

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\prepare_foundation_references.py `
  --source ".\backend\hybrid_data\references\BR-EXAMPLE" `
  --output "D:\review-white\BR-EXAMPLE" `
  --input-mode tight `
  --background both
```

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\hybrid_reference_manager.py add `
  --product-code BR-NEW-0001 `
  --product-name "Tên sản phẩm" `
  --source "D:\approved-crops\BR-NEW-0001"

.\backend\.venv\Scripts\python.exe .\scripts\hybrid_reference_manager.py build
```

Lệnh `add` đồng thời đăng ký SKU Foundation-only vào catalog xác nhận đóng của
web. Vì vậy sản phẩm mới có thể được chọn đúng ở bước review mà không cần sửa
tay frontend hoặc workflow n8n.

Khởi động lại backend để service đọc artifact mới. Sau đó:

1. chạy 20–30 ảnh khay ở `FOUNDATION`;
2. chạy lại cùng bộ ở `COMPARE`;
3. ghi accuracy SKU, sai số count và thời gian;
4. chỉ cho `AUTO` sử dụng khi không có nhầm SKU nguy hiểm;
5. tiếp tục thu dữ liệu xác nhận để train YOLO theo batch nhiều SKU.

Muốn seed lại 14 class hiện tại từ dữ liệu nghiên cứu:

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\hybrid_reference_manager.py bootstrap-existing --max-per-class 24
.\backend\.venv\Scripts\python.exe .\scripts\hybrid_reference_manager.py bootstrap-yolo --max-per-class 24
.\backend\.venv\Scripts\python.exe .\scripts\hybrid_reference_manager.py build
```

## 5. Dữ liệu xác nhận và pseudo-label

Sau khi KiotViet tạo phiếu thành công, backend ghi audit tại
`backend/runtime/hybrid_dataset/`:

- `images/`: ảnh nguồn;
- `labels/`: YOLO labels chỉ khi số box bằng count xác nhận;
- `metadata/`: engine, SKU, count, trạng thái review;
- `classes.json`: class ID ổn định theo productCode.

Endpoint thống kê: `GET /api/v1/bakery/hybrid/dataset`.

Nếu người dùng sửa count khác số box, record là `VERIFIED_COUNT_ONLY` và không
tạo label train-ready. Trước khi train, vẫn cần review mẫu `train_ready`; pseudo
label giúp giảm công sức, không thay thế QA.

## 6. Ngưỡng an toàn

- DINO tray similarity: `0.72`.
- DINO tray margin với class thứ hai: `0.04`.
- DINO từng object: similarity `0.60`, margin `0.02`.
- Lọc diện tích sau semantic: `0.35×–2.50×` median của object hợp lệ.
- Candidate semantic chồng phủ cùng vật thể: giữ score cao hơn khi coverage ≥ `0.45`.
- SAM input tối đa: cạnh dài `1280` px; crop DINO/ảnh annotate vẫn lấy từ ảnh gốc.
- SAM point stride: `96` (thường khoảng 100–140 prompt tùy tỉ lệ ảnh).
- SAM mask quality: `0.55`.
- Loại mask chạm biên: `0.5%` kích thước ảnh.
- Loại bounding box lớn hơn `22%` diện tích ảnh.

Các ngưỡng dựa trên giả định vận hành: ảnh từ trên xuống, thấy trọn sản phẩm,
không che khuất và mỗi khay chỉ có một loại bánh. Nếu quy trình chụp thay đổi,
phải chạy lại bộ validation trước khi chỉnh `.env`.

Các biến tương ứng: `FOUNDATION_SAM_MAX_SIDE`, `FOUNDATION_POINTS_STRIDE`,
`FOUNDATION_INSTANCE_SIMILARITY_THRESHOLD`,
`FOUNDATION_INSTANCE_SIMILARITY_MARGIN`,
`FOUNDATION_INSTANCE_MIN_AREA_FACTOR` và
`FOUNDATION_INSTANCE_MAX_AREA_FACTOR`, `FOUNDATION_INSTANCE_BOX_COVERAGE_NMS`.
Nếu đếm dư, thử tăng instance threshold
từ `0.60` lên `0.65`; nếu bỏ sót, ưu tiên giảm về `0.55` hoặc giảm stride về
`80`, mỗi lần chỉ đổi một biến và chạy lại cùng bộ ground truth.

Kiểm tra GPU:

```powershell
.\backend\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"
```

Máy hiện tại không có `nvidia-smi` và đang dùng PyTorch CPU, vì vậy không đặt
`cuda:0` thủ công. Chỉ dùng CUDA sau khi máy có NVIDIA GPU, driver và PyTorch
CUDA tương thích.

## 7. Cập nhật n8n sau thay đổi giao diện/backend

```powershell
node n8n\generate_workflow4_outbound.mjs
node n8n\test_workflow4_outbound.mjs
```

Import lại `n8n/Workflow 4_ Sharon Bakery Outbound Worker.json`, gán credential
Basic Auth, activate workflow. Generator nhúng cùng frontend local, nên không
sửa trực tiếp HTML/JS bên trong JSON.

## 8. Kiểm tra và phục hồi

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\verify_production.py --require-foundation
cd backend
.\.venv\Scripts\python.exe -m pytest tests -q
cd ..
node n8n\test_workflow4_outbound.mjs
```

Nếu SAM thiếu, chạy `scripts/provision_foundation.py`. Nếu reference artifact
thiếu, phục hồi file NPZ đã backup hoặc build lại từ crop. Khi Foundation lỗi,
`AUTO` vẫn giữ YOLO; `FOUNDATION`/`COMPARE` trả lỗi rõ ràng và không ghi KiotViet.

## 9. Quy hoạch file

- Production source: `backend/`, `n8n/`, `scripts/`, `docs/`.
- Nghiên cứu còn dùng: `pre-annotation/`, `sharon-cv-pipeline/`, `data/` (không
  commit vì lớn).
- File cũ/venv/model/workflow không còn dùng: `_archive/legacy-2026-08-10/`.
- Không xóa archive cho đến khi hệ thống mới chạy ổn qua ít nhất một chu kỳ QA.
