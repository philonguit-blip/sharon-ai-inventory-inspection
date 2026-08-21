# Hướng dẫn cài đặt và khởi động Sharon Bakery AI Inventory

Tài liệu này là quy trình chuẩn để cài lại hệ thống trên máy Windows mới hoặc
khởi động lại sau khi tắt máy.

## 1. Thành phần production

- `backend/`: FastAPI, worker, Hybrid Router, YOLO, SAM2/DINOv2, mapping và giao diện.
- `n8n/Workflow 4_ Sharon Bakery Outbound Worker.json`: workflow production 30 node.
- `start_backend.bat`: khởi động AI backend local tại `127.0.0.1:8080`.
- `start_worker.bat`: kết nối outbound tới n8n và mở giao diện kiểm đếm.
- `scripts/verify_production.py`: kiểm tra model, checksum, mapping và workflow.
- `scripts/provision_foundation.py`: tải/kiểm checksum SAM2 và cache DINOv2.

Mỗi job nhận **1–50 ảnh hoặc một thư mục ảnh của cùng một SKU**, tối đa 200 MB.
Count của từng ảnh được cộng thành tổng job. Không tải nhiều góc của cùng một
khay vật lý nếu không muốn cộng trùng. Người dùng phải xác nhận kết quả trước
khi hệ thống tạo một phiếu nhập nháp KiotViet.

## 2. Cài lần đầu trên máy mới

Yêu cầu:

- Windows 10/11 64-bit.
- Git.
- Python 3.12 x64, có Python Launcher `py.exe`.
- Node.js 20 trở lên để kiểm tra/generate workflow n8n.

Mở PowerShell:

```powershell
git clone https://github.com/philonguit-blip/sharon-ai-inventory-inspection.git
cd sharon-ai-inventory-inspection
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

Setup mặc định cài dependency, tải SAM2.1 Small, cache DINOv2, xác minh 713
reference embeddings và chạy test. Nếu chỉ cần khôi phục YOLO tạm thời có thể
dùng `.\setup_windows.ps1 -SkipFoundation`; `AUTO` vẫn chạy an toàn bằng YOLO.

Sau đó mở `backend/.env` và điền credential thật cho:

- Cloudflare R2;
- KiotViet;
- Basic Auth của ứng dụng và n8n.

Giữ nguyên các giá trị production sau:

```dotenv
YOLO_MODEL_PATH=models/best_YOLO26s_2429_27SKUs.pt
MAX_IMAGES_PER_JOB=50
MAX_JOB_UPLOAD_SIZE_BYTES=209715200
KIOTVIET_CREATE_AS_DRAFT=false
KIOTVIET_AUTO_CREATE_DRAFT=false
R2_TRANSFER_WORKERS=1
BAKERY_INFERENCE_BATCH_SIZE=4
HYBRID_ENABLED=true
HYBRID_DEFAULT_MODE=AUTO
```

Không commit `backend/.env`.

Nếu máy thiếu RAM khi xử lý nhiều ảnh, giảm
`BAKERY_INFERENCE_BATCH_SIZE=4` xuống `2` hoặc `1`. Không tăng quá `8` khi chưa
đo bộ nhớ thực tế.

## 3. Import/cập nhật workflow n8n

1. Chạy `node n8n\generate_workflow4_outbound.mjs`.
2. Import `n8n/Workflow 4_ Sharon Bakery Outbound Worker.json` vào n8n.
3. Gán credential Basic Auth cho các webhook, dùng cùng username/password với
   `APP_AUTH_USERNAME` và `APP_AUTH_PASSWORD`.
4. Activate workflow.
5. Mở:
   `https://n8n.sharon-finefoods.com/webhook/sharon-bakery-inventory`.

Workflow đã nhúng trực tiếp HTML/CSS/JS được sinh từ `backend/frontend`, vì vậy
không cần duy trì một bản giao diện riêng trong JSON.

## 4. Khởi động hằng ngày sau khi bật máy

1. Nếu cần tạo **Phiếu sản xuất**, mở Docker Desktop và chạy
   `start_manufacturing_rpa.bat`. Lần đầu phải chạy
   `..\sharon-bakery-docker_manufacturing\setup-rpa.ps1` để nhập tài khoản web
   KiotViet và đồng bộ token nội bộ.
2. Chạy `start_backend.bat`.
3. Chờ backend báo `Application startup complete`.
4. Chạy `start_worker.bat`.
5. Giữ các cửa sổ terminal đang dùng mở.
6. Chờ giao diện báo **Hệ thống sẵn sàng** rồi mới upload ảnh.
7. Chọn `AUTO` cho công việc hằng ngày. Chỉ dùng `FOUNDATION`/`COMPARE` khi
   onboarding hoặc kiểm định SKU.

Địa chỉ local: <http://127.0.0.1:8080>.

Không mở `http://0.0.0.0:8080`.

## 5. Dừng và khởi động lại

Nhấn `Ctrl+C` ở cửa sổ worker trước, sau đó nhấn `Ctrl+C` ở backend. Khi cần
chạy lại, thực hiện lại hai bước trong mục 4. Không mở nhiều worker hoặc nhiều
backend cùng lúc.

## 6. Kiểm tra sau cập nhật

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\verify_production.py
cd backend
.\.venv\Scripts\python.exe -m pytest tests -q
cd ..
node n8n\generate_workflow4_outbound.mjs
node n8n\test_workflow4_outbound.mjs
node --check backend\frontend\assets\app.js
```

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\verify_production.py --require-foundation
.\backend\.venv\Scripts\python.exe .\scripts\hybrid_reference_manager.py status
```

## 7. Lỗi thường gặp

- `WinError 10048`: cổng 8080 đã có tiến trình khác; không chạy backend lần hai.
- `Worker offline`: kiểm tra cửa sổ worker, Internet, credential n8n và chế độ Sleep.
- `CONFIRMING` kèm lỗi: kiểm tra KiotViet rồi bấm xác nhận lại. Backend chỉ đối
  soát theo `job_id`, không tự tạo phiếu thứ hai.
- `R2 object not found`: kiểm tra manifest/job và object key; lỗi kết nối R2 sẽ
  được báo riêng, không còn bị coi nhầm là file không tồn tại.
- Model checksum sai: phục hồi đúng file trong `backend/models` theo
  `backend/models/MODEL_MANIFEST.json`.
- `Foundation engine is not provisioned`: chạy
  `.\backend\.venv\Scripts\python.exe .\scripts\provision_foundation.py`.
- `REVIEW`: hai engine không đồng thuận. Chọn đúng SKU, sửa count nếu cần rồi
  mới xác nhận; không xem đây là lỗi worker.

## 8. Quy tắc cập nhật model

Chỉ đặt model detector đã sẵn sàng kiểm thử/vận hành trong `backend/models`.
Dataset, pre-annotation, report và model nghiên cứu chưa kiểm tra không được
trộn vào backend. Khi promote model mới phải cập nhật đồng thời manifest,
mapping, test và README.

Chi tiết thêm SKU nhanh, build reference và khai thác pseudo-label nằm trong
[HYBRID_OPERATIONS_GUIDE.md](HYBRID_OPERATIONS_GUIDE.md).

## 9. Thay model hoặc confidence từ bảng Developer

1. Chép model detector `.pt` hoặc `.onnx` vào `backend/models`.
2. Trên web, giữ logo **SB** khoảng 1,2 giây hoặc nhấn `Ctrl+Shift+D`.
3. Nhập `DEVELOPER_SETTINGS_KEY` (mặc định dùng mật khẩu web nếu chưa cấu hình
   khóa riêng).
4. Chọn model, sửa confidence từng class rồi bấm **Kiểm tra và áp dụng**.
5. Chờ thông báo thành công; không tắt backend/worker khi model đang nạp.

Thay đổi được ghi vào `backend/runtime/developer_settings.json`. Nếu model lỗi
hoặc class không khớp mapping, hệ thống từ chối thay đổi và tiếp tục dùng model
trước đó. Job đã bắt đầu cũng không bị đổi model giữa chừng.
