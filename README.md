# Hệ thống AI Kiểm kê Kho (AI Inventory Inspection System)

Hệ thống Computer Vision tự động hóa quy trình kiểm kê kho, giám sát hàng hóa và đồng bộ dữ liệu hạ tầng cho SharonBakery.

> **Triển khai hiện tại (03/08/2026):** hệ thống dùng n8n webhook và worker local
> chủ động kết nối ra ngoài, không còn phụ thuộc Cloudflare Tunnel cho luồng kiểm
> đếm bánh. Xem [N8N_OUTBOUND_WORKER.md](N8N_OUTBOUND_WORKER.md). Phần hướng dẫn
> Tunnel bên dưới chỉ còn dành cho pipeline video cũ/tương thích ngược.

## Vận hành hằng ngày sau khi bật lại máy

Luồng kiểm đếm hiện tại cần hai tiến trình trên máy AI: FastAPI và outbound
worker. Không cần chạy Cloudflare Tunnel.

### Cách nhanh nhất trên Windows

1. Mở thư mục dự án:

   ```text
   C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection
   ```

2. Nhấp đúp `start_backend.bat` và giữ cửa sổ này mở. Chờ thấy:

   ```text
   Uvicorn running on http://127.0.0.1:8080
   ```

3. Nhấp đúp `start_worker.bat` và giữ cửa sổ này mở. Worker sẽ chủ động kết
   nối đến n8n; không mở cổng Internet vào máy.

4. Mở giao diện ở một trong hai địa chỉ:

   - Máy AI: `http://127.0.0.1:8080`
   - Máy khác/điện thoại: `https://sharon-bakery-inventory.pages.dev/`

5. Đăng nhập bằng `APP_AUTH_USERNAME` và `APP_AUTH_PASSWORD` trong
   `backend/.env`. Không gửi file `.env` hoặc chụp màn hình mật khẩu.

6. Kiểm tra góc trên bên phải phải hiện **Hệ thống sẵn sàng** trước khi chọn
   ảnh.

### Chạy bằng PowerShell nếu không dùng file BAT

Terminal 1 — backend:

```powershell
cd C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Terminal 2 — worker:

```powershell
cd C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection
.\start_worker.bat
```

Không truy cập `http://0.0.0.0:8080`; đây chỉ là địa chỉ bind. Không đóng hai
cửa sổ terminal trong lúc cửa hàng còn sử dụng hệ thống.

### Tắt hệ thống đúng cách

Nhấn `Ctrl+C` trong cửa sổ worker, sau đó nhấn `Ctrl+C` trong cửa sổ backend.
Nếu Windows tắt hoặc khởi động lại đột ngột, các job đang chạy sẽ được đánh dấu
`ERROR` và cần gửi lại bộ ảnh.

### Cấu hình vận hành hiện tại

| Cấu hình | Giá trị |
|---|---:|
| Số ảnh tối đa mỗi lượt | 50 |
| Dung lượng tối đa mỗi ảnh | 50 MB |
| Tổng dung lượng mỗi lượt | 160 MB |
| Confidence tối thiểu | 0.55 |
| IOU | 0.50 |
| Kích thước suy luận | 1280 px |

Ảnh được upload thẳng từ trình duyệt lên Cloudflare R2. n8n chỉ giữ metadata,
hàng đợi và trạng thái; outbound worker tải ảnh về máy AI, xử lý tuần tự để giữ
độ ổn định, đồng thời cập nhật tiến độ `1/N`, `2/N` sau từng ảnh.

### Kiểm tra và khắc phục nhanh

- **Outbound AI worker is offline:** kiểm tra `start_worker.bat` còn chạy và
  máy AI không ở chế độ Sleep.
- **Không mở được local:** kiểm tra backend có báo cổng 8080 đang được dùng hay
  không; chỉ chạy một backend.
- **Job đứng lâu:** xem `backend/runtime/jobs/<job_id>/job.json`; nếu timestamp
  không đổi, dừng worker rồi backend, sau đó khởi động lại theo đúng thứ tự.
- **Job not found:** tải lại trang. Frontend sẽ tự chờ và gửi lại cùng job theo
  cơ chế idempotent.
- **Đổi mật khẩu:** phải cập nhật đồng thời `APP_AUTH_PASSWORD` trong `.env` và
  credential Basic Auth của workflow n8n, rồi restart backend và worker.
- **Máy khác vẫn thấy cấu hình cũ:** nhấn `Ctrl+F5` hoặc đóng tab rồi mở lại URL
  Pages.

### Bảo mật credential

- Chỉ giữ credential thật trong `backend/.env`; không gửi file này qua email/chat và
  không đưa vào workflow JSON hoặc mã nguồn.
- Các file workflow cũ và bản sao `.env` trong repository đã được thay credential bằng
  `REDACTED_ROTATE_REQUIRED`. Không dùng những file này để chạy production.
- Nếu repository từng được chia sẻ ra ngoài, hãy tạo khóa R2 mới, cập nhật
  `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY`, kiểm tra upload thành công rồi mới thu hồi
  khóa cũ để tránh làm gián đoạn hệ thống.

### Cài lại môi trường khi `.venv` bị mất

```powershell
cd C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sau khi sửa giao diện trong `backend/frontend`, triển khai lại Cloudflare Pages:

```powershell
cd C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection
npx --yes wrangler@4.30.0 pages deploy backend\frontend --project-name sharon-bakery-inventory --branch main
```

---

# 1. Kiến trúc hệ thống (System Architecture)

Hệ thống được thiết kế theo kiến trúc microservices nhẹ, giao tiếp thông qua webhook và RESTful API.

## Các thành phần chính

### ChatOps Interface
- **Telegram API**
- Giao diện tương tác thời gian thực để gửi ảnh và nhận báo cáo.

### Orchestration Layer
- **n8n**
- Điều phối workflow, ETL pipeline và tự động hóa quy trình Purchase Order.

### Storage Layer
- **Cloudflare R2** → Lưu trữ ảnh gốc
- **Supabase / PostgreSQL** → Lưu dữ liệu kiểm kê

### AI Inference Layer
- **FastAPI**
- **Ultralytics YOLOv8**
- **ONNX Runtime**

Chức năng:
- Phát hiện vật thể
- Phân loại sản phẩm
- Tổng hợp số lượng

### Network Layer
- **Cloudflare Zero Trust Tunnel**
- Cung cấp kết nối bảo mật từ bên ngoài vào FastAPI nội bộ

---

# 2. Điều kiện tiên quyết (Prerequisites)

Đảm bảo môi trường triển khai có sẵn các thành phần sau:

## Môi trường runtime
- Python 3.10+
- Node.js (nếu chạy n8n local)
- Hoặc tài khoản n8n Cloud

## Hạ tầng
- Cloudflared CLI
- Tài khoản Supabase
- Tài khoản Cloudflare R2

---

# 3. Hướng dẫn triển khai (Deployment Guide)

---

## 3.1 Thiết lập cơ sở dữ liệu (Supabase Setup)

1. Truy cập **Supabase Dashboard**
2. Chọn project
3. Mở **SQL Editor**
4. Chạy đoạn SQL sau:

```sql
CREATE TABLE inventory_records (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('Asia/Ho_Chi_Minh', NOW()),
    telegram_user_id BIGINT NOT NULL,
    raw_image_url TEXT NOT NULL,
    object_class TEXT NOT NULL,
    quantity INTEGER NOT NULL
);
````

Sau khi tạo xong:

Vào:

```text
Project Settings → API
```

Lấy các thông tin:

* Project URL
* Service Role Secret

---

## 3.2 Triển khai AI Backend (FastAPI)

Di chuyển vào thư mục backend:

```bash
cd backend
```

### Tạo môi trường ảo

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Chuẩn bị model AI

Đảm bảo file model nằm đúng vị trí:

```text
models/yolov8n.onnx
```

### Khởi động FastAPI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### Kiểm tra trạng thái hệ thống

```text
http://localhost:8080/health
```

---

## 3.3 Thiết lập Cloudflare Tunnel

Expose server local để n8n có thể gọi được:

```bash
cloudflared tunnel --url http://127.0.0.1:8080
```

Sau khi chạy, copy URL:

```text
https://xxxxx.trycloudflare.com
```

URL này sẽ dùng cho node HTTP Request trong n8n.

---

## 3.4 Thiết lập workflow n8n

Mở giao diện n8n.

### Cấu hình Credentials

#### Telegram API

Nhập:

* Bot Token

#### Cloudflare R2

Nhập:

* Access Key
* Secret Key
* Endpoint URL

#### Supabase API

Nhập:

* Host URL
* Service Role Secret

---

### Import workflow

Import file JSON workflow vào canvas n8n.

---

### Cập nhật endpoint backend

Trong node HTTP Request:

```text
https://<cloudflare-tunnel-url>/api/v1/predict
```

---

### Kích hoạt workflow

Bật chế độ Active để workflow bắt đầu chạy.

---

# 4. Quy trình thu thập dữ liệu & huấn luyện (Data Collection & Training SOP)

Hiện tại hệ thống đang sử dụng model COCO pre-trained.

Để fine-tune theo SKU đặc thù của SharonBakery:

---

## Quay video baseline

Thực hiện:

```text
10 giây / SKU
```

Mục tiêu:

* Lấy đặc trưng hình dạng chuẩn
* Tạo dữ liệu sạch

---

## Quay video ngữ cảnh thực tế

Thực hiện:

```text
15 giây / khu vực kệ hoặc pallet
```

Mục tiêu:

* Thu dữ liệu che khuất (occlusion)
* Thu bố cục thực tế
* Thu dữ liệu môi trường thật

---

## Trích xuất khung hình

Sử dụng FFmpeg:

```bash
ffmpeg -i input.mp4 -vf fps=2 output_%04d.jpg
```

---

## Quy chuẩn gán nhãn (Annotation)

Sử dụng multi-class bounding box.

Nguyên tắc:

* Gán nhãn tất cả sản phẩm nhìn thấy
* Bounding box ôm sát vật thể
* Giữ lại các mẫu khó
* Giữ lại các trường hợp che khuất dưới 80%
* Giữ nguyên ngữ cảnh shelf clutter

---

# 5. Các vấn đề hiện tại (Known Issues)

## Tích hợp Microsoft Teams

Hiện tại đang dùng Telegram làm giải pháp tạm thời do việc xác thực webhook nội bộ với Microsoft Teams chưa ổn định.

Cần triển khai:

* Azure Bot Service
* Microsoft Graph API ổn định hơn

---

## Truy cập web từ xa

Giao diện vận hành đã được triển khai tại:

```text
https://sharon-bakery-inventory.pages.dev/
```

Luồng từ xa dùng Cloudflare Pages + webhook n8n + outbound worker. FastAPI
không cần mở cổng internet và không còn phụ thuộc vào Quick Tunnel. Trên máy AI
cần chạy đồng thời backend và `start_worker.bat`; xem hướng dẫn chi tiết trong
`N8N_OUTBOUND_WORKER.md`.

---

# 6. Lộ trình tiếp theo (Next Steps)

## Nâng cấp AI

* Huấn luyện model YOLOv8 custom cho hơn 40 SKU
* Tăng độ chính xác inference

## Tích hợp OCR

* Thêm OCR để xác thực chéo nhãn sản phẩm

## Hạ tầng

* Docker hóa toàn bộ hệ thống
* Theo dõi heartbeat và cảnh báo khi outbound worker ngừng hoạt động

## Vận hành

* Thực thi OPS-001 để harden deployment
* Chuẩn hóa môi trường production
