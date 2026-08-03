# Sharon Bakery - local image counting pipeline

Pipeline này nhận một hoặc nhiều ảnh qua FastAPI, chạy model bakery, cộng số
lượng theo mã KiotViet và tạo file Excel nhập hàng từ template.

Giới hạn hiện tại là `50 MB/ảnh`, tối đa `50 ảnh/job` và tổng dung lượng
`160 MB/job`. Ảnh được upload trực tiếp lên R2 nên không đi qua request body
của n8n/Cloudflare Pages. Với bộ ảnh lớn hơn 160 MB, chia thành nhiều job.

## 1. Khởi động backend

Mở PowerShell tại thư mục `backend`:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-milestone2.txt
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Swagger UI:

```text
http://127.0.0.1:8080/docs
```

Giao diện vận hành dành cho người dùng:

```text
http://127.0.0.1:8080/
```

Giao diện production qua Cloudflare Pages:

```text
https://sharon-bakery-inventory.pages.dev/
```

Trên giao diện này có thể:

- Chọn một/nhiều ảnh hoặc chọn cả thư mục ảnh.
- Theo dõi tiến trình AI và số ảnh đã xử lý.
- Xem mã hàng, tên hàng, tổng số lượng và ảnh đã đánh dấu.
- Tải file Excel nhập hàng.
- Tự động gọi workflow n8n bằng gateway cùng origin; credential không nằm trong trình duyệt.
- Upload tối đa bốn ảnh song song trực tiếp lên R2.
- Tự kiểm tra dữ liệu và tạo phiếu nhập nháp khi job hoàn tất.
- Khôi phục job đang chạy nếu người dùng tải lại trang.

Khi `KIOTVIET_AUTO_CREATE_DRAFT=true`, nút **Kiểm đếm và tạo phiếu nháp** sẽ chạy toàn
bộ pipeline. Mỗi `job_id` chỉ có thể tạo tối đa một phiếu; submit lại cùng job chỉ trả
trạng thái hiện có và không chạy AI lần hai.

Kiểm tra model và template:

```text
GET /api/v1/bakery/health
```

## 2. Gửi ảnh

Một ảnh:

```powershell
curl.exe -X POST "http://127.0.0.1:8080/api/v1/bakery/jobs" `
  -F "files=@C:\Images\tray_01.jpg"
```

Nhiều ảnh hoặc toàn bộ ảnh trong một folder được gửi thành nhiều trường
`files` trong cùng multipart request:

```powershell
curl.exe -X POST "http://127.0.0.1:8080/api/v1/bakery/jobs" `
  -F "files=@C:\Images\tray_01.jpg" `
  -F "files=@C:\Images\tray_02.jpg"
```

API trả HTTP `202` cùng `job_id` và `status_url`. Việc detect và tạo Excel
tiếp tục chạy ở background.

## 3. Xem kết quả và tải file

```text
GET /api/v1/bakery/jobs/{job_id}
GET /api/v1/bakery/jobs/{job_id}/excel
GET /api/v1/bakery/jobs/{job_id}/annotated/{filename}
```

Các trạng thái job:

- `QUEUED`: đã nhận ảnh.
- `PROCESSING`: đang detect.
- `COMPLETED`: Excel và ảnh annotated đã sẵn sàng.
- `ERROR`: không tạo Excel; xem trường `error`.

## 4. Dữ liệu lưu local

Mỗi job nằm tại:

```text
backend/runtime/jobs/{job_id}/
  job.json
  detections.json
  original/
  annotated/
  output/
```

Ảnh trùng byte trong cùng một job bị từ chối để tránh đếm hai lần. Nếu một
ảnh lỗi, cả job chuyển sang `ERROR` và không tạo file nhập hàng chưa đầy đủ.

## 5. Lưu ý khi đưa lên cloud

`BackgroundTasks` phù hợp với chạy local và một process. Khi triển khai nhiều
worker hoặc cần tự phục hồi sau restart, thay worker local bằng hàng đợi bền
vững như Redis + RQ/Celery. API upload và cấu trúc job hiện tại có thể giữ
nguyên; chỉ thay cơ chế thực thi `_process_job`.

## 6. R2 artifact storage

Khi cấu hình R2 hợp lệ, mỗi job tự upload bốn nhóm artifact:

```text
purchase-intake/{job_id}/original/
purchase-intake/{job_id}/annotated/
purchase-intake/{job_id}/output/
purchase-intake/{job_id}/metadata/
```

Danh sách object đã lưu được trả về trong trường `r2_objects` của job.

## 7. Tự động tạo phiếu nhập nháp KiotViet

Khi `KIOTVIET_AUTO_CREATE_DRAFT=true`, sau khi AI, Excel và R2 hoàn tất, backend
tự động:

1. Đọc chi nhánh Warehouse và đối chiếu mã/tên hàng trực tiếp trên KiotViet.
2. Xác nhận cấu hình bắt buộc là phiếu nháp (`isDraft=true`).
3. Chỉ khi hai bước trên hợp lệ mới tạo một phiếu nhập nháp.
4. Ghi mã phiếu hoặc lỗi vào trường `kiotviet` trong trạng thái job.

Mỗi job chỉ được tạo một phiếu. Endpoint bên dưới được giữ lại để kiểm tra lại
hoặc thử lại thủ công khi lần tự động gặp lỗi.

### Kiểm tra thủ công

Endpoint preview chỉ xác thực token, chi nhánh, mã/tên sản phẩm và dựng payload;
không tạo chứng từ:

```text
GET /api/v1/bakery/jobs/{job_id}/kiotviet-preview
```

POST với `confirm=false` cũng là dry-run:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8080/api/v1/bakery/jobs/{job_id}/kiotviet" `
  -H "Content-Type: application/json" `
  -d '{"confirm":false}'
```

Chỉ lệnh sau mới tạo phiếu nhập nháp thật trên KiotViet:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8080/api/v1/bakery/jobs/{job_id}/kiotviet" `
  -H "Content-Type: application/json" `
  -d '{"confirm":true}'
```

Với cấu hình hiện tại: chi nhánh `Warehouse`, nhà cung cấp để trống, đơn giá
`0`, tiền trả trước `0`, và `isDraft=1`.
