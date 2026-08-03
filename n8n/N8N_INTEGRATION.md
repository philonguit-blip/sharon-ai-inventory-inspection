# Tích hợp Sharon Bakery qua n8n

## Kiến trúc

Không gửi file ảnh lớn xuyên qua n8n. Trình duyệt gọi n8n để xin presigned URL,
upload từng ảnh thẳng lên R2, rồi gọi n8n lần hai để bắt đầu job AI. Cách này tránh
giới hạn request mặc định 16 MiB của n8n và giảm tải RAM cho máy chạy n8n.

Giao diện production không gọi n8n trực tiếp. Nó gọi gateway cùng origin tại
`/api/v1/orchestrator`; gateway giữ Basic Auth trên server rồi chuyển tiếp JSON nhỏ đến
n8n. Người dùng không cần PowerShell và credential n8n không xuất hiện trong JavaScript.

Các webhook production sau khi workflow được activate:

- `GET https://n8n.sharon-finefoods.com/webhook/bakery-health`
- `POST https://n8n.sharon-finefoods.com/webhook/bakery-upload-init`
- `POST https://n8n.sharon-finefoods.com/webhook/bakery-submit`
- `GET https://n8n.sharon-finefoods.com/webhook/bakery-job-status?job_id={job_id}`

## Import workflow

1. Đăng nhập `https://n8n.sharon-finefoods.com`.
2. Chọn **Import from File** và chọn `Workflow 4_ Sharon Bakery R2 Intake.json`.
3. Trong **Credentials**, tạo credential loại **Basic Auth** với tên
   `Sharon Bakery Backend API`.
4. Lấy username và password từ hai biến `APP_AUTH_USERNAME` và
   `APP_AUTH_PASSWORD` trong `backend/.env`. Không dán hai giá trị này vào JSON
   workflow hoặc ghi chúng vào Git.
5. Gán credential này cho bốn node HTTP Request:
   `Backend - Health`, `Backend - Presign R2`, `Backend - Start AI Job`, và
   `Backend - Read Job`.
6. Chạy thử riêng node `00 - Health`. Chỉ activate workflow khi node trả về
   `ready: true`, `r2_configured: true`, và `kiotviet_configured: true`.

Workflow production trỏ đến Named Tunnel cố định:

```text
https://inventory.sharon-finefoods.com
```

Khởi động tunnel bằng `start_tunnel.bat`; tunnel chuyển hostname trên về backend tại
`http://127.0.0.1:8080`.

## CORS bắt buộc trên R2

R2 phải cho phép trang web upload bằng presigned `PUT`. Token R2 hiện tại không có
quyền sửa CORS bucket, vì vậy cấu hình phần này trong Cloudflare Dashboard:

```json
[
  {
    "AllowedOrigins": [
      "https://sharon-finefoods.com",
      "https://www.sharon-finefoods.com",
      "https://inventory.sharon-finefoods.com",
      "https://n8n.sharon-finefoods.com",
      "http://127.0.0.1:8080",
      "http://localhost:8080"
    ],
    "AllowedMethods": ["PUT"],
    "AllowedHeaders": ["Content-Type"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

CORS không làm bucket thành public. Mỗi file vẫn cần presigned URL hợp lệ và URL
upload hết hạn sau 15 phút.

## Giao diện một nút dùng n8n

Giao diện gọi gateway cùng origin mặc định:

```js
window.SHARON_ORCHESTRATOR_BASE = "/api/v1/orchestrator";
```

Luồng sau nút **Kiểm đếm và tạo phiếu nháp** hoàn toàn tự động: xin URL, upload tối đa
bốn ảnh song song lên R2, submit job, theo dõi trạng thái, rồi hiện Excel, ảnh annotated
và mã phiếu nháp KiotViet. Job đang chạy được lưu trong `localStorage` để tải lại trang
không làm mất tiến trình.

## Quy tắc an toàn khi kiểm thử

- Chỉ gọi `bakery-health` để test kết nối ban đầu.
- `bakery-upload-init` chỉ cấp URL và chưa tạo phiếu KiotViet.
- `bakery-submit` bắt đầu YOLO, xuất Excel và có thể tự động tạo phiếu nhập nháp
  KiotViet. Chỉ gọi endpoint này với bộ ảnh thật cần nhập kho.
- Trước khi public cho nhiều máy, bảo vệ các webhook bằng Cloudflare Access hoặc
  một cơ chế xác thực tương đương; không để webhook tạo job mở hoàn toàn trên Internet.
