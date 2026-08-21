# Tích hợp Sharon Bakery qua n8n

## Kiến trúc

Ảnh lớn không đi xuyên qua n8n. Trình duyệt gọi n8n/gateway để xin presigned URL,
upload từng ảnh trực tiếp lên Cloudflare R2, sau đó gửi danh sách `object_key` để tạo
job AI. Cách này tránh đưa binary lớn qua n8n và giảm tải RAM cho máy orchestration.

Luồng production:

```text
Web
→ xin presigned URLs
→ upload 1–50 ảnh/thư mục cùng SKU thẳng lên R2
→ n8n tạo PROCESS task
→ outbound worker lấy task
→ local FastAPI chạy Hybrid theo từng ảnh và cộng count an toàn
→ trả DIRECT / FAMILY / AMBIGUOUS / NO_DETECTION
→ web hiển thị kết quả
→ người dùng xác nhận
→ n8n tạo CONFIRM task
→ outbound worker gọi local /confirm
→ backend tra productCode → productId
→ tạo phiếu nhập nháp KiotViet
→ trả kết quả cuối về web
```

Một job được phép có **1–50 ảnh của cùng một SKU**, tổng tối đa 200 MB. Mỗi ảnh
phải đại diện một khay/lô cần cộng riêng. Backend chặn xác nhận nếu các ảnh
không cùng SKU/family hoặc có ảnh chưa nhận diện an toàn.

Mỗi tiến trình worker có một session ID chứa hostname + PID. Khi worker mới
heartbeat, workflow đưa những task còn bị lease bởi phiên worker cũ về
`QUEUED`. Endpoint kết quả cũng cho worker đang đăng ký nhận lại một stale
lease và worker tự heartbeat/retry đúng một lần. Cơ chế này ngăn vòng lặp
`Task lease does not belong to this worker` sau khi restart hoặc đổi workflow.

Giao diện production ưu tiên gọi gateway cùng origin tại `/api/v1/orchestrator`.
Gateway giữ Basic Auth trên server và chuyển tiếp JSON nhỏ tới n8n. Với static
deployment ở remote mode, frontend có thể gọi trực tiếp webhook n8n bằng phiên
Basic Auth do người vận hành nhập.

## Webhook production

Sau khi workflow được import, gán credential và activate:

- `GET  /webhook/bakery-health`
- `POST /webhook/bakery-upload-init`
- `GET  /webhook/bakery-request-status?request_id={request_id}`
- `POST /webhook/bakery-submit`
- `GET  /webhook/bakery-job-status?job_id={job_id}`
- `POST /webhook/bakery-confirm`
- `GET  /webhook/bakery-worker-next?worker_id={worker_id}`
- `POST /webhook/bakery-worker-result`
- `POST /webhook/bakery-worker-heartbeat`

`bakery-confirm` là bước mới bắt buộc. `bakery-submit` chỉ chạy AI và **không**
được phép tự tạo phiếu KiotViet.

## State machine

```text
WAITING_FOR_WORKER / READY
→ QUEUED
→ PROCESSING
→
  ├─ AWAITING_CONFIRMATION
  │    ├─ DIRECT → xác nhận count
  │    └─ FAMILY → chọn SKU thành viên + xác nhận count
  │
  ├─ NEEDS_RETAKE
  │    └─ ảnh ambiguous / no detection / khác class trong cùng job
  │
  └─ ERROR

AWAITING_CONFIRMATION
→ CONFIRMING
→ COMPLETED
```

Nếu tạo phiếu bị gián đoạn ở bước `CONFIRM`, job giữ trạng thái `CONFIRMING` và
lưu `confirmation_error`. Lần thử sau backend chỉ đối soát receipt theo `job_id`;
nếu chưa tìm thấy receipt, backend không POST lại để tránh tạo trùng.

## Request xác nhận

Direct SKU:

```json
{
  "job_id": "0123456789abcdef0123456789abcdef",
  "confirm": true
}
```

Family SKU:

```json
{
  "job_id": "0123456789abcdef0123456789abcdef",
  "confirm": true,
  "product_code": "BR-SD-0000167"
}
```

n8n kiểm tra `product_code` được chọn có thuộc `decision.members` trước khi tạo
task `CONFIRM`. Backend tiếp tục kiểm tra lại một lần nữa và tra live product trên
KiotViet trước khi tạo phiếu.

## Import Workflow 4

1. Chạy file generator nếu cần tái tạo JSON:

   ```bash
   node generate_workflow4_outbound.mjs
   ```

2. Import file:

   `Workflow 4_ Sharon Bakery Outbound Worker.json`

3. Tạo credential **Basic Auth** với tên:

   `Sharon Bakery Backend API`

4. Dùng cùng username/password với `APP_AUTH_USERNAME` và
   `APP_AUTH_PASSWORD`. Không ghi credential thật trực tiếp vào workflow JSON
   hoặc Git.

5. Gán credential cho tất cả webhook cần xác thực sau khi import.

6. Chạy:

   ```bash
   node test_workflow4_outbound.mjs
   ```

   Test phải kết thúc bằng:

   ```text
   Workflow 4 outbound queue + confirmation simulation passed.
   ```

7. Activate workflow sau khi `bakery-health` trả:

   - `ready: true`
   - `r2_configured: true`
   - `kiotviet_configured: true`
   - `kiotviet_auto_create_draft: false`

## CORS bắt buộc trên R2

R2 phải cho phép frontend upload qua presigned `PUT`.

Ví dụ:

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

CORS không làm bucket public. Upload vẫn cần presigned URL hợp lệ.

## Frontend

Gateway mặc định:

```js
window.SHARON_ORCHESTRATOR_BASE = "/api/v1/orchestrator";
```

Frontend hiện xử lý các trạng thái:

- `AWAITING_CONFIRMATION`: hiển thị dominant class, count, purity, confidence,
  kết quả từng ảnh và ảnh annotated.
- `DIRECT`: cho xác nhận trực tiếp.
- `FAMILY`: bắt buộc chọn một `product_code` trong `decision.members`.
- `NEEDS_RETAKE`: không hiển thị nút tạo phiếu; yêu cầu tải/chụp lại.
- `CONFIRMING`: tiếp tục poll.
- `COMPLETED`: hiển thị sản phẩm đã xác nhận, Excel nếu có và mã phiếu KiotViet.

## Quy tắc an toàn

- `bakery-upload-init` chỉ cấp URL upload.
- `bakery-submit` chỉ chạy AI.
- Không có write KiotViet trước `bakery-confirm`.
- Một job chỉ chứa một ảnh của một class/family.
- COMMON family không được tự ép thành một SKU cụ thể bằng vision.
- Mọi phiếu phải được tạo ở chế độ draft.
- Trước khi public cho nhiều máy, bảo vệ webhook bằng Basic Auth và/hoặc
  Cloudflare Access.
