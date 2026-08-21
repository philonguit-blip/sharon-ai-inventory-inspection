# n8n outbound AI worker

Máy AI không nhận kết nối inbound. Worker local chủ động poll workflow n8n để
xử lý ba loại task:

- `PRESIGN`: cấp URL upload một ảnh khay lên R2;
- `PROCESS`: tải đúng object trong manifest và chạy chế độ Hybrid đã chọn;
- `CONFIRM`: đối chiếu SKU rồi tạo đúng một phiếu nhập nháp KiotViet.

## Workflow production

File canonical:

```text
n8n/Workflow 4_ Sharon Bakery Outbound Worker.json
```

Workflow có 36 node, gồm Web UI, queue kiểm đếm và relay cấu hình Developer tại:

```text
https://n8n.sharon-finefoods.com/webhook/sharon-bakery-inventory
```

Không sửa trực tiếp HTML/CSS/JS trong JSON. Sửa `backend/frontend`, sau đó chạy:

```powershell
node n8n\generate_workflow4_outbound.mjs
node n8n\test_workflow4_outbound.mjs
```

Import lại JSON, gán Basic Auth và Activate workflow.

## Chạy worker

```powershell
.\start_worker.bat
```

Worker yêu cầu backend tại `http://127.0.0.1:8080`, dùng cùng Basic Auth với
n8n và gửi heartbeat định kỳ. Mỗi job nhận tối đa 50 ảnh cùng SKU, 50 MB/ảnh và 200 MB tổng batch.
Payload giữ nguyên `inference_mode` (`AUTO`, `YOLO`, `FOUNDATION`, `COMPARE`)
từ lúc presign đến lúc backend xử lý.

Khi confirmation lỗi/gián đoạn, n8n giữ `CONFIRMING`. Backend đối soát receipt
theo marker job trước khi cho hoàn tất và không tự POST phiếu thứ hai.
