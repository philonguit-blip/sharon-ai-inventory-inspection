# Local bakery pipeline

Backend local nhận 1–50 ảnh cùng SKU và chạy Hybrid Router cho từng ảnh. `AUTO`
dùng YOLO trước, chỉ gọi SAM2+DINOv2 nếu kết quả không chắc. Batch Aggregator
chỉ cộng count khi mọi ảnh tương thích với cùng một SKU/family. Hệ thống trả
một trong năm decision: `DIRECT`, `FAMILY`, `REVIEW`, `AMBIGUOUS`, `NO_DETECTION`.

`DIRECT`, `FAMILY` và `REVIEW` dừng ở `AWAITING_CONFIRMATION`. Chỉ endpoint `/confirm` mới
được tạo phiếu nhập nháp KiotViet. `AMBIGUOUS` và `NO_DETECTION` chuyển sang
`NEEDS_RETAKE`.

## Chạy thủ công

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --workers 1
```

Health:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/healthz
```

Không chạy nhiều Uvicorn worker vì model và job state local được thiết kế cho
một tiến trình. Hướng dẫn đầy đủ nằm tại
`../HUONG_DAN_CAI_DAT_VA_KHOI_DONG.md`.
