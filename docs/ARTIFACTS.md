# Quản lý model, dữ liệu và artifact

## Production

Repository production phải chứa:

- `backend/models/best_YOLO26s_PROD_14_SKUs_v3.pt`;
- `backend/models/MODEL_MANIFEST.json`;
- `backend/models/FOUNDATION_MANIFEST.json` (URL/checksum SAM2 external);
- `backend/hybrid_data/reference_embeddings.npz` (713 DINOv2 vectors, 40 reference classes);
- `backend/config/product_mapping.json`;
- source backend/frontend, workflow n8n, test và tài liệu.

Model production hiện tại:

```text
SHA256 21C06D56454B41507A906171DCF45AAF8A3B883A3EBF75917CEBF9A6939BE1EA
14 visual classes -> 18 business SKUs
```

Chạy `backend/.venv/Scripts/python.exe scripts/verify_production.py` để kiểm tra
checksum và cấu trúc trước khi khởi động hoặc commit.

`sam2.1_s.pt` không commit vì là weight ngoài 88 MB; `setup_windows.ps1` tải từ
Ultralytics và kiểm SHA256. Reference crop gốc là dữ liệu private, bị Git bỏ
qua; chỉ artifact embedding nhỏ được version-control để máy mới chạy được.

## Research local

Các thư mục `data/`, `pre-annotation/` và `sharon-cv-pipeline/` chứa dataset,
pre-annotation, Streamlit evaluation và checkpoint thử nghiệm. Chúng bị Git bỏ
qua, không được backend production tham chiếu.

Mỗi dataset/model được promote phải ghi lại:

- phiên bản dataset và nguồn;
- class order;
- lệnh/config train;
- checkpoint và SHA256;
- kết quả ground truth theo class;
- threshold được chọn;
- ngày/người promote.

## Archive

Backup cũ, cloudflared cũ, model nghiên cứu cũ và report sinh tự động được giữ
tại `_archive/` trong thời gian cần đối chiếu. Thư mục này không chạy cùng hệ
thống và không được commit.

Không đặt model có tên gần giống nhau trong `backend/models`; thư mục đó chỉ giữ
YOLO production và SAM2 được hai manifest tham chiếu.
