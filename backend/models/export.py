from ultralytics import YOLO

# Khởi tạo mô hình và xuất ra định dạng ONNX
model = YOLO("yolov8n.pt")
model.export(format="onnx")