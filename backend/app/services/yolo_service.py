import os
import cv2
import requests
import numpy as np
from ultralytics import YOLO

class YoloInferenceService:
    def __init__(self):
        # 1. Định vị động (Dynamic Path) tới file ONNX để tránh lỗi đường dẫn tuyệt đối
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "..", "..", "models", "yolov8n.onnx")
        self.model_path = os.path.abspath(self.model_path) # Chuẩn hóa đường dẫn Windows
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {self.model_path}")
        
        print(f"[AI CORE] Đang nạp mô hình từ: {self.model_path}")
        
        # 2. Nạp mô hình vào RAM (Sử dụng ONNX Runtime backend đã cài đặt)
        self.model = YOLO(self.model_path, task="detect")
        self.names = self.model.names # Lấy từ điển nhãn

    def download_image_to_memory(self, url: str) -> np.ndarray:
        """
        Tải ảnh trực tiếp từ Cloudflare R2 vào bộ nhớ RAM (Không ghi xuống ổ cứng)
        """
        try:
            # timeout=10 để chống kẹt luồng nếu rớt mạng
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Giải mã luồng byte thành ma trận điểm ảnh (Pixels)
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Định dạng hình ảnh không hợp lệ hoặc bị lỗi Data Lake.")
            return img
            
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải ảnh từ mạng: {str(e)}")

    def run_inference(self, image_url: str) -> dict:
        """
        Phiên bản Tối ưu Payload (Loại bỏ mảng detections rác, nâng ngưỡng tự tin)
        """
        # 1. Thu thập ma trận điểm ảnh
        img = self.download_image_to_memory(image_url)
        
        # 2. Kích hoạt AI dự đoán (Set conf=0.45 để lọc rác từ mô hình COCO)
        results = self.model(img, conf=0.45, verbose=False)
        boxes = results[0].boxes
        
        # 3. Thuật toán phân tích cấu trúc (Aggregation)
        total_items = len(boxes)
        summary_counts = {}
        total_confidence_sum = 0.0
        
        # Vòng lặp siêu tốc O(N)
        for box in boxes:
            class_id = int(box.cls[0]) if box.cls is not None else -1
            class_name = self.names.get(class_id, f"unknown_{class_id}")
            
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            total_confidence_sum += confidence
            
            if class_name not in summary_counts:
                summary_counts[class_name] = {
                    "count": 0,
                    "sum_conf": 0.0,
                    "min_conf": 1.0,
                    "max_conf": 0.0
                }
                
            # Cộng dồn
            summary_counts[class_name]["count"] += 1
            summary_counts[class_name]["sum_conf"] += confidence
            
            # Ghi nhận min/max
            if confidence < summary_counts[class_name]["min_conf"]:
                summary_counts[class_name]["min_conf"] = confidence
            if confidence > summary_counts[class_name]["max_conf"]:
                summary_counts[class_name]["max_conf"] = confidence
            
        # 4. Đóng gói JSON (Chỉ lấy điểm trung bình, payload siêu nhẹ O(C))
        predictions_output = {}
        for cls_name, data in summary_counts.items():
            avg_conf = data["sum_conf"] / data["count"] if data["count"] > 0 else 0.0
            
            predictions_output[cls_name] = {
                "count": data["count"],
                "avg_confidence": round(avg_conf, 4),
                "min_confidence": round(data["min_conf"], 4),
                "max_confidence": round(data["max_conf"], 4)
            }
            
        overall_confidence = (total_confidence_sum / total_items) if total_items > 0 else 0.0
        
        return {
            "total_count": total_items,
            "overall_confidence": round(overall_confidence, 4),
            "predictions": predictions_output
        }

    def analyze_frame(self, img: np.ndarray) -> dict:
        """
        Nhận trực tiếp ma trận ảnh từ luồng Video hoặc Ảnh tĩnh.
        Bóc tách và trả về danh sách độ tự tin của các vật thể.
        """
        # BẢO VỆ AI: Chống crash nếu frame/ảnh bị hỏng hoặc trống rỗng
        if img is None or img.size == 0:
            return {}

        # Kích hoạt AI dự đoán (Giữ nguyên cấu hình conf=0.45 lọc rác)
        results = self.model(img, conf=0.45, verbose=False)
        boxes = results[0].boxes
        
        # Dictionary lưu mảng độ tự tin cho từng class vật thể trong 1 frame
        frame_detections = {}
        
        for box in boxes:
            class_id = int(box.cls[0]) if box.cls is not None else -1
            class_name = self.names.get(class_id, f"unknown_{class_id}")
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            
            if class_name not in frame_detections:
                frame_detections[class_name] = []
            frame_detections[class_name].append(confidence)
            
        return frame_detections

    def analyze_video_with_tracking(self, video_path: str) -> dict:
        """
        Quét Video kết hợp thuật toán ByteTrack để đếm đối tượng Unique (Chống đếm trùng).
        Sử dụng Generator (stream=True) để giải phóng RAM trong quá trình quét.
        """
        print("[AI CORE] Bắt đầu kích hoạt ByteTrack cho luồng Video...")
        
        # 1. Gọi thẳng Engine YOLOv8 kèm cấu hình Tracker
        results = self.model.track(
            source=video_path,
            conf=0.45,
            tracker="bytetrack.yaml", 
            persist=True,   # Giữ bộ nhớ ID giữa các frame
            stream=True,    # Tối quan trọng: Stream để không tràn RAM với video lớn
            verbose=False
        )
        
        # Cấu trúc lưu trữ ID: { "carton_box": { 1: [0.85, 0.88, ...], 2: [0.9] } }
        tracked_objects = {}
        
        # 2. Quét qua từng Frame (Chỉ tính toán khi yield generator)
        for frame_result in results:
            boxes = frame_result.boxes
            
            # Bỏ qua nếu frame trống hoặc AI chưa cấp được ID (tracking chưa hội tụ)
            if boxes is None or boxes.id is None:
                continue
                
            for box, track_id in zip(boxes, boxes.id):
                class_id = int(box.cls[0])
                class_name = self.names.get(class_id, f"unknown_{class_id}")
                conf = float(box.conf[0])
                t_id = int(track_id)
                
                # Khởi tạo Dictionary nếu chưa có
                if class_name not in tracked_objects:
                    tracked_objects[class_name] = {}
                if t_id not in tracked_objects[class_name]:
                    tracked_objects[class_name][t_id] = []
                    
                # Ghi nhận độ tự tin của vật thể ID này vào mảng lịch sử
                tracked_objects[class_name][t_id].append(conf)
                
        # 3. Tổng hợp thành Output tương thích ngược với luồng n8n hiện tại
        unique_detections = {}
        for class_name, objects in tracked_objects.items():
            unique_detections[class_name] = []
            
            # Mỗi t_id đại diện cho ĐÚNG 1 kiện hàng có thật ngoài đời
            for t_id, conf_list in objects.items():
                # Lấy độ tự tin CAO NHẤT trong suốt vòng đời xuất hiện của nó
                best_conf = max(conf_list) 
                unique_detections[class_name].append(best_conf)
                
        return unique_detections