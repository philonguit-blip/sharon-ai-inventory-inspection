import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageOps
import imagehash

class VideoPreProcessor:
    def __init__(self, blur_threshold: float = 35.0, max_dimension: int = 1024, target_fps: int = 1, similarity_threshold: int = 8):
        self.blur_threshold = blur_threshold
        self.max_dimension = max_dimension
        self.target_fps = target_fps
        self.similarity_threshold = similarity_threshold

    def _is_blurry(self, frame: np.ndarray) -> bool:
        """Giữ nguyên logic lõi tính toán độ sắc nét"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold

    def _resize_and_pad(self, frame: np.ndarray) -> Image.Image:
        """Giữ nguyên logic lõi chuẩn hóa kích thước"""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
        return image

    def process_video(self, video_path: str, output_dir: str) -> list:
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        cap = cv2.VideoCapture(video_path)
        source_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_stride = max(1, int(source_fps / self.target_fps))
        
        extracted_images = []
        frame_idx = 0
        saved_count = 1
        
        last_saved_hash = None  # Biến lưu trữ "vân tay" của bức ảnh cuối cùng được lưu

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_stride == 0:
                    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    
                    # 1. Lọc nhòe (Blur filter)
                    if self._is_blurry(frame):
                        frame_idx += 1
                        continue

                    # 2. Tiền xử lý để lấy ảnh Pillow chuẩn bị lưu
                    processed_img = self._resize_and_pad(frame)
                    
                    # 3. Tính toán "Vân tay" (pHash) của ảnh hiện tại
                    current_hash = imagehash.phash(processed_img)
                    
                    # 4. THUẬT TOÁN LỌC TRÙNG (DEDUPLICATION)
                    if last_saved_hash is not None:
                        # Tính khoảng cách khác biệt giữa 2 bức ảnh
                        hash_diff = current_hash - last_saved_hash 
                        
                        # Nếu sự khác biệt quá nhỏ (ảnh giống nhau) -> Bỏ qua không lưu!
                        if hash_diff < self.similarity_threshold:
                            print(f"   [Deduplicate] Bỏ qua frame {timestamp_ms}ms vì góc quay không thay đổi.")
                            frame_idx += 1
                            continue
                    
                    # Nếu ảnh vượt qua bài test (ảnh mới, góc quay mới), tiến hành lưu
                    file_name = f"{video_name}_frame_{saved_count:04d}_{timestamp_ms}ms.jpg"
                    file_path = os.path.join(output_dir, file_name)
                    processed_img.save(file_path, "JPEG", quality=95)
                    
                    extracted_images.append(file_path)
                    
                    # 5. Cập nhật lại "vân tay" cho lần so sánh tiếp theo
                    last_saved_hash = current_hash
                    saved_count += 1

                frame_idx += 1
        finally:
            cap.release()

        return extracted_images

def process_batch(input_folder: str, output_folder: str, processor: VideoPreProcessor):
    """Hàm điều phối xử lý hàng loạt mới"""
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder not found: {input_folder}")
        return

    # 1. Auto scan file: Chỉ quét các file .mp4
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    total_videos = len(video_files)
    
    if total_videos == 0:
        print(f"[INFO] Found 0 videos in {input_folder}")
        return

    print(f"[INFO] Found {total_videos} videos")
    
    success_count = 0
    failed_count = 0
    total_extracted = 0

    # 2. Duyệt qua từng video (Error Isolation)
    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        print(f"[INFO] Processing: {video_filename}")
        
        # Đường dẫn output cho riêng video này
        video_output_dir = os.path.join(output_folder, video_name)
        
        # Bắt lỗi độc lập (nếu 1 video hỏng, vòng lặp vẫn chạy tiếp file khác)
        try:
            extracted_images = processor.process_video(video_path, video_output_dir)
            num_frames = len(extracted_images)
            
            print(f"[INFO] Extracted {num_frames} frames")
            
            # Format đường dẫn tương đối (VD: output/bot_chia_khoa_do) cho Log đẹp
            display_path = os.path.relpath(video_output_dir, start=os.getcwd()).replace("\\", "/")
            print(f"[SUCCESS] Saved to {display_path}")
            
            success_count += 1
            total_extracted += num_frames
            
        except Exception as e:
            # 4. In log lỗi tường minh kèm theo nguyên nhân thực tế
            print(f"[ERROR] Failed to read video {video_filename} | Lỗi chi tiết: {str(e)}")
            failed_count += 1
            
    # 5. Summary cuối cùng
    print("\n========== SUMMARY ==========")
    print(f"Total videos: {total_videos}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total extracted frames: {total_extracted}")
    print("=============================")


# --- Kích hoạt hệ thống ---
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Định nghĩa cấu trúc folder
    INPUT_FOLDER = r"C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection\data\Test5"
    OUTPUT_FOLDER = r"C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection\data\Test5"

    # ─── ĐOẠN CODE TEST NHANH ───
    print(f"[*] Thư mục quét video thực tế: {INPUT_FOLDER}")
    print(f"[*] Trạng thái tìm thấy thư mục: {os.path.exists(INPUT_FOLDER)}")
    
    # Khởi tạo class tiền xử lý
    processor = VideoPreProcessor(blur_threshold=20.0, max_dimension=1024, target_fps=1)
    
    # Chạy xử lý hàng loạt
    process_batch(INPUT_FOLDER, OUTPUT_FOLDER, processor)