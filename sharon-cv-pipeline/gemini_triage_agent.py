import os
import json
import time
from pathlib import Path
from pydantic import BaseModel
from PIL import Image
from tqdm import tqdm
from google import genai
from google.genai import types
from google.genai import errors

# 1. Định nghĩa cấu trúc JSON đầu ra bắt buộc
class ImageTriage(BaseModel):
    is_whole_package_visible: bool
    is_text_readable: bool
    is_color_patch_clear: bool

# 2. Khởi tạo API Client
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Chưa tìm thấy GEMINI_API_KEY trong hệ thống.")

client = genai.Client(api_key=api_key)
generation_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=ImageTriage,
    temperature=0.0,
)

# Đường dẫn thư mục kho bãi
BASE_DIR = Path(r"C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection\sharon-cv-pipeline\raw_images_08072026")
OUTPUT_JSONL = "triage_results.jsonl" # Dùng định dạng JSON Lines để ghi nối liên tục

# [BỔ SUNG] Đường dẫn ảnh mẫu Few-shot
SAMPLE_PASS_PATH = Path("samples/sample_pass.jpg")
SAMPLE_FAIL_PATH = Path("samples/sample_fail.jpg")

def get_few_shot_context() -> list:
    """Tạo bối cảnh Few-shot, nạp ảnh vào RAM 1 lần duy nhất để tối ưu I/O."""
    prompt = """
Bạn là một System Agent chuyên kiểm định chất lượng Dữ liệu Thị giác Máy tính (Computer Vision Dataset) cho hệ thống kho bãi.
Nhiệm vụ của bạn là quan sát vật thể CHÍNH nằm ở trung tâm bức ảnh và đánh giá 3 tiêu chí sau.

### ĐỊNH NGHĨA TIÊU CHÍ (CHÚ Ý KỸ):
1. is_whole_package_visible (Cho AI YOLO): Trả về `true` CHỈ KHI thấy được ít nhất 80% ranh giới vật lý (các cạnh, góc) của bao bì chính. Nếu vật thể bị cắt xén bởi viền ảnh hoặc bị vật khác đè lên che mất hơn 20% diện tích, hãy trả về `false`.
2. is_text_readable (Cho AI OCR): Trả về `true` CHỈ KHI các cụm từ định danh (Tên thương hiệu, Mã sản phẩm) hiển thị rõ nét, không bị nhòe do chuyển động (motion blur) và chữ không bị biến dạng đến mức không thể nhận diện ký tự. Bỏ qua các dòng chữ thành phần nhỏ.
3. is_color_patch_clear (Cho AI Color): Trả về `true` CHỈ KHI mảng màu đặc trưng của vỏ bao bì hiển thị rõ ràng, không bị cháy sáng (overexposed) lóa trắng hoặc chìm vào bóng tối đen đặc.

### RÀNG BUỘC ĐẦU RA (STRICT OUTPUT CONSTRAINTS):
Tuyệt đối KHÔNG giao tiếp bằng ngôn ngữ tự nhiên. CHỈ trả về duy nhất một chuỗi JSON hợp lệ theo đúng cấu trúc (Schema) dưới đây. Hãy cung cấp lý luận (reasoning) ngắn gọn (dưới 15 từ) trước khi chốt kết quả boolean.

{
  "shape_reasoning": "string",
  "is_whole_package_visible": boolean,
  "text_reasoning": "string",
  "is_text_readable": boolean,
  "color_reasoning": "string",
  "is_color_patch_clear": boolean
}
"""
    # Load ảnh bằng PIL giống cách code cũ xử lý ảnh target
    try:
        img_pass = Image.open(SAMPLE_PASS_PATH)
        img_fail = Image.open(SAMPLE_FAIL_PATH)
    except FileNotFoundError:
        print(f"\n[CẢNH BÁO] Không tìm thấy ảnh mẫu tại '{SAMPLE_PASS_PATH}' hoặc '{SAMPLE_FAIL_PATH}'.")
        print("-> Hệ thống sẽ tự động hạ cấp xuống chạy Zero-shot.\n")
        return [prompt]
    
    # Cấu trúc hội thoại Few-shot định hướng Model
    few_shot_history = [
        prompt,
        "--- MẪU 1: ẢNH ĐẠT CHUẨN (PASS) ---",
        img_pass,
        json.dumps({
            "is_whole_package_visible": True,
            "is_text_readable": True,
            "is_color_patch_clear": True
        }, ensure_ascii=False),
        
        "--- MẪU 2: ẢNH LỖI (FAIL) ---",
        img_fail,
        json.dumps({
            "is_whole_package_visible": False,
            "is_text_readable": False,
            "is_color_patch_clear": False
        }, ensure_ascii=False),
        
        "--- BÂY GIỜ, HÃY PHÂN TÍCH BỨC ẢNH THỰC TẾ NÀY ---"
    ]
    return few_shot_history

def analyze_image(image_path: Path, few_shot_context: list, retries: int = 5) -> dict:
    """Gửi ảnh cho Gemini với cơ chế Exponential Backoff cho cả lỗi 429 và 500 INTERNAL."""
    for attempt in range(retries):
        try:
            img = Image.open(image_path)
            current_request = few_shot_context + [img]
            
            response = client.models.generate_content(
                model='gemma-4-31b-it',
                contents=current_request,
                config=generation_config
            )
            
            raw_text = response.text.strip()
            
            # Tìm vị trí dấu mở ngoặc nhọn đầu tiên
            start_idx = raw_text.find('{')
            if start_idx == -1:
                raise json.JSONDecodeError("Không tìm thấy khối JSON trong phản hồi của mô hình.", raw_text, 0)
            
            decoder = json.JSONDecoder()
            res_dict, _ = decoder.raw_decode(raw_text[start_idx:])
            return res_dict
            
        except errors.ClientError as e:
            # Xử lý lỗi 429 (Rate Limit)
            if e.code == 429:
                tqdm.write(f"\nTràn bộ đệm API (429). Tự động ngủ đông 60 giây để phục hồi... ({image_path.name})")
                time.sleep(60)
            else:
                tqdm.write(f"\nLỖI API CLIENT [{image_path.name}]: {e}")
                if attempt == retries - 1:
                    return {"is_whole_package_visible": False, "is_text_readable": False, "is_color_patch_clear": False}
                time.sleep(5)
                
        except Exception as e:
            # TÍNH TOÁN THỜI GIAN NGỦ LŨY TIẾN: Lần 0 = 5s, Lần 1 = 10s, Lần 2 = 20s, Lần 3 = 40s...
            wait_time = (2 ** attempt) * 5
            
            # Kiểm tra nếu chuỗi thông báo chứa lỗi 500 hoặc INTERNAL từ Google
            error_msg = str(e).upper()
            if "500" in error_msg or "INTERNAL" in error_msg:
                tqdm.write(f"\n[Google Server Glitch 500] Máy chủ Google đang quá tải khi xử lý {image_path.name}.")
                tqdm.write(f"-> Kích hoạt luồng khôi phục: Ngủ đông {wait_time} giây và thử lại (Lần {attempt + 1}/{retries})...")
            else:
                tqdm.write(f"\nLỖI HỆ THỐNG / PARSE JSON [{image_path.name}]: {e}")
                tqdm.write(f"-> Thử lại sau {wait_time} giây (Lần {attempt + 1}/{retries})...")
            
            if attempt == retries - 1:
                tqdm.write(f"Thất bại hoàn toàn sau {retries} lần thử với ảnh [{image_path.name}]. Chấp nhận nhãn mặc định False.")
                return {"is_whole_package_visible": False, "is_text_readable": False, "is_color_patch_clear": False}
            
            time.sleep(wait_time)

def load_processed_keys() -> set:
    """Đọc file log để tìm các ảnh đã xử lý thành công trước đó (Phục hồi)."""
    processed = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed.add(list(data.keys())[0])
                except:
                    pass
    return processed

def main():
    # [BỔ SUNG] Load bộ Context Few-shot 1 lần duy nhất để tái sử dụng cho toàn bộ vòng lặp
    few_shot_context = get_few_shot_context()
    
    image_files = [p for p in BASE_DIR.rglob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    # Lọc ra những ảnh chưa làm
    processed_keys = load_processed_keys()
    pending_files = [p for p in image_files if str(p.relative_to(BASE_DIR)) not in processed_keys]
    
    print(f"Tổng số ảnh: {len(image_files)} | Đã xử lý: {len(processed_keys)} | Còn lại: {len(pending_files)}")
    
    pbar = tqdm(pending_files, desc="Processing Images")
    
    # Mở file dưới chế độ 'a' (Append) để ghi chèn vào cuối file
    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as f:
        for img_path in pbar:
            # [THAY ĐỔI] Truyền bối cảnh Few-shot vào trong hàm analyze_image
            result = analyze_image(img_path, few_shot_context)
            
            # Lấy trạng thái của cả 3 cờ
            yolo_flag = "✅" if result.get('is_whole_package_visible') else "❌"
            ocr_flag = "✅" if result.get('is_text_readable') else "❌"
            color_flag = "✅" if result.get('is_color_patch_clear') else "❌"
            
            # In ra màn hình Terminal đầy đủ 3 luồng
            tqdm.write(f"📸 [{img_path.name}] YOLO: {yolo_flag} | OCR: {ocr_flag} | Color: {color_flag}")
            
            relative_key = str(img_path.relative_to(BASE_DIR))
            
            # Ghi ngay lập tức xuống ổ cứng
            record = {relative_key: result}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush() # Ép OS ghi buffer xuống đĩa vật lý
            
            # Ngủ 4 giây = 15 requests / phút (Đảm bảo an toàn qua ải Free Tier 20 RPM)
            time.sleep(4) 
            
    print(f"\nHoàn tất! Toàn bộ kết quả định tuyến nằm tại {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()