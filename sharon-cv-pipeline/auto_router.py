import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# 1. Cấu hình đường dẫn
BASE_DIR = Path(r"C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection\sharon-cv-pipeline\raw_images")
JSONL_FILE = "triage_results.jsonl"

# 2. Khởi tạo đích đến
DEST_YOLO = Path(r"C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection\sharon-cv-pipeline\usable_for_YOLO")
DEST_OCR = Path(r"C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection\sharon-cv-pipeline\usable_for_OCR")
DEST_COLOR = Path(r"C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection\sharon-cv-pipeline\usable_for_Color")
DEST_UNUSABLE = Path(r"C:\Users\SHARON-AI\Downloads\SharonBakery_AI_documents\sharon-AI-inventory-inspection\sharon-cv-pipeline\unusable_images")

def create_dirs_if_not_exist():
    """Tự động tạo các thư mục đích nếu chúng chưa tồn tại."""
    for d in [DEST_YOLO, DEST_OCR, DEST_COLOR, DEST_UNUSABLE]:
        d.mkdir(parents=True, exist_ok=True)

def route_files():
    create_dirs_if_not_exist()
    
    if not os.path.exists(JSONL_FILE):
        print(f"Chưa tìm thấy file {JSONL_FILE}.")
        return

    # Khởi tạo dictionary để theo dõi các file đã copy.
    # Việc khởi tạo giá trị của mỗi key là một list trống giúp chúng ta 
    # gom nhóm danh sách các phần tử theo từng mảng một cách an toàn
    # thay vì vô tình ghi đè duplicate keys lên nhau trong quá trình lọc.
    routing_summary = {'YOLO': [], 'OCR': [], 'Color': [], 'Unusable': []}
    
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Bắt đầu định tuyến {len(lines)} ảnh đã được AI phân loại...")
    
    for line in tqdm(lines, desc="Routing Files"):
        try:
            data = json.loads(line.strip())
            relative_path_str = list(data.keys())[0]
            flags = data[relative_path_str]
            
            source_path = BASE_DIR / relative_path_str
            
            if not source_path.exists():
                continue

            # Flattening Name: Đổi "MenTrang\img1.jpg" thành "MenTrang_img1.jpg"
            # Để tránh lỗi ghi đè nếu các SKU có file trùng tên.
            safe_file_name = relative_path_str.replace("\\", "_").replace("/", "_")

            # Biến kiểm tra xem ảnh có xài được cho bất kỳ luồng nào không
            is_usable = False

            # Thực thi Copy và gom nhóm kết quả vào dictionary
            if flags.get('is_whole_package_visible'):
                target = DEST_YOLO / safe_file_name
                if not target.exists():
                    shutil.copy2(source_path, target)
                routing_summary['YOLO'].append(safe_file_name)
                is_usable = True
                
            if flags.get('is_text_readable'):
                target = DEST_OCR / safe_file_name
                if not target.exists():
                    shutil.copy2(source_path, target)
                routing_summary['OCR'].append(safe_file_name)
                is_usable = True
                
            if flags.get('is_color_patch_clear'):
                target = DEST_COLOR / safe_file_name
                if not target.exists():
                    shutil.copy2(source_path, target)
                routing_summary['Color'].append(safe_file_name)
                is_usable = True
                
            # Phân luồng Unusable: Nếu không vượt qua được bài test nào ở trên
            if not is_usable:
                target = DEST_UNUSABLE / safe_file_name
                if not target.exists():
                    shutil.copy2(source_path, target)
                routing_summary['Unusable'].append(safe_file_name)
                
        except Exception as e:
            print(f"Error processing line: {line.strip()}")
            pass # Bỏ qua các dòng lỗi (nếu có) do đang ghi dở

    # In báo cáo
    print("\nHoàn tất định tuyến! Tóm tắt số lượng ảnh trong các khoang chứa:")
    for key, item_list in routing_summary.items():
        print(f"- Luồng {key}: {len(item_list)} ảnh.")

if __name__ == "__main__":
    route_files()