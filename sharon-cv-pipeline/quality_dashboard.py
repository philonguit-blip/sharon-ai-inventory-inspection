import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Cấu hình đường dẫn
JSONL_FILE = "triage_results.jsonl"
OUTPUT_DASHBOARD = "data_quality_report.png"

def generate_pie_dashboard():
    # 1. Khởi tạo bộ đếm
    stats = {
        'YOLO': {'pass': 0, 'fail': 0},
        'OCR': {'pass': 0, 'fail': 0},
        'Color': {'pass': 0, 'fail': 0}
    }
    
    # 2. Đọc và Parse dữ liệu
    if not os.path.exists(JSONL_FILE):
        print(f"Lỗi: Không tìm thấy file {JSONL_FILE}.")
        return

    total_processed = 0
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                relative_path = list(data.keys())[0]
                flags = data[relative_path]
                
                # Cập nhật số liệu YOLO
                if flags.get('is_whole_package_visible'): stats['YOLO']['pass'] += 1
                else: stats['YOLO']['fail'] += 1
                
                # Cập nhật số liệu OCR
                if flags.get('is_text_readable'): stats['OCR']['pass'] += 1
                else: stats['OCR']['fail'] += 1
                
                # Cập nhật số liệu Color
                if flags.get('is_color_patch_clear'): stats['Color']['pass'] += 1
                else: stats['Color']['fail'] += 1
                
                total_processed += 1
            except Exception as e:
                continue

    if total_processed == 0:
        print("Cảnh báo: File JSONL trống, chưa có dữ liệu để vẽ.")
        return

    # 3. Trực quan hóa bằng Matplotlib (Vẽ 3 Pie Charts)
    # Thiết lập kích thước figure (Ngang x Dọc)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Báo cáo Chất lượng Dữ liệu Đầu vào (Tổng số ảnh: {total_processed})', 
                 fontsize=16, fontweight='bold', y=1.05)

    # Cấu hình UI cho Pie Chart
    labels = ['Đạt chuẩn\n(Sẵn sàng Train)', 'Hao hụt\n(Cần loại bỏ)']
    colors = ['#2ecc71', '#e74c3c'] # Xanh lục (Pass) và Đỏ (Fail)
    explode = (0, 0.1) # Tách mảnh "Hao hụt" ra một chút để nhấn mạnh rủi ro

    # Vẽ Chart 1: YOLO
    axes[0].pie([stats['YOLO']['pass'], stats['YOLO']['fail']], 
                labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, explode=explode, shadow=True)
    axes[0].set_title('Luồng A: YOLO (Hình dáng bao bì)', fontsize=12)

    # Vẽ Chart 2: OCR
    axes[1].pie([stats['OCR']['pass'], stats['OCR']['fail']], 
                labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, explode=explode, shadow=True)
    axes[1].set_title('Luồng B: OCR (Độ nét văn bản)', fontsize=12)

    # Vẽ Chart 3: Color
    axes[2].pie([stats['Color']['pass'], stats['Color']['fail']], 
                labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, explode=explode, shadow=True)
    axes[2].set_title('Luồng C: Color (Màu sắc chuẩn)', fontsize=12)

    # 4. Lưu và Hiển thị
    plt.tight_layout()
    plt.savefig(OUTPUT_DASHBOARD, dpi=300, bbox_inches='tight')
    print(f"Đã kết xuất biểu đồ thành công: {OUTPUT_DASHBOARD}")
    
    # Hiển thị UI Pop-up
    plt.show()

if __name__ == "__main__":
    generate_pie_dashboard()