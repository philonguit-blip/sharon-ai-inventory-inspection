import os
import re
import sys
import logging
import argparse  # Thư viện lõi phân tích cú pháp dòng lệnh CLI
import mimetypes
from datetime import datetime, timezone
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from unidecode import unidecode

# Cấu hình hệ thống log chuẩn hóa
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class CloudflareR2StorageManager:
    def __init__(self):
        """Khởi chạy hệ thống: Nạp cấu hình môi trường bí mật và thiết lập kết nối Cloud"""
        load_dotenv()  # Tìm và nạp file .env cục bộ vào hệ thống
        
        # Đọc dữ liệu định danh bảo mật
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.access_key = os.getenv("R2_ACCESS_KEY_ID")
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("R2_BUCKET_NAME")
        
        # Cấu hình Endpoint URL chuẩn S3 của Cloudflare R2 (Không chứa dấu / ở cuối)
        self.endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"
        
        # Kích hoạt cổng kết nối API
        self.s3_client = self._connect_r2()

    def _connect_r2(self):
        """Khởi tạo S3 Client kết nối bảo mật lớp lõi tới Cloudflare R2"""
        try:
            logging.info("Đang thiết lập cổng truyền bảo mật tới Cloudflare R2 Endpoint...")
            return boto3.client(
                service_name="s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="auto",
                config=Config(signature_version="s3v4")
            )
        except Exception as e:
            logging.critical(f"Khởi tạo kết nối thất bại, vui lòng kiểm tra lại Key trong file .env: {str(e)}")
            raise e

    def _sanitize_filename(self, filename: str) -> str:
        """[Private] Chuẩn hóa tên file an toàn cho Cloud URL, bóc tách Unicode tiếng Việt"""
        name, ext = os.path.splitext(filename)
        
        # Chuyển tiếng Việt có dấu thành không dấu và viết thường
        name = unidecode(name).lower()
        
        # Thay thế các chuỗi khoảng trắng hoặc dấu gạch dưới thành một dấu gạch ngang đơn '-'
        name = re.sub(r'[\s_\-]+', '-', name)
        
        # Bộ lọc an toàn: Chỉ giữ lại chữ thường a-z, số 0-9 và dấu gạch ngang
        name = re.sub(r'[^a-z0-9\-]', '', name)
        name = name.strip('-')
        
        return f"{name}{ext.lower()}"

    def _generate_object_key(self, local_file_path: str, partition: str) -> str:
        """[Private] Sử dụng F-string kết hợp UTC Datetime để tự động sinh chuỗi cấu trúc cây thư mục ảo"""
        raw_filename = os.path.basename(local_file_path)
        clean_filename = self._sanitize_filename(raw_filename)
        
        # Lấy thời gian hiện thời theo chuẩn quốc tế UTC chống lệch múi giờ giữa các Server
        utc_now = datetime.now(timezone.utc)
        year = utc_now.strftime("%Y")
        month = utc_now.strftime("%m")
        day = utc_now.strftime("%d")
        
        return f"{partition}/{year}/{month}/{day}/{clean_filename}"

    def upload_file(self, local_file_path: str, partition: str = "raw") -> str:
        """
        [Public API] Thực thi kiểm tra bảo mật, tính toán cấu trúc và đẩy tệp trực tiếp lên Cloud.
        """
        if not os.path.exists(local_file_path):
            logging.error(f"Lỗi I/O: Không tìm thấy file nguồn tại đường dẫn cục bộ: {local_file_path}")
            return None
        
        r2_object_key = self._generate_object_key(local_file_path, partition)
        
        content_type, _ = mimetypes.guess_type(local_file_path)
        if content_type is None:
            content_type = "application/octet-stream"
            
        try:
            logging.info(f"Đang truyền tải lên R2 Bucket [{self.bucket_name}]...")
            logging.info(f"Mục tiêu định tuyến cấu trúc: {r2_object_key}")
            
            self.s3_client.upload_file(
                Filename=local_file_path,
                Bucket=self.bucket_name,
                Key=r2_object_key,
                ExtraArgs={"ContentType": content_type}
            )
            
            logging.info("[SUCCESS] Tải lên thành công! File đã được lưu trữ an toàn toàn cục.")
            return r2_object_key
            
        except ClientError as e:
            logging.error(f"Lỗi API Cloudflare R2 ({e.response['Error']['Code']}): {e.response['Error']['Message']}")
            return None
        except Exception as e:
            logging.error(f"Lỗi đường truyền mạng không xác định: {str(e)}")
            return None

def main():
    """Khối điều hướng xử lý logic CLI chuyên nghiệp"""
    # 1. Khởi tạo bộ phân tích cú pháp dòng lệnh
    parser = argparse.ArgumentParser(
        description="R2 Storage CLI Automation Tool - Phát triển bởi Sharon-AI. "
                    "Tự động chuẩn hóa tên file tiếng Việt và phân vùng lưu trữ theo ngày (YYYY/MM/DD)."
    )
    
    # 2. Định nghĩa các tham số đầu vào (Arguments Configuration)
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Đường dẫn vật lý của tệp tin cục bộ cần tải lên Cloud (Ví dụ: bacao.xlsx, data/img.png)"
    )
    
    parser.add_argument(
        "-p", "--partition", 
        type=str, 
        default="raw",
        choices=["raw", "reports"],
        help="Lựa chọn phân vùng thư mục gốc trên R2. Chỉ chấp nhận 'raw' hoặc 'reports' (Mặc định: raw)"
    )

    # Phân tích cú pháp các tham số người dùng gõ từ Terminal
    args = parser.parse_args()

    # 3. Thực thi Mệnh đề phòng vệ cấp CLI (Fail-Fast Principle)
    # Kiểm tra sự tồn tại của file ngay tại tầng CLI trước khi khởi tạo kết nối mạng mạng kết nối R2
    if not os.path.exists(args.file_path):
        print(f"\nLỖI HỆ THỐNG: Tệp tin không tồn tại tại đường dẫn: '{args.file_path}'")
        print("Vui lòng kiểm tra lại tên file hoặc đường dẫn thư mục hiện hành.\n")
        sys.exit(1)  # Trả về mã lỗi 1 để thông báo cho các hệ thống tự động/n8n biết script bị lỗi

    print("\n=== KÍCH HOẠT HỆ THỐNG TRUYỀN TẢI DỮ LIỆU CLI CLOUDFLARE R2 ===")
    
    # 4. Khởi tạo Storage Manager và thực thi nhiệm vụ
    storage_manager = CloudflareR2StorageManager()
    uploaded_key = storage_manager.upload_file(local_file_path=args.file_path, partition=args.partition)
    
    if uploaded_key:
        print("\n----------------------------------------------------------------")
        print(f"TRUYỀN TẢI FILE THÀNH CÔNG!")
        print(f"Object Key trong DB: {uploaded_key}")
        print(f"URL Endpoint:         {storage_manager.endpoint_url}/{storage_manager.bucket_name}/{uploaded_key}")
        print("----------------------------------------------------------------\n")
        sys.exit(0)  # Trả về mã thành công 0
    else:
        print("\nLỖI: Tiến trình tải file lên Cloudflare R2 thất bại.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()