import os
import json
import logging
import urllib.request
from urllib.error import URLError, HTTPError
from dotenv import load_dotenv

# Cau hinh log thuan van ban
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def verify_supabase_connection():
    # 1. Nap bien moi truong tu file .env
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not service_role_key:
        logging.error("Loi: Thieu cau hinh SUPABASE_URL hoac SUPABASE_KEY trong file .env")
        return False
        
    # 构建 PostgREST API Endpoint
    endpoint = f"{supabase_url}/rest/v1/execution_logs"
    
    # 2. Cau hinh Headers dung de ghi log qua Service Role Key
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    # 3. Payload du lieu kiem tra
    test_payload = {
        "job_id": "CLI-TEST-314",
        "source_file": "test_python_314_core.png",
        "process_status": "success",
        "error_message": None,
        "detected_count": 0
    }
    
    # Chuyen doi payload sang dang chuoi bytes UTF-8 de truyen qua luong mang
    data_bytes = json.dumps(test_payload).encode('utf-8')
    
    # Khoi tao thuc the Request object cua urllib
    req = urllib.request.Request(
        url=endpoint, 
        data=data_bytes, 
        headers=headers, 
        method="POST"
    )
    
    try:
        logging.info("Dang thiet lap ket noi va ghi du lieu qua thu vien loi urllib...")
        
        # Thuc thi gui request voi cu phap context manager, dat timeout 10 giay
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.getcode()
            response_body = response.read().decode('utf-8')
            
            if status_code == 201:
                logging.info("[SUCCESS] KET NOI SUPABASE THANH CONG!")
                logging.info(f"Du lieu Postgres phan hoi: {response_body}")
                return True
            else:
                logging.error(f"[FAILED] Giao tiep bi tu choi. HTTP Status: {status_code}")
                return False
                
    except HTTPError as e:
        # Boc tach loi chi tiet tra ve tu phia server Supabase
        error_response = e.read().decode('utf-8')
        logging.error(f"[FAILED] Loi HTTP tu Supabase Gateway ({e.code}): {error_response}")
        return False
    except URLError as e:
        # Xu ly loi sai URL hoac mat ket noi mang Internet
        logging.error(f"Loi ket noi mang: {e.reason}")
        return False
    except Exception as e:
        logging.error(f"Loi he thong khong xac dinh: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n=== KIEM TRA TRANG THAI KET NOI HE THONG SUPABASE LOGS ===")
    verify_supabase_connection()
    print("===========================================================\n")