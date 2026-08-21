@echo off
setlocal EnableExtensions
title Sharon Bakery - n8n Outbound AI Worker
color 0A

rem ============================================================
rem Sharon Bakery AI Inventory - Outbound worker launcher
rem
rem This worker:
rem   - ensures the local FastAPI backend is running
rem   - polls the n8n outbound queue
rem   - handles PRESIGN / PROCESS / CONFIRM / DEVELOPER_SETTINGS tasks
rem   - opens the integrated Sharon Bakery counting web
rem ============================================================

set "PROJECT_DIR=%~dp0"
set "BACKEND_DIR=%PROJECT_DIR%backend"
set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"
set "WORKER_FILE=%BACKEND_DIR%\app\queue_worker.py"
set "BACKEND_STARTER=%PROJECT_DIR%start_backend.bat"

set "BACKEND_HEALTH_URL=http://127.0.0.1:8080/healthz"
set "WEB_URL=https://n8n.sharon-finefoods.com/webhook/sharon-bakery-inventory"

set "PYTHONUNBUFFERED=1"
set "PYTHONFAULTHANDLER=1"
set "SHARON_WORKER_DEBUG=1"
set "SHARON_BACKEND_DEBUG=1"
set "SHARON_WORKER_LOG_FILE=%BACKEND_DIR%\runtime\logs\outbound-worker-debug.log"
set "YOLO_CONFIG_DIR=%BACKEND_DIR%\runtime\ultralytics-config"
set "MPLCONFIGDIR=%BACKEND_DIR%\runtime\matplotlib-config"

echo ===========================================================
echo [SYSTEM] SHARON BAKERY OUTBOUND AI WORKER
echo [BACKEND] %BACKEND_HEALTH_URL%
echo [WEB] %WEB_URL%
echo [MODE] PRESIGN / PROCESS / CONFIRM / DEVELOPER_SETTINGS
echo [DEBUG] ENABLED - task, HTTP timing, image progress, decisions, tracebacks
echo [LOG] %SHARON_WORKER_LOG_FILE%
echo ===========================================================
echo.

rem ------------------------------------------------------------
rem 1. Validate local files
rem ------------------------------------------------------------

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Khong tim thay Python virtualenv:
  echo         %PYTHON_EXE%
  echo.
  pause
  exit /b 1
)

if not exist "%WORKER_FILE%" (
  echo [ERROR] Khong tim thay worker:
  echo         %WORKER_FILE%
  echo.
  echo Dam bao queue_worker.py moi da duoc copy vao backend\app.
  pause
  exit /b 1
)

if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
if not exist "%BACKEND_DIR%\runtime\logs" mkdir "%BACKEND_DIR%\runtime\logs"

rem ------------------------------------------------------------
rem 2. Ensure backend is running
rem ------------------------------------------------------------

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $r = Invoke-RestMethod -Uri '%BACKEND_HEALTH_URL%' -TimeoutSec 3; if ($r.status -eq 'ok') { exit 0 } } catch {}; exit 1" ^
  >nul 2>&1

if errorlevel 1 (
  echo [INFO] Backend chua chay.

  if not exist "%BACKEND_STARTER%" (
    echo [ERROR] Khong tim thay:
    echo         %BACKEND_STARTER%
    echo.
    echo Hay copy start_backend.bat moi vao cung thu muc voi file nay.
    pause
    exit /b 1
  )

  echo [INFO] Dang mo mot cua so backend rieng...
  start "Sharon Bakery - FastAPI AI Backend" cmd /k call "%BACKEND_STARTER%"

  echo [INFO] Dang cho backend load YOLO model va san sang...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "$deadline=(Get-Date).AddSeconds(120); while((Get-Date) -lt $deadline) { try { $r=Invoke-RestMethod -Uri '%BACKEND_HEALTH_URL%' -TimeoutSec 2; if($r.status -eq 'ok') { exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1" ^
    >nul 2>&1

  if errorlevel 1 (
    echo.
    echo [ERROR] Backend khong san sang sau 120 giay.
    echo [ERROR] Xem cua so "Sharon Bakery - FastAPI AI Backend" de doc log.
    pause
    exit /b 1
  )
)

echo [OK] Backend da san sang.

rem ------------------------------------------------------------
rem 3. Check the integrated n8n Web UI
rem ------------------------------------------------------------

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $r=Invoke-WebRequest -UseBasicParsing -Uri '%WEB_URL%' -TimeoutSec 10; if($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) { exit 0 } } catch {}; exit 1" ^
  >nul 2>&1

if errorlevel 1 (
  echo [WARNING] Chua truy cap duoc web n8n:
  echo           %WEB_URL%
  echo [WARNING] Hay kiem tra workflow "KV-INSP-00 Outbound AI Worker Queue"
  echo           da duoc import va Activate tren n8n hay chua.
  echo.
) else (
  echo [OK] Web n8n dang truy cap duoc.
)

rem ------------------------------------------------------------
rem 4. Open the counting web
rem ------------------------------------------------------------

echo [INFO] Dang mo web kiem dem...
start "" "%WEB_URL%"

rem ------------------------------------------------------------
rem 5. Start outbound worker
rem ------------------------------------------------------------

echo.
echo [INFO] Dang khoi dong outbound worker...
echo [INFO] Giu cua so nay mo de web xu ly upload, YOLO, confirm va Developer Settings.
echo [INFO] Worker se gui heartbeat len n8n sau khi khoi dong.
echo [INFO] DEBUG log se hien task/job ID, request timing, progress tung anh, class/count va loi traceback.
echo [INFO] Log cung duoc luu tai: %SHARON_WORKER_LOG_FILE%
echo [INFO] Neu web hien "he thong chua san sang", doi 5-10 giay roi refresh.
echo.

pushd "%BACKEND_DIR%"
"%PYTHON_EXE%" -m app.queue_worker --poll-seconds 2 --debug --log-file "%SHARON_WORKER_LOG_FILE%"
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
if "%EXIT_CODE%"=="0" (
  echo [INFO] Worker da dung binh thuong.
  echo [INFO] Neu log bao mot outbound worker khac dang chay,
  echo        hay dung cua so worker cu truoc khi chay lai file nay.
) else (
  echo [ERROR] Worker da dung voi ma loi %EXIT_CODE%.
  echo [ERROR] Kiem tra n8n, Basic Auth, backend va file backend\.env.
)

pause
exit /b %EXIT_CODE%
