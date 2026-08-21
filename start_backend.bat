@echo off
setlocal EnableExtensions
title Sharon Bakery - FastAPI AI Backend
color 0A

rem ============================================================
rem Sharon Bakery AI Inventory - Backend launcher
rem
rem Expected project layout:
rem   <project>\
rem     start_backend.bat
rem     start_worker.bat
rem     backend\
rem       .venv\
rem       app\
rem       .env
rem ============================================================

set "PROJECT_DIR=%~dp0"
set "BACKEND_DIR=%PROJECT_DIR%backend"
set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"
set "APP_MAIN=%BACKEND_DIR%\app\main.py"
set "ENV_FILE=%BACKEND_DIR%\.env"

set "HOST=127.0.0.1"
set "PORT=8080"
set "HEALTH_URL=http://%HOST%:%PORT%/healthz"
set "LOCAL_WEB_URL=http://%HOST%:%PORT%/"

set "YOLO_CONFIG_DIR=%BACKEND_DIR%\runtime\ultralytics-config"
set "MPLCONFIGDIR=%BACKEND_DIR%\runtime\matplotlib-config"
set "PYTHONUNBUFFERED=1"
set "PYTHONFAULTHANDLER=1"
set "SHARON_BACKEND_DEBUG=1"

echo ===========================================================
echo [SYSTEM] SHARON BAKERY AI BACKEND
echo [BACKEND] %HEALTH_URL%
echo [LOCAL UI] %LOCAL_WEB_URL%
echo [DEBUG] BACKEND JOB TRACE ENABLED
echo ===========================================================
echo.

rem ------------------------------------------------------------
rem 1. Validate project files
rem ------------------------------------------------------------

if not exist "%BACKEND_DIR%\" (
  echo [ERROR] Khong tim thay thu muc backend:
  echo         %BACKEND_DIR%
  echo.
  pause
  exit /b 1
)

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Khong tim thay Python virtualenv:
  echo         %PYTHON_EXE%
  echo.
  echo Tao virtualenv trong backend va cai dependencies truoc khi chay lai.
  pause
  exit /b 1
)

if not exist "%APP_MAIN%" (
  echo [ERROR] Khong tim thay FastAPI entrypoint:
  echo         %APP_MAIN%
  echo.
  pause
  exit /b 1
)

if not exist "%ENV_FILE%" (
  echo [WARNING] Khong tim thay %ENV_FILE%
  echo [WARNING] Backend van se thu khoi dong bang environment hien tai.
  echo.
)

if not exist "%BACKEND_DIR%\runtime" mkdir "%BACKEND_DIR%\runtime"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"

rem ------------------------------------------------------------
rem 2. Do not start a duplicate healthy backend
rem ------------------------------------------------------------

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $r = Invoke-RestMethod -Uri '%HEALTH_URL%' -TimeoutSec 3; if ($r.status -eq 'ok') { exit 0 } } catch {}; exit 1" ^
  >nul 2>&1

if not errorlevel 1 (
  echo [OK] Backend da dang chay tai %HEALTH_URL%
  echo [INFO] Khong khoi dong them Uvicorn trung lap.
  echo.
  exit /b 0
)

rem ------------------------------------------------------------
rem 3. Detect another process occupying port 8080
rem ------------------------------------------------------------

set "PORT_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do set "PORT_PID=%%P"

if defined PORT_PID (
  echo [ERROR] Cong %PORT% dang bi PID %PORT_PID% chiem dung,
  echo         nhung %HEALTH_URL% khong tra ve Sharon Bakery health check.
  echo.
  echo Kiem tra tien trinh bang lenh:
  echo   tasklist /FI "PID eq %PORT_PID%"
  echo.
  echo Sau khi xu ly tien trinh do, chay lai start_backend.bat.
  pause
  exit /b 1
)

rem ------------------------------------------------------------
rem 4. Start FastAPI
rem ------------------------------------------------------------

echo [INFO] Dang khoi dong FastAPI...
echo [INFO] Lan khoi dong dau co the mat them thoi gian de load YOLO model.
echo [INFO] Giu cua so nay mo trong suot thoi gian su dung web.
echo.

pushd "%BACKEND_DIR%"

"%PYTHON_EXE%" -m uvicorn app.main:app ^
  --host %HOST% ^
  --port %PORT% ^
  --workers 1 ^
  --log-level info

set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
if "%EXIT_CODE%"=="0" (
  echo [INFO] Backend da dung.
) else (
  echo [ERROR] Backend da dung voi ma loi %EXIT_CODE%.
  echo [ERROR] Xem log phia tren de xac dinh loi model, config, R2 hoac KiotViet.
)

pause
exit /b %EXIT_CODE%
