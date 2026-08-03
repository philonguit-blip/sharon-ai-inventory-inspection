@echo off
setlocal
title Sharon Bakery - n8n Outbound AI Worker
color 0A

set "PROJECT_DIR=%~dp0"
set "BACKEND_DIR=%PROJECT_DIR%backend"
set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Khong tim thay Python tai %PYTHON_EXE%
  echo Hay tao lai virtualenv trong thu muc backend truoc.
  pause
  exit /b 1
)

echo ===================================================
echo [SYSTEM] KHOI DONG N8N OUTBOUND AI WORKER
echo [MODE] May local chu dong ket noi ra n8n; khong can tunnel.
echo [BACKEND] http://127.0.0.1:8080
echo ===================================================
echo.

pushd "%BACKEND_DIR%"
"%PYTHON_EXE%" -m app.queue_worker
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
echo [ERROR] Worker da dung voi ma %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
