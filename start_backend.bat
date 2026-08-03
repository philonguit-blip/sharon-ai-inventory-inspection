@echo off
setlocal
title Sharon Bakery - FastAPI AI Backend
color 0A

set "PROJECT_DIR=%~dp0"
set "BACKEND_DIR=%PROJECT_DIR%backend"
set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Khong tim thay Python tai %PYTHON_EXE%
  echo Hay tao virtualenv va cai requirements trong thu muc backend.
  pause
  exit /b 1
)

echo ===================================================
echo [SYSTEM] KHOI DONG SHARON BAKERY AI BACKEND
echo [LOCAL] http://127.0.0.1:8080
echo [PUBLIC] https://sharon-bakery-inventory.pages.dev/
echo ===================================================
echo.

pushd "%BACKEND_DIR%"
"%PYTHON_EXE%" -m uvicorn app.main:app --host 127.0.0.1 --port 8080
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
echo [ERROR] Backend da dung voi ma %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
