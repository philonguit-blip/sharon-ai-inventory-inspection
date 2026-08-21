@echo off
setlocal EnableExtensions

title Sharon Bakery - FOUNDATION Test Lab v2
color 0B

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%backend\.venv\Scripts\python.exe" (
    set "PROJECT_DIR=%SCRIPT_DIR%"
) else (
    for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_DIR=%%~fI\"
)

set "BACKEND_DIR=%PROJECT_DIR%backend"
set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"
set "APP_FILE=%SCRIPT_DIR%foundation_streamlit_test.py"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python virtualenv not found:
    echo         %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo [ERROR] Streamlit app not found:
    echo         %APP_FILE%
    echo.
    pause
    exit /b 1
)

echo ===========================================================
echo [SYSTEM] SHARON BAKERY FOUNDATION TEST LAB v2
echo [BACKEND] %BACKEND_DIR%
echo [APP] %APP_FILE%
echo [MODE] FOUNDATION ONLY - SAM2 + DINOv2
echo [DEBUG] LOAD / SAM / DINO / RENDER timings enabled
echo ===========================================================
echo.

pushd "%PROJECT_DIR%"

"%PYTHON_EXE%" -c "import streamlit, pandas" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit or pandas is not installed.
    echo.
    echo Install with:
    echo   "%PYTHON_EXE%" -m pip install streamlit pandas
    echo.
    popd
    pause
    exit /b 1
)

set "PYTHONUNBUFFERED=1"
set "PYTHONFAULTHANDLER=1"

echo [INFO] Starting Streamlit on http://127.0.0.1:8502
echo.

"%PYTHON_EXE%" -m streamlit run "%APP_FILE%" ^
  --server.address 127.0.0.1 ^
  --server.port 8502 ^
  --browser.gatherUsageStats false

set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
echo [INFO] Streamlit exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
