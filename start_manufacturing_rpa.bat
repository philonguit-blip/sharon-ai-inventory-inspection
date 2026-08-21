@echo off
setlocal
set "RPA_DIR=%~dp0..\sharon-bakery-docker_manufacturing"
if not exist "%RPA_DIR%\start-kiotviet-rpa.bat" (
  echo [ERROR] Manufacturing RPA project not found: %RPA_DIR%
  pause
  exit /b 1
)
call "%RPA_DIR%\start-kiotviet-rpa.bat"
