@echo off
title Cloudflare Named Tunnel - Sharon AI Inventory
color 0A

echo ===================================================
echo [SYSTEM] KHOI DONG NAMED TUNNEL
echo [URL] https://inventory.sharon-finefoods.com
echo [ORIGIN] http://127.0.0.1:8080
echo ===================================================
echo.

"%~dp0cloudflared\cloudflared.exe" tunnel --config "%~dp0cloudflared\config.yml" run

echo.
echo [ERROR] Named Tunnel da dung. Kiem tra backend cong 8080 va file log.
pause
