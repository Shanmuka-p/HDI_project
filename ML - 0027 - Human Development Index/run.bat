@echo off
title HDI Predictor Web App
echo ===================================================
echo   Starting Human Development Index Predictor App
echo   Access the UI at: http://127.0.0.1:5000
echo ===================================================
echo.
python "%~dp0Flask\app.py"
pause
