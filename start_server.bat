@echo off
title Fake News Detection Server

echo ===================================================
echo     Fake News Intelligence - Startup Script
echo ===================================================

echo [1/3] Checking and installing dependencies...
pip install -r requirements.txt --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo Error installing dependencies. Please check your Python/pip installation!
    pause
    exit /b %errorlevel%
)

echo [2/3] Opening browser...
start http://127.0.0.1:5000

echo [3/3] Starting the Flask Backend...
echo The initial AI Model Training may take up to 20 seconds. Please wait!
python codes.py

pause
