@echo off
title IGNTU AI Attendance System Launcher
echo ===================================================
echo   IGNTU BCA Department - AI Attendance System
echo ===================================================
echo.
echo Starting Python backend server...

:: Navigate to the directory where this script is located
cd /d "%~dp0"

:: Start backend uvicorn server in the background
start /b "" "%~dp0.venv\Scripts\python.exe" backend/main.py

:: Wait 3 seconds for the server to initialize
timeout /t 3 /nobreak > nul

echo.
echo Opening browser...
:: Launch default web browser to the local server URL
start http://127.0.0.1:8000

echo.
echo ===================================================
echo   System is running! You can minimize this window.
echo   To STOP the server, close this window.
echo ===================================================
echo.
pause
