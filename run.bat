@echo off
title EduPulse AI - Startup Manager
color 0b

echo ===================================================
echo       Starting EduPulse AI System...
echo ===================================================
echo.

echo [1/2] Launching FastAPI Backend (Port 8000)...
start "EduPulse Backend" cmd /k "cd /d e:\UNI Important\5th semester\Machine Learning\backend && echo Installing backend dependencies... && pip install -r requirements.txt -q && echo. && echo ✓ Starting FastAPI server on http://localhost:8000 && echo. && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 >nul

echo [2/2] Launching Vite Frontend (Port 5173)...
start "EduPulse Frontend" cmd /k "cd /d e:\UNI Important\5th semester\Machine Learning\frontend && echo Installing frontend dependencies... && npm install --silent && echo. && echo ✓ Starting Vite dev server... && echo. && npm run dev"

echo.
echo ===================================================
echo  Both services are booting in separate windows!
echo.
echo  IMPORTANT: Wait for the backend window to say
echo  "Uvicorn running on http://0.0.0.0:8000"
echo  BEFORE using the frontend.
echo.
echo  Backend API Docs:  http://localhost:8000/docs
echo  Frontend Web App:  http://localhost:5173
echo.
echo  Close the two new windows to stop the servers.
echo ===================================================
pause
