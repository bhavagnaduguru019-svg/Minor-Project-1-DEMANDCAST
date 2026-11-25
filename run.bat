@echo off
title DEMANDCAST Server
echo ========================================================
echo   DEMANDCAST - AI Demand Forecasting Platform
echo ========================================================
echo.
echo Starting FastAPI Backend + React Dashboard on Port 8000...
echo.
echo Application URL: http://localhost:8000
echo Swagger API Docs: http://localhost:8000/docs
echo.
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
