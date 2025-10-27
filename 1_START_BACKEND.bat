@echo off
REM Backend Server Startup Script
REM Starts the FastAPI backend on http://127.0.0.1:8000

title OMS Chat Bot - Backend Server
cd /d "d:\OMS Chat Bot\backend"

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  BACKEND SERVER - OMS CHAT BOT RAG APPLICATION            ║
echo ║                                                            ║
echo ║  FastAPI Server starting on http://127.0.0.1:8000        ║
echo ║  API Documentation: http://127.0.0.1:8000/docs           ║
echo ║                                                            ║
echo ║  To stop: Press Ctrl+C                                   ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Starting FastAPI server...
echo.

REM Start the server with reload enabled for development
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

echo.
echo Backend server stopped.
pause
