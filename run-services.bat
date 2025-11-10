@echo off
REM Simple batch file to run both services
REM Kill any existing processes
taskkill /F /IM node.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1

timeout /t 2 >nul

echo.
echo ========================================
echo   Starting OMS Chatbot Services
echo ========================================
echo.

REM Start backend in a new window
echo Starting Backend (FastAPI on port 8000)...
start "OMS Backend" cmd /k "cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

REM Wait for backend to start
timeout /t 5 >nul

REM Start frontend in a new window
echo Starting Frontend (Next.js on port 3000)...
start "OMS Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 5 >nul

echo.
echo ========================================
echo   Services Started Successfully!
echo ========================================
echo.
echo Frontend: http://localhost:3000
echo Backend:  http://127.0.0.1:8000
echo API Docs: http://127.0.0.1:8000/docs
echo.
echo Open http://localhost:3000 in your browser!
echo.
pause
