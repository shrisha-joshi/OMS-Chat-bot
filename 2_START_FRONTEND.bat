@echo off
REM Frontend Server Startup Script
REM Starts the Next.js frontend on http://localhost:3000

title OMS Chat Bot - Frontend Server
cd /d "d:\OMS Chat Bot\frontend"

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  FRONTEND SERVER - OMS CHAT BOT RAG APPLICATION           ║
echo ║                                                            ║
echo ║  Next.js Application starting on http://localhost:3000   ║
echo ║                                                            ║
echo ║  To stop: Press Ctrl+C                                   ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

echo Starting Next.js frontend...
echo.

REM Start the development server
call npm run dev

echo.
echo Frontend server stopped.
pause
