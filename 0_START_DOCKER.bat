@echo off
REM Database Services Startup Script
REM Starts MongoDB, Qdrant, and Redis using Docker Compose

title OMS Chat Bot - Database Services
cd /d "d:\OMS Chat Bot"

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  DATABASE SERVICES - OMS CHAT BOT RAG APPLICATION          ║
echo ║                                                            ║
echo ║  Starting Docker containers...                            ║
echo ║  - MongoDB    (localhost:27017)                           ║
echo ║  - Qdrant     (http://localhost:6333)                     ║
echo ║  - Redis      (localhost:6379)                            ║
echo ║                                                            ║
echo ║  Note: Keep this window open while using the app         ║
echo ║  To stop: Press Ctrl+C                                   ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Stop any existing containers
echo Cleaning up existing containers...
docker-compose down 2>nul
timeout /t 2 /nobreak

REM Start new containers
echo.
echo Starting Docker containers...
docker-compose up -d

echo.
echo ⏳ Waiting 30 seconds for services to initialize...
timeout /t 30 /nobreak

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  ✅ DATABASE SERVICES STARTED!                             ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Services are now running:
echo.
echo   MongoDB  : localhost:27017
echo   Qdrant   : http://localhost:6333/dashboard
echo   Redis    : localhost:6379
echo.
echo Keep this window open. Services will continue running.
echo.
pause
