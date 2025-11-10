# START BOTH BACKEND AND FRONTEND FOR OMS CHATBOT
# This script starts both servers in separate windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OMS CHATBOT - COMPLETE STARTUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get current directory
$rootDir = "d:\OMS Chat Bot"

Write-Host "Starting Backend Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$rootDir\start-backend.ps1'"

Write-Host "Waiting 5 seconds for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "Starting Frontend Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$rootDir\frontend'; npm run dev"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SERVERS STARTED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend API:  http://127.0.0.1:8000" -ForegroundColor White
Write-Host "API Docs:     http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "Frontend:     http://localhost:3000" -ForegroundColor White
Write-Host "Admin Panel:  http://localhost:3000/admin" -ForegroundColor White
Write-Host ""
Write-Host "Press Enter to close this window (servers will keep running)" -ForegroundColor Yellow
Read-Host
