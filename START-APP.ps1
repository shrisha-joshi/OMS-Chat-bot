# Quick Start Script - OMS Chat Bot
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  OMS CHAT BOT - QUICK START" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Start Backend
Write-Host "`nStarting Backend Server (Port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\OMS Chat Bot\backend'; Write-Host 'BACKEND SERVER' -ForegroundColor Green; python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000" -WindowStyle Normal

# Wait for backend to initialize
Write-Host "Waiting for backend to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 10

# Build Frontend (if needed)
Write-Host "`nChecking Frontend Build..." -ForegroundColor Yellow
if (-not (Test-Path "d:\OMS Chat Bot\frontend\.next\BUILD_ID")) {
    Write-Host "Building frontend (first-time setup)..." -ForegroundColor Gray
    cd "d:\OMS Chat Bot\frontend"
    npm run build | Out-Null
}

# Start Frontend
Write-Host "Starting Frontend Server (Port 3000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\OMS Chat Bot\frontend'; Write-Host 'FRONTEND SERVER (Production Mode)' -ForegroundColor Green; npm run start" -WindowStyle Normal

# Wait for frontend to start
Write-Host "Waiting for frontend to start..." -ForegroundColor Gray
Start-Sleep -Seconds 12

# Test servers
Write-Host "`nTesting Servers..." -ForegroundColor Yellow
try {
    $backend = (Invoke-WebRequest -Uri http://127.0.0.1:8000/health -UseBasicParsing -TimeoutSec 5).StatusCode
    Write-Host "[OK] Backend:  http://127.0.0.1:8000 (Status: $backend)" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Backend not accessible" -ForegroundColor Red
}

try {
    $frontend = (Invoke-WebRequest -Uri http://localhost:3000 -UseBasicParsing -TimeoutSec 5).StatusCode
    Write-Host "[OK] Frontend: http://localhost:3000 (Status: $frontend)" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Frontend not accessible" -ForegroundColor Red
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  APPLICATION READY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nAccess URLs:" -ForegroundColor Yellow
Write-Host "  Frontend:  http://localhost:3000" -ForegroundColor Cyan
Write-Host "  Backend:   http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "  API Docs:  http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "`nTwo PowerShell windows are now running your servers." -ForegroundColor Gray
Write-Host "Do NOT close them while using the application." -ForegroundColor Gray

# Open browser
Write-Host "`nOpening browser..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
Start-Process "http://localhost:3000"

Write-Host "`nPress any key to close this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
