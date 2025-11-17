# Start OMS Chatbot Application
# This script starts both backend and frontend servers

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Starting OMS Chatbot Application" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Kill any existing processes
Write-Host "[1/4] Cleaning up old processes..." -ForegroundColor Yellow
Get-Process -Name python,node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Start Backend
Write-Host "`n[2/4] Starting Backend (FastAPI)..." -ForegroundColor Yellow
$backendPath = "d:\OMS Chat Bot\backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; python -m uvicorn app.main:app --reload"
Write-Host "  Backend starting at http://127.0.0.1:8000" -ForegroundColor Green

Start-Sleep -Seconds 5

# Start Frontend
Write-Host "`n[3/4] Starting Frontend (Next.js)..." -ForegroundColor Yellow
$frontendPath = "d:\OMS Chat Bot\frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; Remove-Item -Recurse -Force .next -ErrorAction SilentlyContinue; npm run dev"
Write-Host "  Frontend starting at http://localhost:3000" -ForegroundColor Green

# Wait and check health
Write-Host "`n[4/4] Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "`nChecking backend health..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
    Write-Host "  ✓ Backend is running!" -ForegroundColor Green
    Write-Host "    MongoDB: $($response.databases.mongodb)" -ForegroundColor Cyan
    Write-Host "    Qdrant: $($response.databases.qdrant)" -ForegroundColor Cyan
} catch {
    Write-Host "  ! Backend not ready yet (may need a few more seconds)" -ForegroundColor Yellow
}

Write-Host "`nChecking frontend..." -ForegroundColor Yellow
try {
    $frontendResponse = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    if ($frontendResponse.StatusCode -eq 200) {
        Write-Host "  ✓ Frontend is running!" -ForegroundColor Green
    }
} catch {
    Write-Host "  ! Frontend not ready yet (may need a few more seconds)" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Application Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nAccess Points:" -ForegroundColor White
Write-Host "  • Frontend:    http://localhost:3000" -ForegroundColor Cyan
Write-Host "  • Backend API: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "  • API Docs:    http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "  • Metrics:     http://127.0.0.1:8000/metrics" -ForegroundColor Cyan

Write-Host "`nServers are running in separate windows." -ForegroundColor Yellow
Write-Host "Close those windows to stop the servers.`n" -ForegroundColor Yellow
