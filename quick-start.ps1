# Quick Start Script - OMS Chat Bot (FIXED VERSION)
Write-Host "Starting OMS Chat Bot with ROOT CAUSE FIXES..." -ForegroundColor Cyan
Write-Host "  - Next.js upgraded to 14.2.18 (Node 22 compatible)" -ForegroundColor Green
Write-Host "  - Port standardized to 3000" -ForegroundColor Green
Write-Host "  - All caches cleared`n" -ForegroundColor Green

# Kill any existing processes on ports 8000 and 3000
Write-Host "Cleaning up old processes..." -ForegroundColor Yellow
$processes8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
$processes3000 = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($pid in $processes8000) {
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
}
foreach ($pid in $processes3000) {
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2

# Start Backend
Write-Host "`nStarting Backend (Port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\OMS Chat Bot\backend'; Write-Host 'Backend Starting...' -ForegroundColor Cyan; python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"

# Wait for backend to initialize
Start-Sleep -Seconds 5

# Start Frontend
Write-Host "Starting Frontend (Port 3000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\OMS Chat Bot\frontend'; Write-Host 'Frontend Starting...' -ForegroundColor Cyan; npm run dev"

# Wait for frontend to start
Start-Sleep -Seconds 8

Write-Host "`nSERVERS SHOULD BE RUNNING!" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "  Backend:  http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "  API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "`nCheck the two PowerShell windows that opened." -ForegroundColor Magenta
Write-Host "If you see errors, press any key to open browser anyway..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Open browser
Start-Process http://localhost:3000
