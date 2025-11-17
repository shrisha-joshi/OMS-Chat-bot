# Start OMS Chatbot - Simple Version
Write-Host "`nStarting OMS Chatbot Application...`n" -ForegroundColor Cyan

# Clean up
Write-Host "[1/3] Stopping old processes..." -ForegroundColor Yellow
Get-Process -Name python,node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Start Backend
Write-Host "[2/3] Starting Backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\OMS Chat Bot\backend'; python -m uvicorn app.main:app --reload"
Write-Host "  Backend: http://127.0.0.1:8000" -ForegroundColor Green
Start-Sleep -Seconds 5

# Start Frontend
Write-Host "[3/3] Starting Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\OMS Chat Bot\frontend'; npm run dev"
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Green

Write-Host "`nWaiting 10 seconds for services to initialize...`n" -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Application Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nOpen in browser:" -ForegroundColor White
Write-Host "  http://localhost:3000`n" -ForegroundColor Cyan
Write-Host "Servers running in separate windows." -ForegroundColor Yellow
Write-Host "Close those windows to stop.`n" -ForegroundColor Yellow
