# OMS ChatBot - Simple Startup
Write-Host "`n=== OMS CHATBOT STARTUP ===" -ForegroundColor Cyan

# Clean up
Write-Host "Cleaning ports..." -ForegroundColor Yellow
$b = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($b) { Stop-Process -Id $b.OwningProcess -Force -ErrorAction SilentlyContinue }
$f = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue  
if ($f) { Stop-Process -Id $f.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

# Start Backend
Write-Host "Starting Backend..." -ForegroundColor Yellow
$cmd1 = "cd 'D:\OMS Chat Bot\backend'; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd1

# Wait
Write-Host "Waiting 45s for backend..." -ForegroundColor Yellow
for ($i = 0; $i -lt 9; $i++) {
    Start-Sleep -Seconds 5
    try {
        $h = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop
        if ($h.status -eq "alive") { Write-Host "Backend OK!" -ForegroundColor Green; break }
    } catch { Write-Host "." -NoNewline -ForegroundColor Gray }
}

# Start Frontend  
Write-Host "`nStarting Frontend..." -ForegroundColor Yellow
$cmd2 = "cd 'D:\OMS Chat Bot\frontend'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd2
Start-Sleep -Seconds 10

# Open browser
Write-Host "`nOpening browser..." -ForegroundColor Cyan
Start-Process "http://localhost:3000"

Write-Host "`n✅ Done! Check the service windows." -ForegroundColor Green
Read-Host "Press Enter to exit"
