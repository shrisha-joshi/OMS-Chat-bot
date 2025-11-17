# OMS Chat Bot - Complete Startup Script
# Kills old processes, starts backend with all improvements, runs tests

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "  OMS CHAT BOT - FULL STARTUP" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "`nWith improvements:" -ForegroundColor Yellow
Write-Host "  ✓ File size limits (10MB max)" -ForegroundColor Green
Write-Host "  ✓ Retry logic (3 attempts with backoff)" -ForegroundColor Green
Write-Host "  ✓ Rate limiting (10 req/60s)" -ForegroundColor Green
Write-Host "  ✓ Performance monitoring" -ForegroundColor Green
Write-Host "  ✓ Tokenizer fixed" -ForegroundColor Green
Write-Host "  ✓ Background tasks fixed`n" -ForegroundColor Green

# Step 1: Kill existing Python processes
Write-Host "[1/5] Stopping existing backend..." -ForegroundColor Cyan
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "  ✓ Old processes terminated`n" -ForegroundColor Green

# Step 2: Verify environment
Write-Host "[2/5] Checking environment..." -ForegroundColor Cyan
if (-not (Test-Path "d:\OMS Chat Bot\backend\.env")) {
    Write-Host "  ⚠️  WARNING: .env file not found!" -ForegroundColor Yellow
    Write-Host "     Copy backend/env.sample to backend/.env and configure`n" -ForegroundColor Yellow
}
else {
    Write-Host "  ✓ Environment file found`n" -ForegroundColor Green
}

# Step 3: Start backend
Write-Host "[3/5] Starting backend server..." -ForegroundColor Cyan
cd "d:\OMS Chat Bot\backend"

$backendProcess = Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Write-Host 'OMS Chat Bot Backend Server' -ForegroundColor Green; Write-Host 'Port: 8000' -ForegroundColor Yellow; Write-Host ''; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
) -PassThru -WindowStyle Normal

Write-Host "  ✓ Backend starting (PID: $($backendProcess.Id))`n" -ForegroundColor Green

# Step 4: Wait for backend to be ready
Write-Host "[4/5] Waiting for backend to start..." -ForegroundColor Cyan
$maxAttempts = 30
$attempt = 0
$ready = $false

while ($attempt -lt $maxAttempts -and -not $ready) {
    $attempt++
    Start-Sleep -Seconds 1
    
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 -ErrorAction Stop
        if ($health.status -eq "alive" -or $health.status -eq "degraded") {
            $ready = $true
            Write-Host "  ✓ Backend is ready!" -ForegroundColor Green
            Write-Host "    Status: $($health.status)" -ForegroundColor White
            Write-Host "    MongoDB: $($health.databases.mongodb)" -ForegroundColor $(if($health.databases.mongodb -eq 'connected'){'Green'}else{'Yellow'})
            Write-Host "    Qdrant: $($health.databases.qdrant)" -ForegroundColor $(if($health.databases.qdrant -eq 'connected'){'Green'}else{'Yellow'})
            Write-Host "    Redis: $($health.databases.redis)" -ForegroundColor $(if($health.databases.redis -eq 'connected'){'Green'}else{'Yellow'})
        }
    }
    catch {
        Write-Host "  ." -NoNewline -ForegroundColor Gray
    }
}

if (-not $ready) {
    Write-Host "`n  ✗ Backend failed to start after 30 seconds!" -ForegroundColor Red
    Write-Host "    Check backend window for errors" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 5: Run tests
Write-Host "[5/5] Running test suite..." -ForegroundColor Cyan
cd "d:\OMS Chat Bot"

if (Test-Path "test_rag_pipeline.py") {
    Write-Host ""
    python test_rag_pipeline.py
    $testResult = $LASTEXITCODE
    
    Write-Host ""
    if ($testResult -eq 0) {
        Write-Host "==================================================" -ForegroundColor Green
        Write-Host "  ✓ ALL TESTS PASSED!" -ForegroundColor Green
        Write-Host "==================================================" -ForegroundColor Green
    }
    else {
        Write-Host "==================================================" -ForegroundColor Yellow
        Write-Host "  ⚠️  SOME TESTS FAILED" -ForegroundColor Yellow
        Write-Host "==================================================" -ForegroundColor Yellow
    }
}
else {
    Write-Host "  ⚠️  Test file not found, skipping tests" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  APPLICATION READY" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend:  http://127.0.0.1:8000" -ForegroundColor White
Write-Host "Docs:     http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "Health:   http://127.0.0.1:8000/health" -ForegroundColor White
Write-Host "Metrics:  http://127.0.0.1:8000/metrics" -ForegroundColor White
Write-Host ""
Write-Host "To test upload:" -ForegroundColor Yellow
Write-Host "  curl -X POST http://127.0.0.1:8000/admin/documents/upload-json \" -ForegroundColor Gray
Write-Host "    -H 'Content-Type: application/json' \" -ForegroundColor Gray
Write-Host "    -d '{\"filename\":\"test.txt\",\"content_base64\":\"...\",\"content_type\":\"text/plain\"}'" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop (backend window will remain open)" -ForegroundColor Gray
Write-Host ""

# Keep script running
try {
    while ($true) {
        Start-Sleep -Seconds 60
        
        # Periodic health check
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 -ErrorAction Stop
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Backend: $($health.status) | MongoDB: $($health.databases.mongodb) | Qdrant: $($health.databases.qdrant)" -ForegroundColor DarkGray
        }
        catch {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Backend: NOT RESPONDING!" -ForegroundColor Red
        }
    }
}
catch {
    Write-Host "`nShutting down..." -ForegroundColor Yellow
}
