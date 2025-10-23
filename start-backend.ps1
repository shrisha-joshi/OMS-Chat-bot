# Start Backend Server - Keep running with auto-restart
Write-Host "=========================================="
Write-Host "Starting OMS Chat Bot Backend Server"
Write-Host "URL: http://0.0.0.0:8000"
Write-Host "=========================================="
Write-Host ""

$maxRetries = 10
$retryCount = 0

while ($retryCount -lt $maxRetries) {
    Write-Host "[INFO] Starting backend attempt $($retryCount + 1) of $maxRetries..."
    
    try {
        cd "d:\OMS Chat Bot\backend"
        & python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level warning
        
        Write-Host "[WARNING] Backend exited unexpectedly. Restarting in 5 seconds..."
        Start-Sleep -Seconds 5
        $retryCount++
    }
    catch {
        Write-Host "[ERROR] Error: $_"
        Write-Host "[WARNING] Restarting backend in 5 seconds..."
        Start-Sleep -Seconds 5
        $retryCount++
    }
}

Write-Host "[ERROR] Backend failed after $maxRetries attempts. Exiting."
