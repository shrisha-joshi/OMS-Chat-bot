# Simple System Status Check

Write-Host "`n=== OMS CHATBOT STATUS ===" -ForegroundColor Cyan

# Frontend
Write-Host "`nFrontend: " -NoNewline
try {
    $null = Invoke-WebRequest -Uri "http://localhost:3001" -TimeoutSec 2 -UseBasicParsing
    Write-Host "RUNNING" -ForegroundColor Green
    Write-Host "  URL: http://localhost:3001"
} catch {
    Write-Host "NOT RUNNING" -ForegroundColor Red
}

# Backend
Write-Host "`nBackend: " -NoNewline
try {
    $info = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 2
    Write-Host "RUNNING" -ForegroundColor Green
    Write-Host "  URL: http://127.0.0.1:8000"
    Write-Host "  Documents: $($info.databases.mongodb.objects)"
    Write-Host "  Vectors: $($info.databases.qdrant.points_count)"
} catch {
    Write-Host "NOT RUNNING" -ForegroundColor Red
}

Write-Host "`n========================`n"
