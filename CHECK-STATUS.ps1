# CHECK-STATUS.ps1 - Quick System Health Check
# Run this anytime to check if your Graph RAG system is working

Clear-Host
Write-Host "`n╔═══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   OMS CHATBOT - SYSTEM STATUS CHECK      ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Check Frontend
Write-Host "🌐 Frontend Check..." -NoNewline
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3001" -TimeoutSec 3 -UseBasicParsing
    Write-Host " ✅ RUNNING" -ForegroundColor Green
    Write-Host "   URL: http://localhost:3001" -ForegroundColor Gray
} catch {
    Write-Host " ❌ NOT RESPONDING" -ForegroundColor Red
    Write-Host "   Fix: cd frontend ; npm run dev" -ForegroundColor Yellow
}

# Check Backend
Write-Host "`n🔧 Backend Check..." -NoNewline
try {
    $info = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 3
    Write-Host " ✅ RUNNING" -ForegroundColor Green
    Write-Host "   URL: http://127.0.0.1:8000" -ForegroundColor Gray
    Write-Host "   API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Gray
    
    # Show database stats
    Write-Host "`n📊 Database Status:" -ForegroundColor Cyan
    Write-Host "   MongoDB:  $($info.databases.mongodb.objects) documents" -ForegroundColor White
    Write-Host "   Qdrant:   $($info.databases.qdrant.points_count) vectors" -ForegroundColor White
    $redisMemory = $info.databases.redis.memory_used
    Write-Host "   Redis:    Connected - $redisMemory" -ForegroundColor White
    
    # Check Graph RAG
    if ($info.configuration.use_graph_search) {
        Write-Host "   Graph RAG: ENABLED" -ForegroundColor Green
    } else {
        Write-Host "   Graph RAG: DISABLED" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host " ❌ NOT RESPONDING" -ForegroundColor Red
    Write-Host "   Fix: cd backend ; python run_server.py" -ForegroundColor Yellow
}

Write-Host "`n═══════════════════════════════════════════" -ForegroundColor Cyan

# Summary
$frontendRunning = $false
$backendRunning = $false

try {
    $null = Invoke-WebRequest -Uri "http://localhost:3001" -TimeoutSec 2 -UseBasicParsing
    $frontendRunning = $true
} catch {}

try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 2
    $backendRunning = $true
} catch {}

if ($frontendRunning -and $backendRunning) {
    Write-Host "`nALL SYSTEMS OPERATIONAL!" -ForegroundColor Green
    Write-Host "   Ready to chat at: http://localhost:3001`n" -ForegroundColor White
} elseif ($backendRunning) {
    Write-Host "`nBackend OK, Frontend starting..." -ForegroundColor Yellow
    Write-Host "   Wait 10 seconds and try again`n" -ForegroundColor White
} else {
    Write-Host "`nSERVICES NOT RUNNING" -ForegroundColor Red
    Write-Host "   Run: START-GRAPH-RAG.ps1" -ForegroundColor Yellow
    Write-Host ""
}
