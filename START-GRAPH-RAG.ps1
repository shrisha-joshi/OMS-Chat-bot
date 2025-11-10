# Quick start script for Graph RAG Testing
# Starts backend and frontend for end-to-end testing

Write-Host ""
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  GRAPH RAG SYSTEM - QUICK START" -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

# Start Backend
Write-Host "[1/2] Starting Backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\OMS Chat Bot\backend' ; python run_server.py"
Start-Sleep -Seconds 15

# Check if backend is running
try {
    $info = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 3
    Write-Host "  ✓ Backend RUNNING on port 8000" -ForegroundColor Green
    Write-Host "    Documents: $($info.databases.mongodb.objects)" -ForegroundColor White
    Write-Host "    Vectors: $($info.databases.qdrant.points_count)" -ForegroundColor White
} catch {
    Write-Host "  ✗ Backend not responding" -ForegroundColor Red
    exit 1
}

# Start Frontend
Write-Host ""
Write-Host "[2/2] Starting Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\OMS Chat Bot\frontend' ; npm run dev"
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  SYSTEM READY FOR TESTING!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "ACCESS URLs:" -ForegroundColor Cyan
Write-Host "  Frontend:  http://localhost:3001" -ForegroundColor White
Write-Host "  Admin:     http://localhost:3001/admin" -ForegroundColor White
Write-Host "  API Docs:  http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "GRAPH RAG FEATURES:" -ForegroundColor Cyan
Write-Host "  ✓ Entity Extraction (LLM-based)" -ForegroundColor Green
Write-Host "  ✓ Knowledge Graph Building (Neo4j)" -ForegroundColor Green  
Write-Host "  ✓ Hybrid Retrieval (Vector + Graph)" -ForegroundColor Green
Write-Host "  ✓ Graph-aware Context" -ForegroundColor Green
Write-Host ""
Write-Host "TESTING WORKFLOW:" -ForegroundColor Cyan
Write-Host "  1. Open Admin Panel" -ForegroundColor White
Write-Host "  2. Upload a document (PDF/DOCX/TXT)" -ForegroundColor White
Write-Host "  3. Wait for processing (entities extracted)" -ForegroundColor White
Write-Host "  4. Go to Chat and ask questions" -ForegroundColor White
Write-Host "  5. See graph-enhanced responses!" -ForegroundColor White
Write-Host ""
Write-Host "System is running. Close terminal windows to stop." -ForegroundColor Yellow
