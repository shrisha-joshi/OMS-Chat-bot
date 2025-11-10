# OMS Chatbot - Complete Startup Script
# Starts both backend and frontend servers

Write-Host "`n" -NoNewline
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                              ║" -ForegroundColor Cyan
Write-Host "║          🚀 OMS CHATBOT - STARTING APPLICATION 🚀           ║" -ForegroundColor Cyan
Write-Host "║                                                              ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if backend is already running
Write-Host "📡 Checking backend status..." -ForegroundColor Yellow
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 2
    Write-Host "   ✅ Backend already running on port 8000" -ForegroundColor Green
} catch {
    Write-Host "   ⚙️  Starting backend server..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\OMS Chat Bot\backend'; Write-Host '🔥 BACKEND SERVER' -ForegroundColor Cyan; & 'D:\OMS Chat Bot\.venv\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
    Write-Host "   ⏳ Waiting for backend to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 15
    
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 5
        Write-Host "   ✅ Backend started successfully!" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Backend failed to start" -ForegroundColor Red
        exit 1
    }
}

# Check if frontend is already running
Write-Host "`n🌐 Checking frontend status..." -ForegroundColor Yellow
try {
    $null = Invoke-WebRequest -Uri "http://localhost:3001" -TimeoutSec 2 -UseBasicParsing
    Write-Host "   ✅ Frontend already running on port 3001" -ForegroundColor Green
} catch {
    Write-Host "   ⚙️  Starting frontend server..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\OMS Chat Bot\frontend'; Write-Host '🌟 FRONTEND SERVER' -ForegroundColor Cyan; npm run dev"
    Write-Host "   ⏳ Waiting for frontend to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 8
    
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:3001" -TimeoutSec 5 -UseBasicParsing
        Write-Host "   ✅ Frontend started successfully!" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️  Frontend may still be starting..." -ForegroundColor Yellow
    }
}

# Display final status
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                                                              ║" -ForegroundColor Green
Write-Host "║              ✨ APPLICATION READY! ✨                        ║" -ForegroundColor Green
Write-Host "║                                                              ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "🔗 ACCESS POINTS:" -ForegroundColor Cyan
Write-Host "   📱 Frontend:        http://localhost:3001" -ForegroundColor White
Write-Host "   📱 Admin Panel:     http://localhost:3001/admin" -ForegroundColor White
Write-Host "   🔌 Backend API:     http://127.0.0.1:8000" -ForegroundColor White
Write-Host "   📚 API Docs:        http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host ""

# Get system info
try {
    $info = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 3
    Write-Host "📊 SYSTEM STATUS:" -ForegroundColor Cyan
    Write-Host "   📦 Documents:       $($info.databases.mongodb.objects) in MongoDB" -ForegroundColor White
    Write-Host "   🔍 Vectors:         $($info.databases.qdrant.points_count) in Qdrant" -ForegroundColor White
    Write-Host "   💾 Database Size:   $([math]::Round($info.databases.mongodb.dataSize/1KB, 2)) KB" -ForegroundColor White
    Write-Host ""
} catch {
    Write-Host "⚠️  Could not fetch system status" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "✅ Both servers are running in separate windows" -ForegroundColor Green
Write-Host "📝 Keep those windows open to keep the servers running" -ForegroundColor Yellow
Write-Host "🛑 Press Ctrl+C in each window to stop the servers" -ForegroundColor Yellow
Write-Host ""

# Open browser
Write-Host "🌐 Opening browser..." -ForegroundColor Cyan
Start-Process "http://localhost:3001"

Write-Host ""
Write-Host "Press any key to exit this script (servers will continue running)..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
