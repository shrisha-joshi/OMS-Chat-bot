# Complete Connection Test Script
# Tests all connections: Frontend, Backend, WebSocket, LMStudio, Databases

Write-Host "`n╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     CONNECTION ARCHITECTURE VERIFICATION TEST            ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

$results = @()
$passCount = 0
$totalTests = 8

# Test 1: Backend Health
Write-Host "🔍 Test 1: Backend Health Check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 5
    if ($health.status -eq "healthy") {
        Write-Host "  ✅ PASS: Backend is healthy" -ForegroundColor Green
        $results += "Backend Health: ✅ PASS"
        $passCount++
    } else {
        Write-Host "  ❌ FAIL: Backend status is $($health.status)" -ForegroundColor Red
        $results += "Backend Health: ❌ FAIL"
    }
} catch {
    Write-Host "  ❌ FAIL: Backend not responding - $($_.Exception.Message)" -ForegroundColor Red
    $results += "Backend Health: ❌ FAIL (not responding)"
}

# Test 2: MongoDB Connection
Write-Host "`n🔍 Test 2: MongoDB Atlas Connection..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 5
    if ($health.databases.mongodb -eq "ok") {
        Write-Host "  ✅ PASS: MongoDB Atlas connected" -ForegroundColor Green
        $results += "MongoDB: ✅ PASS"
        $passCount++
    } else {
        Write-Host "  ❌ FAIL: MongoDB status is $($health.databases.mongodb)" -ForegroundColor Red
        $results += "MongoDB: ❌ FAIL"
    }
} catch {
    Write-Host "  ❌ FAIL: Cannot check MongoDB - backend not responding" -ForegroundColor Red
    $results += "MongoDB: ❌ FAIL"
}

# Test 3: Qdrant Connection
Write-Host "`n🔍 Test 3: Qdrant Cloud Connection..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 5
    if ($health.databases.qdrant -eq "ok") {
        Write-Host "  ✅ PASS: Qdrant Cloud connected" -ForegroundColor Green
        $results += "Qdrant: ✅ PASS"
        $passCount++
    } else {
        Write-Host "  ❌ FAIL: Qdrant status is $($health.databases.qdrant)" -ForegroundColor Red
        $results += "Qdrant: ❌ FAIL"
    }
} catch {
    Write-Host "  ❌ FAIL: Cannot check Qdrant - backend not responding" -ForegroundColor Red
    $results += "Qdrant: ❌ FAIL"
}

# Test 4: Redis Connection
Write-Host "`n🔍 Test 4: Redis Cloud Connection..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 5
    if ($health.databases.redis -eq "ok") {
        Write-Host "  ✅ PASS: Redis Cloud connected" -ForegroundColor Green
        $results += "Redis: ✅ PASS"
        $passCount++
    } else {
        Write-Host "  ❌ FAIL: Redis status is $($health.databases.redis)" -ForegroundColor Red
        $results += "Redis: ❌ FAIL"
    }
} catch {
    Write-Host "  ❌ FAIL: Cannot check Redis - backend not responding" -ForegroundColor Red
    $results += "Redis: ❌ FAIL"
}

# Test 5: CORS Configuration
Write-Host "`n🔍 Test 5: CORS Configuration..." -ForegroundColor Yellow
try {
    $headers = @{
        "Origin" = "http://127.0.0.1:3000"
        "Access-Control-Request-Method" = "POST"
    }
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -Method Options -Headers $headers -TimeoutSec 5
    $corsOrigin = $response.Headers."Access-Control-Allow-Origin"
    
    if (($corsOrigin -eq "http://127.0.0.1:3000") -or ($corsOrigin -eq "*")) {
        Write-Host "  ✅ PASS: CORS properly configured" -ForegroundColor Green
        $results += "CORS: ✅ PASS"
        $passCount++
    } else {
        Write-Host "  ❌ FAIL: CORS headers incorrect" -ForegroundColor Red
        $results += "CORS: ❌ FAIL"
    }
} catch {
    Write-Host "  ⚠️  WARNING: Could not test CORS (may still work)" -ForegroundColor Yellow
    $results += "CORS: ⚠️  WARNING"
}

# Test 6: LMStudio Connection
Write-Host "`n🔍 Test 6: LMStudio Local AI..." -ForegroundColor Yellow
try {
    $lmstudio = Invoke-RestMethod -Uri "http://127.0.0.1:1234/v1/models" -Method Get -TimeoutSec 5
    if ($lmstudio) {
        Write-Host "  ✅ PASS: LMStudio is running" -ForegroundColor Green
        Write-Host "  ℹ️  Models loaded: $($lmstudio.data.Count)" -ForegroundColor Cyan
        $results += "LMStudio: ✅ PASS"
        $passCount++
    }
} catch {
    Write-Host "  ❌ FAIL: LMStudio not responding - Is it running?" -ForegroundColor Red
    Write-Host "  ℹ️  Start LMStudio and load a model, then retest" -ForegroundColor Yellow
    $results += "LMStudio: ❌ FAIL (not running)"
}

# Test 7: Frontend Connection
Write-Host "`n🔍 Test 7: Frontend Server..." -ForegroundColor Yellow
try {
    $frontend = Invoke-WebRequest -Uri "http://127.0.0.1:3000" -Method Get -TimeoutSec 5
    if ($frontend.StatusCode -eq 200) {
        Write-Host "  ✅ PASS: Frontend server running" -ForegroundColor Green
        $results += "Frontend: ✅ PASS"
        $passCount++
    }
} catch {
    Write-Host "  ❌ FAIL: Frontend not responding - Is npm run dev running?" -ForegroundColor Red
    $results += "Frontend: ❌ FAIL (not running)"
}

# Test 8: WebSocket Monitoring Endpoint
Write-Host "`n🔍 Test 8: WebSocket Monitoring..." -ForegroundColor Yellow
try {
    $wsMetrics = Invoke-RestMethod -Uri "http://127.0.0.1:8000/monitoring/websocket/metrics" -Method Get -TimeoutSec 5
    if ($wsMetrics) {
        Write-Host "  ✅ PASS: WebSocket manager operational" -ForegroundColor Green
        Write-Host "  ℹ️  Active connections: $($wsMetrics.overall.active_connections)" -ForegroundColor Cyan
        $results += "WebSocket: ✅ PASS"
        $passCount++
    }
} catch {
    Write-Host "  ❌ FAIL: WebSocket metrics unavailable - $($_.Exception.Message)" -ForegroundColor Red
    $results += "WebSocket: ❌ FAIL"
}

# Summary
Write-Host "`n╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    TEST SUMMARY                           ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

foreach ($result in $results) {
    Write-Host "  $result"
}

Write-Host "`n📊 Score: $passCount / $totalTests tests passed" -ForegroundColor $(if ($passCount -eq $totalTests) { "Green" } elseif ($passCount -ge 5) { "Yellow" } else { "Red" })

if ($passCount -eq $totalTests) {
    Write-Host "`n🎉 EXCELLENT! All connections are working properly!" -ForegroundColor Green
    Write-Host "   System is ready for production use." -ForegroundColor Green
} elseif ($passCount -ge 6) {
    Write-Host "`n✅ GOOD! Most connections working." -ForegroundColor Yellow
    Write-Host "   Review failed tests above and fix them." -ForegroundColor Yellow
} elseif ($passCount -ge 4) {
    Write-Host "`n⚠️  WARNING! Several connections failing." -ForegroundColor Yellow
    Write-Host "   System may work but not optimally." -ForegroundColor Yellow
} else {
    Write-Host "`n❌ CRITICAL! Multiple connection failures." -ForegroundColor Red
    Write-Host "   System will not function properly." -ForegroundColor Red
}

Write-Host "`n📚 For detailed troubleshooting, see:" -ForegroundColor Cyan
Write-Host "   - CONNECTION_ARCHITECTURE_FIXED.md" -ForegroundColor Cyan
Write-Host "   - API_WEBSOCKET_ENHANCEMENTS.md" -ForegroundColor Cyan

Write-Host "`n🔧 Quick Fixes:" -ForegroundColor Yellow
if ($results | Where-Object { $_ -like "*Backend*FAIL*" }) {
    Write-Host "   - Backend: cd backend; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload" -ForegroundColor White
}
if ($results | Where-Object { $_ -like "*Frontend*FAIL*" }) {
    Write-Host "   - Frontend: cd frontend; npm run dev" -ForegroundColor White
}
if ($results | Where-Object { $_ -like "*LMStudio*FAIL*" }) {
    Write-Host "   - LMStudio: Start LMStudio application and load a model" -ForegroundColor White
}

Write-Host ""
