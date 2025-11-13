# Complete System Diagnostic Script
# Tests every component of the RAG system

Write-Host "`n" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "           OMS CHATBOT - COMPLETE SYSTEM DIAGNOSTIC" -ForegroundColor White
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

$testResults = @{
    "Backend Health" = $false
    "Direct LLM" = $false
    "Full RAG Query" = $false
    "Frontend Running" = $false
    "LMStudio Reachable" = $false
}

# Test 1: Backend Health
Write-Host "TEST 1: Backend Health Check" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------"
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Host " ✅ Backend is alive: $($health.status)" -ForegroundColor Green
    $testResults["Backend Health"] = $true
} catch {
    Write-Host " ❌ Backend is DOWN!" -ForegroundColor Red
    Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 2: LMStudio Direct
Write-Host "TEST 2: LMStudio Direct Connection" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------"
try {
    $lmModels = Invoke-RestMethod -Uri "http://192.168.56.1:1234/v1/models" -TimeoutSec 5
    Write-Host " ✅ LMStudio is responding!" -ForegroundColor Green
    Write-Host "    Models available: $($lmModels.data.Count)" -ForegroundColor White
    $lmModels.data | ForEach-Object {
        Write-Host "      - $($_.id)" -ForegroundColor Gray
    }
    $testResults["LMStudio Reachable"] = $true
} catch {
    Write-Host " ❌ Cannot reach LMStudio!" -ForegroundColor Red
    Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "    Make sure LMStudio is running on 192.168.56.1:1234" -ForegroundColor Yellow
}
Write-Host ""

# Test 3: Backend → LMStudio (Direct LLM Test)
Write-Host "TEST 3: Backend → LMStudio (Direct LLM Test)" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------"
try {
    $llmTest = Invoke-RestMethod -Uri "http://localhost:8000/chat/test-llm" `
        -Method POST `
        -Body '{"query":"Say hello in 5 words"}' `
        -ContentType "application/json" `
        -TimeoutSec 60
    
    if ($llmTest.status -eq "success" -and $llmTest.response.Length -gt 0) {
        Write-Host " ✅ LLM Test PASSED!" -ForegroundColor Green
        Write-Host "    Response: $($llmTest.response.Substring(0, [Math]::Min(80, $llmTest.response.Length)))..." -ForegroundColor White
        $testResults["Direct LLM"] = $true
    } else {
        Write-Host " ❌ LLM returned empty or error response" -ForegroundColor Red
        Write-Host "    Status: $($llmTest.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host " ❌ LLM Test FAILED!" -ForegroundColor Red
    Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 4: Full RAG Query
Write-Host "TEST 4: Full RAG Query (Frontend → Backend → LLM)" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------"
try {
    $ragQuery = @{
        query = "Hello"
        session_id = "diagnostic-test-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        stream = $false
    } | ConvertTo-Json

    Write-Host "   Sending RAG query..." -ForegroundColor Gray
    $ragResponse = Invoke-RestMethod -Uri "http://localhost:8000/chat/query" `
        -Method POST `
        -Body $ragQuery `
        -ContentType "application/json" `
        -TimeoutSec 180

    if ($ragResponse.response -and $ragResponse.response -notlike "*error*" -and $ragResponse.response -notlike "*apologize*") {
        Write-Host " ✅ RAG Query PASSED!" -ForegroundColor Green
        Write-Host "    Response: $($ragResponse.response.Substring(0, [Math]::Min(100, $ragResponse.response.Length)))..." -ForegroundColor White
        Write-Host "    Processing time: $([math]::Round($ragResponse.processing_time, 2))s" -ForegroundColor White
        Write-Host "    Sources: $($ragResponse.sources.Count)" -ForegroundColor White
        Write-Host "    Tokens: $($ragResponse.tokens_generated)" -ForegroundColor White
        $testResults["Full RAG Query"] = $true
    } else {
        Write-Host " ❌ RAG Query returned ERROR response" -ForegroundColor Red
        Write-Host "    Response: $($ragResponse.response)" -ForegroundColor Yellow
        Write-Host "    This means the RAG pipeline is failing internally" -ForegroundColor Yellow
        Write-Host "    Check backend logs for the actual error" -ForegroundColor Yellow
    }
} catch {
    Write-Host " ❌ RAG Query FAILED!" -ForegroundColor Red
    Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.ErrorDetails.Message) {
        Write-Host "    Details: $($_.ErrorDetails.Message)" -ForegroundColor Yellow
    }
}
Write-Host ""

# Test 5: Frontend
Write-Host "TEST 5: Frontend Accessibility" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------"
try {
    $frontend = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 5 -UseBasicParsing
    Write-Host " ✅ Frontend is running!" -ForegroundColor Green
    Write-Host "    Status Code: $($frontend.StatusCode)" -ForegroundColor White
    Write-Host "    Content Length: $($frontend.Content.Length) bytes" -ForegroundColor White
    $testResults["Frontend Running"] = $true
} catch {
    Write-Host " ❌ Frontend is NOT accessible!" -ForegroundColor Red
    Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "    Start frontend with: cd frontend && npm run dev" -ForegroundColor Yellow
}
Write-Host ""

# Test 6: Check Ports
Write-Host "TEST 6: Port Status" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------"
$ports = @(8000, 3000, 1234)
foreach ($port in $ports) {
    $listening = netstat -ano | findstr ":$port.*LISTENING"
    if ($listening) {
        Write-Host " ✅ Port $port is LISTENING" -ForegroundColor Green
        $processId = ($listening -split '\s+')[-1]
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "    Process: $($process.ProcessName) (PID: $processId)" -ForegroundColor Gray
        }
    } else {
        Write-Host " ❌ Port $port is NOT listening" -ForegroundColor Red
    }
}
Write-Host ""

# Summary
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "                         TEST SUMMARY" -ForegroundColor White
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

$passed = ($testResults.Values | Where-Object { $_ -eq $true }).Count
$total = $testResults.Count

foreach ($test in $testResults.GetEnumerator()) {
    $symbol = if ($test.Value) { " ✅" } else { " ❌" }
    $color = if ($test.Value) { "Green" } else { "Red" }
    Write-Host "$symbol $($test.Key)" -ForegroundColor $color
}

Write-Host ""
Write-Host "Score: $passed / $total tests passed" -ForegroundColor $(if ($passed -eq $total) { "Green" } else { "Yellow" })
Write-Host ""

# Diagnosis
if ($testResults["Backend Health"] -and $testResults["Direct LLM"] -and -not $testResults["Full RAG Query"]) {
    Write-Host "DIAGNOSIS:" -ForegroundColor Yellow
    Write-Host "  Backend and LLM are working, but RAG pipeline is failing." -ForegroundColor White
    Write-Host "  This is likely due to:" -ForegroundColor White
    Write-Host "    1. Database connection issue (MongoDB/Qdrant)" -ForegroundColor White
    Write-Host "    2. Missing or corrupted indexes" -ForegroundColor White
    Write-Host "    3. Service initialization error" -ForegroundColor White
    Write-Host ""
    Write-Host "NEXT STEPS:" -ForegroundColor Cyan
    Write-Host "  1. Check backend terminal for error logs" -ForegroundColor White
    Write-Host "  2. Verify .env file has correct database credentials" -ForegroundColor White
    Write-Host "  3. Test database connections individually" -ForegroundColor White
    Write-Host ""
} elseif ($passed -eq $total) {
    Write-Host "🎉 ALL TESTS PASSED! System is fully functional!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now:" -ForegroundColor White
    Write-Host "  1. Open http://localhost:3000 in your browser" -ForegroundColor White
    Write-Host "  2. Start chatting with the bot" -ForegroundColor White
    Write-Host "  3. Upload documents via /admin" -ForegroundColor White
    Write-Host ""
} elseif (-not $testResults["Backend Health"]) {
    Write-Host "DIAGNOSIS:" -ForegroundColor Yellow
    Write-Host "  Backend is not running!" -ForegroundColor White
    Write-Host ""
    Write-Host "NEXT STEPS:" -ForegroundColor Cyan
    Write-Host "  1. cd backend" -ForegroundColor White
    Write-Host "  2. python -m uvicorn app.main:app --reload" -ForegroundColor White
    Write-Host ""
} elseif (-not $testResults["LMStudio Reachable"]) {
    Write-Host "DIAGNOSIS:" -ForegroundColor Yellow
    Write-Host "  LMStudio is not running or not reachable!" -ForegroundColor White
    Write-Host ""
    Write-Host "NEXT STEPS:" -ForegroundColor Cyan
    Write-Host "  1. Start LMStudio application" -ForegroundColor White
    Write-Host "  2. Load a model (e.g., mistral-7b-instruct)" -ForegroundColor White
    Write-Host "  3. Verify it's running on 192.168.56.1:1234" -ForegroundColor White
    Write-Host ""
}

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""
