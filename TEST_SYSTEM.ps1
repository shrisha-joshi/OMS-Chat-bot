# ============================================================================
# OMS Chat Bot - Comprehensive Test Suite
# ============================================================================
# Tests all components of the RAG chatbot system
# ============================================================================

param(
    [switch]$Quick,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
function Write-TestHeader {
    param([string]$Text)
    Write-Host "`n" -NoNewline
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    for($i=0; $i -lt 70; $i++) { Write-Host "=" -NoNewline -ForegroundColor Cyan }
    Write-Host ""
    Write-Host "  $Text" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    for($i=0; $i -lt 70; $i++) { Write-Host "=" -NoNewline -ForegroundColor Cyan }
    Write-Host "`n"
}

function Write-TestCase {
    param([string]$Text)
    Write-Host "`n► TEST: " -NoNewline -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor White
}

function Write-Pass {
    param([string]$Text)
    Write-Host "  ✅ PASS: " -NoNewline -ForegroundColor Green
    Write-Host $Text -ForegroundColor White
}

function Write-Fail {
    param([string]$Text)
    Write-Host "  ❌ FAIL: " -NoNewline -ForegroundColor Red
    Write-Host $Text -ForegroundColor Red
}

function Write-Info {
    param([string]$Text)
    Write-Host "  ℹ️  " -NoNewline -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Gray
}

# ============================================================================
# TEST STATISTICS
# ============================================================================
$script:totalTests = 0
$script:passedTests = 0
$script:failedTests = 0
$script:skippedTests = 0

function Record-TestResult {
    param([bool]$Passed, [bool]$Skipped = $false)
    $script:totalTests++
    if ($Skipped) {
        $script:skippedTests++
    } elseif ($Passed) {
        $script:passedTests++
    } else {
        $script:failedTests++
    }
}

# ============================================================================
# BANNER
# ============================================================================
Clear-Host
Write-Host ""
Write-Host "  _____ _____ ____ _____   ____  _   _ ___ _____ _____ " -ForegroundColor Cyan
Write-Host " |_   _| ____/ ___|_   _| / ___|| | | |_ _|_   _| ____|" -ForegroundColor Cyan
Write-Host "   | | |  _| \___ \ | |   \___ \| | | || |  | | |  _|  " -ForegroundColor Cyan
Write-Host "   | | | |___ ___) || |    ___) | |_| || |  | | | |___ " -ForegroundColor Cyan
Write-Host "   |_| |_____|____/ |_|   |____/ \___/|___| |_| |_____|" -ForegroundColor Cyan
Write-Host ""
Write-Host "  OMS Chat Bot - Comprehensive Test Suite" -ForegroundColor Green
Write-Host ""

# ============================================================================
# SECTION 1: INFRASTRUCTURE TESTS
# ============================================================================
Write-TestHeader "SECTION 1: INFRASTRUCTURE TESTS"

Write-TestCase "Backend Server Availability"
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 5
    Write-Pass "Backend is running on port 8000"
    Write-Info "App Environment: $($response.app_info.app_env)"
    Write-Info "Debug Mode: $($response.app_info.debug_mode)"
    Record-TestResult -Passed $true
} catch {
    Write-Fail "Backend is not accessible: $($_.Exception.Message)"
    Write-Info "Make sure backend is running: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
    Record-TestResult -Passed $false
}

Write-TestCase "Frontend Server Availability"
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3001" -TimeoutSec 5 -UseBasicParsing
    Write-Pass "Frontend is running on port 3001"
    Write-Info "HTTP Status: $($response.StatusCode)"
    Record-TestResult -Passed $true
} catch {
    Write-Fail "Frontend is not accessible: $($_.Exception.Message)"
    Write-Info "Make sure frontend is running: npm run dev"
    Record-TestResult -Passed $false
}

# ============================================================================
# SECTION 2: DATABASE CONNECTIVITY
# ============================================================================
Write-TestHeader "SECTION 2: DATABASE CONNECTIVITY"

Write-TestCase "MongoDB Connection"
try {
    $sysInfo = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 5
    if ($sysInfo.databases.mongodb.connected) {
        Write-Pass "MongoDB is connected"
        Write-Info "Database: $($sysInfo.databases.mongodb.database)"
        Write-Info "Documents: $($sysInfo.databases.mongodb.objects)"
        Write-Info "Data Size: $([math]::Round($sysInfo.databases.mongodb.dataSize/1KB, 2)) KB"
        Record-TestResult -Passed $true
    } else {
        Write-Fail "MongoDB is not connected"
        Record-TestResult -Passed $false
    }
} catch {
    Write-Fail "Could not check MongoDB status"
    Record-TestResult -Passed $false
}

Write-TestCase "Qdrant Vector DB Connection"
try {
    $sysInfo = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 5
    if ($sysInfo.databases.qdrant.connected) {
        Write-Pass "Qdrant is connected"
        Write-Info "URL: $($sysInfo.databases.qdrant.url)"
        Write-Info "Collections: $($sysInfo.databases.qdrant.collections)"
        Write-Info "Vector Points: $($sysInfo.databases.qdrant.points_count)"
        Record-TestResult -Passed $true
    } else {
        Write-Fail "Qdrant is not connected"
        Record-TestResult -Passed $false
    }
} catch {
    Write-Fail "Could not check Qdrant status"
    Record-TestResult -Passed $false
}

Write-TestCase "Redis Cache Connection"
try {
    $sysInfo = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 5
    if ($sysInfo.databases.redis.connected) {
        Write-Pass "Redis is connected"
        Write-Info "Cache active and ready"
        Record-TestResult -Passed $true
    } else {
        Write-Fail "Redis is not connected"
        Record-TestResult -Passed $false
    }
} catch {
    Write-Fail "Could not check Redis status"
    Record-TestResult -Passed $false
}

# ============================================================================
# SECTION 3: LLM CONNECTIVITY
# ============================================================================
Write-TestHeader "SECTION 3: LLM CONNECTIVITY"

Write-TestCase "Direct LLM Connection (Bypass RAG)"
try {
    $startTime = Get-Date
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat/test-llm" `
                                  -Method POST `
                                  -Body '{"query":"Say hello in 3 words only"}' `
                                  -ContentType "application/json" `
                                  -TimeoutSec 60
    $elapsed = ((Get-Date) - $startTime).TotalSeconds
    
    if ($response.status -eq "success") {
        Write-Pass "LMStudio connection successful ($([math]::Round($elapsed,2))s)"
        Write-Info "Response: $($response.response)"
        Record-TestResult -Passed $true
    } else {
        Write-Fail "LMStudio returned error status"
        Record-TestResult -Passed $false
    }
} catch {
    Write-Fail "LMStudio connection failed: $($_.Exception.Message)"
    Write-Info "Make sure LMStudio is running with a loaded model"
    Write-Info "Expected URL: http://192.168.56.1:1234/v1"
    Record-TestResult -Passed $false
}

# ============================================================================
# SECTION 4: API ENDPOINTS
# ============================================================================
Write-TestHeader "SECTION 4: API ENDPOINTS"

Write-TestCase "Health Check Endpoint"
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
    if ($response.status -eq "healthy") {
        Write-Pass "Health endpoint working"
        Record-TestResult -Passed $true
    } else {
        Write-Fail "Health check returned unexpected status"
        Record-TestResult -Passed $false
    }
} catch {
    Write-Fail "Health check failed: $($_.Exception.Message)"
    Record-TestResult -Passed $false
}

Write-TestCase "System Info Endpoint"
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/system/info" -TimeoutSec 5
    if ($response.app_info) {
        Write-Pass "System info endpoint working"
        Write-Info "Uptime: $([math]::Round($response.app_info.uptime_seconds, 0))s"
        Record-TestResult -Passed $true
    } else {
        Write-Fail "System info incomplete"
        Record-TestResult -Passed $false
    }
} catch {
    Write-Fail "System info failed: $($_.Exception.Message)"
    Record-TestResult -Passed $false
}

# ============================================================================
# SECTION 5: CHAT FUNCTIONALITY
# ============================================================================
Write-TestHeader "SECTION 5: CHAT FUNCTIONALITY"

Write-TestCase "Simple Chat Query (Non-Document Based)"
try {
    $startTime = Get-Date
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat/query" `
                                  -Method POST `
                                  -Body '{"query":"What is 2+2? Answer with just the number.","session_id":"test-simple"}' `
                                  -ContentType "application/json" `
                                  -TimeoutSec 120
    $elapsed = ((Get-Date) - $startTime).TotalSeconds
    
    if ($response.response) {
        Write-Pass "Chat query completed ($([math]::Round($elapsed,2))s)"
        Write-Info "Response: $($response.response.Substring(0,[Math]::Min(100,$response.response.Length)))..."
        Write-Info "Processing time: $($response.processing_time)s"
        
        if ($response.tokens_generated -gt 0) {
            Write-Info "Tokens generated: $($response.tokens_generated) ✓"
            Record-TestResult -Passed $true
        } else {
            Write-Fail "No tokens generated (LLM may not have responded)"
            Record-TestResult -Passed $false
        }
    } else {
        Write-Fail "Chat query returned empty response"
        Record-TestResult -Passed $false
    }
} catch {
    Write-Fail "Chat query failed: $($_.Exception.Message)"
    Record-TestResult -Passed $false
}

if (-not $Quick) {
    Write-TestCase "Knowledge-Based Query"
    try {
        $startTime = Get-Date
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat/query" `
                                      -Method POST `
                                      -Body '{"query":"Tell me about the company policies","session_id":"test-knowledge"}' `
                                      -ContentType "application/json" `
                                      -TimeoutSec 120
        $elapsed = ((Get-Date) - $startTime).TotalSeconds
        
        if ($response.response) {
            Write-Pass "Knowledge query completed ($([math]::Round($elapsed,2))s)"
            Write-Info "Sources found: $($response.sources.Count)"
            Write-Info "Response length: $($response.response.Length) chars"
            
            if ($response.sources.Count -gt 0) {
                Write-Info "Retrieved relevant documents ✓"
            }
            Record-TestResult -Passed $true
        } else {
            Write-Fail "Knowledge query returned empty"
            Record-TestResult -Passed $false
        }
    } catch {
        Write-Fail "Knowledge query failed: $($_.Exception.Message)"
        Record-TestResult -Passed $false
    }
}

# ============================================================================
# SECTION 6: PERFORMANCE TESTS
# ============================================================================
Write-TestHeader "SECTION 6: PERFORMANCE TESTS"

Write-TestCase "Response Time - First Query (No Cache)"
try {
    $sessionId = "perf-test-$(Get-Random)"
    $startTime = Get-Date
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat/query" `
                                  -Method POST `
                                  -Body "{`"query`":`"Hello, how are you?`",`"session_id`":`"$sessionId`"}" `
                                  -ContentType "application/json" `
                                  -TimeoutSec 120
    $elapsed = ((Get-Date) - $startTime).TotalSeconds
    
    if ($elapsed -lt 30) {
        Write-Pass "First query response time acceptable: $([math]::Round($elapsed,2))s"
        Record-TestResult -Passed $true
    } elseif ($elapsed -lt 60) {
        Write-Pass "First query completed but slow: $([math]::Round($elapsed,2))s"
        Write-Info "Acceptable for first query with warmup"
        Record-TestResult -Passed $true
    } else {
        Write-Fail "First query too slow: $([math]::Round($elapsed,2))s"
        Record-TestResult -Passed $false
    }
} catch {
    Write-Fail "Performance test failed: $($_.Exception.Message)"
    Record-TestResult -Passed $false
}

if (-not $Quick) {
    Write-TestCase "Response Time - Subsequent Query (With Cache)"
    try {
        $startTime = Get-Date
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat/query" `
                                      -Method POST `
                                      -Body '{"query":"What is your name?","session_id":"perf-test-2"}' `
                                      -ContentType "application/json" `
                                      -TimeoutSec 60
        $elapsed = ((Get-Date) - $startTime).TotalSeconds
        
        if ($elapsed -lt 15) {
            Write-Pass "Subsequent query fast: $([math]::Round($elapsed,2))s"
            Record-TestResult -Passed $true
        } elseif ($elapsed -lt 30) {
            Write-Pass "Subsequent query acceptable: $([math]::Round($elapsed,2))s"
            Record-TestResult -Passed $true
        } else {
            Write-Fail "Subsequent query slow: $([math]::Round($elapsed,2))s"
            Record-TestResult -Passed $false
        }
    } catch {
        Write-Fail "Cached query test failed: $($_.Exception.Message)"
        Record-TestResult -Passed $false
    }
}

# ============================================================================
# SECTION 7: ERROR HANDLING
# ============================================================================
Write-TestHeader "SECTION 7: ERROR HANDLING"

Write-TestCase "Invalid Request Handling"
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat/query" `
                                  -Method POST `
                                  -Body '{"invalid":"data"}' `
                                  -ContentType "application/json" `
                                  -TimeoutSec 10
    Write-Fail "Should have rejected invalid request"
    Record-TestResult -Passed $false
} catch {
    if ($_.Exception.Message -like "*422*" -or $_.Exception.Message -like "*400*") {
        Write-Pass "Correctly rejected invalid request"
        Record-TestResult -Passed $true
    } else {
        Write-Fail "Unexpected error: $($_.Exception.Message)"
        Record-TestResult -Passed $false
    }
}

Write-TestCase "Empty Query Handling"
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat/query" `
                                  -Method POST `
                                  -Body '{"query":"","session_id":"empty-test"}' `
                                  -ContentType "application/json" `
                                  -TimeoutSec 10
    
    if ($response.response) {
        Write-Pass "Handled empty query gracefully"
        Record-TestResult -Passed $true
    } else {
        Write-Fail "Did not handle empty query"
        Record-TestResult -Passed $false
    }
} catch {
    if ($_.Exception.Message -like "*422*") {
        Write-Pass "Correctly rejected empty query"
        Record-TestResult -Passed $true
    } else {
        Write-Fail "Unexpected error: $($_.Exception.Message)"
        Record-TestResult -Passed $false
    }
}

# ============================================================================
# FINAL REPORT
# ============================================================================
Write-TestHeader "TEST SUMMARY"

Write-Host ""
Write-Host "  TOTAL TESTS:    " -NoNewline -ForegroundColor White
Write-Host "$script:totalTests" -ForegroundColor Cyan
Write-Host "  PASSED:         " -NoNewline -ForegroundColor White
Write-Host "$script:passedTests" -ForegroundColor Green
Write-Host "  FAILED:         " -NoNewline -ForegroundColor White
Write-Host "$script:failedTests" -ForegroundColor Red
if ($script:skippedTests -gt 0) {
    Write-Host "  SKIPPED:        " -NoNewline -ForegroundColor White
    Write-Host "$script:skippedTests" -ForegroundColor Yellow
}
Write-Host ""

$successRate = if ($script:totalTests -gt 0) { 
    [math]::Round(($script:passedTests / $script:totalTests) * 100, 1) 
} else { 
    0 
}

Write-Host "  SUCCESS RATE:   " -NoNewline -ForegroundColor White
if ($successRate -ge 90) {
    Write-Host "$successRate%" -ForegroundColor Green
} elseif ($successRate -ge 70) {
    Write-Host "$successRate%" -ForegroundColor Yellow
} else {
    Write-Host "$successRate%" -ForegroundColor Red
}
Write-Host ""

if ($script:failedTests -eq 0) {
    Write-Host "  🎉 " -NoNewline -ForegroundColor Green
    Write-Host "ALL TESTS PASSED! SYSTEM IS FULLY FUNCTIONAL!" -ForegroundColor Green
} elseif ($script:failedTests -le 2) {
    Write-Host "  ✅ " -NoNewline -ForegroundColor Green
    Write-Host "SYSTEM IS MOSTLY FUNCTIONAL (MINOR ISSUES)" -ForegroundColor Yellow
} else {
    Write-Host "  ⚠️  " -NoNewline -ForegroundColor Yellow
    Write-Host "SYSTEM HAS ISSUES THAT NEED ATTENTION" -ForegroundColor Red
}
Write-Host ""

Write-Host "=" -NoNewline -ForegroundColor Cyan
for($i=0; $i -lt 70; $i++) { Write-Host "=" -NoNewline -ForegroundColor Cyan }
Write-Host ""
Write-Host ""

# Exit code
exit $script:failedTests
