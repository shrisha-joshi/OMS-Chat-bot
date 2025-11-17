# ============================================================================
# COMPLETE REST API ENDPOINT TESTING SCRIPT
# Tests all backend endpoints with comprehensive validation
# ============================================================================

$baseUrl = "http://127.0.0.1:8000"
$testResults = @()

function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Method,
        [string]$Endpoint,
        [hashtable]$Body = $null,
        [int]$ExpectedStatus = 200
    )
    
    Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "Testing: $Name" -ForegroundColor Yellow
    Write-Host "Method: $Method | Endpoint: $Endpoint" -ForegroundColor Gray
    
    try {
        $params = @{
            Uri = "$baseUrl$Endpoint"
            Method = $Method
            TimeoutSec = 30
            ContentType = "application/json"
        }
        
        if ($Body) {
            $params.Body = ($Body | ConvertTo-Json -Depth 10)
            Write-Host "Body: $($params.Body)" -ForegroundColor Gray
        }
        
        $response = Invoke-RestMethod @params
        
        Write-Host "✅ SUCCESS" -ForegroundColor Green
        Write-Host "Response:" -ForegroundColor Cyan
        Write-Host ($response | ConvertTo-Json -Depth 3) -ForegroundColor White
        
        $testResults += [PSCustomObject]@{
            Test = $Name
            Status = "PASS"
            StatusCode = 200
            Error = $null
        }
        
        return $response
        
    } catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "❌ FAILED" -ForegroundColor Red
        Write-Host "Status Code: $statusCode" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        
        $testResults += [PSCustomObject]@{
            Test = $Name
            Status = "FAIL"
            StatusCode = $statusCode
            Error = $_.Exception.Message
        }
        
        return $null
    }
}

# ============================================================================
# START TESTING
# ============================================================================

Write-Host "`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║        COMPREHENSIVE REST API ENDPOINT TESTING               ║" -ForegroundColor Cyan
Write-Host "║        OMS Chat Bot - Backend API Verification               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

# ============================================================================
# 1. HEALTH & SYSTEM ENDPOINTS
# ============================================================================

Write-Host "`n`n┌─────────────────────────────────────────────────────────┐" -ForegroundColor Magenta
Write-Host "│  1. HEALTH & SYSTEM ENDPOINTS                           │" -ForegroundColor Magenta
Write-Host "└─────────────────────────────────────────────────────────┘" -ForegroundColor Magenta

$health = Test-Endpoint -Name "Health Check" -Method "GET" -Endpoint "/health"
Test-Endpoint -Name "Readiness Check" -Method "GET" -Endpoint "/ready"
Test-Endpoint -Name "Root Endpoint" -Method "GET" -Endpoint "/"
Test-Endpoint -Name "System Info" -Method "GET" -Endpoint "/system/info"

# ============================================================================
# 2. CHAT ENDPOINTS
# ============================================================================

Write-Host "`n`n┌─────────────────────────────────────────────────────────┐" -ForegroundColor Magenta
Write-Host "│  2. CHAT ENDPOINTS                                      │" -ForegroundColor Magenta
Write-Host "└─────────────────────────────────────────────────────────┘" -ForegroundColor Magenta

# Test LLM directly
Test-Endpoint -Name "Test LLM Direct" -Method "POST" -Endpoint "/chat/test-llm" `
    -Body @{ query = "Hello, can you hear me?" }

# Test chat query
$chatResponse = Test-Endpoint -Name "Chat Query" -Method "POST" -Endpoint "/chat/query" `
    -Body @{
        query = "What is RAG?"
        session_id = "test-session-001"
        stream = $false
    }

if ($chatResponse) {
    $sessionId = $chatResponse.session_id
    
    # Test session history
    Test-Endpoint -Name "Get Session History" -Method "GET" -Endpoint "/chat/history/$sessionId"
    
    # Test chat health
    Test-Endpoint -Name "Chat Service Health" -Method "GET" -Endpoint "/chat/health"
}

# Test query suggestions
Test-Endpoint -Name "Query Suggestions" -Method "GET" -Endpoint "/chat/suggestions?query=what+is`&limit=5"

# ============================================================================
# 3. ADMIN ENDPOINTS
# ============================================================================

Write-Host "`n`n┌─────────────────────────────────────────────────────────┐" -ForegroundColor Magenta
Write-Host "│  3. ADMIN ENDPOINTS                                     │" -ForegroundColor Magenta
Write-Host "└─────────────────────────────────────────────────────────┘" -ForegroundColor Magenta

# List documents
$docsResponse = Test-Endpoint -Name "List Documents" -Method "GET" -Endpoint "/admin/documents/list?skip=0`&limit=10"

if ($docsResponse -and $docsResponse.documents -and $docsResponse.documents.Count -gt 0) {
    $docId = $docsResponse.documents[0]._id
    
    # Get document status
    Test-Endpoint -Name "Get Document Status" -Method "GET" -Endpoint "/admin/documents/status/$docId"
}

# ============================================================================
# 4. MONITORING ENDPOINTS
# ============================================================================

Write-Host "`n`n┌─────────────────────────────────────────────────────────┐" -ForegroundColor Magenta
Write-Host "│  4. MONITORING ENDPOINTS                                │" -ForegroundColor Magenta
Write-Host "└─────────────────────────────────────────────────────────┘" -ForegroundColor Magenta

# Note: WebSocket endpoints need special testing, skipping for REST API test

# ============================================================================
# 5. FEEDBACK ENDPOINTS
# ============================================================================

Write-Host "`n`n┌─────────────────────────────────────────────────────────┐" -ForegroundColor Magenta
Write-Host "│  5. FEEDBACK ENDPOINTS                                  │" -ForegroundColor Magenta
Write-Host "└─────────────────────────────────────────────────────────┘" -ForegroundColor Magenta

# Submit feedback
Test-Endpoint -Name "Submit Feedback" -Method "POST" -Endpoint "/feedback/submit" `
    -Body @{
        session_id = "test-session-001"
        query = "What is RAG?"
        response = "RAG stands for Retrieval-Augmented Generation"
        rating = "helpful"
        correction = $null
    }

# ============================================================================
# TEST SUMMARY
# ============================================================================

Write-Host "`n`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    TEST RESULTS SUMMARY                       ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green

$passCount = ($testResults | Where-Object { $_.Status -eq "PASS" }).Count
$failCount = ($testResults | Where-Object { $_.Status -eq "FAIL" }).Count
$totalCount = $testResults.Count

Write-Host "`nTotal Tests: $totalCount" -ForegroundColor Cyan
Write-Host "Passed: $passCount" -ForegroundColor Green
Write-Host "Failed: $failCount" -ForegroundColor $(if($failCount -gt 0){"Red"}else{"Green"})
Write-Host "Success Rate: $([math]::Round(($passCount/$totalCount)*100, 2))%" -ForegroundColor Cyan

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "Detailed Results:" -ForegroundColor Yellow
$testResults | Format-Table -AutoSize

if ($failCount -eq 0) {
    Write-Host "`n🎉 ALL TESTS PASSED! ALL ENDPOINTS ARE WORKING PERFECTLY! 🎉" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Some tests failed. Review the details above." -ForegroundColor Yellow
}

Write-Host "`n════════════════════════════════════════════════════════" -ForegroundColor Cyan
