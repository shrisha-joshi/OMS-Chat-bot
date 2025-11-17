# REST API Endpoint Testing Script
# Tests all OMS Chat Bot backend endpoints

$baseUrl = "http://127.0.0.1:8000"
$results = @()

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "REST API ENDPOINT TESTING" -ForegroundColor Cyan
Write-Host "=========================================`n" -ForegroundColor Cyan

function Test-API {
    param([string]$Name, [string]$Method, [string]$Url, $Body)
    Write-Host "Testing: $Name" -ForegroundColor Yellow
    try {
        if ($Body) {
            $response = Invoke-RestMethod -Uri $Url -Method $Method -Body ($Body | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30
        } else {
            $response = Invoke-RestMethod -Uri $Url -Method $Method -TimeoutSec 30
        }
        Write-Host "  PASS" -ForegroundColor Green
        $script:results += @{Test=$Name; Status="PASS"}
        return $response
    } catch {
        Write-Host "  FAIL: $($_.Exception.Message)" -ForegroundColor Red
        $script:results += @{Test=$Name; Status="FAIL"}
        return $null
    }
}

# 1. HEALTH ENDPOINTS
Write-Host "`n1. HEALTH & SYSTEM ENDPOINTS" -ForegroundColor Magenta
Write-Host "----------------------------" -ForegroundColor Magenta
Test-API "Health Check" "GET" "$baseUrl/health"
Test-API "Readiness Check" "GET" "$baseUrl/ready"
Test-API "Root Endpoint" "GET" "$baseUrl/"
Test-API "System Info" "GET" "$baseUrl/system/info"

# 2. CHAT ENDPOINTS  
Write-Host "`n2. CHAT ENDPOINTS" -ForegroundColor Magenta
Write-Host "----------------------------" -ForegroundColor Magenta
Test-API "Test LLM" "POST" "$baseUrl/chat/test-llm" @{query="Hello"}
$chatResp = Test-API "Chat Query" "POST" "$baseUrl/chat/query" @{query="What is RAG?"; session_id="test-001"; stream=$false}
if ($chatResp) {
    Test-API "Session History" "GET" "$baseUrl/chat/history/$($chatResp.session_id)"
}
Test-API "Chat Health" "GET" "$baseUrl/chat/health"

# 3. ADMIN ENDPOINTS
Write-Host "`n3. ADMIN ENDPOINTS" -ForegroundColor Magenta
Write-Host "----------------------------" -ForegroundColor Magenta
$docs = Test-API "List Documents" "GET" "$baseUrl/admin/documents/list"
if ($docs -and $docs.documents -and $docs.documents.Count -gt 0) {
    $docId = $docs.documents[0]._id
    Test-API "Document Status" "GET" "$baseUrl/admin/documents/status/$docId"
}

# 4. FEEDBACK ENDPOINTS
Write-Host "`n4. FEEDBACK ENDPOINTS" -ForegroundColor Magenta
Write-Host "----------------------------" -ForegroundColor Magenta
Test-API "Submit Feedback" "POST" "$baseUrl/feedback/submit" @{
    session_id="test-001"
    query="Test"
    response="Test response"
    rating="helpful"
}

# SUMMARY
Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "TEST SUMMARY" -ForegroundColor Cyan
Write-Host "=========================================`n" -ForegroundColor Cyan

$passed = ($results | Where-Object {$_.Status -eq "PASS"}).Count
$failed = ($results | Where-Object {$_.Status -eq "FAIL"}).Count
$total = $results.Count

Write-Host "Total Tests: $total" -ForegroundColor White
Write-Host "Passed: $passed" -ForegroundColor Green
Write-Host "Failed: $failed" -ForegroundColor $(if($failed -gt 0){"Red"}else{"Green"})

if ($failed -eq 0) {
    Write-Host "`nALL TESTS PASSED!" -ForegroundColor Green
} else {
    Write-Host "`nSome tests failed." -ForegroundColor Yellow
}

Write-Host "`n=========================================`n" -ForegroundColor Cyan
