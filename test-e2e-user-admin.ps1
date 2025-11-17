# End-to-End Testing Script - User & Admin Stories
# Tests complete workflows for both user and admin personas

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "     END-TO-END TESTING: USER & ADMIN STORIES" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

$BASE_URL = "http://127.0.0.1:8000"
$results = @()
$issuesFound = @()
$passCount = 0
$totalTests = 0

# Helper function to test endpoint
function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Method,
        [string]$Url,
        [hashtable]$Headers = @{},
        [object]$Body = $null
    )
    
    $global:totalTests++
    Write-Host "Testing: $Name..." -ForegroundColor Yellow
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            Headers = $Headers
            TimeoutSec = 10
        }
        
        if ($Body) {
            $params['Body'] = ($Body | ConvertTo-Json -Depth 10)
            $params['ContentType'] = 'application/json'
        }
        
        $response = Invoke-RestMethod @params
        Write-Host "  PASS: $Name" -ForegroundColor Green
        $global:results += "${Name}: PASS"
        $global:passCount++
        return $response
    } catch {
        Write-Host "  FAIL: $Name - $($_.Exception.Message)" -ForegroundColor Red
        $global:results += "${Name}: FAIL"
        $global:issuesFound += @{
            Test = $Name
            Error = $_.Exception.Message
            StatusCode = $_.Exception.Response.StatusCode.value__
        }
        return $null
    }
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " PHASE 1: SYSTEM HEALTH & CONNECTIVITY" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

# Test 1: Backend Health
$health = Test-Endpoint -Name "Backend Health" -Method "GET" -Url "$BASE_URL/health"
if ($health) {
    Write-Host "  Databases: MongoDB=$($health.databases.mongodb), Qdrant=$($health.databases.qdrant), Redis=$($health.databases.redis)" -ForegroundColor Cyan
}

# Test 2: API Documentation
Test-Endpoint -Name "API Documentation" -Method "GET" -Url "$BASE_URL/docs" | Out-Null

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host " PHASE 2: AUTHENTICATION & USER MANAGEMENT" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

# Test 3: Admin Login (using hardcoded credentials)
$adminLogin = @{
    username = "admin"
    password = "admin123"
}

$adminAuth = Test-Endpoint -Name "Admin Login" -Method "POST" -Url "$BASE_URL/auth/token" -Body $adminLogin
$adminToken = $null
if ($adminAuth -and $adminAuth.access_token) {
    $adminToken = $adminAuth.access_token
    Write-Host "  Admin Token Generated: ${adminToken.Substring(0, 20)}..." -ForegroundColor Cyan
    Write-Host "  Role: $($adminAuth.user.role)" -ForegroundColor Cyan
}

# Test 4: Get Current User Info
if ($adminToken) {
    $authHeaders = @{
        "Authorization" = "Bearer $adminToken"
    }
    $currentUser = Test-Endpoint -Name "Get Current User" -Method "GET" -Url "$BASE_URL/auth/me" -Headers $authHeaders
    if ($currentUser) {
        Write-Host "  Username: $($currentUser.username), Role: $($currentUser.role)" -ForegroundColor Cyan
    }
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host " PHASE 3: ADMIN STORY - DOCUMENT MANAGEMENT" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

if ($adminToken) {
    $authHeaders = @{
        "Authorization" = "Bearer $adminToken"
    }
    
    # Test 5: List Documents (should show existing documents)
    $docs = Test-Endpoint -Name "List Documents" -Method "GET" -Url "$BASE_URL/admin/documents/list" -Headers $authHeaders
    if ($docs) {
        Write-Host "  Documents in system: $($docs.documents.Count)" -ForegroundColor Cyan
    }
    
    # Test 6: Upload Document (simulated with JSON)
    Write-Host "Testing: Document Upload..." -ForegroundColor Yellow
    $testDoc = @{
        filename = "test_policy_$(Get-Random -Maximum 999).txt"
        content_base64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("This is a test company policy document about employee benefits and vacation time. Employees are entitled to 15 days of paid vacation per year. Health insurance is provided for all full-time employees."))
        content_type = "text/plain"
    }
    
    try {
        $uploadResponse = Invoke-RestMethod -Uri "$BASE_URL/admin/documents/upload-json" `
            -Method POST `
            -Headers $authHeaders `
            -Body ($testDoc | ConvertTo-Json) `
            -ContentType "application/json" `
            -TimeoutSec 30
        
        if ($uploadResponse) {
            Write-Host "  PASS: Document Upload" -ForegroundColor Green
            Write-Host "  Uploaded: $($testDoc.filename)" -ForegroundColor Cyan
            $global:results += "Document Upload: PASS"
            $global:passCount++
            $global:totalTests++
            
            # Save document ID for later tests
            $script:uploadedDocId = $uploadResponse.document_id
        }
    } catch {
        Write-Host "  FAIL: Document Upload - $($_.Exception.Message)" -ForegroundColor Red
        $global:results += "Document Upload: FAIL"
        $global:totalTests++
        $global:issuesFound += @{
            Test = "Document Upload"
            Error = $_.Exception.Message
        }
    }
    
    # Wait for processing
    Write-Host "`n  Waiting 5 seconds for document processing..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # Test 7: List Documents (should have new document)
    $docsAfter = Test-Endpoint -Name "List Documents (After Upload)" -Method "GET" -Url "$BASE_URL/admin/documents/list" -Headers $authHeaders
    if ($docsAfter) {
        Write-Host "  Documents in system: $($docsAfter.documents.Count)" -ForegroundColor Cyan
    }
    
    # Test 8: Get Document Status
    if ($script:uploadedDocId) {
        $docDetails = Test-Endpoint -Name "Get Document Status" -Method "GET" -Url "$BASE_URL/admin/documents/status/$($script:uploadedDocId)" -Headers $authHeaders
        if ($docDetails) {
            Write-Host "  Status: $($docDetails.status), Chunks: $($docDetails.chunks_count)" -ForegroundColor Cyan
        }
    }
} else {
    Write-Host "  SKIPPED: Admin tests (no admin token)" -ForegroundColor Yellow
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host " PHASE 4: USER STORY - RAG CHAT & QUERY" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

if ($adminToken) {
    $userHeaders = @{
        "Authorization" = "Bearer $adminToken"
    }
    
    # Test 9: Chat Query #1 - Simple question
    Write-Host "Testing: RAG Chat Query (Simple Question)..." -ForegroundColor Yellow
    $chatQuery1 = @{
        query = "What is the company's vacation policy?"
        session_id = "test_session_$(Get-Random -Maximum 9999)"
        use_graph_search = $false
        max_context_tokens = 2000
    }
    
    try {
        $chatResponse1 = Invoke-RestMethod -Uri "$BASE_URL/chat/query" `
            -Method POST `
            -Headers $userHeaders `
            -Body ($chatQuery1 | ConvertTo-Json) `
            -ContentType "application/json" `
            -TimeoutSec 60
        
        if ($chatResponse1) {
            Write-Host "  PASS: RAG Chat Query (Simple)" -ForegroundColor Green
            Write-Host "  Response: $($chatResponse1.response.Substring(0, [Math]::Min(150, $chatResponse1.response.Length)))..." -ForegroundColor Cyan
            Write-Host "  Sources: $($chatResponse1.sources.Count), Tokens: $($chatResponse1.tokens_generated)" -ForegroundColor Cyan
            $global:results += "RAG Chat Simple: PASS"
            $global:passCount++
            $global:totalTests++
            $script:chatResponse = $chatResponse1
        }
    } catch {
        Write-Host "  FAIL: RAG Chat Query - $($_.Exception.Message)" -ForegroundColor Red
        $global:results += "RAG Chat Simple: FAIL"
        $global:totalTests++
        $global:issuesFound += @{
            Test = "RAG Chat Simple"
            Error = $_.Exception.Message
        }
    }
    
    # Test 10: Chat Query #2 - Follow-up question
    if ($chatResponse1) {
        Write-Host "`nTesting: RAG Chat Query (Follow-up)..." -ForegroundColor Yellow
        $chatQuery2 = @{
            query = "How many days specifically?"
            session_id = $chatQuery1.session_id
            use_graph_search = $false
            max_context_tokens = 2000
        }
        
        try {
            $chatResponse2 = Invoke-RestMethod -Uri "$BASE_URL/chat/query" `
                -Method POST `
                -Headers $userHeaders `
                -Body ($chatQuery2 | ConvertTo-Json) `
                -ContentType "application/json" `
                -TimeoutSec 60
            
            if ($chatResponse2) {
                Write-Host "  PASS: RAG Chat Query (Follow-up)" -ForegroundColor Green
                Write-Host "  Response: $($chatResponse2.response.Substring(0, [Math]::Min(100, $chatResponse2.response.Length)))..." -ForegroundColor Cyan
                $global:results += "RAG Chat Follow-up: PASS"
                $global:passCount++
                $global:totalTests++
            }
        } catch {
            Write-Host "  FAIL: Follow-up Query - $($_.Exception.Message)" -ForegroundColor Red
            $global:results += "RAG Chat Follow-up: FAIL"
            $global:totalTests++
        }
    }
    
    # Test 11: Chat History
    $history = Test-Endpoint -Name "Chat History" -Method "GET" -Url "$BASE_URL/chat/history/$($chatQuery1.session_id)" -Headers $userHeaders
    if ($history) {
        Write-Host "  Messages in history: $($history.messages.Count)" -ForegroundColor Cyan
    }
    
    # Test 12: List Sessions
    $sessions = Test-Endpoint -Name "Chat Sessions List" -Method "GET" -Url "$BASE_URL/chat/sessions/list" -Headers $userHeaders
    if ($sessions) {
        Write-Host "  Total sessions: $($sessions.sessions.Count)" -ForegroundColor Cyan
    }
} else {
    Write-Host "  SKIPPED: Chat tests (no auth token)" -ForegroundColor Yellow
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host " PHASE 5: WEBSOCKET & MONITORING" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

# Test 13: WebSocket Metrics
$wsMetrics = Test-Endpoint -Name "WebSocket Metrics" -Method "GET" -Url "$BASE_URL/monitoring/websocket/metrics"
if ($wsMetrics) {
    Write-Host "  Active connections: $($wsMetrics.overall.active_connections)" -ForegroundColor Cyan
    Write-Host "  Total messages: $($wsMetrics.overall.total_messages_sent)" -ForegroundColor Cyan
}

# Test 14: System Info
$sysInfo = Test-Endpoint -Name "System Information" -Method "GET" -Url "$BASE_URL/system/info"
if ($sysInfo) {
    Write-Host "  Version: $($sysInfo.version), Uptime: $($sysInfo.uptime_seconds)s" -ForegroundColor Cyan
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host " PHASE 6: LLM INTEGRATION" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

# Test 15: LLM Health
Write-Host "Testing: LMStudio Connection..." -ForegroundColor Yellow
try {
    $lmHealth = Invoke-RestMethod -Uri "http://127.0.0.1:1234/v1/models" -TimeoutSec 5
    Write-Host "  PASS: LMStudio Connection" -ForegroundColor Green
    Write-Host "  Models loaded: $($lmHealth.data.Count)" -ForegroundColor Cyan
    $global:results += "LMStudio Connection: PASS"
    $global:passCount++
    $global:totalTests++
} catch {
    Write-Host "  FAIL: LMStudio Connection" -ForegroundColor Red
    $global:results += "LMStudio Connection: FAIL"
    $global:totalTests++
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "                    TEST SUMMARY" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

Write-Host "Overall Score: $passCount / $totalTests tests passed" -ForegroundColor $(if ($passCount -eq $totalTests) { "Green" } elseif ($passCount -ge ($totalTests * 0.8)) { "Yellow" } else { "Red" })

Write-Host "`nDetailed Results:" -ForegroundColor Cyan
foreach ($result in $results) {
    $status = if ($result -like "*PASS*") { "Green" } else { "Red" }
    Write-Host "  $result" -ForegroundColor $status
}

if ($issuesFound.Count -gt 0) {
    Write-Host "`n================================================================" -ForegroundColor Red
    Write-Host " ISSUES FOUND: $($issuesFound.Count)" -ForegroundColor Red
    Write-Host "================================================================`n" -ForegroundColor Red
    
    foreach ($issue in $issuesFound) {
        Write-Host "Test: $($issue.Test)" -ForegroundColor Yellow
        Write-Host "  Error: $($issue.Error)" -ForegroundColor Red
        if ($issue.StatusCode) {
            Write-Host "  Status Code: $($issue.StatusCode)" -ForegroundColor Red
        }
        Write-Host ""
    }
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host " USER STORY VERIFICATION" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

Write-Host "Admin Story Checklist:" -ForegroundColor Yellow
Write-Host "  [$(if ($adminAuth) {'X'} else {' '})] Login as admin" -ForegroundColor $(if ($adminAuth) {'Green'} else {'Red'})
Write-Host "  [$(if ($currentUser) {'X'} else {' '})] Verify user identity" -ForegroundColor $(if ($currentUser) {'Green'} else {'Red'})
Write-Host "  [$(if ($docs) {'X'} else {' '})] List documents" -ForegroundColor $(if ($docs) {'Green'} else {'Red'})
Write-Host "  [$(if ($uploadResponse) {'X'} else {' '})] Upload new document" -ForegroundColor $(if ($uploadResponse) {'Green'} else {'Red'})
Write-Host "  [$(if ($docDetails) {'X'} else {' '})] View document processing status" -ForegroundColor $(if ($docDetails) {'Green'} else {'Red'})
Write-Host "  [$(if ($wsMetrics) {'X'} else {' '})] Monitor system metrics" -ForegroundColor $(if ($wsMetrics) {'Green'} else {'Red'})

Write-Host "`nUser Story Checklist:" -ForegroundColor Yellow
Write-Host "  [$(if ($adminAuth) {'X'} else {' '})] Authenticated access" -ForegroundColor $(if ($adminAuth) {'Green'} else {'Red'})
Write-Host "  [$(if ($chatResponse1) {'X'} else {' '})] Query documents with RAG" -ForegroundColor $(if ($chatResponse1) {'Green'} else {'Red'})
Write-Host "  [$(if ($chatResponse2) {'X'} else {' '})] Follow-up questions (context)" -ForegroundColor $(if ($chatResponse2) {'Green'} else {'Red'})
Write-Host "  [$(if ($history) {'X'} else {' '})] View chat history" -ForegroundColor $(if ($history) {'Green'} else {'Red'})
Write-Host "  [$(if ($sessions) {'X'} else {' '})] List all sessions" -ForegroundColor $(if ($sessions) {'Green'} else {'Red'})

$adminComplete = $adminAuth -and $currentUser -and $docs -and $docDetails -and $wsMetrics
$userComplete = $adminAuth -and $chatResponse1 -and $chatResponse2 -and $history -and $sessions

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host " FINAL VERDICT" -ForegroundColor Cyan
Write-Host "================================================================`n" -ForegroundColor Cyan

if ($adminComplete -and $userComplete -and $passCount -ge ($totalTests * 0.9)) {
    Write-Host "SUCCESS! Both user and admin stories work correctly!" -ForegroundColor Green
    Write-Host "System is ready for production use." -ForegroundColor Green
} elseif ($passCount -ge ($totalTests * 0.7)) {
    Write-Host "PARTIAL SUCCESS! Most features working but some issues found." -ForegroundColor Yellow
    Write-Host "Review failed tests and fix issues before production." -ForegroundColor Yellow
} else {
    Write-Host "CRITICAL ISSUES! System has major problems." -ForegroundColor Red
    Write-Host "System NOT ready for production use." -ForegroundColor Red
}

Write-Host "`nFor detailed documentation, see:" -ForegroundColor Cyan
Write-Host "  - TESTING_PLAYBOOK.md" -ForegroundColor White
Write-Host "  - CONNECTION_ARCHITECTURE_FIXED.md" -ForegroundColor White
Write-Host "  - DEBUG_AND_FIX_GUIDE.md`n" -ForegroundColor White
