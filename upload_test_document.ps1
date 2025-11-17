# Upload Test Document Script
# Uploads test_document.txt to the backend for RAG testing

$baseUrl = "http://127.0.0.1:8000"
$filePath = "d:\OMS Chat Bot\test_document.txt"

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "TEST DOCUMENT UPLOAD" -ForegroundColor Cyan
Write-Host "=========================================`n" -ForegroundColor Cyan

# Check if file exists
if (-not (Test-Path $filePath)) {
    Write-Host "ERROR: Test document not found at $filePath" -ForegroundColor Red
    exit 1
}

$fileContent = Get-Content $filePath -Raw
$fileBytes = [System.Text.Encoding]::UTF8.GetBytes($fileContent)
$fileSize = $fileBytes.Length

Write-Host "File: test_document.txt" -ForegroundColor Yellow
Write-Host "Size: $fileSize bytes" -ForegroundColor Yellow
Write-Host "Content preview:" -ForegroundColor Yellow
Write-Host ($fileContent.Substring(0, [Math]::Min(200, $fileContent.Length))) -ForegroundColor Gray
Write-Host "...(truncated)`n" -ForegroundColor Gray

# Create multipart form data
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

$bodyLines = @(
    "--$boundary",
    "Content-Disposition: form-data; name=`"file`"; filename=`"test_document.txt`"",
    "Content-Type: text/plain",
    "",
    $fileContent,
    "--$boundary--"
)

$body = $bodyLines -join $LF

Write-Host "Uploading to $baseUrl/admin/documents/upload..." -ForegroundColor Cyan

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/admin/documents/upload" `
        -Method POST `
        -ContentType "multipart/form-data; boundary=$boundary" `
        -Body $body `
        -TimeoutSec 60
    
    Write-Host "`nSUCCESS!" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Cyan
    Write-Host ($response | ConvertTo-Json -Depth 3) -ForegroundColor White
    
    # Extract doc_id from response
    $docId = $response.doc_id
    if ($docId) {
        Write-Host "`nDocument ID: $docId" -ForegroundColor Yellow
        
        # Wait for processing to start
        Write-Host "`nWaiting 5 seconds for processing to start..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
        
        # Check document status
        Write-Host "`nChecking document status..." -ForegroundColor Cyan
        try {
            $status = Invoke-RestMethod -Uri "$baseUrl/admin/documents/status/$docId" `
                -Method GET `
                -TimeoutSec 10
            
            Write-Host "`nDocument Status:" -ForegroundColor Cyan
            Write-Host "  Status: $($status.ingest_status)" -ForegroundColor $(
                if ($status.ingest_status -eq "SUCCESS") { "Green" }
                elseif ($status.ingest_status -eq "PROCESSING") { "Yellow" }
                else { "Red" }
            )
            Write-Host "  Filename: $($status.filename)" -ForegroundColor White
            
            if ($status.stages) {
                Write-Host "`nProcessing Stages:" -ForegroundColor Cyan
                foreach ($stage in $status.stages_status) {
                    $color = if ($stage.status -eq "SUCCESS") { "Green" } 
                             elseif ($stage.status -eq "PROCESSING") { "Yellow" }
                             else { "Red" }
                    Write-Host "  $($stage.stage): $($stage.status)" -ForegroundColor $color
                }
            }
            
        } catch {
            Write-Host "Could not fetch status: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        
        # Now test a query
        Write-Host "`n`n=========================================" -ForegroundColor Cyan
        Write-Host "TESTING RAG QUERY WITH UPLOADED DOCUMENT" -ForegroundColor Cyan
        Write-Host "=========================================`n" -ForegroundColor Cyan
        
        Write-Host "Waiting 10 more seconds for indexing to complete..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        
        Write-Host "`nSending test query: 'What is RAG?'" -ForegroundColor Cyan
        
        try {
            $chatResponse = Invoke-RestMethod -Uri "$baseUrl/chat/query" `
                -Method POST `
                -Body (@{
                    query = "What is RAG?"
                    session_id = "test-upload-session"
                    stream = $false
                } | ConvertTo-Json) `
                -ContentType "application/json" `
                -TimeoutSec 120
            
            Write-Host "`nQUERY SUCCESS!" -ForegroundColor Green
            Write-Host "`nResponse:" -ForegroundColor Cyan
            Write-Host $chatResponse.response -ForegroundColor White
            
            Write-Host "`nSources Found: $($chatResponse.sources.Count)" -ForegroundColor Cyan
            if ($chatResponse.sources.Count -gt 0) {
                Write-Host "Source Details:" -ForegroundColor Cyan
                $chatResponse.sources | ForEach-Object {
                    Write-Host "  - $($_.filename) (score: $($_.score))" -ForegroundColor Gray
                }
            }
            
            Write-Host "`nProcessing Time: $([math]::Round($chatResponse.processing_time, 2))s" -ForegroundColor Cyan
            Write-Host "Tokens Generated: $($chatResponse.tokens_generated)" -ForegroundColor Cyan
            
        } catch {
            Write-Host "`nQuery failed: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "This might be due to timeout (expected for first query)" -ForegroundColor Yellow
        }
    }
    
    Write-Host "`n=========================================" -ForegroundColor Cyan
    Write-Host "DOCUMENT UPLOAD AND TESTING COMPLETE" -ForegroundColor Cyan
    Write-Host "=========================================`n" -ForegroundColor Cyan
    
} catch {
    Write-Host "`nERROR: Upload failed" -ForegroundColor Red
    Write-Host "Message: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check if backend is running on http://127.0.0.1:8000" -ForegroundColor Gray
    Write-Host "2. Verify MongoDB, Qdrant, and Redis connections" -ForegroundColor Gray
    Write-Host "3. Check backend logs for errors" -ForegroundColor Gray
    exit 1
}
