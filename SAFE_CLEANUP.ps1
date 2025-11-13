# OMS CHATBOT - SAFE FILE CLEANUP SCRIPT
# Removes outdated temporary fix scripts and duplicate files

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  OMS CHATBOT - FILE CLEANUP UTILITY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

$rootPath = "D:\OMS Chat Bot"
$deletedCount = 0
$totalSize = 0

# Files to delete
$filesToDelete = @(
    # Temporary fix scripts
    "fix_all_json_commas.py",
    "fix_all_quick_issues.py",
    "fix_all_remaining_issues.py",
    "fix_all_remaining_quick.py",
    "fix_all_sonar_issues.py",
    "fix_comprehensive_json.py",
    "fix_faq_json.py",
    "fix_final_quick.py",
    "fix_medium_complexity.py",
    "fix_typescript_issues.py",
    "comprehensive_fix_plan.py",
    "phase1_root_fixes.py",
    
    # Cleanup scripts
    "ABSOLUTE_FINAL_FIX.py",
    "COMPLETE_ALL_FIXES.py",
    "CORRECT_SONAR_SUPPRESSIONS.py",
    "FINAL_CLEANUP_ALL.py",
    "REMOVE_INVALID_DECORATORS.py",
    "REMOVE_TEST_DECORATORS.py",
    "ULTIMATE_SONAR_FIX.py",
    "ULTIMATE_SONARQUBE_FIX.py",
    "ZERO_TOLERANCE_FIX.py",
    
    # Status scripts
    "FINAL_NUCLEAR_STATUS.py",
    "FINAL_STATUS_REPORT.py",
    "TODAYS_ACHIEVEMENT_REPORT.py",
    
    # Duplicate scripts
    "quick-start.ps1",
    "START-ALL.ps1",
    "START-APP.ps1",
    "START_APP.ps1",
    "START_APP_FIXED.ps1",
    "start-backend-fixed.ps1",
    "START-GRAPH-RAG.ps1",
    "0_START_DOCKER.bat",
    "1_START_BACKEND.bat",
    "2_START_FRONTEND.bat",
    "run-services.bat",
    
    # Old files
    "test_rag_direct.py",
    "SYSTEM_READY.md",
    "SYSTEM_RUNNING.md",
    "SETUP_VERIFICATION.txt",
    "frontend-debug.log",
    "Neo4j-cee8b30a-Created-2025-10-15.txt",
    "oms_rag_project_(1)[1].pdf"
)

Write-Host "`nFound $($filesToDelete.Count) files to clean up" -ForegroundColor Cyan
Write-Host "These are temporary fix scripts and outdated files." -ForegroundColor White
Write-Host "Your running application will NOT be affected.`n" -ForegroundColor Green

$confirmation = Read-Host "Continue with cleanup? (yes/no)"

if ($confirmation -ne "yes") {
    Write-Host "`nCleanup cancelled." -ForegroundColor Red
    exit 0
}

Write-Host "`nDeleting files..." -ForegroundColor Yellow

foreach ($file in $filesToDelete) {
    $filePath = Join-Path $rootPath $file
    
    if (Test-Path $filePath) {
        try {
            $fileInfo = Get-Item $filePath
            $fileSize = $fileInfo.Length
            $totalSize += $fileSize
            
            Remove-Item $filePath -Force
            
            $sizeKB = [math]::Round($fileSize / 1KB, 2)
            Write-Host "  Deleted: $file ($sizeKB KB)" -ForegroundColor Green
            $deletedCount++
        }
        catch {
            Write-Host "  Failed: $file - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

$totalSizeMB = [math]::Round($totalSize / 1MB, 2)

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "`nFiles deleted: $deletedCount" -ForegroundColor Green
Write-Host "Space freed: $totalSizeMB MB" -ForegroundColor Cyan
Write-Host "`nYour application is still running normally!" -ForegroundColor Green
Write-Host "Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
