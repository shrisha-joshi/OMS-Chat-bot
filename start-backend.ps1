# Start Backend Server for OMS Chatbot
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting OMS Chatbot Backend Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to backend directory
Set-Location "d:\OMS Chat Bot\backend"

Write-Host "Starting FastAPI server on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "API Documentation: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the server directly using our custom runner
python run_server.py

Read-Host "`nPress Enter to close"
