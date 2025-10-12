# ============================================================
#  Windows Setup Script for RAG + Graph Chatbot
#  Description: Automates setup for backend, frontend, and services on Windows
#  Run as Administrator in PowerShell
# ============================================================

param(
    [switch]$SkipDatabases,
    [switch]$DevMode
)

Write-Host "🚀 Starting Windows setup for RAG + Graph Chatbot..." -ForegroundColor Green

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "❌ This script must be run as Administrator. Please run PowerShell as Administrator." -ForegroundColor Red
    exit 1
}

# Function to check if a command exists
function Test-Command {
    param($Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Function to download and extract files
function Download-And-Extract {
    param($Url, $OutputPath, $ExtractPath)
    try {
        Write-Host "📦 Downloading from $Url..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri $Url -OutFile $OutputPath -UseBasicParsing
        if ($OutputPath.EndsWith('.zip')) {
            Expand-Archive -Path $OutputPath -DestinationPath $ExtractPath -Force
        }
        return $true
    }
    catch {
        Write-Host "❌ Failed to download: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# -------------------------------
# 1. SYSTEM REQUIREMENTS CHECK
# -------------------------------
Write-Host "🔍 Checking system requirements..." -ForegroundColor Cyan

# Check Python
if (-not (Test-Command python)) {
    Write-Host "📦 Installing Python..." -ForegroundColor Yellow
    $pythonUrl = "https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe"
    $pythonInstaller = "$env:TEMP\python-installer.exe"
    
    if (Download-And-Extract -Url $pythonUrl -OutputPath $pythonInstaller) {
        Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait
        Write-Host "✅ Python installed successfully" -ForegroundColor Green
        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
    }
} else {
    Write-Host "✅ Python is already installed" -ForegroundColor Green
}

# Check Node.js
if (-not (Test-Command node)) {
    Write-Host "📦 Installing Node.js..." -ForegroundColor Yellow
    $nodeUrl = "https://nodejs.org/dist/v20.11.0/node-v20.11.0-x64.msi"
    $nodeInstaller = "$env:TEMP\node-installer.msi"
    
    if (Download-And-Extract -Url $nodeUrl -OutputPath $nodeInstaller) {
        Start-Process -FilePath "msiexec.exe" -ArgumentList "/i `"$nodeInstaller`" /quiet" -Wait
        Write-Host "✅ Node.js installed successfully" -ForegroundColor Green
        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
    }
} else {
    Write-Host "✅ Node.js is already installed" -ForegroundColor Green
}

# Check Git
if (-not (Test-Command git)) {
    Write-Host "📦 Installing Git..." -ForegroundColor Yellow
    $gitUrl = "https://github.com/git-for-windows/git/releases/latest/download/Git-2.44.0-64-bit.exe"
    $gitInstaller = "$env:TEMP\git-installer.exe"
    
    if (Download-And-Extract -Url $gitUrl -OutputPath $gitInstaller) {
        Start-Process -FilePath $gitInstaller -ArgumentList "/SILENT" -Wait
        Write-Host "✅ Git installed successfully" -ForegroundColor Green
    }
} else {
    Write-Host "✅ Git is already installed" -ForegroundColor Green
}

# -------------------------------
# 2. PROJECT DEPENDENCIES
# -------------------------------
Write-Host "📦 Installing project dependencies..." -ForegroundColor Cyan

# Backend dependencies
if (Test-Path "backend\requirements.txt") {
    Write-Host "Installing Python backend dependencies..." -ForegroundColor Yellow
    Set-Location backend
    
    # Create virtual environment
    if (-not (Test-Path "venv")) {
        python -m venv venv
    }
    
    # Activate virtual environment and install dependencies
    & "venv\Scripts\Activate.ps1"
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    
    Set-Location ..
    Write-Host "✅ Backend dependencies installed" -ForegroundColor Green
}

# Frontend dependencies
if (Test-Path "frontend\package.json") {
    Write-Host "Installing Node.js frontend dependencies..." -ForegroundColor Yellow
    Set-Location frontend
    
    npm install
    
    Set-Location ..
    Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green
}

# -------------------------------
# 3. DATABASE SERVICES (Optional)
# -------------------------------
if (-not $SkipDatabases) {
    Write-Host "🗄️ Setting up database services..." -ForegroundColor Cyan
    
    # Create data directories
    $dataDir = "C:\ProgramData\RAGChatbot"
    if (-not (Test-Path $dataDir)) {
        New-Item -ItemType Directory -Path $dataDir -Force
    }
    
    # MongoDB
    Write-Host "📦 Setting up MongoDB..." -ForegroundColor Yellow
    $mongoDir = "$dataDir\MongoDB"
    if (-not (Test-Path "$mongoDir\bin\mongod.exe")) {
        $mongoUrl = "https://fastdl.mongodb.org/windows/mongodb-windows-x86_64-7.0.5.zip"
        $mongoZip = "$env:TEMP\mongodb.zip"
        
        if (Download-And-Extract -Url $mongoUrl -OutputPath $mongoZip -ExtractPath $mongoDir) {
            # Move files from extracted folder to MongoDB directory
            $extractedFolder = Get-ChildItem $mongoDir | Where-Object {$_.PSIsContainer} | Select-Object -First 1
            Move-Item "$mongoDir\$($extractedFolder.Name)\*" $mongoDir -Force
            Remove-Item "$mongoDir\$($extractedFolder.Name)" -Force
            
            Write-Host "✅ MongoDB downloaded and extracted" -ForegroundColor Green
        }
    }
    
    # Redis
    Write-Host "📦 Setting up Redis..." -ForegroundColor Yellow
    $redisDir = "$dataDir\Redis"
    if (-not (Test-Path "$redisDir\redis-server.exe")) {
        $redisUrl = "https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip"
        $redisZip = "$env:TEMP\redis.zip"
        
        if (Download-And-Extract -Url $redisUrl -OutputPath $redisZip -ExtractPath $redisDir) {
            Write-Host "✅ Redis downloaded and extracted" -ForegroundColor Green
        }
    }
    
    # Qdrant
    Write-Host "📦 Setting up Qdrant..." -ForegroundColor Yellow
    $qdrantDir = "$dataDir\Qdrant"
    if (-not (Test-Path "$qdrantDir\qdrant.exe")) {
        $qdrantUrl = "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip"
        $qdrantZip = "$env:TEMP\qdrant.zip"
        
        if (Download-And-Extract -Url $qdrantUrl -OutputPath $qdrantZip -ExtractPath $qdrantDir) {
            Write-Host "✅ Qdrant downloaded and extracted" -ForegroundColor Green
        }
    }
    
    # ArangoDB
    Write-Host "📦 Setting up ArangoDB..." -ForegroundColor Yellow
    $arangoDir = "$dataDir\ArangoDB"
    if (-not (Test-Path "$arangoDir\bin\arangod.exe")) {
        Write-Host "Please download ArangoDB manually from: https://www.arangodb.com/download-major/" -ForegroundColor Yellow
        Write-Host "Install it to $arangoDir for automatic detection." -ForegroundColor Yellow
    }
} else {
    Write-Host "⏭️ Skipping database setup (use -SkipDatabases was specified)" -ForegroundColor Yellow
}

# -------------------------------
# 4. ENVIRONMENT CONFIGURATION
# -------------------------------
Write-Host "⚙️ Setting up environment configuration..." -ForegroundColor Cyan

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✅ Environment file created from template" -ForegroundColor Green
    Write-Host "📝 Please edit .env file with your specific configuration" -ForegroundColor Yellow
}

# -------------------------------
# 5. CREATE STARTUP SCRIPTS
# -------------------------------
Write-Host "📄 Creating startup scripts..." -ForegroundColor Cyan

# Backend startup script
$backendScript = @"
@echo off
cd /d "%~dp0backend"
call venv\Scripts\activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"@
$backendScript | Out-File -FilePath "start-backend.bat" -Encoding ASCII

# Frontend startup script
$frontendScript = @"
@echo off
cd /d "%~dp0frontend"
npm run dev
"@
$frontendScript | Out-File -FilePath "start-frontend.bat" -Encoding ASCII

# Database startup script
$dbScript = @"
@echo off
echo Starting database services...
start "MongoDB" /D "C:\ProgramData\RAGChatbot\MongoDB\bin" mongod.exe --dbpath "C:\ProgramData\RAGChatbot\MongoDB\data"
start "Redis" /D "C:\ProgramData\RAGChatbot\Redis" redis-server.exe
start "Qdrant" /D "C:\ProgramData\RAGChatbot\Qdrant" qdrant.exe
echo Database services started. Check individual windows for status.
"@
$dbScript | Out-File -FilePath "start-databases.bat" -Encoding ASCII

Write-Host "✅ Startup scripts created" -ForegroundColor Green

# -------------------------------
# 6. FINAL INSTRUCTIONS
# -------------------------------
Write-Host ""
Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env file with your database credentials" -ForegroundColor White
Write-Host "2. Start LMStudio and load Mistral-3B model" -ForegroundColor White
Write-Host "3. Run start-databases.bat to start database services" -ForegroundColor White
Write-Host "4. Run start-backend.bat to start the FastAPI backend" -ForegroundColor White
Write-Host "5. Run start-frontend.bat to start the Next.js frontend" -ForegroundColor White
Write-Host ""
Write-Host "🌐 Access Points:" -ForegroundColor Cyan
Write-Host "   Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "   Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "📝 Important Notes:" -ForegroundColor Yellow
Write-Host "   - Make sure LMStudio is running on port 1234" -ForegroundColor White
Write-Host "   - Database services need to be started before backend" -ForegroundColor White
Write-Host "   - Check firewall settings if services don't start" -ForegroundColor White