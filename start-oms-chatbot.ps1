# ============================================================
#  OMS Chat Bot - Quick Start Script
#  Run this script to start all services
# ============================================================

param(
    [switch]$StopServices,
    [switch]$RestartServices
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 OMS Advanced RAG Chat Bot - Quick Start" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker info | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to check if port is in use
function Test-PortInUse {
    param($Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient('localhost', $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

# Stop services if requested
if ($StopServices -or $RestartServices) {
    Write-Host "🛑 Stopping services..." -ForegroundColor Yellow
    
    # Stop Docker containers
    try {
        docker-compose down
        Write-Host "✅ Docker containers stopped" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Docker containers might not be running" -ForegroundColor Yellow
    }
    
    if ($StopServices) {
        Write-Host "✅ All services stopped!" -ForegroundColor Green
        exit 0
    }
}

# Check Docker
Write-Host "🐳 Checking Docker..." -ForegroundColor Cyan
if (-not (Test-DockerRunning)) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    Write-Host "   Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Docker is running" -ForegroundColor Green

# Check if Python is installed
Write-Host "🐍 Checking Python..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.9+ first." -ForegroundColor Red
    Write-Host "   Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if Node.js is installed
Write-Host "📦 Checking Node.js..." -ForegroundColor Cyan
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js 18+ first." -ForegroundColor Red
    Write-Host "   Download from: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Start Docker containers
Write-Host "🗄️  Starting database services..." -ForegroundColor Cyan
try {
    docker-compose up -d
    Write-Host "✅ Database services started" -ForegroundColor Green
    
    # Wait for services to be ready
    Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    
    # Check if services are responding
    $services = @(
        @{Name="MongoDB"; Port=27017},
        @{Name="Qdrant"; Port=6333},
        @{Name="ArangoDB"; Port=8529}, 
        @{Name="Redis"; Port=6379}
    )
    
    foreach ($service in $services) {
        if (Test-PortInUse -Port $service.Port) {
            Write-Host "✅ $($service.Name) is ready on port $($service.Port)" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $($service.Name) might not be ready yet on port $($service.Port)" -ForegroundColor Yellow
        }
    }
    
} catch {
    Write-Host "❌ Failed to start database services: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Check if backend virtual environment exists
Write-Host "🔧 Checking backend setup..." -ForegroundColor Cyan
$backendPath = "backend"
$venvPath = "$backendPath\venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "⚠️  Virtual environment not found. Creating..." -ForegroundColor Yellow
    try {
        Push-Location $backendPath
        python -m venv venv
        & ".\venv\Scripts\Activate.ps1"
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
        Pop-Location
        Write-Host "✅ Backend environment created" -ForegroundColor Green
    } catch {
        Pop-Location
        Write-Host "❌ Failed to create backend environment: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Check if frontend dependencies are installed
Write-Host "🎨 Checking frontend setup..." -ForegroundColor Cyan
$frontendPath = "frontend"
if (-not (Test-Path "$frontendPath\node_modules")) {
    Write-Host "⚠️  Frontend dependencies not found. Installing..." -ForegroundColor Yellow
    try {
        Push-Location $frontendPath
        npm install
        Pop-Location
        Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green
    } catch {
        Pop-Location
        Write-Host "❌ Failed to install frontend dependencies: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Setup Complete! Now start the services:" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Manual Steps Required:" -ForegroundColor Cyan
Write-Host "1. 🤖 Start LMStudio:" -ForegroundColor Yellow
Write-Host "   - Open LMStudio application" -ForegroundColor Gray
Write-Host "   - Go to 'Local Server' tab" -ForegroundColor Gray
Write-Host "   - Load a model (Mistral-7B-Instruct recommended)" -ForegroundColor Gray
Write-Host "   - Click 'Start Server'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. 🐍 Start Backend (New Terminal):" -ForegroundColor Yellow
Write-Host "   cd `"d:\OMS Chat Bot\backend`"" -ForegroundColor Gray
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "   python -m uvicorn app.main:app --reload" -ForegroundColor Gray
Write-Host ""
Write-Host "3. 🎨 Start Frontend (New Terminal):" -ForegroundColor Yellow
Write-Host "   cd `"d:\OMS Chat Bot\frontend`"" -ForegroundColor Gray
Write-Host "   npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
Write-Host "   Frontend:  http://localhost:3000" -ForegroundColor Green
Write-Host "   Backend:   http://localhost:8000" -ForegroundColor Green
Write-Host "   API Docs:  http://localhost:8000/docs" -ForegroundColor Green
Write-Host "   ArangoDB:  http://localhost:8529" -ForegroundColor Green
Write-Host "   Qdrant:    http://localhost:6333/dashboard" -ForegroundColor Green
Write-Host ""
Write-Host "✅ Database services are running in the background!" -ForegroundColor Green