# OMS Chat Bot - Complete Windows Setup Script
# This script sets up the entire RAG + Knowledge Graph chatbot system on Windows

param(
    [switch]$SkipDependencies,
    [switch]$DevMode = $true
)

Write-Host "=== OMS Chat Bot - Windows Setup ===" -ForegroundColor Green
Write-Host "Setting up RAG + Knowledge Graph Chatbot System" -ForegroundColor Yellow

# Check if running as Administrator
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin -and -not $SkipDependencies) {
    Write-Host "This script requires Administrator privileges for dependency installation." -ForegroundColor Red
    Write-Host "Please run PowerShell as Administrator or use -SkipDependencies flag." -ForegroundColor Yellow
    exit 1
}

# Function to check if command exists
function Test-Command($command) {
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

# Function to install Chocolatey
function Install-Chocolatey {
    if (-not (Test-Command "choco")) {
        Write-Host "Installing Chocolatey..." -ForegroundColor Yellow
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        refreshenv
    } else {
        Write-Host "Chocolatey already installed" -ForegroundColor Green
    }
}

# Function to install Python
function Install-Python {
    if (-not (Test-Command "python")) {
        Write-Host "Installing Python 3.11..." -ForegroundColor Yellow
        choco install python --version=3.11.7 -y
        refreshenv
    } else {
        $pythonVersion = python --version 2>&1
        Write-Host "Python already installed: $pythonVersion" -ForegroundColor Green
    }
}

# Function to install Node.js
function Install-NodeJS {
    if (-not (Test-Command "node")) {
        Write-Host "Installing Node.js..." -ForegroundColor Yellow
        choco install nodejs -y
        refreshenv
    } else {
        $nodeVersion = node --version
        Write-Host "Node.js already installed: $nodeVersion" -ForegroundColor Green
    }
}

# Function to install Git
function Install-Git {
    if (-not (Test-Command "git")) {
        Write-Host "Installing Git..." -ForegroundColor Yellow
        choco install git -y
        refreshenv
    } else {
        $gitVersion = git --version
        Write-Host "Git already installed: $gitVersion" -ForegroundColor Green
    }
}

# Function to setup Python virtual environment
function Setup-PythonEnvironment {
    Write-Host "Setting up Python virtual environment..." -ForegroundColor Yellow
    
    Set-Location "backend"
    
    # Create virtual environment if it doesn't exist
    if (-not (Test-Path "venv")) {
        python -m venv venv
    }
    
    # Activate virtual environment
    & ".\venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install requirements
    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    # Install spaCy English model
    Write-Host "Installing spaCy English model..." -ForegroundColor Yellow
    python -m spacy download en_core_web_sm
    
    Set-Location ".."
}

# Function to setup Node.js environment
function Setup-NodeEnvironment {
    Write-Host "Setting up Node.js environment..." -ForegroundColor Yellow
    
    Set-Location "frontend"
    
    # Install dependencies
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
    npm install
    
    # Build the application if not in dev mode
    if (-not $DevMode) {
        Write-Host "Building frontend application..." -ForegroundColor Yellow
        npm run build
    }
    
    Set-Location ".."
}

# Function to create environment file
function Create-EnvironmentFile {
    Write-Host "Creating environment configuration..." -ForegroundColor Yellow
    
    $envContent = @"
# Database Configuration
MONGODB_URI=mongodb://localhost:27017/oms_chatbot
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
ARANGO_URL=http://localhost:8529
ARANGO_USERNAME=root
ARANGO_PASSWORD=password
ARANGO_DATABASE=oms_chatbot
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Admin Configuration
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
ADMIN_EMAIL=admin@example.com

# LLM Configuration
LMSTUDIO_API_URL=http://localhost:1234/v1/chat/completions
LMSTUDIO_API_KEY=

# Processing Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=5
EMBEDDING_DIMENSION=384

# Feature Flags
USE_RERANKER=true
USE_GRAPH_SEARCH=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_OCR=true
ENABLE_FEEDBACK=true

# Performance
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.9
MAX_TOKENS=512

# Development
DEBUG=true
LOG_LEVEL=INFO
ENVIRONMENT=development
"@

    Set-Content -Path "backend\.env" -Value $envContent -Encoding UTF8
    
    # Create frontend environment file
    $frontendEnvContent = @"
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
"@
    
    Set-Content -Path "frontend\.env.local" -Value $frontendEnvContent -Encoding UTF8
}

# Function to install databases
function Install-Databases {
    Write-Host "Setting up databases..." -ForegroundColor Yellow
    
    # Install MongoDB
    if (-not (Test-Command "mongod")) {
        Write-Host "Installing MongoDB..." -ForegroundColor Yellow
        choco install mongodb -y
        
        # Start MongoDB service
        Write-Host "Starting MongoDB service..." -ForegroundColor Yellow
        Start-Service -Name "MongoDB"
        Set-Service -Name "MongoDB" -StartupType Automatic
    }
    
    # Install Redis
    if (-not (Get-Service "Redis" -ErrorAction SilentlyContinue)) {
        Write-Host "Installing Redis..." -ForegroundColor Yellow
        
        # Download Redis for Windows
        $redisUrl = "https://github.com/MicrosoftArchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.msi"
        $redisInstaller = "$env:TEMP\redis-installer.msi"
        
        Invoke-WebRequest -Uri $redisUrl -OutFile $redisInstaller
        Start-Process -FilePath "msiexec.exe" -ArgumentList "/i", $redisInstaller, "/quiet" -Wait
        Remove-Item $redisInstaller
    }
    
    Write-Host "Note: Please install Qdrant and ArangoDB manually:" -ForegroundColor Yellow
    Write-Host "  - Qdrant: Download from https://qdrant.tech/documentation/guides/installation/" -ForegroundColor Cyan
    Write-Host "  - ArangoDB: Download from https://www.arangodb.com/download-major/" -ForegroundColor Cyan
}

# Function to download LMStudio
function Setup-LMStudio {
    Write-Host "LMStudio setup information:" -ForegroundColor Yellow
    Write-Host "  1. Download LMStudio from: https://lmstudio.ai/" -ForegroundColor Cyan
    Write-Host "  2. Install and launch LMStudio" -ForegroundColor Cyan
    Write-Host "  3. Download the Mistral-3B model from the Models tab" -ForegroundColor Cyan
    Write-Host "  4. Load the model and start the local server on port 1234" -ForegroundColor Cyan
    Write-Host "  5. The chatbot will connect to http://localhost:1234" -ForegroundColor Cyan
}

# Function to create startup scripts
function Create-StartupScripts {
    Write-Host "Creating startup scripts..." -ForegroundColor Yellow
    
    # Backend startup script
    $backendScript = @"
@echo off
echo Starting OMS Chatbot Backend...
cd /d "%~dp0backend"
call venv\Scripts\activate.bat
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause
"@
    
    Set-Content -Path "start-backend.bat" -Value $backendScript -Encoding UTF8
    
    # Frontend startup script
    $frontendScript = @"
@echo off
echo Starting OMS Chatbot Frontend...
cd /d "%~dp0frontend"
npm run dev
pause
"@
    
    Set-Content -Path "start-frontend.bat" -Value $frontendScript -Encoding UTF8
    
    # Combined startup script
    $combinedScript = @"
@echo off
echo Starting OMS Chatbot System...
echo.
echo Starting Backend...
start "Backend" cmd /c "start-backend.bat"
timeout /t 5 /nobreak > nul
echo Starting Frontend...
start "Frontend" cmd /c "start-frontend.bat"
echo.
echo Both services starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause
"@
    
    Set-Content -Path "start-chatbot.bat" -Value $combinedScript -Encoding UTF8
    
    Write-Host "Created startup scripts:" -ForegroundColor Green
    Write-Host "  - start-backend.bat: Start backend only" -ForegroundColor Cyan
    Write-Host "  - start-frontend.bat: Start frontend only" -ForegroundColor Cyan
    Write-Host "  - start-chatbot.bat: Start both services" -ForegroundColor Cyan
}

# Function to create README
function Create-ReadmeFile {
    Write-Host "Creating README file..." -ForegroundColor Yellow
    
    $readmeContent = @"
# OMS Chat Bot - RAG + Knowledge Graph Chatbot

A production-ready chatbot system with Retrieval-Augmented Generation (RAG) and Knowledge Graph capabilities, optimized for Windows development.

## 🚀 Quick Start

### Prerequisites Installed
✅ Python 3.11+
✅ Node.js 18+
✅ MongoDB
✅ Redis
⚠️ Qdrant (Manual installation required)
⚠️ ArangoDB (Manual installation required)
⚠️ LMStudio + Mistral-3B model

### Starting the System

1. **Start all services:**
   ```cmd
   start-chatbot.bat
   ```

2. **Or start services individually:**
   ```cmd
   # Backend only
   start-backend.bat
   
   # Frontend only  
   start-frontend.bat
   ```

### Access Points
- **Frontend UI:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## 🔧 Configuration

### Environment Files
- `backend\.env` - Backend configuration
- `frontend\.env.local` - Frontend configuration

### Default Credentials
- **Admin Username:** admin
- **Admin Password:** admin123

## 📁 Project Structure

```
OMS Chat Bot/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Database clients
│   │   ├── services/       # Business logic
│   │   └── workers/        # Background tasks
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment config
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # Next.js 13+ app directory
│   │   ├── components/    # React components
│   │   ├── contexts/      # React contexts
│   │   └── lib/          # Utilities and API
│   ├── package.json       # Node dependencies
│   └── .env.local        # Frontend environment
└── setup-windows.ps1      # This setup script
```

## 🧠 Features

### Core Functionality
- **RAG Pipeline:** Document ingestion, chunking, embedding, retrieval
- **Knowledge Graph:** Entity extraction, relationship mapping
- **Multi-format Support:** PDF, DOCX, TXT, CSV, Excel, PowerPoint, Images
- **Real-time Chat:** WebSocket streaming responses
- **Admin Dashboard:** Document management, analytics

### Technology Stack
- **Backend:** FastAPI, Python 3.11+
- **Frontend:** Next.js 14, React, TypeScript
- **Databases:** MongoDB, Qdrant, ArangoDB, Redis
- **ML/AI:** sentence-transformers, spaCy, LMStudio
- **Authentication:** JWT with refresh tokens

## 📊 Database Setup

### Required Databases
1. **MongoDB:** Document storage (GridFS for files)
2. **Qdrant:** Vector database for embeddings
3. **ArangoDB:** Graph database for relationships
4. **Redis:** Caching and pub/sub messaging

### Database Installation
- **MongoDB & Redis:** Installed automatically by setup script
- **Qdrant:** Download from https://qdrant.tech/documentation/guides/installation/
- **ArangoDB:** Download from https://www.arangodb.com/download-major/

## 🤖 LMStudio Setup

1. Download LMStudio from https://lmstudio.ai/
2. Install and launch LMStudio
3. Go to the "Models" tab and download "Mistral-3B" or similar model
4. Load the model in the "Chat" tab
5. Start the local server (usually on port 1234)
6. The backend will connect to http://localhost:1234

## 🔍 Usage

### Document Upload
1. Access the admin panel at http://localhost:3000/admin
2. Login with admin credentials
3. Upload documents (PDF, DOCX, etc.)
4. Monitor processing status in real-time

### Chat Interface
1. Open http://localhost:3000
2. Start chatting - the system will:
   - Search relevant documents
   - Extract entities from your query
   - Query the knowledge graph
   - Generate contextual responses

### API Usage
The backend provides REST APIs for integration:
- `/chat/query` - Send chat messages
- `/admin/documents` - Manage documents
- `/feedback/submit` - Submit feedback

## 🛠️ Development

### Backend Development
```cmd
cd backend
venv\Scripts\activate
python -m uvicorn app.main:app --reload
```

### Frontend Development
```cmd
cd frontend
npm run dev
```

### Adding New Features
1. Backend: Add routes in `backend/app/api/`
2. Frontend: Add components in `frontend/src/components/`
3. Database: Extend clients in `backend/app/core/`

## 📈 Monitoring

### Health Checks
- Backend health: http://localhost:8000/health
- Database connections: http://localhost:8000/health/databases
- Processing status: Real-time via WebSocket

### Logs
- Backend logs: Console output with configurable log levels
- Frontend logs: Browser console and Network tab

## 🚨 Troubleshooting

### Common Issues
1. **Port conflicts:** Change ports in environment files
2. **Database connection:** Check if all databases are running
3. **Model loading:** Ensure LMStudio server is running
4. **Dependencies:** Run setup script again with admin privileges

### Getting Help
1. Check logs for error messages
2. Verify all services are running
3. Check environment file configurations
4. Ensure all prerequisites are installed

## 🔒 Security Notes

### Production Deployment
- Change default admin credentials
- Use strong SECRET_KEY
- Enable HTTPS
- Configure firewall rules
- Use production database settings

### Environment Variables
- Never commit .env files to version control
- Use different credentials for production
- Enable authentication for all databases

## 📝 License

This project is for educational and development purposes.
Ensure proper licensing for all dependencies in production use.
"@

    Set-Content -Path "README.md" -Value $readmeContent -Encoding UTF8
}

# Main execution
try {
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan
    
    # Install dependencies if not skipped
    if (-not $SkipDependencies) {
        Install-Chocolatey
        Install-Python
        Install-NodeJS
        Install-Git
        Install-Databases
    }
    
    # Setup environments
    Setup-PythonEnvironment
    Setup-NodeEnvironment
    
    # Create configuration files
    Create-EnvironmentFile
    Create-StartupScripts
    Create-ReadmeFile
    
    # Display setup information
    Setup-LMStudio
    
    Write-Host ""
    Write-Host "=== Setup Complete! ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host "1. Install Qdrant and ArangoDB manually (see README.md)" -ForegroundColor Cyan
    Write-Host "2. Download and setup LMStudio with Mistral-3B model" -ForegroundColor Cyan
    Write-Host "3. Start the system with: .\start-chatbot.bat" -ForegroundColor Cyan
    Write-Host "4. Access the application at: http://localhost:3000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Admin Credentials:" -ForegroundColor Yellow
    Write-Host "  Username: admin" -ForegroundColor Cyan
    Write-Host "  Password: admin123" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Documentation: See README.md for detailed instructions" -ForegroundColor Yellow
    
} catch {
    Write-Host "Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please check the error and try again" -ForegroundColor Yellow
    exit 1
}

Write-Host "Setup completed successfully! 🎉" -ForegroundColor Green