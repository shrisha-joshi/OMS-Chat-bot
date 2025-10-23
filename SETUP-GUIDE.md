# 🚀 OMS Advanced RAG Chat Bot - Complete Setup Guide

## 📋 **Prerequisites**

### **Required Software:**
1. **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop)
2. **Python 3.9-3.11** - [Download here](https://www.python.org/downloads/)
3. **Node.js 18+** - [Download here](https://nodejs.org/)
4. **LMStudio** - [Download here](https://lmstudio.ai/)

---

## 🔧 **Step 1: Install Prerequisites**

### **1.1 Install Docker Desktop**
```powershell
# Download and install Docker Desktop from docker.com
# Make sure to enable WSL2 integration
docker --version  # Should show version after installation
```

### **1.2 Install Python** 
```powershell
# Download Python 3.11 from python.org
# ✅ CHECK "Add Python to PATH" during installation
python --version  # Should show 3.11.x
```

### **1.3 Install Node.js**
```powershell
# Download Node.js 18+ from nodejs.org  
node --version    # Should show 18+
npm --version     # Should be available
```

---

## 🗄️ **Step 2: Start Database Services**

### **2.1 Start All Databases with Docker**
```powershell
cd "d:\OMS Chat Bot"
docker-compose up -d
```

**This will start:**
- ✅ **MongoDB** on `localhost:27017` 
- ✅ **Qdrant** on `localhost:6333`
- ✅ **ArangoDB** on `localhost:8529` 
- ✅ **Redis** on `localhost:6379`

### **2.2 Verify Services**
```powershell
# Check if all containers are running
docker ps

# You should see 4 containers:
# - oms_mongodb
# - oms_qdrant  
# - oms_arangodb
# - oms_redis
```

---

## 🤖 **Step 3: Setup LMStudio**

### **3.1 Install and Configure LMStudio**
1. **Download & Install** LMStudio from [lmstudio.ai](https://lmstudio.ai/)
2. **Open LMStudio** application
3. **Download a Model:**
   - Go to "Discover" tab
   - Search and download: **"Mistral-7B-Instruct"** or **"Llama-2-7B-Chat"**
4. **Start Local Server:**
   - Go to "Local Server" tab
   - Load your downloaded model
   - Click "Start Server"
   - Default URL: `http://localhost:1234`

---

## 🐍 **Step 4: Backend Setup**

### **4.1 Setup Python Environment**
```powershell
cd "d:\OMS Chat Bot\backend"

# Create virtual environment
python -m venv venv

# Activate virtual environment  
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### **4.2 Start Backend Server**
```powershell
# Make sure you're in backend directory with venv activated
cd "d:\OMS Chat Bot\backend"
.\venv\Scripts\Activate.ps1

# Start FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Backend should be running at: `http://localhost:8000`**

---

## 🎨 **Step 5: Frontend Setup** 

### **5.1 Install Dependencies**
```powershell
cd "d:\OMS Chat Bot\frontend"
npm install
```

### **5.2 Start Frontend Development Server**
```powershell
# In frontend directory
npm run dev
```

**Frontend should be running at: `http://localhost:3000`**

---

## ✅ **Step 6: Verification & Testing**

### **6.1 Check Database Connections**
- **MongoDB**: `http://localhost:27017` (Use MongoDB Compass if needed)
- **Qdrant**: `http://localhost:6333/dashboard` (Web dashboard) 
- **ArangoDB**: `http://localhost:8529` (Web interface, user: root, password: your_arango_password)
- **Redis**: Use Redis CLI or Redis Desktop Manager

### **6.2 Test API Endpoints**
```powershell
# Test backend health
curl http://localhost:8000/health

# Test chat endpoint (should return 401 - need auth)
curl http://localhost:8000/chat/query
```

### **6.3 Access Applications**
- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`

---

## 🔧 **Configuration Files to Check**

### **Backend Configuration:**
- ✅ `.env` - Database connections and API keys
- ✅ `backend/app/config.py` - Application settings
- ✅ `backend/requirements.txt` - Python dependencies

### **Frontend Configuration:**
- ✅ `frontend/package.json` - Node.js dependencies
- ✅ `frontend/next.config.js` - Next.js configuration  
- ✅ `frontend/tailwind.config.js` - Styling configuration

---

## 🚨 **Common Issues & Solutions**

### **Issue 1: Docker containers won't start**
```powershell
# Stop all containers and restart
docker-compose down
docker-compose up -d

# Check logs for specific service
docker logs oms_mongodb
```

### **Issue 2: Python virtual environment issues**
```powershell
# Delete and recreate venv
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### **Issue 3: Port conflicts**
```powershell
# Check what's using ports
netstat -ano | findstr :8000
netstat -ano | findstr :3000
netstat -ano | findstr :6333
```

### **Issue 4: LMStudio connection failed**
- Make sure LMStudio server is running on `localhost:1234`
- Check if firewall is blocking the connection
- Verify model is loaded and server is started

---

## 🎯 **Quick Start Commands**

### **Start Everything (After initial setup):**
```powershell
# Terminal 1: Start databases  
cd "d:\OMS Chat Bot"
docker-compose up -d

# Terminal 2: Start backend
cd "d:\OMS Chat Bot\backend"
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload

# Terminal 3: Start frontend
cd "d:\OMS Chat Bot\frontend"
npm run dev

# Terminal 4: Start LMStudio (manually through GUI)
```

---

## 🔍 **Advanced Features Available**

Your system now includes:
- ✅ **Query Intelligence** - Automatic query rewriting and decomposition
- ✅ **Hybrid Search** - BM25 + Vector + Metadata filtering  
- ✅ **Context Optimization** - Autocut filtering and Chain-of-Thought prompting
- ✅ **Hierarchical Indexing** - Multi-level document chunking
- ✅ **Evaluation System** - Real-time response quality assessment
- ✅ **Redis Caching** - Performance optimization with intelligent caching
- ✅ **Neo4j Integration** - Knowledge graph functionality
- ✅ **Advanced UI** - Evaluation metrics display and feedback collection

---

## 📞 **Need Help?**

1. **Check logs**: Each service has logs in Docker or terminal output
2. **Verify connections**: Make sure all services are running and accessible
3. **Test components**: Use the provided health check endpoints
4. **Review configuration**: Double-check `.env` file settings

**Your Advanced RAG Chat Bot is ready to use! 🎉**