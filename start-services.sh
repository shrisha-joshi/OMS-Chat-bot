#!/bin/bash
# Simple startup script for OMS Chat Bot

echo "🚀 Starting OMS Chat Bot Services..."

# Start databases
echo "📚 Starting databases..."
docker-compose up -d

echo "⏳ Waiting for databases to be ready..."
sleep 10

echo "✅ Database services are running!"
echo ""
echo "🔧 Next steps:"
echo "1. Start LMStudio manually and load a model"
echo "2. Run: cd backend && .\venv\Scripts\Activate.ps1 && python -m uvicorn app.main:app --reload"  
echo "3. Run: cd frontend && npm run dev"
echo ""
echo "🌐 Access at http://localhost:3000"