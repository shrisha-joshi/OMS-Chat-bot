# OMS Chat Bot - RAG System

A production-ready Retrieval-Augmented Generation (RAG) chatbot system with document upload, vector search, and LLM integration.

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- MongoDB Atlas
- Qdrant Cloud
- Redis Cloud
- LMStudio running with Mistral-7B

### Installation

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### Configuration

Create `.env` file in backend directory:
```
# MongoDB
MONGODB_URI=mongodb+srv://...
MONGODB_DB=oms_chatbot

# Qdrant
QDRANT_URL=https://...
QDRANT_API_KEY=...

# Redis
REDIS_URL=redis://...

# LLM
LMSTUDIO_API_URL=http://192.168.56.1:1234/v1
```

### Running the Application

**Backend:**
```bash
cd backend
python -m uvicorn app.main:app --reload
```
Backend runs on: `http://localhost:8000`

**Frontend:**
```bash
cd frontend
npm run dev
```
Frontend runs on: `http://localhost:3001`

## Features

✅ **Chat Interface** - Real-time AI responses
✅ **Document Upload** - PDF, DOCX, TXT, HTML, JSON support
✅ **Vector Search** - Fast similarity-based retrieval
✅ **RAG Pipeline** - Context-aware responses with sources
✅ **Admin Panel** - Manage documents and monitor ingestion
✅ **Public API** - RESTful endpoints for integration

## Key Endpoints

### Chat
- `POST /chat/query` - Send a question
- `GET /chat/sessions/{session_id}/history` - Get chat history

### Admin
- `POST /admin/documents/upload` - Upload document
- `GET /admin/documents` - List documents
- `DELETE /admin/documents/{doc_id}` - Delete document
- `GET /admin/stats` - System statistics

### Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /health` - System health check

## Architecture

```
Frontend (Next.js)
    ↓
Backend (FastAPI)
    ├─ MongoDB (documents)
    ├─ Qdrant (vectors)
    ├─ Redis (cache)
    └─ Mistral-7B (LLM)
```

## System Components

- **LLM**: Mistral-7B (32K context window)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Reranker**: cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Vector DB**: Qdrant (HNSW algorithm)
- **Document DB**: MongoDB
- **Cache**: Redis

## Configuration Files

- `backend/app/config.py` - Application settings
- `backend/.env` - Environment variables
- `frontend/.env.local` - Frontend variables
- `.gitignore` - Git ignore patterns

## Security Note

- Admin endpoints are **PUBLIC** for development/testing
- Add authentication before production deployment
- Sensitive data should be in `.env` files (excluded from git)

## Documentation

Core documentation files:
- `SETUP-GUIDE.md` - Initial setup instructions
- `QUICK-START-CLOUD.md` - Cloud deployment guide

## Support

For issues or questions, check:
1. Backend logs: `http://localhost:8000/docs`
2. Frontend console: Browser DevTools (F12)
3. Health check: `http://localhost:8000/health`

---

**Status**: ✅ Production Ready
**Last Updated**: October 23, 2025
