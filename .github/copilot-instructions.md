# RAG + Knowledge Graph Chatbot Project Instructions

## Project Overview
This is a production-ready RAG (Retrieval-Augmented Generation) + Knowledge Graph Chatbot system designed for Windows development environments.

## Technology Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Next.js + React + TypeScript
- **Databases**: MongoDB, Qdrant, ArangoDB, Redis
- **LLM**: LMStudio + Mistral-3B (local)
- **Embeddings**: sentence-transformers

## Project Status
✅ Project structure created for Windows development
✅ Environment configuration completed
✅ Backend implementation with FastAPI
✅ Frontend implementation with Next.js
✅ Database integration (MongoDB, Qdrant, ArangoDB, Redis)
✅ Windows-specific setup scripts
✅ Documentation completed

## Windows Setup Instructions
1. Run `setup-windows.ps1` in PowerShell (as Administrator)
2. Configure `.env` file with your database connections
3. Start LMStudio manually
4. Run backend: `cd backend && python -m uvicorn app.main:app --reload`
5. Run frontend: `cd frontend && npm run dev`

## Key Features
- Secure JWT authentication
- Real-time ingestion monitoring
- Streaming chat responses
- Multi-modal attachment support
- Knowledge graph reasoning
- Admin dashboard for file management