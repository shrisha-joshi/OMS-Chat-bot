
# RAG + Knowledge Graph Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Backend-FastAPI-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Frontend-Next.js-black?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Database-MongoDB-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/VectorDB-Qdrant-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GraphDB-ArangoDB-teal?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LLM-LMStudio-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge"/>
</p>

---

## 🚀 Project Overview

This is a full-stack, production-grade Retrieval-Augmented Generation (RAG) chatbot platform with integrated knowledge graph analytics. It enables enterprise-grade document Q&A, semantic search, and knowledge discovery over unstructured and structured data. The system is modular, scalable, and designed for extensibility and security.

---

## 🏗️ System Architecture

![Architecture Diagram](https://raw.githubusercontent.com/your-org/your-repo/main/docs/architecture.png)

**Key Components:**

- **Frontend:** Next.js (React), TailwindCSS, React Context, Axios API client
- **Backend:** FastAPI (Python), modular routers, async service layer, dependency injection
- **Databases:**
  - MongoDB: Document storage (GridFS for large files)
  - Qdrant: Vector search for semantic retrieval
  - ArangoDB: Knowledge graph (entity/relation analytics)
  - Redis: Caching for fast response and session state
- **LLM:** LMStudio (local API, supports open LLMs)
- **RAG Pipeline:**
  1. Document upload (PDF, DOCX, TXT, JSON)
  2. Chunking and preprocessing
  3. Embedding generation (SentenceTransformer)
  4. Vector search (Qdrant, top-K retrieval)
  5. Context assembly
  6. LLM response generation
  7. Caching and session management

**Security:**
- PII masking, input validation, and resilience patterns
- No credentials in code; all secrets in `.env`
- Public admin endpoints for dev only (see [CLEANUP_COMPLETE.md](./CLEANUP_COMPLETE.md))

**DevOps:**
- Docker Compose for local DBs
- PowerShell scripts for Windows automation
- Modular, testable codebase with CI-ready structure

---

## 🛠️ Features

- **Document Upload & Ingestion:** Drag-and-drop UI, supports PDF, DOCX, TXT, JSON
- **Semantic Search:** Qdrant-powered vector retrieval, hybrid search with metadata filters
- **Knowledge Graph:** Entity/relation extraction, ArangoDB visualization, graph-based Q&A
- **Chatbot Interface:** LLM-powered, context-aware, supports session history
- **Admin Dashboard:** Ingestion status, pipeline health, metrics, and logs
- **Security:** PII masking, input validation, rate limiting
- **Scalability:** Async FastAPI, modular routers, background workers
- **Extensibility:** Add new document types, LLMs, or DBs with minimal changes

---

## 🏁 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for databases, optional but recommended)
- LMStudio (for LLM, [Download](https://lmstudio.ai/))

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/rag-kg-chatbot.git
   cd rag-kg-chatbot
   ```
2. **Configure environment variables:**
   - Copy and edit backend/.env:
     ```bash
     cp backend/env.sample backend/.env
     ```
   - Set MongoDB, Qdrant, ArangoDB, Redis URIs, and LMSTUDIO_API_URL.
3. **Start databases (optional):**
   ```bash
   docker-compose up -d
   ```
4. **Start backend:**
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload
   ```
5. **Start frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
6. **Access the application:**
   - Frontend: [http://localhost:3000](http://localhost:3000)
   - API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📚 Documentation & References

- [System Overview](./FINAL_SUCCESS_STATUS.md)
- [Testing Playbook](./TESTING_PLAYBOOK.md)
- [Debug & Fix Guide](./DEBUG_AND_FIX_GUIDE.md)
- [Ingest Worker Setup](./INGEST_WORKER_SETUP.md)
- [API Reference (Swagger)](http://localhost:8000/docs)
- [Frontend Components](frontend/COMPONENTS.md)
- [Backend Patterns](#backend-patterns)
- [Frontend Patterns](#frontend-patterns)

---

## 🧑‍💻 Contributing

We welcome contributions from the community! Please follow these guidelines:

1. **Fork** the repository
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Write clear, well-documented code**
4. **Add/Update tests** as appropriate
5. **Lint and format your code**
6. **Commit and push:**
   ```bash
   git commit -am 'Add new feature'
   git push origin feature/your-feature
   ```
7. **Open a Pull Request** with a clear description

### Code Style
- **Python:** [PEP8](https://peps.python.org/pep-0008/), Black, isort
- **TypeScript:** [Airbnb](https://github.com/airbnb/javascript), Prettier, ESLint

### Issue Tracking
- [Report a Bug](https://github.com/your-org/rag-kg-chatbot/issues/new?template=bug_report.md)
- [Request a Feature](https://github.com/your-org/rag-kg-chatbot/issues/new?template=feature_request.md)
- [Start a Discussion](https://github.com/your-org/rag-kg-chatbot/discussions)

---

## 🧩 Project Structure

```
rag-kg-chatbot/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routers
│   │   ├── core/          # DB clients, config, model manager
│   │   ├── middleware/    # Rate limiting, security
│   │   ├── migrations/    # DB schema migrations
│   │   ├── schemas/       # Pydantic models
│   │   ├── services/      # Business logic, LLM, ingestion
│   │   ├── utils/         # Validation, metrics, helpers
│   │   └── workers/       # Background ingestion
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── src/
│   │   ├── app/           # Next.js pages
│   │   ├── components/    # React components
│   │   ├── contexts/      # React context providers
│   │   ├── lib/           # API client, logger
│   │   └── types/         # TypeScript types
│   ├── package.json
│   └── ...
├── docker-compose.yml
├── start-backend.ps1
├── start-frontend.ps1
└── ...
```

---

## ⚙️ Backend Patterns

- **Configuration:** All settings via `backend/app/config.py` (Pydantic, env aliases)
- **Router Pattern:** Each API module exports `router = APIRouter()`
- **Dependency Injection:** Use `Depends()` for DB clients and services
- **Service Layer:** Singleton pattern for core services (chat, ingestion)
- **Database Clients:** Async, modular, with graceful degradation
- **Lifespan Management:** Startup/shutdown hooks for DBs and background tasks
- **Testing:** Modular, with e2e and unit tests in `backend/tests/`

## ⚛️ Frontend Patterns

- **API Client:** Singleton Axios instance with auth, error handling
- **Component Structure:** Modular, reusable React components
- **State Management:** React Context for auth, chat, and global state
- **Styling:** TailwindCSS, HeroIcons, Lucide React
- **Validation:** HTML5 + Axios error handling

---

## 🛡️ Security & Best Practices

- No credentials in code; use `.env` for all secrets
- Public admin endpoints for development only
- All DB clients degrade gracefully if unavailable
- LLM calls are async/threaded for performance
- Input validation and PII masking throughout pipeline
- Rate limiting and resilience patterns in middleware

---

## 📦 Deployment

- **Local:** Use provided PowerShell scripts or Docker Compose
- **Production:**
  - Secure admin endpoints
  - Set strong secrets in `.env`
  - Use HTTPS and secure DB connections
  - Monitor logs and metrics

---

## 🪟 Windows Setup

1. Run `setup-windows-complete.ps1` as Administrator
2. Configure `backend/.env` with all required URIs and secrets
3. Start LMStudio manually
4. Use `start-backend.ps1` and `start-frontend.ps1` to launch services

---

## 🧪 Testing & Quality Assurance

- **Unit & Integration Tests:**
  ```bash
  cd backend
  python -m unittest discover tests
  ```
- **End-to-End:** See [TESTING_PLAYBOOK.md](./TESTING_PLAYBOOK.md)
- **Linting:**
  - Python: `black`, `isort`, `flake8`
  - TypeScript: `eslint`, `prettier`
- **Code Quality:** SonarQube integration recommended

---

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/)
- [Qdrant](https://qdrant.tech/)
- [ArangoDB](https://www.arangodb.com/)
- [LMStudio](https://lmstudio.ai/)
- [MongoDB](https://www.mongodb.com/)
- [TailwindCSS](https://tailwindcss.com/)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://github.com/your-org/rag-kg-chatbot/issues/new?template=bug_report.md"><img src="https://img.shields.io/badge/Report%20Bug-red?style=for-the-badge"/></a>
  <a href="https://github.com/your-org/rag-kg-chatbot/issues/new?template=feature_request.md"><img src="https://img.shields.io/badge/Request%20Feature-blue?style=for-the-badge"/></a>
  <a href="https://github.com/your-org/rag-kg-chatbot/discussions"><img src="https://img.shields.io/badge/Discussions-green?style=for-the-badge"/></a>
</p>
