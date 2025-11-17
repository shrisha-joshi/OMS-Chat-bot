# RAG + Knowledge Graph Chatbot - AI Agent Guide

## Quick Links

### Documentation Files (START HERE)
1. **`FINAL_SUCCESS_STATUS.md`** - Complete system overview and status
2. **`CLEANUP_COMPLETE.md`** - Latest cleanup report (21.1% code reduction)
3. **`TESTING_PLAYBOOK.md`** - Step-by-step end-to-end testing guide (9 phases)
4. **`DEBUG_AND_FIX_GUIDE.md`** - Comprehensive troubleshooting (8 parts, error reference)
5. **`INGEST_WORKER_SETUP.md`** - Worker configuration, monitoring, performance tuning
6. **`backend/env.sample`** - All environment variables with documentation

### Reference Guides
- Architecture Overview: See section below
- Backend Patterns: See "Backend Patterns (Python FastAPI)" section
- Frontend Patterns: See "Frontend Patterns (Next.js + TypeScript)" section
- Common Pitfalls: See "Common Patterns to Avoid" at end

## Architecture Overview

**RAG Pipeline** (Simplified - Nov 2025): Documents → Chunks → Embeddings (Qdrant) → Retrieval → LLM Response
**Query Flow**: Frontend → FastAPI `/chat/query` → ChatService.process_query() → Direct pipeline:
  1. Fast-path check (greetings → direct LLM)
  2. Check Redis cache
  3. Preprocess query
  4. Generate embedding (SentenceTransformer)
  5. Qdrant vector search (TOP_K=10)
  6. Build context
  7. LLM generation
  8. Cache result
**Admin Upload**: Frontend → FastAPI `/admin/documents/upload` → MongoDB/GridFS → IngestService
**Code Cleanup**: Phase 1-8 complex pipeline removed (Nov 2025), simplified to direct flow

## Backend Patterns (Python FastAPI)

### Configuration & Defaults
- **Settings**: `backend/app/config.py` uses Pydantic BaseSettings with env aliases (e.g., `MONGODB_URI` → `mongodb_uri`)
- **Key params**: `MAX_LLM_OUTPUT_TOKENS=2048` (token budget), `CHUNK_SIZE=750`, `TOP_K_RETRIEVAL=10`
- **Database optional**: All DB clients gracefully degrade; app runs without MongoDB/Qdrant if `.env` missing
- **Debug mode**: `APP_ENV=development` enables `/docs` and `/system/info` endpoints

### Router & Dependency Injection
- **Router pattern**: Each API module (`chat.py`, `admin.py`, `auth.py`) exports `router = APIRouter()` 
- **Inclusion**: `backend/app/main.py` includes routers with prefixes (e.g., `app.include_router(admin.router, prefix="/admin")`)
- **Dependencies**: Use `Depends()` for DB clients: `mongo: MongoDBClient = Depends(get_mongodb_client)`
- **Service initialization**: Singleton pattern via lazy-initialized global (e.g., `get_chat_service()` returns cached instance)

### Key Endpoints & Response Models
- `POST /chat/query` (ChatRequest) → ChatResponse with `response`, `sources`, `session_id`, `tokens_generated`
- `POST /admin/documents/upload` (multipart) → returns `{"success": bool, "filename": str, "size": int}`
- `POST /admin/documents/upload-json` (JSON) → expects `{"filename", "content_base64", "content_type"}`
- Admin endpoints are **intentionally public** (no auth) for development; re-enable before production

### Database Client Pattern
Located in `backend/app/core/db_*.py`:
- Each client (`MongoDBClient`, `QdrantDBClient`, `ArangoDBClient`, `RedisClient`) is a class with async methods
- Constructor initializes `self.client = None`; `async def connect()` establishes connection with timeout/error handling
- Dependency: `async def get_mongo_client() -> MongoDBClient:` returns global singleton from `backend.app.core.db_mongo`
- Graceful degradation: clients fail independently (connection timeout caught, logged as warning, app continues)

### Lifespan Manager
`backend/app/main.py` @asynccontextmanager lifespan():
- **Startup**: Concurrent DB connects with per-service timeouts (MongoDB=8s, Qdrant=6s, Redis=4s)
- **Lazy workers**: Ingest worker initialized on-demand, not at startup
- **Shutdown**: Cancels background tasks, disconnects all DBs

## Frontend Patterns (Next.js + TypeScript)

### API Client
`frontend/src/lib/api.ts` exports singleton `ApiClient` with Axios:
- **Auth interceptor**: Adds `Authorization: Bearer {token}` from localStorage
- **Token refresh**: 401 response triggers refresh attempt or redirects to `/login`
- **Upload method**: `async upload(url, file, onProgress?)` creates FormData automatically

### Component Structure
- **Pages** (`src/app/page.tsx`): Server or client components; fetch data in getServerSideProps or use hooks
- **Components**: Reusable React elements; document-upload at `frontend/src/components/admin/DocumentUpload.tsx` (no-auth POST to `/admin/documents/upload`)
- **Contexts**: Store global state (e.g., auth, chat history) in `src/contexts/`
- **Types**: Define request/response shapes in `src/types/` then import into components

### Styling
- TailwindCSS + PostCSS (no CSS Modules unless needed)
- HeroIcons for UI icons, Lucide React for other symbols
- Form validation via HTML5 + Axios error handling, no dedicated form library

## Critical Workflows

### Running Locally
```bash
# Terminal 1: Backend (auto-reloads on .py changes)
cd backend && python -m uvicorn app.main:app --reload

# Terminal 2: Frontend (auto-reloads on .tsx changes)
cd frontend && npm run dev

# Browser: http://localhost:3000 (frontend) | http://localhost:8000/docs (API docs)
```

### Adding a New Admin Endpoint
1. Add async handler to `backend/app/api/admin.py` decorated with `@router.post("/new-endpoint")`
2. Define request/response Pydantic models above the handler
3. Use `Depends()` for DB clients; handler auto-receives them
4. Frontend: Import `ApiClient` singleton in component, call `apiClient.post('/admin/new-endpoint', data)`

### Debugging LLM Token Issues
- Config: `max_context_tokens=2000`, `max_llm_output_tokens=2048` in `backend/app/config.py`
- Service: `backend/app/services/chat_service.py` limits input context and output tokens before calling LLM
- Monitor: `POST /chat/query` response includes `tokens_generated` field

## Windows-Specific Setup
1. Run `setup-windows-complete.ps1` as Administrator to install Python, Node, MongoDB (local option)
2. Create `backend/.env` with DB URLs and `LMSTUDIO_API_URL=http://192.168.56.1:1234/v1` (LMStudio default)
3. Start LMStudio separately (manual step, not automated)
4. Run backend & frontend per "Running Locally" section above

## Common Patterns to Avoid
- ❌ Do NOT hardcode credentials in code; use `.env` and config.py
- ❌ Do NOT import `app.main:app` at module level in routers (circular import); use `Depends()` instead
- ❌ Do NOT add auth middleware to public endpoints (`/admin/documents/upload` intentionally has no auth)
- ❌ Do NOT call sync LLM code without wrapping in `asyncio.to_thread()` or using async-compatible library