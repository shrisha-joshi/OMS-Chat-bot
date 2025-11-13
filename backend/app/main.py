"""
Main FastAPI application entry point.
This module sets up the FastAPI application, middleware, database connections,
and includes all API routes for the RAG chatbot system.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import asyncio
from typing import AsyncGenerator
from datetime import datetime, timezone

from .config import settings
from .core.db_mongo import mongodb_client
from .core.db_qdrant import qdrant_client
from .core.cache_redis import redis_client
from .api import chat, admin, auth, feedback, monitoring, websocket
from .workers.ingest_worker import start_ingest_worker
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.debug_mode else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# ========== SERVICE VALIDATION HELPERS (Complexity Reduction) ==========

async def _validate_mongodb() -> tuple[bool, str]:
    """Validate MongoDB connection."""
    try:
        logger.info("Validating MongoDB...")
        await asyncio.wait_for(mongodb_client.connect(), timeout=5.0)
        logger.info("  ✓ MongoDB: Connected and healthy")
        return True, "✓ OK"
    except asyncio.TimeoutError:
        return False, "Connection timeout - check if running on localhost:27017"
    except Exception as e:
        return False, f"Connection failed - {str(e)}"

async def _validate_qdrant() -> tuple[bool, str]:
    """Validate Qdrant connection."""
    try:
        logger.info("Validating Qdrant...")
        await asyncio.wait_for(qdrant_client.connect(), timeout=5.0)
        logger.info("  ✓ Qdrant: Connected and healthy")
        return True, "✓ OK"
    except asyncio.TimeoutError:
        return False, "Connection timeout - check if running on localhost:6333"
    except Exception as e:
        return False, f"Connection failed - {str(e)}"

async def _validate_redis() -> tuple[bool, str]:
    """Validate Redis connection."""
    try:
        logger.info("Validating Redis...")
        await asyncio.wait_for(redis_client.connect(), timeout=5.0)
        logger.info("  ✓ Redis: Connected and healthy")
        return True, "✓ OK"
    except asyncio.TimeoutError:
        return False, "Connection timeout - check if running on localhost:6379"
    except Exception as e:
        return False, f"Connection failed - {str(e)}"

def _validate_embedding_model() -> tuple[bool, str]:
    """Validate embedding model loading."""
    try:
        logger.info("Validating Embedding Model...")
        from sentence_transformers import SentenceTransformer
        model_name = settings.embedding_model_name or "all-MiniLM-L6-v2"
        logger.info(f"  Loading: {model_name}")
        model = SentenceTransformer(model_name)
        test_embedding = model.encode("test")
        assert len(test_embedding) > 0, "Embedding generation failed"
        logger.info(f"  ✓ Embedding Model: {model_name} (dimension: {len(test_embedding)})")
        return True, "✓ OK"
    except Exception as e:
        return False, str(e)

async def _validate_llm_provider() -> tuple[bool, str]:
    """Validate LLM provider availability."""
    try:
        logger.info("Validating LLM Provider...")
        llm_url = settings.lmstudio_api_url or "http://localhost:1234/v1"
        async with httpx.AsyncClient(timeout=5) as client:
            response = await asyncio.wait_for(
                client.get(f"{llm_url}/models"),
                timeout=3.0
            )
            if response.status_code in [200, 404]:  # 404 OK if LMStudio not fully loaded
                logger.info(f"  ✓ LLM Provider: Available at {llm_url}")
                return True, "✓ OK"
            return False, f"HTTP {response.status_code} from {llm_url}"
    except asyncio.TimeoutError:
        return False, f"Timeout connecting to {settings.lmstudio_api_url}"
    except Exception as e:
        return False, str(e)

def _validate_spacy_model() -> tuple[bool, str]:
    """Validate spaCy model loading."""
    try:
        logger.info("Validating spaCy Model...")
        import spacy
        try:
            spacy.load("en_core_web_sm")
            logger.info("  ✓ spaCy Model: en_core_web_sm loaded")
            return True, "✓ OK"
        except OSError:
            return False, "Model 'en_core_web_sm' not installed. Install with: python -m spacy download en_core_web_sm"
    except Exception as e:
        return False, str(e)

def _log_validation_errors(errors: list[str]):
    """Log validation errors with actionable checklist."""
    logger.error("❌ STARTUP FAILED - Required services not available:\n")
    for i, error in enumerate(errors, 1):
        logger.error(f"  {i}. {error}")
    
    logger.error("\n" + "=" * 80)
    logger.error("REQUIRED SERVICES CHECKLIST:")
    logger.error("=" * 80)
    logger.error("  [ ] MongoDB running on localhost:27017")
    logger.error("  [ ] Qdrant running on localhost:6333")
    logger.error("  [ ] Redis running on localhost:6379")
    logger.error("  [ ] LMStudio running on localhost:1234")
    logger.error("  [ ] spaCy model 'en_core_web_sm' installed")
    logger.error("\nSee SETUP-GUIDE.md for installation instructions")
    logger.error("=" * 80)

async def validate_required_services():
    """
    Validate that all required services are available and healthy.
    Fail fast at startup with clear error messages.
    """
    logger.info("=" * 80)
    logger.info("🔍 STARTUP VALIDATION: Checking all required services...")
    logger.info("=" * 80)
    
    # Run all validations
    validations = {
        "mongodb": await _validate_mongodb(),
        "qdrant": await _validate_qdrant(),
        "redis": await _validate_redis(),
        "embedding_model": _validate_embedding_model(),
        "llm_provider": await _validate_llm_provider(),
        "spacy_model": _validate_spacy_model()
    }
    
    # Collect errors
    errors = [f"{name}: {msg}" for name, (success, msg) in validations.items() if not success]
    
    # Report results
    logger.info("=" * 80)
    logger.info("STARTUP VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    if errors:
        _log_validation_errors(errors)
        raise RuntimeError("Required services validation failed. See logs above.")
    
    logger.info("✅ All required services validated successfully!")
    logger.info("=" * 80)


# ========== LIFESPAN PHASE HELPERS (Complexity Reduction) ==========

def _log_configuration():
    """Log application configuration."""
    logger.info("\nPhase 1: Configuration Check")
    logger.info("-" * 80)
    logger.info(f"  APP_ENV: {settings.app_env}")
    logger.info(f"  DEBUG_MODE: {settings.debug_mode}")
    logger.info(f"  MONGODB_URI configured: {bool(settings.mongodb_uri)}")
    logger.info(f"  QDRANT_URL: {settings.qdrant_url if settings.qdrant_url else 'NOT SET'}")
    logger.info(f"  REDIS_URL configured: {bool(settings.redis_url)}")
    logger.info(f"  LMSTUDIO_API_URL: {settings.lmstudio_api_url}")

async def _safe_connect(client_connect_coro, name: str, timeout_seconds: float = 5.0) -> bool:
    """Safely connect to a service with timeout."""
    try:
        await asyncio.wait_for(client_connect_coro, timeout=timeout_seconds)
        logger.info(f"  ✓ Connected: {name}")
        return True
    except asyncio.TimeoutError:
        logger.warning(f"  ⚠️ {name} connection timeout (will retry later)")
        return False
    except Exception as e:
        logger.warning(f"  ⚠️ {name} connection error: {type(e).__name__} (will retry later)")
        return False

async def _connect_databases_dev():
    """Connect to databases in development mode (longer timeouts)."""
    connect_tasks = [
        _safe_connect(mongodb_client.connect(), "MongoDB", timeout_seconds=30.0),
        _safe_connect(qdrant_client.connect(), "Qdrant", timeout_seconds=30.0),
        _safe_connect(redis_client.connect(), "Redis", timeout_seconds=20.0)
    ]
    results = await asyncio.gather(*connect_tasks)
    logger.info(f"  Connection attempts: MongoDB={results[0]}, Qdrant={results[1]}, Redis={results[2]}")
    
    # Try Neo4j (optional)
    try:
        from app.core.db_neo4j import neo4j_client
        neo4j_result = await _safe_connect(neo4j_client.connect(), "Neo4j", timeout_seconds=10.0)
        logger.info(f"  Neo4j connection: {neo4j_result}")
    except Exception as e:
        logger.warning(f"  Neo4j connection failed (optional): {e}")

async def _connect_databases_prod():
    """Connect to databases in production mode (strict timeouts)."""
    connect_tasks = [
        _safe_connect(mongodb_client.connect(), "MongoDB", timeout_seconds=8.0),
        _safe_connect(qdrant_client.connect(), "Qdrant", timeout_seconds=6.0),
        _safe_connect(redis_client.connect(), "Redis", timeout_seconds=4.0)
    ]
    results = await asyncio.gather(*connect_tasks)
    
    # Try Neo4j (optional)
    try:
        from app.core.db_neo4j import neo4j_client
        await _safe_connect(neo4j_client.connect(), "Neo4j", timeout_seconds=5.0)
    except Exception:
        pass  # Neo4j is optional
    
    if not all(results):
        raise RuntimeError("One or more critical services failed to connect")

async def _run_database_migrations():
    """Run database migrations."""
    logger.info("\nPhase 4: Database Migrations")
    logger.info("-" * 80)
    try:
        from .migrations.media_schema_migrations import run_all_migrations
        await run_all_migrations(mongodb_client.database)
        logger.info("  ✓ Database migrations completed successfully")
    except Exception as e:
        logger.error(f"  ERROR: Database migrations failed: {e}")
        raise

async def _initialize_media_services():
    """Initialize media services."""
    logger.info("\nPhase 5: Service Initialization")
    logger.info("-" * 80)
    try:
        from .services.media_suggestion_service import media_suggestion_service
        await media_suggestion_service.initialize()
        logger.info("  ✓ Media suggestion service initialized")
    except Exception as e:
        logger.error(f"  ERROR: Failed to initialize media suggestion service: {e}")
        raise

def _configure_background_workers(app: FastAPI):
    """Configure background workers for lazy initialization."""
    logger.info("\nPhase 6: Background Workers Configuration")
    logger.info("-" * 80)
    app.state.ingest_worker_task = None
    app.state.ingest_worker_started = False
    logger.info("  ✓ Background workers configured for lazy initialization")

async def _warmup_services():
    """Warmup services to prevent first-request delays."""
    logger.info("\nPhase 7: Service Warmup")
    logger.info("-" * 80)
    
    try:
        logger.info("🔥 Warming up chat service...")
        from .services.chat_service import ChatService
        
        warmup_service = ChatService()
        await warmup_service.initialize()
        
        logger.info("  ✅ Chat service warmed up and ready!")
        logger.info("  First query will respond in <10 seconds instead of 60+ seconds")
    except Exception as e:
        logger.warning(f"  ⚠️  Chat service warmup failed: {e}")
        logger.warning("  Services will initialize on first request (30-60s delay)")

async def _shutdown_services():
    """Shutdown all services gracefully."""
    logger.info("=" * 80)
    logger.info("SHUTDOWN: Disconnecting from all services...")
    logger.info("=" * 80)
    
    disconnect_tasks = []
    
    # MongoDB
    if mongodb_client.client:
        logger.info("Disconnecting from MongoDB...")
        disconnect_tasks.append(mongodb_client.disconnect())
    
    # Qdrant
    if qdrant_client.client:
        logger.info("Disconnecting from Qdrant...")
        disconnect_tasks.append(qdrant_client.disconnect())
    
    # Redis
    if redis_client.client:
        logger.info("Disconnecting from Redis...")
        disconnect_tasks.append(redis_client.disconnect())
    
    # Neo4j (if available)
    try:
        from app.core.db_neo4j import neo4j_client
        if neo4j_client.driver:
            logger.info("Disconnecting from Neo4j...")
            disconnect_tasks.append(neo4j_client.disconnect())
    except Exception:
        pass
    
    if disconnect_tasks:
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
    
    logger.info("✅ All services disconnected successfully")
    logger.info("=" * 80)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager - orchestrates startup and shutdown."""
    # Startup
    logger.info("=" * 80)
    logger.info("🚀 APPLICATION STARTUP - OMS Chat Bot RAG System")
    logger.info("=" * 80)
    
    try:
        # Phase 1: Configuration
        _log_configuration()
        
        # Phase 2: Service Validation (production only)
        logger.info("\nPhase 2: Service Validation")
        logger.info("-" * 80)
        if settings.app_env == "production":
            logger.info("Production mode: Validating all services...")
            await validate_required_services()
        else:
            logger.info("Development mode: Skipping validation (graceful degradation enabled)")
        
        # Phase 3: Database Connections
        logger.info("\nPhase 3: Connecting to Services")
        logger.info("-" * 80)
        if settings.app_env == "development":
            await _connect_databases_dev()
        else:
            await _connect_databases_prod()
        logger.info("  ✓ Database connection phase completed")
        
        # Phase 4: Migrations
        await _run_database_migrations()
        
        # Phase 5: Initialize Services
        await _initialize_media_services()
        
        # Phase 6: Background Workers
        _configure_background_workers(app)
        
        # Phase 7: Warmup
        await _warmup_services()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ APPLICATION STARTUP COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}", exc_info=True)
        raise
    
    # Shutdown
    await _shutdown_services()


# Create FastAPI application
app = FastAPI(
    title="RAG Graph Chatbot API",
    description="A production-ready RAG + Knowledge Graph chatbot with real-time admin dashboard",
    version="1.0.0",
    docs_url="/docs" if settings.debug_mode else None,
    redoc_url="/redoc" if settings.debug_mode else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        settings.next_public_api_base
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include API routes
app.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

app.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"]
)

app.include_router(
    admin.router,
    prefix="/admin",
    tags=["Administration"]
)

app.include_router(
    feedback.router,
    prefix="/feedback",
    tags=["Feedback"]
)

app.include_router(
    monitoring.router,
    prefix="/monitoring",
    tags=["System Monitoring"]
)

app.include_router(
    websocket.router,
    prefix="",
    tags=["WebSocket"]
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint - liveness probe (app is running)."""
    try:
        mongo_status = "connected" if mongodb_client.is_connected() else "disconnected"
        qdrant_status = "connected" if qdrant_client.is_connected() else "disconnected"
        redis_status = "connected" if redis_client.is_connected() else "disconnected"
        
        databases = {
            "mongodb": mongo_status,
            "qdrant": qdrant_status,
            "redis": redis_status
        }
        
        logger.debug(f"Health check: {databases}")
        
        return {
            "status": "alive",
            "environment": settings.app_env,
            "version": "1.0.0",
            "databases": databases,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/ready")
async def readiness_check():
    """Readiness probe - returns 200 only if critical services available."""
    try:
        critical_services_ok = mongodb_client.is_connected() and redis_client.is_connected()
        
        databases = {
            "mongodb": "ok" if mongodb_client.is_connected() else "down",
            "qdrant": "ok" if qdrant_client.is_connected() else "down",
            "redis": "ok" if redis_client.is_connected() else "down"
        }
        
        if not critical_services_ok:
            logger.warning(f"Not ready - critical services down: {databases}")
            raise HTTPException(status_code=503, detail="Critical services unavailable")
        
        logger.debug("Readiness check passed")
        return {
            "status": "ready",
            "databases": databases,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Readiness check failed")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Graph Chatbot API",
        "version": "1.0.0",
        "docs_url": "/docs" if settings.debug_mode else None,
        "endpoints": {
            "health": "/health",
            "authentication": "/auth",
            "chat": "/chat",
            "admin": "/admin",
            "feedback": "/feedback"
        }
    }

# System info endpoint (debug mode only)
@app.get("/system/info")
async def system_info():
    """Get system information (debug mode only)."""
    if not settings.debug_mode:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        # Get database statistics
        mongo_info = {}
        qdrant_info = {}
        redis_info = {}
        neo4j_info = {}
        
        try:
            # MongoDB info
            if mongodb_client.client:
                stats = await mongodb_client.database.command("dbStats")
                mongo_info = {
                    "database": settings.mongodb_db,
                    "collections": stats.get("collections", 0),
                    "objects": stats.get("objects", 0),
                    "dataSize": stats.get("dataSize", 0)
                }
        except Exception:
            pass
        
        try:
            # Qdrant info
            if qdrant_client.client:
                qdrant_info = await qdrant_client.get_collection_info()
        except Exception:
            pass
        
        try:
            # Redis info
            if redis_client.client:
                redis_info = await redis_client.get_redis_info()
        except Exception:
            pass
        
        try:
            # Neo4j info
            from app.core.db_neo4j import neo4j_client
            if neo4j_client.is_connected():
                neo4j_info = await neo4j_client.get_graph_statistics()
        except Exception:
            pass
        
        return {
            "system": {
                "environment": settings.app_env,
                "debug_mode": settings.debug_mode,
                "host": settings.app_host,
                "port": settings.app_port
            },
            "configuration": {
                "embedding_model": settings.embedding_model_name,
                "embedding_dimension": settings.embedding_dimension,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k_retrieval": settings.top_k_retrieval,
                "use_reranker": settings.use_reranker,
                "use_graph_search": settings.use_graph_search
            },
            "databases": {
                "mongodb": mongo_info,
                "qdrant": qdrant_info,
                "redis": redis_info,
                "neo4j": neo4j_info
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system information")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "status_code": 404
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Custom 500 handler."""
    from fastapi.responses import JSONResponse
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "status_code": 500
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug_mode,
        log_level="info" if settings.debug_mode else "warning"
    )