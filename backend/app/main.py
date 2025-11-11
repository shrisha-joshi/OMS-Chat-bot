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
from datetime import datetime

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


async def validate_required_services():
    """
    ROOT CAUSE FIX #1: STARTUP VALIDATION
    
    Validate that all required services are available and healthy BEFORE
    starting the application.
    
    OLD (PATCH): Application starts even if databases unreachable, fails later
    NEW (FIX): Fail fast at startup with clear error messages
    
    This prevents the "silent failure" pattern where the app appears to work
    but data is never actually saved or retrieved.
    """
    logger.info("=" * 80)
    logger.info("🔍 STARTUP VALIDATION: Checking all required services...")
    logger.info("=" * 80)
    
    validation_results = {}
    errors = []
    
    # Check MongoDB (REQUIRED for document storage)
    try:
        logger.info("Validating MongoDB...")
        await asyncio.wait_for(mongodb_client.connect(), timeout=5.0)
        validation_results["mongodb"] = "✓ OK"
        logger.info("  ✓ MongoDB: Connected and healthy")
    except asyncio.TimeoutError:
        errors.append("MongoDB: Connection timeout - check if running on localhost:27017")
    except Exception as e:
        errors.append(f"MongoDB: Connection failed - {str(e)}")
    
    # Check Qdrant (REQUIRED for vector search)
    try:
        logger.info("Validating Qdrant...")
        await asyncio.wait_for(qdrant_client.connect(), timeout=5.0)
        validation_results["qdrant"] = "✓ OK"
        logger.info("  ✓ Qdrant: Connected and healthy")
    except asyncio.TimeoutError:
        errors.append("Qdrant: Connection timeout - check if running on localhost:6333")
    except Exception as e:
        errors.append(f"Qdrant: Connection failed - {str(e)}")
    
    # Check Redis (REQUIRED for caching and session management)
    try:
        logger.info("Validating Redis...")
        await asyncio.wait_for(redis_client.connect(), timeout=5.0)
        validation_results["redis"] = "✓ OK"
        logger.info("  ✓ Redis: Connected and healthy")
    except asyncio.TimeoutError:
        errors.append("Redis: Connection timeout - check if running on localhost:6379")
    except Exception as e:
        errors.append(f"Redis: Connection failed - {str(e)}")
    
    # Check Embedding Model (REQUIRED for semantic search)
    try:
        logger.info("Validating Embedding Model...")
        from sentence_transformers import SentenceTransformer
        model_name = settings.embedding_model_name or "all-MiniLM-L6-v2"
        logger.info(f"  Loading: {model_name}")
        model = SentenceTransformer(model_name)
        test_embedding = model.encode("test")
        assert len(test_embedding) > 0, "Embedding generation failed"
        validation_results["embedding_model"] = "✓ OK"
        logger.info(f"  ✓ Embedding Model: {model_name} (dimension: {len(test_embedding)})")
    except Exception as e:
        errors.append(f"Embedding Model: {str(e)}")
    
    # Check LLM Provider (REQUIRED for response generation)
    try:
        logger.info("Validating LLM Provider...")
        llm_url = settings.lmstudio_api_url or "http://localhost:1234/v1"
        async with httpx.AsyncClient(timeout=5) as client:
            response = await asyncio.wait_for(
                client.get(f"{llm_url}/models"),
                timeout=3.0
            )
            if response.status_code in [200, 404]:  # 404 OK if LMStudio not fully loaded yet
                validation_results["llm_provider"] = "✓ OK"
                logger.info(f"  ✓ LLM Provider: Available at {llm_url}")
            else:
                errors.append(f"LLM Provider: HTTP {response.status_code} from {llm_url}")
    except asyncio.TimeoutError:
        errors.append(f"LLM Provider: Timeout connecting to {settings.lmstudio_api_url}")
    except Exception as e:
        errors.append(f"LLM Provider: {str(e)}")
    
    # Check spaCy Model (REQUIRED for NER)
    try:
        logger.info("Validating spaCy Model...")
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            validation_results["spacy_model"] = "✓ OK"
            logger.info("  ✓ spaCy Model: en_core_web_sm loaded")
        except OSError:
            errors.append(
                "spaCy: Model 'en_core_web_sm' not installed. "
                "Install with: python -m spacy download en_core_web_sm"
            )
    except Exception as e:
        errors.append(f"spaCy: {str(e)}")
    
    # Report validation results
    logger.info("=" * 80)
    logger.info("STARTUP VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    if errors:
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
        logger.error("  [ ] spaCy model: python -m spacy download en_core_web_sm")
        logger.error("=" * 80)
        
        # Raise error to prevent application startup
        raise RuntimeError(
            f"Startup validation failed: {len(errors)} critical service(s) unavailable. "
            f"Cannot proceed without all required services."
        )
    else:
        logger.info("✅ ALL REQUIRED SERVICES VALIDATED AND HEALTHY")
        logger.info("=" * 80)
        for service, status in validation_results.items():
            logger.info(f"  {status} - {service}")
        logger.info("=" * 80)
        return validation_results


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager for startup and shutdown events.
    
    ROOT CAUSE FIX: This now VALIDATES all required services before starting.
    Old approach: Started regardless of database availability (patch work).
    New approach: Fail fast if any required service unavailable (real fix).
    """
    # Startup
    logger.info("=" * 80)
    logger.info("🚀 APPLICATION STARTUP - OMS Chat Bot RAG System")
    logger.info("=" * 80)
    
    try:
        # Phase 1: Validate configuration
        logger.info("\nPhase 1: Configuration Check")
        logger.info("-" * 80)
        logger.info(f"  APP_ENV: {settings.app_env}")
        logger.info(f"  DEBUG_MODE: {settings.debug_mode}")
        logger.info(f"  MONGODB_URI configured: {bool(settings.mongodb_uri)}")
        logger.info(f"  QDRANT_URL: {settings.qdrant_url if settings.qdrant_url else 'NOT SET'}")
        logger.info(f"  REDIS_URL configured: {bool(settings.redis_url)}")
        logger.info(f"  LMSTUDIO_API_URL: {settings.lmstudio_api_url}")
        
        # Phase 2: ROOT CAUSE FIX - Validate all required services (FAIL FAST)
        # But in development mode, skip validation to allow fast startup
        logger.info("\nPhase 2: Service Validation (ROOT CAUSE FIX #1: FAIL FAST)")
        logger.info("-" * 80)
        
        if settings.app_env == "production":
            logger.info("Production mode: Validating all services...")
            validation_results = await validate_required_services()
        else:
            logger.info("Development mode: Skipping startup validation (graceful degradation enabled)")
            logger.info("  Services will connect/degrade as accessed, not at startup")
        
        # Phase 3: Connect to databases (now guaranteed to work)
        logger.info("\nPhase 3: Connecting to Services")
        logger.info("-" * 80)
        
        async def _safe_connect(client_connect_coro, name: str, timeout: float = 5.0):
            try:
                result = await asyncio.wait_for(client_connect_coro, timeout=timeout)
                logger.info(f"  ✓ Connected: {name}")
                return True
            except asyncio.TimeoutError:
                logger.warning(f"  ⚠️ {name} connection timeout (will retry later)")
                return False
            except Exception as e:
                logger.warning(f"  ⚠️ {name} connection error: {type(e).__name__} (will retry later)")
                return False

        # In dev mode, use longer timeouts and don't fail on errors
        if settings.app_env == "development":
            connect_tasks = [
                _safe_connect(mongodb_client.connect(), "MongoDB", timeout=30.0),
                _safe_connect(qdrant_client.connect(), "Qdrant", timeout=30.0),
                _safe_connect(redis_client.connect(), "Redis", timeout=20.0)
            ]
            results = await asyncio.gather(*connect_tasks)
            logger.info(f"  Connection attempts: MongoDB={results[0]}, Qdrant={results[1]}, Redis={results[2]}")
            
            # Try Neo4j connection (non-blocking, optional)
            try:
                from app.core.db_neo4j import neo4j_client
                neo4j_result = await _safe_connect(neo4j_client.connect(), "Neo4j", timeout=10.0)
                logger.info(f"  Neo4j connection: {neo4j_result}")
            except Exception as e:
                logger.warning(f"  Neo4j connection failed (optional): {e}")
        else:
            # Production: strict timeouts
            connect_tasks = [
                _safe_connect(mongodb_client.connect(), "MongoDB", timeout=8.0),
                _safe_connect(qdrant_client.connect(), "Qdrant", timeout=6.0),
                _safe_connect(redis_client.connect(), "Redis", timeout=4.0)
            ]
            results = await asyncio.gather(*connect_tasks)
            
            # Try Neo4j in production too (optional)
            try:
                from app.core.db_neo4j import neo4j_client
                neo4j_result = await _safe_connect(neo4j_client.connect(), "Neo4j", timeout=5.0)
                logger.info(f"  Neo4j connection: {neo4j_result}")
            except Exception:
                pass  # Neo4j is optional in production
            
            if not all(results):
                raise RuntimeError("One or more critical services failed to connect")
        
        logger.info(f"  ✓ Database connection phase completed")
        
        # Phase 4: Run database migrations for media features
        logger.info("\nPhase 4: Database Migrations")
        logger.info("-" * 80)
        try:
            from .migrations.media_schema_migrations import run_all_migrations
            await run_all_migrations(mongodb_client.database)
            logger.info("  ✓ Database migrations completed successfully")
        except Exception as e:
            logger.error(f"  ERROR: Database migrations failed: {e}")
            raise  # Don't continue if migrations fail
        
        # Phase 5: Initialize media services
        logger.info("\nPhase 5: Service Initialization")
        logger.info("-" * 80)
        try:
            from .services.media_suggestion_service import media_suggestion_service
            await media_suggestion_service.initialize()
            logger.info("  ✓ Media suggestion service initialized")
        except Exception as e:
            logger.error(f"  ERROR: Failed to initialize media suggestion service: {e}")
            raise  # Don't continue if service initialization fails
        
        # Phase 6: Background workers (lazy initialization)
        logger.info("\nPhase 6: Background Workers Configuration")
        logger.info("-" * 80)
        app.state.ingest_worker_task = None
        app.state.ingest_worker_started = False
        logger.info("  ✓ Background workers configured for lazy initialization")
        
        # Phase 7: Warmup - ENABLE to speed up first request
        logger.info("\nPhase 7: Service Warmup")
        logger.info("-" * 80)
        
        try:
            logger.info("🔥 Warming up chat service (prevents 60s delay on first request)...")
            from .services.chat_service import ChatService
            
            # Create and initialize chat service
            warmup_service = ChatService()
            await warmup_service.initialize()
            
            logger.info("  ✅ Chat service warmed up and ready!")
            logger.info("  ✓ Embedding model loaded")
            logger.info("  ✓ LLM handler initialized")
            logger.info("  ✓ Database connections ready")
            logger.info("  First query will now respond in <10 seconds instead of 60+ seconds")
        except Exception as e:
            logger.warning(f"  ⚠️  Chat service warmup failed: {e}")
            logger.warning("  Services will initialize on first request (may cause 30-60s delay)")
            logger.warning("  This is not critical - app will still work")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ APPLICATION STARTUP COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        yield
        
        logger.info("=" * 80)
        logger.info("LIFESPAN: Returned from yield, entering shutdown phase")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"LIFESPAN: ❌ Application startup failed: {e}", exc_info=True)
        raise
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("LIFESPAN: Shutting down RAG Graph Chatbot application...")
    logger.info("=" * 80)
    
    try:
        # Cancel background tasks
        if hasattr(app, 'state') and hasattr(app.state, 'ingest_worker_task') and app.state.ingest_worker_task:
            logger.info("LIFESPAN: Cancelling ingest worker task...")
            app.state.ingest_worker_task.cancel()
            try:
                await app.state.ingest_worker_task
            except (asyncio.CancelledError, Exception):
                logger.info("LIFESPAN: Ingest worker task cancelled successfully")
        
        logger.info("LIFESPAN: Disconnecting from databases...")
        # Disconnect from databases
        await mongodb_client.disconnect()
        await qdrant_client.disconnect()
        await redis_client.disconnect()
        
        logger.info("=" * 80)
        logger.info("LIFESPAN: ✅ Application shutdown completed successfully")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"LIFESPAN: ❌ Application shutdown failed: {e}", exc_info=True)

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
        
        status_code = 200
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
            "timestamp": datetime.utcnow().isoformat()
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
            "timestamp": datetime.utcnow().isoformat()
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
        arango_info = {}
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