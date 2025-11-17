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
from .api import admin_fix_stuck  # Add stuck document recovery endpoints
from .api import rag_diagnostics  # Add RAG pipeline diagnostics
from .workers.ingest_worker import start_ingest_worker
from .middleware import RateLimitMiddleware  # Rate limiting
from .utils.metrics import metrics_collector, PerformanceTimer  # Performance monitoring

# Import enhanced services
from .services.websocket_manager import ws_manager
from .services.health_monitor import health_monitor
from .services.batch_processor import batch_processor

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
        llm_url = settings.lmstudio_api_url or "http://localhost:1234/v1" or "http://192.168.56.1:1234"
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
    
    # Skip migrations if MongoDB is not available
    if mongodb_client is None or mongodb_client.database is None:
        logger.warning("  ⚠️  MongoDB not available, skipping migrations (will retry on reconnect)")
        return
    
    try:
        from .migrations.media_schema_migrations import run_all_migrations
        await run_all_migrations(mongodb_client.database)
        logger.info("  ✓ Database migrations completed successfully")
    except Exception as e:
        logger.warning(f"  ⚠️  Database migrations failed: {e} (non-fatal, continuing startup)")
        # Don't raise - allow app to start even if migrations fail

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
    logger.info("\nPhase 7: Service Warmup & Enhanced Features Initialization")
    logger.info("-" * 80)
    
    # Start WebSocket Manager
    try:
        logger.info("🔗 Starting Enhanced WebSocket Manager...")
        await ws_manager.start_background_tasks()
        logger.info("  ✅ WebSocket manager started with connection pooling & health monitoring")
    except Exception as e:
        logger.warning(f"  ⚠️  WebSocket manager startup failed: {e}")
    
    # Register services for health monitoring
    try:
        logger.info("💊 Registering services for health monitoring...")
        
        # Register MongoDB
        health_monitor.register_service(
            "mongodb",
            health_check=lambda: asyncio.to_thread(lambda: mongodb_client.is_connected()),
            reconnect_handler=mongodb_client.connect
        )
        
        # Register Qdrant
        health_monitor.register_service(
            "qdrant",
            health_check=lambda: asyncio.to_thread(lambda: qdrant_client.is_connected()),
            reconnect_handler=qdrant_client.connect
        )
        
        # Register Redis
        health_monitor.register_service(
            "redis",
            health_check=lambda: asyncio.to_thread(lambda: redis_client.is_connected()),
            reconnect_handler=redis_client.connect
        )
        
        # Start health monitor
        await health_monitor.start()
        logger.info("  ✅ Health monitor started (30s check interval)")
    except Exception as e:
        logger.warning(f"  ⚠️  Health monitor startup failed: {e}")
    
    # Warmup chat service
    try:
        logger.info("🔥 Warming up chat service with research enhancements...")
        from .services.chat_service import ChatService
        
        warmup_service = ChatService()
        await warmup_service.initialize()
        
        logger.info("  ✅ Chat service warmed up with Context Window Optimizer & COS-Mix!")
        logger.info("  First query will respond in <10 seconds with enhanced retrieval")
    except Exception as e:
        logger.warning(f"  ⚠️  Chat service warmup failed: {e}")
        logger.warning("  Services will initialize on first request (30-60s delay)")
    
    # Log batch processor info
    try:
        logger.info("📦 Batch processor initialized:")
        metrics = batch_processor.get_metrics()
        logger.info(f"  Batch size: {metrics['batch_size']}, Max concurrent: {metrics['max_concurrent_batches']}")
        logger.info("  ✅ Ready for 10x faster embedding generation")
    except Exception as e:
        logger.warning(f"  ⚠️  Batch processor check failed: {e}")

async def _shutdown_services():
    """Shutdown all services gracefully."""
    logger.info("=" * 80)
    logger.info("SHUTDOWN: Disconnecting from all services...")
    logger.info("=" * 80)
    
    disconnect_tasks = []
    
    # Stop enhanced services first
    try:
        logger.info("Stopping WebSocket manager...")
        await ws_manager.shutdown()
        logger.info("  ✅ WebSocket manager stopped")
    except Exception as e:
        logger.error(f"Error stopping WebSocket manager: {e}")
    
    try:
        logger.info("Stopping health monitor...")
        await health_monitor.stop()
        logger.info("  ✅ Health monitor stopped")
    except Exception as e:
        logger.error(f"Error stopping health monitor: {e}")
    
    # MongoDB (synchronous disconnect)
    if mongodb_client.client:
        logger.info("Disconnecting from MongoDB...")
        try:
            mongodb_client.disconnect()
        except (Exception, asyncio.CancelledError) as e:
            logger.error(f"Error disconnecting MongoDB: {e}")
    
    # Qdrant (async disconnect)
    if qdrant_client.client:
        logger.info("Disconnecting from Qdrant...")
        disconnect_tasks.append(qdrant_client.disconnect())
    
    # Redis (async disconnect)
    if redis_client.client:
        logger.info("Disconnecting from Redis...")
        disconnect_tasks.append(redis_client.disconnect())
    
    # Neo4j (async disconnect, if available)
    try:
        from app.core.db_neo4j import neo4j_client
        if neo4j_client.driver:
            logger.info("Disconnecting from Neo4j...")
            disconnect_tasks.append(neo4j_client.disconnect())
    except Exception:
        pass
    
    if disconnect_tasks:
        try:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("Disconnection tasks cancelled, forcing cleanup...")
    
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
        
    except asyncio.CancelledError:
        # Python 3.13 compatibility: Handle graceful cancellation during shutdown
        logger.info("Application shutdown initiated (CancelledError caught)")
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}", exc_info=True)
        raise
    finally:
        # Shutdown - always runs even if cancelled
        try:
            await _shutdown_services()
        except asyncio.CancelledError:
            # Suppress cancellation during shutdown
            logger.info("Shutdown completed (cancelled during cleanup)")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="RAG Graph Chatbot API",
    description="A production-ready RAG + Knowledge Graph chatbot with real-time admin dashboard",
    version="1.0.0",
    docs_url="/docs" if settings.debug_mode else None,
    redoc_url="/redoc" if settings.debug_mode else None,
    lifespan=lifespan
)

# Add middleware with proper CORS for WebSocket + HTTP
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://0.0.0.0:3000",
        "http://0.0.0.0:8000",
        "ws://localhost:3000",
        "ws://127.0.0.1:3000",
        "ws://localhost:8000",
        "ws://127.0.0.1:8000",
        settings.next_public_api_base
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting middleware
if settings.rate_limit_enabled:
    app.add_middleware(RateLimitMiddleware)
    logger.info("✅ Rate limiting enabled")

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
    admin_fix_stuck.router,
    prefix="/admin",
    tags=["Administration - Recovery"]
)

app.include_router(
    rag_diagnostics.router,
    prefix="/admin",
    tags=["RAG Diagnostics"]
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
    """Health check endpoint - liveness probe with active connection testing."""
    try:
        # Test MongoDB actively
        mongo_healthy = False
        try:
            if mongodb_client.client:
                await asyncio.wait_for(mongodb_client.client.admin.command('ping'), timeout=2.0)
                mongo_healthy = True
        except Exception:
            pass
        
        # Test Qdrant actively
        qdrant_healthy = False
        try:
            if qdrant_client.client:
                await asyncio.wait_for(asyncio.to_thread(qdrant_client.client.get_collections), timeout=2.0)
                qdrant_healthy = True
        except Exception:
            pass
        
        # Test Redis actively
        redis_healthy = False
        try:
            if redis_client.client:
                await asyncio.wait_for(redis_client.client.ping(), timeout=2.0)
                redis_healthy = True
        except Exception:
            pass
        
        databases = {
            "mongodb": "connected" if mongo_healthy else "disconnected",
            "qdrant": "connected" if qdrant_healthy else "disconnected",
            "redis": "connected" if redis_healthy else "disconnected"
        }
        
        # Determine overall status
        critical_healthy = mongo_healthy and qdrant_healthy
        status = "alive" if critical_healthy else "degraded"
        
        response = {
            "status": status,
            "environment": settings.app_env,
            "version": "1.0.0",
            "databases": databases,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if not critical_healthy:
            response["message"] = "Some critical services are unavailable"
        
        logger.debug(f"Health check: {databases}")
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Metrics endpoint for monitoring
@app.get("/metrics")
async def get_metrics():
    """Get application performance metrics."""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics_collector.get_summary()
    }

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