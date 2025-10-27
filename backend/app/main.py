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
from .core.db_arango import arango_client
from .core.cache_redis import redis_client
from .api import chat, admin, auth, feedback
from .workers.ingest_worker import start_ingest_worker

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.debug_mode else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("=" * 80)
    logger.info("LIFESPAN: Starting RAG Graph Chatbot application...")
    logger.info("=" * 80)
    
    try:
        # Log effective configuration (sanitize secrets)
        import os
        logger.info("=" * 80)
        logger.info("CONFIGURATION CHECK:")
        logger.info(f"  APP_ENV: {settings.app_env}")
        logger.info(f"  DEBUG_MODE: {settings.debug_mode}")
        logger.info(f"  MONGODB_URI configured: {bool(settings.mongodb_uri)}")
        logger.info(f"  QDRANT_URL: {settings.qdrant_url if settings.qdrant_url else 'NOT SET'}")
        logger.info(f"  REDIS_URL configured: {bool(settings.redis_url)}")
        logger.info(f"  ARANGODB_URL configured: {bool(settings.arangodb_url)}")
        logger.info(f"  LMSTUDIO_API_URL: {settings.lmstudio_api_url}")
        logger.info(f"  USE_GRAPH_SEARCH: {settings.use_graph_search}")
        logger.info(f"  USE_RERANKER: {settings.use_reranker}")
        logger.info(f"  CHUNK_SIZE: {settings.chunk_size}")
        logger.info(f"  TOP_K_RETRIEVAL: {settings.top_k_retrieval}")
        logger.info(f"  MAX_LLM_OUTPUT_TOKENS: {settings.max_llm_output_tokens}")
        logger.info("=" * 80)
        
        # Connect to all databases (continue even if some fail)
        logger.info("LIFESPAN: Connecting to databases...")
        
        # Try to connect to databases concurrently with per-service timeouts
        logger.info("LIFESPAN: Attempting database connections concurrently (short timeouts)...")

        async def _safe_connect(client_connect_coro, name: str, timeout: float = 5.0):
            try:
                result = await asyncio.wait_for(client_connect_coro, timeout=timeout)
                logger.info(f"LIFESPAN: {name} connection completed")
                return result
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ {name} connection timed out after {timeout}s - continuing without {name}")
                return False
            except Exception as e:
                logger.warning(f"⚠️ {name} connection failed: {e}")
                return False

        # Schedule concurrent connections
        connect_tasks = [
            _safe_connect(mongodb_client.connect(), "MongoDB", timeout=8.0),
            _safe_connect(qdrant_client.connect(), "Qdrant", timeout=6.0),
            _safe_connect(arango_client.connect(), "ArangoDB", timeout=4.0),
            _safe_connect(redis_client.connect(), "Redis", timeout=4.0)
        ]

        mongo_success, qdrant_success, arango_success, redis_success = await asyncio.gather(*connect_tasks)

        if not mongo_success:
            logger.warning("⚠️ MongoDB connection failed - continuing without MongoDB")
        if not qdrant_success:
            logger.warning("⚠️ Qdrant connection failed - continuing without Qdrant")
        if not arango_success:
            logger.warning("⚠️ ArangoDB connection failed - continuing without ArangoDB")
        if not redis_success:
            logger.warning("⚠️ Redis connection failed - continuing without Redis")
        
        # Start background workers (lazy initialization on first use)
        logger.info("LIFESPAN: Background workers configured for lazy initialization")
        app.state.ingest_worker_task = None
        app.state.ingest_worker_started = False
        
        # Schedule a background warmup to initialize heavy services (embeddings, LLMs)
        async def _warmup_chat_service():
            try:
                logger.info("LIFESPAN: Starting background warmup for chat service (non-blocking)")
                # Import here to avoid circular imports at module load
                from .api.chat import get_chat_service
                # Call dependency function to initialize and swallow exceptions
                await get_chat_service()
                logger.info("LIFESPAN: Chat service warmup completed")
            except Exception as e:
                logger.warning(f"LIFESPAN: Chat service warmup failed: {e}")

        # Fire-and-forget warmup task
        try:
            asyncio.create_task(_warmup_chat_service())
        except Exception:
            pass
        
        logger.info("=" * 80)
        logger.info("LIFESPAN: ✅ Application startup completed successfully")
        logger.info("=" * 80)
        logger.info("LIFESPAN: About to yield control to Uvicorn...")
        
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
        await arango_client.disconnect()
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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint - liveness probe (app is running)."""
    try:
        mongo_status = "connected" if mongodb_client.is_connected() else "disconnected"
        qdrant_status = "connected" if qdrant_client.is_connected() else "disconnected"
        arango_status = "connected" if arango_client.is_connected() else "disconnected"
        redis_status = "connected" if redis_client.is_connected() else "disconnected"
        
        status_code = 200
        databases = {
            "mongodb": mongo_status,
            "qdrant": qdrant_status,
            "arangodb": arango_status,
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
            "redis": "ok" if redis_client.is_connected() else "down",
            "arangodb": "ok" if arango_client.is_connected() else "down"
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
            # ArangoDB info
            if arango_client.client:
                arango_info = await arango_client.get_graph_statistics()
        except Exception:
            pass
        
        try:
            # Redis info
            if redis_client.client:
                redis_info = await redis_client.get_redis_info()
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
                "arangodb": arango_info,
                "redis": redis_info
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