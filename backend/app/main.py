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
    logger.info("Starting RAG Graph Chatbot application...")
    
    try:
        # Connect to all databases
        logger.info("Connecting to databases...")
        await mongodb_client.connect()
        await qdrant_client.connect()
        await arango_client.connect()
        await redis_client.connect()
        
        # Start background workers
        logger.info("Starting background workers...")
        ingest_worker_task = asyncio.create_task(start_ingest_worker())
        
        logger.info("✅ Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down RAG Graph Chatbot application...")
    
    try:
        # Cancel background tasks
        if 'ingest_worker_task' in locals():
            ingest_worker_task.cancel()
            try:
                await ingest_worker_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect from databases
        await mongodb_client.disconnect()
        await qdrant_client.disconnect()
        await arango_client.disconnect()
        await redis_client.disconnect()
        
        logger.info("✅ Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Application shutdown failed: {e}")

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", settings.next_public_api_base],
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
    """Health check endpoint for monitoring."""
    try:
        # Check database connections
        mongo_status = "connected" if mongodb_client.client else "disconnected"
        qdrant_status = "connected" if qdrant_client.client else "disconnected"
        arango_status = "connected" if arango_client.client else "disconnected"
        redis_status = "connected" if redis_client.client else "disconnected"
        
        return {
            "status": "healthy",
            "environment": settings.app_env,
            "version": "1.0.0",
            "databases": {
                "mongodb": mongo_status,
                "qdrant": qdrant_status,
                "arangodb": arango_status,
                "redis": redis_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

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
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "status_code": 404
    }

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug_mode,
        log_level="info" if settings.debug_mode else "warning"
    )