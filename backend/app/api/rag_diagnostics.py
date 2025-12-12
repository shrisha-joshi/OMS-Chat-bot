"""
RAG Pipeline Diagnostics Endpoint
Provides deep visibility into RAG components: Qdrant, BM25, Embeddings, Documents
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime

from ..core.db_qdrant import get_qdrant_client, QdrantDBClient
from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..core.model_manager import get_model_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/rag/status")
async def get_rag_pipeline_status(
    qdrant_client: QdrantDBClient = Depends(get_qdrant_client),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
) -> Dict[str, Any]:
    """
    Deep diagnostic check of RAG pipeline components.
    Returns detailed status of:
    - Qdrant vector store (collection, vector count, indexed count)
    - MongoDB documents (total, by status)
    - Embedding model (loaded, model name)
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "qdrant": {},
        "mongodb": {},
        "embeddings": {},
        "overall_health": "unknown"
    }
    
    issues = []
    
    # ===== QDRANT CHECK =====
    try:
        if not qdrant_client.is_connected():
            await qdrant_client.connect()
        
        # Get collection info
        from qdrant_client import models
        collection_name = "documents"  # From config
        
        try:
            collection_info = qdrant_client.client.get_collection(collection_name)
            result["qdrant"] = {
                "status": "connected",
                "collection_exists": True,
                "collection_name": collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count if hasattr(collection_info, 'indexed_vectors_count') else collection_info.vectors_count,
                "points_count": collection_info.points_count if hasattr(collection_info, 'points_count') else collection_info.vectors_count,
                "vector_size": collection_info.config.params.vectors.size if hasattr(collection_info.config.params, 'vectors') else 768,
                "distance": str(collection_info.config.params.vectors.distance) if hasattr(collection_info.config.params, 'vectors') else "unknown"
            }
            
            if result["qdrant"]["vectors_count"] == 0:
                issues.append("Qdrant has 0 vectors - no documents indexed")
        except Exception as e:
            result["qdrant"] = {
                "status": "connected",
                "collection_exists": False,
                "collection_name": collection_name,
                "error": str(e)
            }
            issues.append(f"Qdrant collection '{collection_name}' does not exist")
    except Exception as e:
        result["qdrant"] = {
            "status": "error",
            "error": str(e)
        }
        issues.append(f"Qdrant connection failed: {str(e)}")
    
    # ===== MONGODB CHECK =====
    try:
        if not mongo_client.is_connected():
            await mongo_client.connect()
        
        # Count documents by status
        from motor.motor_asyncio import AsyncIOMotorCollection
        collection: AsyncIOMotorCollection = mongo_client.db["documents"]
        
        total_docs = await collection.count_documents({})
        
        # Group by status
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        status_counts = {}
        async for doc in collection.aggregate(pipeline):
            status_key = doc["_id"] if doc["_id"] else "UNKNOWN"
            status_counts[status_key] = doc["count"]
        
        result["mongodb"] = {
            "status": "connected",
            "total_documents": total_docs,
            "by_status": status_counts
        }
        
        if total_docs == 0:
            issues.append("MongoDB has 0 documents")
        elif status_counts.get("COMPLETED", 0) == 0:
            issues.append(f"MongoDB has {total_docs} documents but 0 COMPLETED")
    except Exception as e:
        result["mongodb"] = {
            "status": "error",
            "error": str(e)
        }
        issues.append(f"MongoDB check failed: {str(e)}")
    
    # ===== EMBEDDING MODEL CHECK =====
    try:
        model_manager = await get_model_manager()
        embed_model = model_manager.get_embedding_model()
        
        if embed_model:
            result["embeddings"] = {
                "status": "loaded",
                "model_loaded": True,
                "model_name": "all-MiniLM-L6-v2", # Hardcoded for now or get from config
                "embedding_dimension": 384 # Hardcoded for now
            }
        else:
            result["embeddings"] = {
                "status": "not_available",
                "error": "Embedding model not initialized"
            }
            issues.append("Embedding model not loaded")
    except Exception as e:
        result["embeddings"] = {
            "status": "error",
            "error": str(e)
        }
        issues.append(f"Embedding check failed: {str(e)}")
    
    # ===== OVERALL HEALTH =====
    if not issues:
        result["overall_health"] = "healthy"
    elif len(issues) <= 2:
        result["overall_health"] = "degraded"
    else:
        result["overall_health"] = "critical"
    
    result["issues"] = issues
    result["issues_count"] = len(issues)
    
    return result


@router.post("/rag/rebuild-bm25")
async def rebuild_bm25_index(
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
) -> Dict[str, Any]:
    """
    Rebuild BM25 index - Not implemented in current architecture.
    """
    return {
        "success": False,
        "error": "BM25 is not used in the current architecture (Vector Search + Reranking only)",
        "documents_indexed": 0
    }
