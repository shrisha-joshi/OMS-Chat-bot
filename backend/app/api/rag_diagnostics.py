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
from ..services.ingest_service import IngestService

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
    - BM25 index (initialized, document count)
    - Embedding model (loaded, model name)
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "qdrant": {},
        "mongodb": {},
        "bm25": {},
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
    
    # ===== BM25 INDEX CHECK =====
    try:
        # Instantiate IngestService to access BM25 handler
        ingest_service = IngestService()
        bm25_handler = ingest_service.bm25_handler
        
        if bm25_handler and hasattr(bm25_handler, 'index'):
            result["bm25"] = {
                "status": "ready" if bm25_handler.index else "not_initialized",
                "initialized": bm25_handler.index is not None,
                "document_count": len(bm25_handler.doc_ids) if hasattr(bm25_handler, 'doc_ids') and bm25_handler.doc_ids else 0,
                "corpus_size": len(bm25_handler.corpus) if hasattr(bm25_handler, 'corpus') and bm25_handler.corpus else 0
            }
            
            if not bm25_handler.index:
                issues.append("BM25 index not initialized")
            elif result["bm25"]["document_count"] == 0:
                issues.append("BM25 index has 0 documents")
        else:
            result["bm25"] = {
                "status": "not_available",
                "error": "BM25 handler not initialized in IngestService"
            }
            issues.append("BM25 handler not available")
    except Exception as e:
        result["bm25"] = {
            "status": "error",
            "error": str(e)
        }
        issues.append(f"BM25 check failed: {str(e)}")
    
    # ===== EMBEDDING MODEL CHECK =====
    try:
        # Access embedding handler from freshly instantiated IngestService
        embed_handler = ingest_service.embedding_handler
        
        if embed_handler and hasattr(embed_handler, 'model'):
            result["embeddings"] = {
                "status": "loaded" if embed_handler.model else "not_loaded",
                "model_loaded": embed_handler.model is not None,
                "model_name": str(embed_handler.model_name) if hasattr(embed_handler, 'model_name') else "unknown",
                "embedding_dimension": embed_handler.embedding_dim if hasattr(embed_handler, 'embedding_dim') else 768
            }
            
            if not embed_handler.model:
                issues.append("Embedding model not loaded")
        else:
            result["embeddings"] = {
                "status": "not_available",
                "error": "Embedding handler not initialized"
            }
            issues.append("Embedding handler not available")
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
    Rebuild BM25 index from all COMPLETED documents in MongoDB.
    """
    try:
        if not mongo_client.is_connected():
            await mongo_client.connect()
        
        from motor.motor_asyncio import AsyncIOMotorCollection
        collection: AsyncIOMotorCollection = mongo_client.db["documents"]
        
        # Get all completed documents
        completed_docs = []
        async for doc in collection.find({"status": "COMPLETED"}):
            if "chunks" in doc and doc["chunks"]:
                for chunk in doc["chunks"]:
                    completed_docs.append({
                        "doc_id": str(doc["_id"]),
                        "text": chunk.get("text", "")
                    })
        
        if not completed_docs:
            return {
                "success": False,
                "error": "No COMPLETED documents with chunks found",
                "documents_checked": await collection.count_documents({})
            }
        
        # Rebuild BM25 - instantiate IngestService
        ingest_service = IngestService()
        bm25_handler = ingest_service.bm25_handler
        if not bm25_handler:
            return {
                "success": False,
                "error": "BM25 handler not available"
            }
        
        # Prepare corpus
        corpus = [doc["text"] for doc in completed_docs]
        doc_ids = [doc["doc_id"] for doc in completed_docs]
        
        # Rebuild (assuming BM25Handler has a method to rebuild)
        # If not, we'd need to reinitialize it
        bm25_handler.corpus = corpus
        bm25_handler.doc_ids = doc_ids
        
        # Re-create index
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25_handler.index = BM25Okapi(tokenized_corpus)
        
        return {
            "success": True,
            "documents_indexed": len(completed_docs),
            "unique_doc_ids": len(set(doc_ids))
        }
    except Exception as e:
        logger.error(f"Failed to rebuild BM25: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
