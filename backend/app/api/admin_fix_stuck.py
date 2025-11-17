"""
Admin endpoints for fixing stuck documents and system recovery.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
from datetime import datetime, timezone, timedelta

from ..core.db_mongo import get_mongodb_client, MongoDBClient

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/documents/reset-stuck")
async def reset_stuck_documents(
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    Reset all documents stuck in PROCESSING status for more than 10 minutes.
    This fixes documents that failed without proper error handling.
    """
    try:
        # Find documents stuck in PROCESSING for more than 10 minutes
        ten_minutes_ago = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        
        stuck_docs = []
        cursor = mongo_client.database.documents.find({
            "processing_status": "PROCESSING",
            "updated_at": {"$lt": ten_minutes_ago}
        })
        
        async for doc in cursor:
            stuck_docs.append({
                "id": str(doc["_id"]),
                "filename": doc.get("filename", "unknown"),
                "updated_at": doc.get("updated_at", "")
            })
        
        if not stuck_docs:
            return {
                "message": "No stuck documents found",
                "reset_count": 0,
                "documents": []
            }
        
        # Reset all stuck documents to PENDING
        result = await mongo_client.database.documents.update_many(
            {
                "processing_status": "PROCESSING",
                "updated_at": {"$lt": ten_minutes_ago}
            },
            {
                "$set": {
                    "processing_status": "FAILED",
                    "error_message": "Processing timeout - document was stuck for more than 10 minutes",
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        logger.info(f"✅ Reset {result.modified_count} stuck documents to FAILED status")
        
        return {
            "message": f"Reset {result.modified_count} stuck documents",
            "reset_count": result.modified_count,
            "documents": stuck_docs
        }
        
    except Exception as e:
        logger.error(f"Failed to reset stuck documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/check-stuck")
async def check_stuck_documents(
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    Check for documents stuck in PROCESSING status.
    Returns list without modifying anything.
    """
    try:
        # Find documents stuck in PROCESSING for more than 5 minutes
        five_minutes_ago = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        
        stuck_docs = []
        cursor = mongo_client.database.documents.find({
            "processing_status": "PROCESSING",
            "updated_at": {"$lt": five_minutes_ago}
        })
        
        async for doc in cursor:
            stuck_docs.append({
                "id": str(doc["_id"]),
                "filename": doc.get("filename", "unknown"),
                "updated_at": doc.get("updated_at", ""),
                "size": doc.get("size", 0),
                "content_type": doc.get("content_type", "unknown")
            })
        
        return {
            "stuck_count": len(stuck_docs),
            "documents": stuck_docs,
            "message": f"Found {len(stuck_docs)} documents stuck in processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to check stuck documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/delete-failed")
async def delete_all_failed(
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    Delete all documents with FAILED status to clean up the system.
    """
    try:
        # Get count first
        count = await mongo_client.database.documents.count_documents({
            "processing_status": "FAILED"
        })
        
        if count == 0:
            return {
                "message": "No failed documents to delete",
                "deleted_count": 0
            }
        
        # Delete all failed documents
        result = await mongo_client.database.documents.delete_many({
            "processing_status": "FAILED"
        })
        
        logger.info(f"✅ Deleted {result.deleted_count} failed documents")
        
        return {
            "message": f"Deleted {result.deleted_count} failed documents",
            "deleted_count": result.deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to delete failed documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
