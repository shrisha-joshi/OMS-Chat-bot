"""Admin router for document uploads and management.

Provides endpoints for document upload and processing.
These endpoints are intentionally public for local/dev use.
Add authentication and validation before using in production.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import base64
import logging
from datetime import datetime

from ..core.db_mongo import get_mongodb_client, MongoDBClient

logger = logging.getLogger(__name__)
router = APIRouter()


class UploadJSONRequest(BaseModel):
    filename: str
    content_base64: str
    content_type: Optional[str] = "application/octet-stream"


@router.post("/documents/upload")
async def upload_document_multipart(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    mongo_client: MongoDBClient = Depends(get_mongodb_client),
    redis_client = Depends(lambda: None)  # Optional Redis dependency
):
    """Accept a multipart/form-data file, save to MongoDB, and queue for processing."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        content = await file.read()
        size = len(content)
        
        if size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        if size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
        
        # Save to MongoDB GridFS
        doc_id = await mongo_client.save_document(
            filename=file.filename,
            content=content,
            content_type=file.content_type or "application/octet-stream",
            size=size
        )
        
        logger.info(f"✅ Document saved to MongoDB: {file.filename} ({size} bytes) - ID: {doc_id}")
        
        # Invalidate document list cache (documents may have changed)
        try:
            from ..core.cache_redis import get_redis_client
            redis_client = await get_redis_client()
            if redis_client.is_connected():
                # Clear the document list cache patterns
                await redis_client.clear_pattern("docs:list:*")
                logger.info(f"🔄 Cleared document list cache")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
        
        # Queue for background processing
        background_tasks.add_task(
            _process_document,
            doc_id,
            file.filename,
            content,
            mongo_client
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "size": size,
            "document_id": str(doc_id),
            "status": "processing"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/documents/upload-json")
async def upload_document_json(
    payload: UploadJSONRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """Accept JSON payload with base64-encoded content."""
    try:
        content = base64.b64decode(payload.content_base64)
        size = len(content)
        
        if size == 0:
            raise HTTPException(status_code=400, detail="Content is empty")
        
        # Save to MongoDB GridFS
        doc_id = await mongo_client.save_document(
            filename=payload.filename,
            content=content,
            content_type=payload.content_type or "application/octet-stream",
            size=size
        )
        
        logger.info(f"✅ JSON document saved to MongoDB: {payload.filename} ({size} bytes) - ID: {doc_id}")
        
        # Queue for background processing
        background_tasks.add_task(
            _process_document,
            doc_id,
            payload.filename,
            content,
            mongo_client
        )
        
        return {
            "success": True,
            "filename": payload.filename,
            "size": size,
            "document_id": str(doc_id),
            "status": "processing"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ JSON upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def _process_document(doc_id, filename, content, mongo_client):
    """Background task to process uploaded document with chunking and embeddings."""
    try:
        logger.info(f"🔄 Starting document processing: {filename} (doc_id: {doc_id})")
        
        # Lazy initialize ingest service (import here to avoid circular imports)
        from ..services.ingest_service import IngestService
        from ..core.db_qdrant import get_qdrant_client
        from ..core.db_arango import get_arango_client
        from ..core.cache_redis import get_redis_client
        
        # Create ingest service instance
        ingest_service = IngestService()
        await ingest_service.initialize()
        
        # Process the document (chunks, embeddings, Qdrant storage)
        success = await ingest_service.process_document(str(doc_id))
        
        if success:
            logger.info(f"✅ Document processing completed: {filename} (doc_id: {doc_id})")
        else:
            logger.warning(f"⚠️ Document processing partial success: {filename} (doc_id: {doc_id})")
            
    except Exception as e:
        logger.error(f"❌ Error in document processing: {filename} (doc_id: {doc_id}) - {e}", exc_info=True)


@router.get("/documents/list")
async def list_documents(
    mongo_client: MongoDBClient = Depends(get_mongodb_client),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    uploader: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    List all uploaded documents with metadata.
    
    **Query Parameters:**
    - skip: Number of documents to skip (pagination)
    - limit: Maximum documents to return (1-100)
    - uploader: Filter by uploader username
    - status: Filter by ingest_status (pending, processing, complete, failed)
    
    **Returns:**
    - documents: List of document metadata
    - total: Total number of documents matching filters
    - page: Current page number
    """
    try:
        if not mongo_client.is_connected():
            raise HTTPException(status_code=503, detail="MongoDB not connected")
        
        # Build query filter
        query_filter = {}
        if uploader:
            query_filter["uploader"] = uploader
        if status:
            query_filter["ingest_status"] = status
        
        # Get total count
        total = await mongo_client.database.documents.count_documents(query_filter)
        
        # Fetch documents
        cursor = mongo_client.database.documents.find(query_filter)\
            .sort("uploaded_at", -1)\
            .skip(skip)\
            .limit(limit)
        
        documents = await cursor.to_list(length=limit)
        
        # Format response
        formatted_docs = [
            {
                "id": str(doc.get("_id", "")),
                "filename": doc.get("filename", "unknown"),
                "size": doc.get("size", 0),
                "uploaded_at": doc.get("uploaded_at", ""),
                "ingest_status": doc.get("ingest_status", "pending"),
                "chunks_count": doc.get("chunks_count", 0),
                "file_type": doc.get("file_type", "unknown"),
                "uploader": doc.get("uploader", "system"),
                "error_message": doc.get("error_message", "")
            }
            for doc in documents
        ]
        
        logger.info(f"Listed {len(formatted_docs)} documents (total: {total})")
        
        return {
            "documents": formatted_docs,
            "total": total,
            "page": (skip // limit) + 1 if limit > 0 else 1,
            "skip": skip,
            "limit": limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/status/{doc_id}")
async def get_document_status(
    doc_id: str,
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
) -> Dict[str, Any]:
    """
    Get detailed processing status for a document.
    
    **Path Parameters:**
    - doc_id: Document ID to get status for
    
    **Returns:**
    - current_stage: Current processing stage (EXTRACT, CHUNK, EMBED, STORE_CHUNKS, INDEX_VECTORS, EXTRACT_ENTITIES, COMPLETE, FAILED)
    - stages: List of all stages with their status
    - overall_progress: Percentage complete (0-100)
    - error_message: Error message if stage failed
    - last_updated: Timestamp of last status update
    """
    try:
        if not mongo_client.is_connected():
            raise HTTPException(status_code=503, detail="MongoDB not connected")
        
        from bson import ObjectId
        
        try:
            obj_id = ObjectId(doc_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid document ID format")
        
        # Get document metadata
        doc = await mongo_client.database.documents.find_one({"_id": obj_id})
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Get ingestion logs
        logs = []
        try:
            cursor = mongo_client.database.ingestion_logs.find({"doc_id": str(obj_id)}).sort("timestamp", 1)
            logs = await cursor.to_list(length=None)
        except Exception as e:
            logger.warning(f"Could not retrieve ingestion logs: {e}")
        
        # Define processing stages in order
        expected_stages = ["EXTRACT", "CHUNK", "EMBED", "STORE_CHUNKS", "INDEX_VECTORS", "EXTRACT_ENTITIES"]
        
        # Build status for each stage
        stages_status = []
        current_stage_idx = -1
        current_stage = "PENDING"
        
        for i, stage_name in enumerate(expected_stages):
            stage_log = next((log for log in logs if log.get("stage") == stage_name), None)
            
            if stage_log:
                status = stage_log.get("status", "UNKNOWN")
                stages_status.append({
                    "name": stage_name,
                    "status": status,
                    "message": stage_log.get("message", ""),
                    "metadata": stage_log.get("metadata", {}),
                    "timestamp": stage_log.get("timestamp", "")
                })
                
                if status == "SUCCESS":
                    current_stage_idx = i
                    current_stage = stage_name
                elif status == "PROCESSING":
                    current_stage = stage_name
                    break
                elif status == "FAILED":
                    current_stage = stage_name
                    break
            else:
                stages_status.append({
                    "name": stage_name,
                    "status": "PENDING",
                    "message": "Awaiting processing",
                    "metadata": {},
                    "timestamp": ""
                })
        
        # Calculate overall progress (number of successful stages / total stages)
        successful_stages = sum(1 for s in stages_status if s["status"] == "SUCCESS")
        overall_progress = int((successful_stages / len(expected_stages)) * 100)
        
        # Check if all stages completed
        ingest_status = doc.get("ingest_status", "pending")
        if ingest_status == "complete":
            current_stage = "COMPLETE"
            overall_progress = 100
        elif ingest_status == "failed":
            current_stage = "FAILED"
        
        # Get error message if failed
        error_message = doc.get("error_message", "")
        failed_log = next((log for log in logs if log.get("status") == "FAILED"), None)
        if failed_log and not error_message:
            error_message = failed_log.get("message", "")
        
        logger.info(f"Status for document {doc_id}: {current_stage} ({overall_progress}%)")
        
        return {
            "doc_id": str(obj_id),
            "filename": doc.get("filename", "unknown"),
            "current_stage": current_stage,
            "stages": stages_status,
            "overall_progress": overall_progress,
            "ingest_status": ingest_status,
            "chunks_count": doc.get("chunks_count", 0),
            "error_message": error_message,
            "last_updated": doc.get("updated_at", ""),
            "uploaded_at": doc.get("uploaded_at", "")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")
