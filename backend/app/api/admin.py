"""
Admin API routes for document management and system administration.
This module provides endpoints for file upload, document management,
system monitoring, and real-time status updates.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging
from datetime import datetime
import json
import asyncio
from io import BytesIO

from ..services.auth_service import require_admin_role, get_current_user
from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..core.db_qdrant import get_qdrant_client, QdrantDBClient
from ..core.db_arango import get_arango_client, ArangoDBClient
from ..core.cache_redis import get_redis_client, RedisClient
from ..workers.ingest_worker import enqueue_document, get_queue_status, get_processing_statistics
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    uploaded_at: str
    uploader: str
    ingest_status: str
    error_message: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    doc_id: str
    filename: str
    message: str

class SystemStats(BaseModel):
    mongodb: Dict[str, Any]
    qdrant: Dict[str, Any] 
    arangodb: Dict[str, Any]
    redis: Dict[str, Any]
    processing: Dict[str, Any]

class IngestionLog(BaseModel):
    doc_id: str
    step: str
    status: str
    message: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(require_admin_role),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    Upload a document for processing.
    
    Args:
        file: Uploaded file
        current_user: Authenticated admin user
        mongo_client: MongoDB client
    
    Returns:
        Upload response with document ID
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        content = await file.read()
        if len(content) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Check file type
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_extension not in settings.allowed_file_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed types: {', '.join(settings.allowed_file_extensions)}"
            )
        
        # Store file in GridFS
        gridfs_id = await mongo_client.store_file(
            filename=file.filename,
            content=content,
            metadata={
                "content_type": file.content_type,
                "size": len(content),
                "uploader": current_user["username"]
            }
        )
        
        # Create document record
        document_data = {
            "filename": file.filename,
            "gridfs_id": gridfs_id,
            "file_type": file.content_type,
            "uploader": current_user["username"],
            "file_size": len(content)
        }
        
        doc_id = await mongo_client.create_document(document_data)
        
        # Add to ingestion queue
        await enqueue_document(doc_id)
        
        logger.info(f"File {file.filename} uploaded successfully by {current_user['username']}")
        
        return UploadResponse(
            success=True,
            doc_id=doc_id,
            filename=file.filename,
            message="File uploaded successfully and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status_filter: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(require_admin_role),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    List documents with pagination and filtering.
    
    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        status_filter: Optional status filter
        current_user: Authenticated admin user
        mongo_client: MongoDB client
    
    Returns:
        List of documents
    """
    try:
        documents = await mongo_client.list_documents(skip, limit)
        
        # Apply status filter if provided
        if status_filter:
            documents = [doc for doc in documents if doc.get("ingest_status") == status_filter]
        
        # Convert to response model
        response_docs = []
        for doc in documents:
            response_docs.append(DocumentResponse(
                id=doc["_id"],
                filename=doc.get("filename", ""),
                file_type=doc.get("file_type", ""),
                uploaded_at=doc.get("uploaded_at", datetime.utcnow()).isoformat(),
                uploader=doc.get("uploader", ""),
                ingest_status=doc.get("ingest_status", "UNKNOWN"),
                error_message=doc.get("error_message")
            ))
        
        return response_docs
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@router.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_role),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """Get detailed document information."""
    try:
        document = await mongo_client.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks information
        chunks = await mongo_client.get_chunks(doc_id)
        
        # Get ingestion logs
        logs = await mongo_client.get_ingestion_logs(doc_id)
        
        return {
            "document": document,
            "chunks_count": len(chunks),
            "ingestion_logs": logs,
            "processing_complete": document.get("ingest_status") == "COMPLETED"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    hard_delete: bool = Query(False),
    current_user: Dict[str, Any] = Depends(require_admin_role),
    mongo_client: MongoDBClient = Depends(get_mongodb_client),
    qdrant_client: QdrantDBClient = Depends(get_qdrant_client),
    arango_client: ArangoDBClient = Depends(get_arango_client)
):
    """
    Delete a document (soft or hard delete).
    
    Args:
        doc_id: Document ID to delete
        hard_delete: Whether to perform hard delete (removes all data)
        current_user: Authenticated admin user
        
    Returns:
        Deletion confirmation
    """
    try:
        document = await mongo_client.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if hard_delete:
            # Hard delete: remove all associated data
            
            # Delete vectors from Qdrant
            await qdrant_client.delete_document_vectors(doc_id)
            
            # Delete chunks from MongoDB
            await mongo_client.delete_chunks(doc_id)
            
            # Delete graph data from ArangoDB
            if settings.use_graph_search:
                await arango_client.delete_document_graph_data(doc_id)
            
            # Delete GridFS file
            gridfs_id = document.get("gridfs_id")
            if gridfs_id:
                await mongo_client.delete_file(gridfs_id)
            
            # Delete document record
            await mongo_client.database.documents.delete_one({"_id": document["_id"]})
            
            logger.info(f"Hard deleted document {doc_id} by {current_user['username']}")
            message = "Document permanently deleted"
            
        else:
            # Soft delete: just mark as deleted
            await mongo_client.update_document_status(doc_id, "DELETED")
            
            logger.info(f"Soft deleted document {doc_id} by {current_user['username']}")
            message = "Document marked as deleted"
        
        return {
            "success": True,
            "message": message,
            "doc_id": doc_id,
            "hard_delete": hard_delete
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@router.post("/documents/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_role),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """Reprocess a document (re-run ingestion)."""
    try:
        document = await mongo_client.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reset document status
        await mongo_client.update_document_status(doc_id, "PENDING")
        
        # Add to ingestion queue
        await enqueue_document(doc_id)
        
        logger.info(f"Document {doc_id} queued for reprocessing by {current_user['username']}")
        
        return {
            "success": True,
            "message": "Document queued for reprocessing",
            "doc_id": doc_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reprocess document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reprocess document")

@router.get("/documents/{doc_id}/download")
async def download_document(
    doc_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_role),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """Download original document file."""
    try:
        document = await mongo_client.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get file from GridFS
        gridfs_id = document.get("gridfs_id")
        if not gridfs_id:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_content = await mongo_client.retrieve_file(gridfs_id)
        filename = document.get("filename", f"document_{doc_id}")
        
        # Create streaming response
        def generate():
            yield file_content
        
        return StreamingResponse(
            BytesIO(file_content),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to download document")

@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
    current_user: Dict[str, Any] = Depends(require_admin_role),
    mongo_client: MongoDBClient = Depends(get_mongodb_client),
    qdrant_client: QdrantDBClient = Depends(get_qdrant_client),
    arango_client: ArangoDBClient = Depends(get_arango_client),
    redis_client: RedisClient = Depends(get_redis_client)
):
    """Get comprehensive system statistics."""
    try:
        # Get database statistics
        mongodb_stats = {}
        qdrant_stats = {}
        arangodb_stats = {}
        redis_stats = {}
        
        try:
            # MongoDB stats
            if mongo_client.database:
                db_stats = await mongo_client.database.command("dbStats")
                mongodb_stats = {
                    "database": settings.mongodb_db,
                    "collections": db_stats.get("collections", 0),
                    "objects": db_stats.get("objects", 0),
                    "dataSize": db_stats.get("dataSize", 0),
                    "indexSize": db_stats.get("indexSize", 0)
                }
        except Exception as e:
            logger.warning(f"Failed to get MongoDB stats: {e}")
        
        try:
            # Qdrant stats
            qdrant_stats = await qdrant_client.get_collection_info()
        except Exception as e:
            logger.warning(f"Failed to get Qdrant stats: {e}")
        
        try:
            # ArangoDB stats
            arangodb_stats = await arango_client.get_graph_statistics()
        except Exception as e:
            logger.warning(f"Failed to get ArangoDB stats: {e}")
        
        try:
            # Redis stats
            redis_stats = await redis_client.get_redis_info()
        except Exception as e:
            logger.warning(f"Failed to get Redis stats: {e}")
        
        # Processing statistics
        processing_stats = await get_processing_statistics()
        
        return SystemStats(
            mongodb=mongodb_stats,
            qdrant=qdrant_stats,
            arangodb=arangodb_stats,
            redis=redis_stats,
            processing=processing_stats
        )
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")

@router.get("/logs/{doc_id}", response_model=List[IngestionLog])
async def get_document_logs(
    doc_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_role),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """Get ingestion logs for a specific document."""
    try:
        logs = await mongo_client.get_ingestion_logs(doc_id)
        
        # Convert to response model
        response_logs = []
        for log in logs:
            response_logs.append(IngestionLog(
                doc_id=log.get("doc_id", ""),
                step=log.get("step", ""),
                status=log.get("status", ""),
                message=log.get("message", ""),
                timestamp=log.get("timestamp", datetime.utcnow()).isoformat(),
                metadata=log.get("metadata")
            ))
        
        return response_logs
        
    except Exception as e:
        logger.error(f"Failed to get logs for document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document logs")

# Real-time WebSocket endpoint for admin updates
@router.websocket("/ws/status")
async def websocket_admin_updates(
    websocket: WebSocket,
    redis_client: RedisClient = Depends(get_redis_client)
):
    """
    WebSocket endpoint for real-time admin updates.
    Streams ingestion status updates and system notifications.
    """
    await websocket.accept()
    
    try:
        logger.info("Admin WebSocket client connected")
        
        # Send initial queue status
        queue_status = await get_queue_status()
        await websocket.send_json({
            "type": "queue_status",
            "data": queue_status,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Subscribe to Redis updates
        async def listen_for_updates():
            try:
                async for message in redis_client.subscribe_to_channel("admin_updates"):
                    await websocket.send_json({
                        "type": "update",
                        "data": message,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error in Redis subscription: {e}")
        
        # Start listening for updates
        listen_task = asyncio.create_task(listen_for_updates())
        
        # Keep connection alive and handle client messages
        try:
            while True:
                # Wait for client message or connection close
                message = await websocket.receive_json()
                
                # Handle client requests
                if message.get("type") == "get_queue_status":
                    queue_status = await get_queue_status()
                    await websocket.send_json({
                        "type": "queue_status",
                        "data": queue_status,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
        except WebSocketDisconnect:
            logger.info("Admin WebSocket client disconnected")
        finally:
            # Cancel the listening task
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
            
            # Unsubscribe from Redis
            await redis_client.unsubscribe_from_channel("admin_updates")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1000)
        except:
            pass

@router.get("/queue/status")
async def get_ingestion_queue_status(
    current_user: Dict[str, Any] = Depends(require_admin_role)
):
    """Get current ingestion queue status."""
    try:
        return await get_queue_status()
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queue status")