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
import json
import asyncio
from datetime import datetime

from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..utils.json_sanitizer import validate_json_file
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Constants
DEFAULT_CONTENT_TYPE = "application/octet-stream"


class UploadJSONRequest(BaseModel):
    filename: str
    content_base64: str
    content_type: Optional[str] = DEFAULT_CONTENT_TYPE


@router.post("/documents/upload")
async def upload_document_multipart(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    mongo_client: MongoDBClient = Depends(get_mongodb_client),
    redis_client = Depends(lambda: None),  # Optional Redis dependency
    sync_processing: bool = False  # New parameter for synchronous processing
):
    """
    Accept a multipart/form-data file, save to MongoDB, and queue for processing.
    
    Args:
        sync_processing: If True, process document synchronously before returning.
                        Auto-enabled for files <100KB for instant availability.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        content = await file.read()
        size = len(content)
        
        if size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        if size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
        
        # AUTO-SANITIZE JSON FILES BEFORE SAVING
        if file.filename and file.filename.lower().endswith('.json'):
            logger.info(f"🧹 Auto-sanitizing JSON file: {file.filename}")
            success, sanitized_data, report = validate_json_file(content)
            
            if not success:
                logger.error(f"❌ JSON sanitization failed: {report.get('error')}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid JSON file: {report.get('error', 'Unknown error')}"
                )
            
            # Replace content with sanitized version
            content = json.dumps(sanitized_data, indent=2).encode('utf-8')
            size = len(content)
            
            logger.info(f"✅ JSON sanitized successfully - {len(report.get('cleaning_steps', []))} fixes applied")
            logger.info(f"   Type: {report.get('type')}, Items: {report.get('statistics', {}).get('total_items', 'N/A')}")
        
        # Save to MongoDB GridFS
        doc_id = await mongo_client.save_document(
            filename=file.filename,
            content=content,
            content_type=file.content_type or DEFAULT_CONTENT_TYPE,
            size=size
        )
        
        logger.info(f"✅ Document saved to MongoDB: {file.filename} ({size} bytes) - ID: {doc_id}")
        
        # Invalidate document list cache
        try:
            from ..core.cache_redis import get_redis_client
            redis_cache_client = await get_redis_client()
            if redis_cache_client.is_connected():
                await redis_cache_client.clear_pattern("docs:list:*")
                logger.info("🔄 Cleared document list cache")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
        
        # Auto-detect: Small documents (<100KB) process synchronously for instant availability
        small_document = size < 100_000  # 100KB threshold
        should_process_sync = sync_processing or small_document
        
        if should_process_sync:
            logger.info(f"⚡ FAST-PATH: Processing document synchronously ({size} bytes)")
            try:
                # Process immediately and wait for completion
                success = await _process_document_sync(doc_id, file.filename, mongo_client)
                
                if success:
                    return {
                        "success": True,
                        "filename": file.filename,
                        "size": size,
                        "document_id": str(doc_id),
                        "status": "completed",
                        "processing_time": "immediate",
                        "message": "Document processed and ready for queries"
                    }
                else:
                    return {
                        "success": True,
                        "filename": file.filename,
                        "size": size,
                        "document_id": str(doc_id),
                        "status": "completed_with_warnings",
                        "processing_time": "immediate"
                    }
            except Exception as proc_error:
                logger.error(f"Sync processing failed, falling back to async: {proc_error}")
                # Fall back to async processing
                processing_task = asyncio.create_task(_process_document_with_retry(doc_id, file.filename))
                background_tasks.add_task(lambda: processing_task)
                return {
                    "success": True,
                    "filename": file.filename,
                    "size": size,
                    "document_id": str(doc_id),
                    "status": "processing",
                    "message": "Processing in background (sync failed)"
                }
        else:
            # Large document: Queue for background processing
            logger.info(f"🚀 ASYNC-PATH: Queuing large document for background processing ({size} bytes)")
            processing_task = asyncio.create_task(_process_document_with_retry(doc_id, file.filename))
            background_tasks.add_task(lambda: processing_task)
            
            return {
                "success": True,
                "filename": file.filename,
                "size": size,
                "document_id": str(doc_id),
                "status": "processing",
                "message": "Large document queued for processing"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/documents/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: str,
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """Manually trigger document reprocessing (synchronous for testing)."""
    try:
        # Get document info
        doc = await mongo_client.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        filename = doc.get("filename", "unknown")
        logger.info(f"🔄 Manual reprocessing triggered: {filename}")
        
        # Process synchronously so we can see errors
        from ..services.ingest_service import IngestService
        ingest_service = IngestService()
        await ingest_service.initialize()
        
        # Ensure Qdrant is connected
        if not ingest_service.qdrant_client.is_connected():
            await ingest_service.qdrant_client.connect()
        
        # Update status
        await mongo_client.update_document_status(doc_id, "PROCESSING")
        
        # Process
        success = await ingest_service.process_document(doc_id)
        
        if success:
            await mongo_client.update_document_status(doc_id, "COMPLETED")
            return {"success": True, "message": f"Document {filename} processed successfully"}
        else:
            await mongo_client.update_document_status(doc_id, "FAILED", "Processing returned False")
            return {"success": False, "message": "Processing failed"}
            
    except Exception as e:
        logger.error(f"Reprocessing error: {e}", exc_info=True)
        await mongo_client.update_document_status(doc_id, "FAILED", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload-json")
async def upload_document_json(
    payload: UploadJSONRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    mongo_client: MongoDBClient = Depends(get_mongodb_client),
    sync_processing: bool = False  # New parameter for synchronous processing
):
    """
    Accept JSON payload with base64-encoded content.
    
    Args:
        sync_processing: If True, process document synchronously before returning.
                        Recommended for small documents (<100KB) to eliminate delay.
    """
    try:
        content = base64.b64decode(payload.content_base64)
        size = len(content)
        
        # Validate file size
        if size > settings.max_file_size_bytes:
            logger.warning(f"File too large: {size} bytes (max: {settings.max_file_size_bytes})")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        if size == 0:
            raise HTTPException(status_code=400, detail="Content is empty")
        
        # Save to MongoDB GridFS
        doc_id = await mongo_client.save_document(
            filename=payload.filename,
            content=content,
            content_type=payload.content_type or DEFAULT_CONTENT_TYPE,
            size=size
        )
        
        logger.info(f"✅ JSON document saved to MongoDB: {payload.filename} ({size} bytes) - ID: {doc_id}")
        
        # Auto-detect: Small documents (<100KB) process synchronously for instant availability
        small_document = size < 100_000  # 100KB threshold
        should_process_sync = sync_processing or small_document
        
        if should_process_sync:
            logger.info(f"⚡ FAST-PATH: Processing document synchronously ({size} bytes)")
            try:
                # Process immediately and wait for completion
                success = await _process_document_sync(doc_id, payload.filename, mongo_client)
                
                if success:
                    return {
                        "success": True,
                        "filename": payload.filename,
                        "size": size,
                        "document_id": str(doc_id),
                        "status": "completed",
                        "processing_time": "immediate",
                        "message": "Document processed and ready for queries"
                    }
                else:
                    return {
                        "success": True,
                        "filename": payload.filename,
                        "size": size,
                        "document_id": str(doc_id),
                        "status": "completed_with_warnings",
                        "processing_time": "immediate"
                    }
            except Exception as proc_error:
                logger.error(f"Sync processing failed, falling back to async: {proc_error}")
                # Fall back to async processing (wrap async function for FastAPI background tasks)
                def run_async_fallback():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            _process_document_immediate(doc_id, payload.filename, mongo_client)
                        )
                    finally:
                        loop.close()
                
                background_tasks.add_task(run_async_fallback)
                return {
                    "success": True,
                    "filename": payload.filename,
                    "size": size,
                    "document_id": str(doc_id),
                    "status": "processing",
                    "message": "Processing in background (sync failed)"
                }
        else:
            # Large document: Queue for background processing
            logger.info(f"🚀 ASYNC-PATH: Queuing large document for background processing ({size} bytes)")
            # Create a sync wrapper for FastAPI background tasks
            def run_async_task():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        _process_document_immediate(doc_id, payload.filename, mongo_client)
                    )
                finally:
                    loop.close()
            
            background_tasks.add_task(run_async_task)
            
            return {
                "success": True,
                "filename": payload.filename,
                "size": size,
                "document_id": str(doc_id),
                "status": "processing",
                "message": "Large document queued for processing"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ JSON upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def _attempt_document_processing(doc_id: str, filename: str, mongo_client) -> bool:
    """Single attempt to process a document. Returns True on success."""
    # Update status to PROCESSING
    await mongo_client.update_document_status(doc_id, "PROCESSING")
    
    # Initialize ingest service
    from ..services.ingest_service import IngestService
    ingest_service = IngestService()
    await ingest_service.initialize()
    
    # Ensure Qdrant is connected
    if not ingest_service.qdrant_client.is_connected():
        await ingest_service.qdrant_client.connect()
    
    # Process the document
    logger.info(f"⚙️ Running RAG pipeline for {filename}...")
    success = await ingest_service.process_document(doc_id)
    return success


async def _handle_processing_failure(doc_id: str, filename: str, mongo_client, attempt: int, max_retries: int, error: Exception):
    """Handle processing failure with retry logic or final failure status."""
    logger.error(f"❌ ATTEMPT {attempt} FAILED: {filename} - {error}")
    
    if attempt < max_retries:
        # Retry after delay
        delay = 5 * attempt
        logger.info(f"⏱️ Retrying in {delay} seconds...")
        await asyncio.sleep(delay)
    else:
        # All retries exhausted
        logger.error(f"❌ ALL {max_retries} ATTEMPTS FAILED for {filename}")
        try:
            await mongo_client.update_document_status(doc_id, "FAILED", f"Failed after {max_retries} attempts: {str(error)}")
        except Exception as update_error:
            logger.warning(f"Failed to update status: {update_error}")


async def _process_document_with_retry(doc_id: str, filename: str, max_retries: int = 3):
    """
    Process document with automatic retry on failure.
    This runs in the background using asyncio.create_task.
    """
    from ..core.db_mongo import get_mongodb_client
    
    mongo_client = get_mongodb_client()  # FIX: get_mongodb_client() is SYNC, not async!
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔥 PROCESSING ATTEMPT {attempt}/{max_retries}: {filename} (ID: {doc_id})")
            
            success = await _attempt_document_processing(doc_id, filename, mongo_client)
            
            if success:
                await mongo_client.update_document_status(doc_id, "COMPLETED")
                logger.info(f"✅ SUCCESS (attempt {attempt}): {filename} fully processed!")
                return  # Success - exit retry loop
            else:
                # Partial success - retry
                logger.warning(f"⚠️ PARTIAL SUCCESS (attempt {attempt}): {filename}")
                if attempt < max_retries:
                    await asyncio.sleep(5 * attempt)  # Exponential backoff
                    continue
                else:
                    await mongo_client.update_document_status(doc_id, "COMPLETED", "Partial success after retries")
                    return
                    
        except Exception as e:
            await _handle_processing_failure(doc_id, filename, mongo_client, attempt, max_retries, e)


async def _process_document_sync(doc_id: str, filename: str, mongo_client) -> bool:
    """
    SYNCHRONOUS processing - blocks until document is fully processed.
    Used for small documents to eliminate processing delay.
    
    Returns:
        bool: True if successful, False if partial success
    """
    try:
        logger.info(f"⚡ SYNC PROCESSING STARTED: {filename} (doc_id: {doc_id})")
        
        # Update status to PROCESSING immediately
        await mongo_client.update_document_status(str(doc_id), "PROCESSING")
        
        # Initialize ingest service
        from ..services.ingest_service import IngestService
        
        ingest_service = IngestService()
        await ingest_service.initialize()
        
        # Ensure Qdrant is connected
        if not ingest_service.qdrant_client.is_connected():
            await ingest_service.qdrant_client.connect()
        
        # Process the document with timeout (5 minutes max)
        logger.info("⚙️ Running RAG pipeline: Extract → Chunk → Embed → Index")
        try:
            success = await asyncio.wait_for(
                ingest_service.process_document(str(doc_id)),
                timeout=300.0  # 5 minutes
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ TIMEOUT: Processing took longer than 5 minutes for {filename}")
            await mongo_client.update_document_status(str(doc_id), "FAILED", "Processing timeout after 5 minutes")
            return False
        
        if success:
            await mongo_client.update_document_status(str(doc_id), "COMPLETED")
            logger.info(f"✅ SYNC SUCCESS: {filename} fully processed and ready!")
            
            # Reload BM25 index to include new document
            try:
                from ..services.chat_service import get_chat_service
                chat_service = await get_chat_service()
                await chat_service._reload_bm25_index()
                logger.info("🔄 BM25 index reloaded with new document")
            except Exception as reload_error:
                logger.warning(f"Failed to reload BM25: {reload_error}")
            
            return True
        else:
            await mongo_client.update_document_status(str(doc_id), "COMPLETED", "Partial success")
            logger.warning(f"⚠️ SYNC PARTIAL: {filename} processed with warnings")
            return False
            
    except Exception as e:
        logger.error(f"❌ SYNC FAILED: {filename} - {e}", exc_info=True)
        await mongo_client.update_document_status(str(doc_id), "FAILED", str(e))
        raise


async def _process_document_immediate(doc_id, filename, mongo_client):
    """IMMEDIATE background processing with retry logic."""
    max_retries = settings.max_retry_attempts
    retry_delay = settings.retry_delay_seconds
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"🔄 RETRY {attempt}/{max_retries}: {filename}")
                await asyncio.sleep(retry_delay * (settings.retry_backoff_multiplier ** (attempt - 1)))
            else:
                logger.info(f"🔥 IMMEDIATE PROCESSING STARTED: {filename} (doc_id: {doc_id})")
            
            # Update status to PROCESSING immediately
            await mongo_client.update_document_status(str(doc_id), "PROCESSING")
            logger.info(f"📝 Status: PENDING → PROCESSING for {filename}")
            
            # Lazy initialize ingest service
            from ..services.ingest_service import IngestService
            
            logger.info("🔧 Initializing ingest service...")
            ingest_service = IngestService()
            await ingest_service.initialize()
            logger.info("✅ Ingest service ready")
            
            # Process the document
            logger.info("⚙️ Running RAG pipeline: Extract → Chunk → Embed → Index")
            success = await ingest_service.process_document(str(doc_id))
            
            if success:
                await mongo_client.update_document_status(str(doc_id), "COMPLETED")
                logger.info(f"✅ SUCCESS: {filename} fully processed and indexed!")
                return  # Success - exit retry loop
            else:
                await mongo_client.update_document_status(str(doc_id), "COMPLETED", "Partial success")
                logger.warning(f"⚠️ PARTIAL: {filename} processed with warnings")
                return  # Partial success - exit retry loop
                
        except Exception as e:
            logger.error(f"❌ ATTEMPT {attempt + 1} FAILED: {filename} - {e}", exc_info=True)
            
            if attempt == max_retries - 1:
                # Final attempt failed
                try:
                    await mongo_client.update_document_status(str(doc_id), "FAILED", f"Failed after {max_retries} attempts: {str(e)}")
                except Exception as update_error:
                    logger.warning(f"Failed to update status: {update_error}")
            else:
                # Will retry
                logger.info(f"⏳ Waiting {retry_delay * (settings.retry_backoff_multiplier ** attempt)}s before retry...")


async def _process_document(doc_id, filename, mongo_client):
    """Background task to process uploaded document with chunking and embeddings."""
    try:
        logger.info(f"🔄 BACKGROUND TASK STARTED: Processing {filename} (doc_id: {doc_id})")
        
        # Update status to PROCESSING immediately
        await mongo_client.update_document_status(str(doc_id), "PROCESSING")
        logger.info(f"📝 Status updated to PROCESSING for {filename}")
        
        # Lazy initialize ingest service (import here to avoid circular imports)
        from ..services.ingest_service import IngestService
        from ..core.db_qdrant import get_qdrant_client
        from ..core.cache_redis import get_redis_client
        
        logger.info(f"🔧 Initializing ingest service for {filename}...")
        # Create ingest service instance
        ingest_service = IngestService()
        await ingest_service.initialize()
        logger.info(f"✅ Ingest service initialized for {filename}")
        
        # Process the document (chunks, embeddings, Qdrant storage)
        logger.info(f"⚙️ Starting document processing pipeline for {filename}...")
        success = await ingest_service.process_document(str(doc_id))
        
        if success:
            logger.info(f"✅ Document processing completed successfully: {filename} (doc_id: {doc_id})")
            await mongo_client.update_document_status(str(doc_id), "COMPLETED")
        else:
            logger.warning(f"⚠️ Document processing partial success: {filename} (doc_id: {doc_id})")
            await mongo_client.update_document_status(str(doc_id), "COMPLETED", "Partial success")
            
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in document processing: {filename} (doc_id: {doc_id}) - {e}", exc_info=True)
        try:
            await mongo_client.update_document_status(str(doc_id), "FAILED", str(e))
        except Exception as status_error:
            logger.error(f"Failed to update status: {status_error}")


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


def _build_stage_status(stage_name: str, stage_log: Optional[Dict]) -> Dict[str, Any]:
    """Build status dict for a single processing stage."""
    if stage_log:
        return {
            "name": stage_name,
            "status": stage_log.get("status", "UNKNOWN"),
            "message": stage_log.get("message", ""),
            "metadata": stage_log.get("metadata", {}),
            "timestamp": stage_log.get("timestamp", "")
        }
    else:
        return {
            "name": stage_name,
            "status": "PENDING",
            "message": "Awaiting processing",
            "metadata": {},
            "timestamp": ""
        }


def _determine_current_stage(stages_status: List[Dict], ingest_status: str) -> tuple[str, int]:
    """Determine current stage and overall progress percentage."""
    current_stage = "PENDING"
    
    # Find current stage from stages_status
    for stage in stages_status:
        status = stage["status"]
        if status == "SUCCESS":
            current_stage = stage["name"]
        elif status in ("PROCESSING", "FAILED"):
            current_stage = stage["name"]
            break
    
    # Calculate progress
    successful_stages = sum(1 for s in stages_status if s["status"] == "SUCCESS")
    overall_progress = int((successful_stages / len(stages_status)) * 100) if stages_status else 0
    
    # Override based on ingest_status
    if ingest_status == "complete":
        current_stage = "COMPLETE"
        overall_progress = 100
    elif ingest_status == "failed":
        current_stage = "FAILED"
    
    return current_stage, overall_progress


async def _get_ingestion_logs(mongo_client: MongoDBClient, doc_id: str) -> List[Dict]:
    """Retrieve ingestion logs for a document."""
    try:
        cursor = mongo_client.database.ingestion_logs.find({"doc_id": doc_id}).sort("timestamp", 1)
        return await cursor.to_list(length=None)
    except Exception as e:
        logger.warning(f"Could not retrieve ingestion logs: {e}")
        return []


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
        logs = await _get_ingestion_logs(mongo_client, str(obj_id))
        
        # Define processing stages in order
        expected_stages = ["EXTRACT", "CHUNK", "EMBED", "STORE_CHUNKS", "INDEX_VECTORS", "EXTRACT_ENTITIES"]
        
        # Build status for each stage
        stages_status = []
        for stage_name in expected_stages:
            stage_log = next((log for log in logs if log.get("step") == stage_name), None)
            stages_status.append(_build_stage_status(stage_name, stage_log))
        
        # Determine current stage and progress
        ingest_status = doc.get("ingest_status", "pending")
        current_stage, overall_progress = _determine_current_stage(stages_status, ingest_status)
        
        # Get error message if failed
        error_message = doc.get("error_message", "")
        if not error_message:
            failed_log = next((log for log in logs if log.get("status") == "FAILED"), None)
            if failed_log:
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

