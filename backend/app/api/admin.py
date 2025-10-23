"""""""""

Admin API routes for document management and system administration.

This module provides endpoints for file upload, document management,Admin API routes for document management and system administration.Admin API routes for document management and system administration.

system monitoring, and real-time status updates.

This module provides endpoints for file upload, document management,This module provides endpoints for file upload, document management,

NOTE: ALL ROUTES ARE PUBLIC - NO AUTHENTICATION REQUIRED

"""system monitoring, and real-time status updates.system monitoring, and real-time status updates.



from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect, Query"""

from fastapi.responses import StreamingResponse

from typing import List, Optional, Dict, Any**NOTE: ALL ROUTES ARE PUBLIC - NO AUTHENTICATION REQUIRED**

from pydantic import BaseModel

import logging"""from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect, Query

from datetime import datetime

import jsonfrom fastapi.responses import StreamingResponse

import asyncio

from io import BytesIOfrom fastapi import APIRouter, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect, Queryfrom typing import List, Optional, Dict, Any



from ..core.db_mongo import get_mongodb_client, MongoDBClientfrom fastapi.responses import StreamingResponsefrom pydantic import BaseModel

from ..core.db_qdrant import get_qdrant_client, QdrantDBClient

from ..core.db_arango import get_arango_client, ArangoDBClientfrom typing import List, Optional, Dict, Anyimport logging

from ..core.cache_redis import get_redis_client, RedisClient

from ..workers.ingest_worker import enqueue_document, get_queue_status, get_processing_statisticsfrom pydantic import BaseModelfrom datetime import datetime

from ..config import settings

import loggingimport json

logger = logging.getLogger(__name__)

router = APIRouter()from datetime import datetimeimport asyncio



# Pydantic models for request/responseimport jsonfrom io import BytesIO

class DocumentResponse(BaseModel):

    id: strimport asyncio

    filename: str

    file_type: strfrom io import BytesIOfrom ..core.db_mongo import get_mongodb_client, MongoDBClient

    uploaded_at: str

    uploader: strfrom ..core.db_qdrant import get_qdrant_client, QdrantDBClient

    ingest_status: str

    error_message: Optional[str] = Nonefrom ..core.db_mongo import get_mongodb_client, MongoDBClientfrom ..core.db_arango import get_arango_client, ArangoDBClient



class UploadResponse(BaseModel):from ..core.db_qdrant import get_qdrant_client, QdrantDBClientfrom ..core.cache_redis import get_redis_client, RedisClient

    success: bool

    doc_id: strfrom ..core.db_arango import get_arango_client, ArangoDBClientfrom ..workers.ingest_worker import enqueue_document, get_queue_status, get_processing_statistics

    filename: str

    message: strfrom ..core.cache_redis import get_redis_client, RedisClientfrom ..config import settings



class SystemStats(BaseModel):from ..workers.ingest_worker import enqueue_document, get_queue_status, get_processing_statistics

    mongodb: Dict[str, Any]

    qdrant: Dict[str, Any]from ..config import settingslogger = logging.getLogger(__name__)

    arangodb: Dict[str, Any]

    redis: Dict[str, Any]

    processing: Dict[str, Any]

logger = logging.getLogger(__name__)router = APIRouter()

class IngestionLog(BaseModel):

    doc_id: str

    step: str

    status: strrouter = APIRouter()# Pydantic models for request/response

    message: str

    timestamp: strclass DocumentResponse(BaseModel):

    metadata: Optional[Dict[str, Any]] = None

# Pydantic models for request/response    id: str

@router.post("/documents/upload", response_model=UploadResponse)

async def upload_document(class DocumentResponse(BaseModel):    filename: str

    file: UploadFile = File(...),

    mongo_client: MongoDBClient = Depends(get_mongodb_client)    id: str    file_type: str

):

    """Upload a document for processing (PUBLIC - NO AUTH)."""    filename: str    uploaded_at: str

    try:

        if not file.filename:    file_type: str    uploader: str

            raise HTTPException(status_code=400, detail="No file provided")

            uploaded_at: str    ingest_status: str

        content = await file.read()

        if len(content) > settings.max_file_size_bytes:    uploader: str    error_message: Optional[str] = None

            raise HTTPException(

                status_code=400,    ingest_status: str

                detail=f"File too large. Maximum: {settings.max_file_size_mb}MB"

            )    error_message: Optional[str] = Noneclass UploadResponse(BaseModel):

        

        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''    success: bool

        if file_extension not in settings.allowed_file_extensions:

            raise HTTPException(class UploadResponse(BaseModel):    doc_id: str

                status_code=400,

                detail=f"Unsupported format. Allowed: {', '.join(settings.allowed_file_extensions)}"    success: bool    filename: str

            )

            doc_id: str    message: str

        gridfs_id = await mongo_client.store_file(

            filename=file.filename,    filename: str

            content=content,

            metadata={    message: strclass SystemStats(BaseModel):

                "content_type": file.content_type,

                "size": len(content),    mongodb: Dict[str, Any]

                "uploader": "public"

            }class SystemStats(BaseModel):    qdrant: Dict[str, Any] 

        )

            mongodb: Dict[str, Any]    arangodb: Dict[str, Any]

        document_data = {

            "filename": file.filename,    qdrant: Dict[str, Any]     redis: Dict[str, Any]

            "gridfs_id": gridfs_id,

            "file_type": file.content_type,    arangodb: Dict[str, Any]    processing: Dict[str, Any]

            "uploader": "public",

            "file_size": len(content)    redis: Dict[str, Any]

        }

            processing: Dict[str, Any]class IngestionLog(BaseModel):

        doc_id = await mongo_client.create_document(document_data)

        await enqueue_document(doc_id)    doc_id: str

        

        logger.info(f"File {file.filename} uploaded successfully")class IngestionLog(BaseModel):    step: str

        

        return UploadResponse(    doc_id: str    status: str

            success=True,

            doc_id=doc_id,    step: str    message: str

            filename=file.filename,

            message="File uploaded and queued for processing"    status: str    timestamp: str

        )

            message: str    metadata: Optional[Dict[str, Any]] = None

    except HTTPException:

        raise    timestamp: str

    except Exception as e:

        logger.error(f"Upload failed: {e}")    metadata: Optional[Dict[str, Any]] = None@router.post("/upload", response_model=UploadResponse)

        raise HTTPException(status_code=500, detail="Upload failed")

async def upload_document(

@router.get("/documents", response_model=List[DocumentResponse])

async def list_documents(@router.post("/documents/upload", response_model=UploadResponse)    file: UploadFile = File(...),

    skip: int = Query(0, ge=0),

    limit: int = Query(50, ge=1, le=200),async def upload_document(    mongo_client: MongoDBClient = Depends(get_mongodb_client)

    status_filter: Optional[str] = Query(None),

    mongo_client: MongoDBClient = Depends(get_mongodb_client)    file: UploadFile = File(...),):

):

    """List documents (PUBLIC - NO AUTH)."""    mongo_client: MongoDBClient = Depends(get_mongodb_client)    """

    try:

        documents = await mongo_client.list_documents(skip, limit)):    Upload a document for processing.

        

        if status_filter:    """    

            documents = [doc for doc in documents if doc.get("ingest_status") == status_filter]

            Upload a document for processing (PUBLIC ENDPOINT - NO AUTH REQUIRED).    Args:

        response_docs = []

        for doc in documents:            file: Uploaded file

            response_docs.append(DocumentResponse(

                id=str(doc.get("_id", "")),    Args:        mongo_client: MongoDB client

                filename=doc.get("filename", ""),

                file_type=doc.get("file_type", ""),        file: Uploaded file    

                uploaded_at=doc.get("uploaded_at", datetime.utcnow()).isoformat(),

                uploader=doc.get("uploader", ""),        mongo_client: MongoDB client    Returns:

                ingest_status=doc.get("ingest_status", "UNKNOWN"),

                error_message=doc.get("error_message")            Upload response with document ID

            ))

            Returns:    """

        return response_docs

                Upload response with document ID    try:

    except Exception as e:

        logger.error(f"List failed: {e}")    """        # Validate file

        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

    try:        if not file.filename:

@router.get("/documents/{doc_id}")

async def get_document(        # Validate file            raise HTTPException(status_code=400, detail="No file provided")

    doc_id: str,

    mongo_client: MongoDBClient = Depends(get_mongodb_client)        if not file.filename:        

):

    """Get document details (PUBLIC - NO AUTH)."""            raise HTTPException(status_code=400, detail="No file provided")        # Check file size

    try:

        document = await mongo_client.get_document(doc_id)                content = await file.read()

        if not document:

            raise HTTPException(status_code=404, detail="Document not found")        # Check file size        if len(content) > settings.max_file_size_bytes:

        

        chunks = await mongo_client.get_chunks(doc_id)        content = await file.read()            raise HTTPException(

        logs = await mongo_client.get_ingestion_logs(doc_id)

                if len(content) > settings.max_file_size_bytes:                status_code=400, 

        return {

            "document": document,            raise HTTPException(                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"

            "chunks_count": len(chunks),

            "ingestion_logs": logs,                status_code=400,             )

            "processing_complete": document.get("ingest_status") == "COMPLETED"

        }                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"        

        

    except HTTPException:            )        # Check file type

        raise

    except Exception as e:                file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''

        logger.error(f"Get failed: {e}")

        raise HTTPException(status_code=500, detail="Failed to retrieve document")        # Check file type        if file_extension not in settings.allowed_file_extensions:



@router.delete("/documents/{doc_id}")        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''            raise HTTPException(

async def delete_document(

    doc_id: str,        if file_extension not in settings.allowed_file_extensions:                status_code=400,

    hard_delete: bool = Query(False),

    mongo_client: MongoDBClient = Depends(get_mongodb_client),            raise HTTPException(                detail=f"File type not supported. Allowed types: {', '.join(settings.allowed_file_extensions)}"

    qdrant_client: QdrantDBClient = Depends(get_qdrant_client),

    arango_client: ArangoDBClient = Depends(get_arango_client)                status_code=400,            )

):

    """Delete document (PUBLIC - NO AUTH)."""                detail=f"File type not supported. Allowed types: {', '.join(settings.allowed_file_extensions)}"        

    try:

        document = await mongo_client.get_document(doc_id)            )        # Store file in GridFS

        if not document:

            raise HTTPException(status_code=404, detail="Document not found")                gridfs_id = await mongo_client.store_file(

        

        if hard_delete:        # Store file in GridFS            filename=file.filename,

            await qdrant_client.delete_document_vectors(doc_id)

            await mongo_client.delete_chunks(doc_id)        gridfs_id = await mongo_client.store_file(            content=content,

            

            if settings.use_graph_search:            filename=file.filename,            metadata={

                await arango_client.delete_document_graph_data(doc_id)

                        content=content,                "content_type": file.content_type,

            gridfs_id = document.get("gridfs_id")

            if gridfs_id:            metadata={                "size": len(content),

                await mongo_client.delete_file(gridfs_id)

                            "content_type": file.content_type,                "uploader": "admin"

            await mongo_client.database.documents.delete_one({"_id": document["_id"]})

            message = "Document permanently deleted"                "size": len(content),            }

        else:

            await mongo_client.update_document_status(doc_id, "DELETED")                "uploader": "public"        )

            message = "Document marked as deleted"

                    }        

        return {

            "success": True,        )        # Create document record

            "message": message,

            "doc_id": doc_id,                document_data = {

            "hard_delete": hard_delete

        }        # Create document record            "filename": file.filename,

        

    except HTTPException:        document_data = {            "gridfs_id": gridfs_id,

        raise

    except Exception as e:            "filename": file.filename,            "file_type": file.content_type,

        logger.error(f"Delete failed: {e}")

        raise HTTPException(status_code=500, detail="Delete failed")            "gridfs_id": gridfs_id,            "uploader": "admin",



@router.post("/documents/{doc_id}/reprocess")            "file_type": file.content_type,            "file_size": len(content)

async def reprocess_document(

    doc_id: str,            "uploader": "public",        }

    mongo_client: MongoDBClient = Depends(get_mongodb_client)

):            "file_size": len(content)        

    """Reprocess document (PUBLIC - NO AUTH)."""

    try:        }        doc_id = await mongo_client.create_document(document_data)

        document = await mongo_client.get_document(doc_id)

        if not document:                

            raise HTTPException(status_code=404, detail="Document not found")

                doc_id = await mongo_client.create_document(document_data)        # Add to ingestion queue

        await mongo_client.update_document_status(doc_id, "PENDING")

        await enqueue_document(doc_id)                await enqueue_document(doc_id)

        

        return {        # Add to ingestion queue        

            "success": True,

            "message": "Document queued for reprocessing",        await enqueue_document(doc_id)        logger.info(f"File {file.filename} uploaded successfully")

            "doc_id": doc_id

        }                

        

    except HTTPException:        logger.info(f"File {file.filename} uploaded successfully")        return UploadResponse(

        raise

    except Exception as e:                    success=True,

        logger.error(f"Reprocess failed: {e}")

        raise HTTPException(status_code=500, detail="Reprocess failed")        return UploadResponse(            doc_id=doc_id,



@router.get("/documents/{doc_id}/download")            success=True,            filename=file.filename,

async def download_document(

    doc_id: str,            doc_id=doc_id,            message="File uploaded successfully and queued for processing"

    mongo_client: MongoDBClient = Depends(get_mongodb_client)

):            filename=file.filename,        )

    """Download document file (PUBLIC - NO AUTH)."""

    try:            message="File uploaded successfully and queued for processing"        

        document = await mongo_client.get_document(doc_id)

        if not document:        )    except HTTPException:

            raise HTTPException(status_code=404, detail="Document not found")

                        raise

        gridfs_id = document.get("gridfs_id")

        if not gridfs_id:    except HTTPException:    except Exception as e:

            raise HTTPException(status_code=404, detail="File not found")

                raise        logger.error(f"File upload failed: {e}")

        file_content = await mongo_client.retrieve_file(gridfs_id)

        filename = document.get("filename", f"document_{doc_id}")    except Exception as e:        raise HTTPException(status_code=500, detail="File upload failed")

        

        return StreamingResponse(        logger.error(f"File upload failed: {e}")

            BytesIO(file_content),

            media_type="application/octet-stream",        raise HTTPException(status_code=500, detail="File upload failed")@router.get("/documents", response_model=List[DocumentResponse])

            headers={"Content-Disposition": f"attachment; filename={filename}"}

        )async def list_documents(

        

    except HTTPException:@router.get("/documents", response_model=List[DocumentResponse])    skip: int = Query(0, ge=0),

        raise

    except Exception as e:async def list_documents(    limit: int = Query(50, ge=1, le=200),

        logger.error(f"Download failed: {e}")

        raise HTTPException(status_code=500, detail="Download failed")    skip: int = Query(0, ge=0),    status_filter: Optional[str] = Query(None),



@router.get("/stats")    limit: int = Query(50, ge=1, le=200),    mongo_client: MongoDBClient = Depends(get_mongodb_client)

async def get_system_stats(

    mongo_client: MongoDBClient = Depends(get_mongodb_client),    status_filter: Optional[str] = Query(None),):

    qdrant_client: QdrantDBClient = Depends(get_qdrant_client),

    arango_client: ArangoDBClient = Depends(get_arango_client),    mongo_client: MongoDBClient = Depends(get_mongodb_client)    """

    redis_client: RedisClient = Depends(get_redis_client)

):):    List documents with pagination and filtering.

    """Get system statistics (PUBLIC - NO AUTH)."""

    try:    """    

        mongodb_stats = {}

        qdrant_stats = {}    List documents with pagination and filtering (PUBLIC ENDPOINT - NO AUTH REQUIRED).    Args:

        arangodb_stats = {}

        redis_stats = {}            skip: Number of documents to skip

        

        try:    Args:        limit: Maximum number of documents to return

            if mongo_client.database:

                db_stats = await mongo_client.database.command("dbStats")        skip: Number of documents to skip        status_filter: Optional status filter

                mongodb_stats = {

                    "database": settings.mongodb_db,        limit: Maximum number of documents to return        mongo_client: MongoDB client

                    "collections": db_stats.get("collections", 0),

                    "objects": db_stats.get("objects", 0),        status_filter: Optional status filter    

                    "dataSize": db_stats.get("dataSize", 0),

                    "indexSize": db_stats.get("indexSize", 0)        mongo_client: MongoDB client    Returns:

                }

        except Exception as e:            List of documents

            logger.warning(f"MongoDB stats failed: {e}")

            Returns:    """

        try:

            qdrant_stats = await qdrant_client.get_collection_info()        List of documents    try:

        except Exception as e:

            logger.warning(f"Qdrant stats failed: {e}")    """        documents = await mongo_client.list_documents(skip, limit)

        

        try:    try:        

            arangodb_stats = await arango_client.get_graph_statistics()

        except Exception as e:        documents = await mongo_client.list_documents(skip, limit)        # Apply status filter if provided

            logger.warning(f"ArangoDB stats failed: {e}")

                        if status_filter:

        try:

            redis_stats = await redis_client.get_redis_info()        # Apply status filter if provided            documents = [doc for doc in documents if doc.get("ingest_status") == status_filter]

        except Exception as e:

            logger.warning(f"Redis stats failed: {e}")        if status_filter:        

        

        processing_stats = await get_processing_statistics()            documents = [doc for doc in documents if doc.get("ingest_status") == status_filter]        # Convert to response model

        

        return {                response_docs = []

            "mongodb": mongodb_stats,

            "qdrant": qdrant_stats,        # Convert to response model        for doc in documents:

            "arangodb": arangodb_stats,

            "redis": redis_stats,        response_docs = []            response_docs.append(DocumentResponse(

            "processing": processing_stats

        }        for doc in documents:                id=doc["_id"],

        

    except Exception as e:            response_docs.append(DocumentResponse(                filename=doc.get("filename", ""),

        logger.error(f"Stats failed: {e}")

        raise HTTPException(status_code=500, detail="Failed to retrieve stats")                id=str(doc.get("_id", "")),                file_type=doc.get("file_type", ""),



@router.get("/logs/{doc_id}")                filename=doc.get("filename", ""),                uploaded_at=doc.get("uploaded_at", datetime.utcnow()).isoformat(),

async def get_document_logs(

    doc_id: str,                file_type=doc.get("file_type", ""),                uploader=doc.get("uploader", ""),

    mongo_client: MongoDBClient = Depends(get_mongodb_client)

):                uploaded_at=doc.get("uploaded_at", datetime.utcnow()).isoformat(),                ingest_status=doc.get("ingest_status", "UNKNOWN"),

    """Get document logs (PUBLIC - NO AUTH)."""

    try:                uploader=doc.get("uploader", ""),                error_message=doc.get("error_message")

        logs = await mongo_client.get_ingestion_logs(doc_id)

                        ingest_status=doc.get("ingest_status", "UNKNOWN"),            ))

        response_logs = []

        for log in logs:                error_message=doc.get("error_message")        

            response_logs.append({

                "doc_id": log.get("doc_id", ""),            ))        return response_docs

                "step": log.get("step", ""),

                "status": log.get("status", ""),                

                "message": log.get("message", ""),

                "timestamp": log.get("timestamp", datetime.utcnow()).isoformat(),        return response_docs    except Exception as e:

                "metadata": log.get("metadata")

            })                logger.error(f"Failed to list documents: {e}")

        

        return response_logs    except Exception as e:        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

        

    except Exception as e:        logger.error(f"Failed to list documents: {e}")

        logger.error(f"Logs failed: {e}")

        raise HTTPException(status_code=500, detail="Failed to retrieve logs")        raise HTTPException(status_code=500, detail="Failed to retrieve documents")@router.get("/documents/{doc_id}")



@router.websocket("/ws/status")async def get_document(

async def websocket_admin_updates(

    websocket: WebSocket,@router.get("/documents/{doc_id}")    doc_id: str,

    redis_client: RedisClient = Depends(get_redis_client)

):async def get_document(    current_user: Dict[str, Any] = Depends(require_admin_role),

    """WebSocket for real-time updates (PUBLIC - NO AUTH)."""

    await websocket.accept()    doc_id: str,    mongo_client: MongoDBClient = Depends(get_mongodb_client)

    

    try:    mongo_client: MongoDBClient = Depends(get_mongodb_client)):

        logger.info("WebSocket connected")

        ):    """Get detailed document information."""

        queue_status = await get_queue_status()

        await websocket.send_json({    """Get detailed document information (PUBLIC ENDPOINT - NO AUTH REQUIRED)."""    try:

            "type": "queue_status",

            "data": queue_status,    try:        document = await mongo_client.get_document(doc_id)

            "timestamp": datetime.utcnow().isoformat()

        })        document = await mongo_client.get_document(doc_id)        if not document:

        

        async def listen_for_updates():        if not document:            raise HTTPException(status_code=404, detail="Document not found")

            try:

                async for message in redis_client.subscribe_to_channel("admin_updates"):            raise HTTPException(status_code=404, detail="Document not found")        

                    await websocket.send_json({

                        "type": "update",                # Get chunks information

                        "data": message,

                        "timestamp": datetime.utcnow().isoformat()        # Get chunks information        chunks = await mongo_client.get_chunks(doc_id)

                    })

            except Exception as e:        chunks = await mongo_client.get_chunks(doc_id)        

                logger.error(f"Redis subscription error: {e}")

                        # Get ingestion logs

        listen_task = asyncio.create_task(listen_for_updates())

                # Get ingestion logs        logs = await mongo_client.get_ingestion_logs(doc_id)

        try:

            while True:        logs = await mongo_client.get_ingestion_logs(doc_id)        

                message = await websocket.receive_json()

                                return {

                if message.get("type") == "get_queue_status":

                    queue_status = await get_queue_status()        return {            "document": document,

                    await websocket.send_json({

                        "type": "queue_status",            "document": document,            "chunks_count": len(chunks),

                        "data": queue_status,

                        "timestamp": datetime.utcnow().isoformat()            "chunks_count": len(chunks),            "ingestion_logs": logs,

                    })

                            "ingestion_logs": logs,            "processing_complete": document.get("ingest_status") == "COMPLETED"

        except WebSocketDisconnect:

            logger.info("WebSocket disconnected")            "processing_complete": document.get("ingest_status") == "COMPLETED"        }

        finally:

            listen_task.cancel()        }        

            try:

                await listen_task            except HTTPException:

            except asyncio.CancelledError:

                pass    except HTTPException:        raise

            

            await redis_client.unsubscribe_from_channel("admin_updates")        raise    except Exception as e:

    

    except Exception as e:    except Exception as e:        logger.error(f"Failed to get document {doc_id}: {e}")

        logger.error(f"WebSocket error: {e}")

        try:        logger.error(f"Failed to get document {doc_id}: {e}")        raise HTTPException(status_code=500, detail="Failed to retrieve document")

            await websocket.close(code=1000)

        except:        raise HTTPException(status_code=500, detail="Failed to retrieve document")

            pass

@router.delete("/documents/{doc_id}")

@router.get("/queue/status")

async def get_ingestion_queue_status():@router.delete("/documents/{doc_id}")async def delete_document(

    """Get queue status (PUBLIC - NO AUTH)."""

    try:async def delete_document(    doc_id: str,

        return await get_queue_status()

    except Exception as e:    doc_id: str,    hard_delete: bool = Query(False),

        logger.error(f"Queue status failed: {e}")

        raise HTTPException(status_code=500, detail="Failed to get queue status")    hard_delete: bool = Query(False),    current_user: Dict[str, Any] = Depends(require_admin_role),


    mongo_client: MongoDBClient = Depends(get_mongodb_client),    mongo_client: MongoDBClient = Depends(get_mongodb_client),

    qdrant_client: QdrantDBClient = Depends(get_qdrant_client),    qdrant_client: QdrantDBClient = Depends(get_qdrant_client),

    arango_client: ArangoDBClient = Depends(get_arango_client)    arango_client: ArangoDBClient = Depends(get_arango_client)

):):

    """    """

    Delete a document (soft or hard delete) - PUBLIC ENDPOINT.    Delete a document (soft or hard delete).

        

    Args:    Args:

        doc_id: Document ID to delete        doc_id: Document ID to delete

        hard_delete: Whether to perform hard delete (removes all data)        hard_delete: Whether to perform hard delete (removes all data)

                    current_user: Authenticated admin user

    Returns:        

        Deletion confirmation    Returns:

    """        Deletion confirmation

    try:    """

        document = await mongo_client.get_document(doc_id)    try:

        if not document:        document = await mongo_client.get_document(doc_id)

            raise HTTPException(status_code=404, detail="Document not found")        if not document:

                    raise HTTPException(status_code=404, detail="Document not found")

        if hard_delete:        

            # Hard delete: remove all associated data        if hard_delete:

                        # Hard delete: remove all associated data

            # Delete vectors from Qdrant            

            await qdrant_client.delete_document_vectors(doc_id)            # Delete vectors from Qdrant

                        await qdrant_client.delete_document_vectors(doc_id)

            # Delete chunks from MongoDB            

            await mongo_client.delete_chunks(doc_id)            # Delete chunks from MongoDB

                        await mongo_client.delete_chunks(doc_id)

            # Delete graph data from ArangoDB            

            if settings.use_graph_search:            # Delete graph data from ArangoDB

                await arango_client.delete_document_graph_data(doc_id)            if settings.use_graph_search:

                            await arango_client.delete_document_graph_data(doc_id)

            # Delete GridFS file            

            gridfs_id = document.get("gridfs_id")            # Delete GridFS file

            if gridfs_id:            gridfs_id = document.get("gridfs_id")

                await mongo_client.delete_file(gridfs_id)            if gridfs_id:

                            await mongo_client.delete_file(gridfs_id)

            # Delete document record            

            await mongo_client.database.documents.delete_one({"_id": document["_id"]})            # Delete document record

                        await mongo_client.database.documents.delete_one({"_id": document["_id"]})

            logger.info(f"Hard deleted document {doc_id}")            

            message = "Document permanently deleted"            logger.info(f"Hard deleted document {doc_id} by {current_user['username']}")

                        message = "Document permanently deleted"

        else:            

            # Soft delete: just mark as deleted        else:

            await mongo_client.update_document_status(doc_id, "DELETED")            # Soft delete: just mark as deleted

                        await mongo_client.update_document_status(doc_id, "DELETED")

            logger.info(f"Soft deleted document {doc_id}")            

            message = "Document marked as deleted"            logger.info(f"Soft deleted document {doc_id} by {current_user['username']}")

                    message = "Document marked as deleted"

        return {        

            "success": True,        return {

            "message": message,            "success": True,

            "doc_id": doc_id,            "message": message,

            "hard_delete": hard_delete            "doc_id": doc_id,

        }            "hard_delete": hard_delete

                }

    except HTTPException:        

        raise    except HTTPException:

    except Exception as e:        raise

        logger.error(f"Failed to delete document {doc_id}: {e}")    except Exception as e:

        raise HTTPException(status_code=500, detail="Failed to delete document")        logger.error(f"Failed to delete document {doc_id}: {e}")

        raise HTTPException(status_code=500, detail="Failed to delete document")

@router.post("/documents/{doc_id}/reprocess")

async def reprocess_document(@router.post("/documents/{doc_id}/reprocess")

    doc_id: str,async def reprocess_document(

    mongo_client: MongoDBClient = Depends(get_mongodb_client)    doc_id: str,

):    current_user: Dict[str, Any] = Depends(require_admin_role),

    """Reprocess a document (re-run ingestion) - PUBLIC ENDPOINT."""    mongo_client: MongoDBClient = Depends(get_mongodb_client)

    try:):

        document = await mongo_client.get_document(doc_id)    """Reprocess a document (re-run ingestion)."""

        if not document:    try:

            raise HTTPException(status_code=404, detail="Document not found")        document = await mongo_client.get_document(doc_id)

                if not document:

        # Reset document status            raise HTTPException(status_code=404, detail="Document not found")

        await mongo_client.update_document_status(doc_id, "PENDING")        

                # Reset document status

        # Add to ingestion queue        await mongo_client.update_document_status(doc_id, "PENDING")

        await enqueue_document(doc_id)        

                # Add to ingestion queue

        logger.info(f"Document {doc_id} queued for reprocessing")        await enqueue_document(doc_id)

                

        return {        logger.info(f"Document {doc_id} queued for reprocessing by {current_user['username']}")

            "success": True,        

            "message": "Document queued for reprocessing",        return {

            "doc_id": doc_id            "success": True,

        }            "message": "Document queued for reprocessing",

                    "doc_id": doc_id

    except HTTPException:        }

        raise        

    except Exception as e:    except HTTPException:

        logger.error(f"Failed to reprocess document {doc_id}: {e}")        raise

        raise HTTPException(status_code=500, detail="Failed to reprocess document")    except Exception as e:

        logger.error(f"Failed to reprocess document {doc_id}: {e}")

@router.get("/documents/{doc_id}/download")        raise HTTPException(status_code=500, detail="Failed to reprocess document")

async def download_document(

    doc_id: str,@router.get("/documents/{doc_id}/download")

    mongo_client: MongoDBClient = Depends(get_mongodb_client)async def download_document(

):    doc_id: str,

    """Download original document file - PUBLIC ENDPOINT."""    current_user: Dict[str, Any] = Depends(require_admin_role),

    try:    mongo_client: MongoDBClient = Depends(get_mongodb_client)

        document = await mongo_client.get_document(doc_id)):

        if not document:    """Download original document file."""

            raise HTTPException(status_code=404, detail="Document not found")    try:

                document = await mongo_client.get_document(doc_id)

        # Get file from GridFS        if not document:

        gridfs_id = document.get("gridfs_id")            raise HTTPException(status_code=404, detail="Document not found")

        if not gridfs_id:        

            raise HTTPException(status_code=404, detail="File not found")        # Get file from GridFS

                gridfs_id = document.get("gridfs_id")

        file_content = await mongo_client.retrieve_file(gridfs_id)        if not gridfs_id:

        filename = document.get("filename", f"document_{doc_id}")            raise HTTPException(status_code=404, detail="File not found")

                

        # Create streaming response        file_content = await mongo_client.retrieve_file(gridfs_id)

        def generate():        filename = document.get("filename", f"document_{doc_id}")

            yield file_content        

                # Create streaming response

        return StreamingResponse(        def generate():

            BytesIO(file_content),            yield file_content

            media_type="application/octet-stream",        

            headers={"Content-Disposition": f"attachment; filename={filename}"}        return StreamingResponse(

        )            BytesIO(file_content),

                    media_type="application/octet-stream",

    except HTTPException:            headers={"Content-Disposition": f"attachment; filename={filename}"}

        raise        )

    except Exception as e:        

        logger.error(f"Failed to download document {doc_id}: {e}")    except HTTPException:

        raise HTTPException(status_code=500, detail="Failed to download document")        raise

    except Exception as e:

@router.get("/stats")        logger.error(f"Failed to download document {doc_id}: {e}")

async def get_system_stats(        raise HTTPException(status_code=500, detail="Failed to download document")

    mongo_client: MongoDBClient = Depends(get_mongodb_client),

    qdrant_client: QdrantDBClient = Depends(get_qdrant_client),@router.get("/stats", response_model=SystemStats)

    arango_client: ArangoDBClient = Depends(get_arango_client),async def get_system_stats(

    redis_client: RedisClient = Depends(get_redis_client)    current_user: Dict[str, Any] = Depends(require_admin_role),

):    mongo_client: MongoDBClient = Depends(get_mongodb_client),

    """Get comprehensive system statistics - PUBLIC ENDPOINT."""    qdrant_client: QdrantDBClient = Depends(get_qdrant_client),

    try:    arango_client: ArangoDBClient = Depends(get_arango_client),

        # Get database statistics    redis_client: RedisClient = Depends(get_redis_client)

        mongodb_stats = {}):

        qdrant_stats = {}    """Get comprehensive system statistics."""

        arangodb_stats = {}    try:

        redis_stats = {}        # Get database statistics

                mongodb_stats = {}

        try:        qdrant_stats = {}

            # MongoDB stats        arangodb_stats = {}

            if mongo_client.database:        redis_stats = {}

                db_stats = await mongo_client.database.command("dbStats")        

                mongodb_stats = {        try:

                    "database": settings.mongodb_db,            # MongoDB stats

                    "collections": db_stats.get("collections", 0),            if mongo_client.database:

                    "objects": db_stats.get("objects", 0),                db_stats = await mongo_client.database.command("dbStats")

                    "dataSize": db_stats.get("dataSize", 0),                mongodb_stats = {

                    "indexSize": db_stats.get("indexSize", 0)                    "database": settings.mongodb_db,

                }                    "collections": db_stats.get("collections", 0),

        except Exception as e:                    "objects": db_stats.get("objects", 0),

            logger.warning(f"Failed to get MongoDB stats: {e}")                    "dataSize": db_stats.get("dataSize", 0),

                            "indexSize": db_stats.get("indexSize", 0)

        try:                }

            # Qdrant stats        except Exception as e:

            qdrant_stats = await qdrant_client.get_collection_info()            logger.warning(f"Failed to get MongoDB stats: {e}")

        except Exception as e:        

            logger.warning(f"Failed to get Qdrant stats: {e}")        try:

                    # Qdrant stats

        try:            qdrant_stats = await qdrant_client.get_collection_info()

            # ArangoDB stats        except Exception as e:

            arangodb_stats = await arango_client.get_graph_statistics()            logger.warning(f"Failed to get Qdrant stats: {e}")

        except Exception as e:        

            logger.warning(f"Failed to get ArangoDB stats: {e}")        try:

                    # ArangoDB stats

        try:            arangodb_stats = await arango_client.get_graph_statistics()

            # Redis stats        except Exception as e:

            redis_stats = await redis_client.get_redis_info()            logger.warning(f"Failed to get ArangoDB stats: {e}")

        except Exception as e:        

            logger.warning(f"Failed to get Redis stats: {e}")        try:

                    # Redis stats

        # Processing statistics            redis_stats = await redis_client.get_redis_info()

        processing_stats = await get_processing_statistics()        except Exception as e:

                    logger.warning(f"Failed to get Redis stats: {e}")

        return {        

            "mongodb": mongodb_stats,        # Processing statistics

            "qdrant": qdrant_stats,        processing_stats = await get_processing_statistics()

            "arangodb": arangodb_stats,        

            "redis": redis_stats,        return SystemStats(

            "processing": processing_stats            mongodb=mongodb_stats,

        }            qdrant=qdrant_stats,

                    arangodb=arangodb_stats,

    except Exception as e:            redis=redis_stats,

        logger.error(f"Failed to get system stats: {e}")            processing=processing_stats

        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")        )

        

@router.get("/logs/{doc_id}")    except Exception as e:

async def get_document_logs(        logger.error(f"Failed to get system stats: {e}")

    doc_id: str,        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")

    mongo_client: MongoDBClient = Depends(get_mongodb_client)

):@router.get("/logs/{doc_id}", response_model=List[IngestionLog])

    """Get ingestion logs for a specific document - PUBLIC ENDPOINT."""async def get_document_logs(

    try:    doc_id: str,

        logs = await mongo_client.get_ingestion_logs(doc_id)    current_user: Dict[str, Any] = Depends(require_admin_role),

            mongo_client: MongoDBClient = Depends(get_mongodb_client)

        # Convert to response model):

        response_logs = []    """Get ingestion logs for a specific document."""

        for log in logs:    try:

            response_logs.append({        logs = await mongo_client.get_ingestion_logs(doc_id)

                "doc_id": log.get("doc_id", ""),        

                "step": log.get("step", ""),        # Convert to response model

                "status": log.get("status", ""),        response_logs = []

                "message": log.get("message", ""),        for log in logs:

                "timestamp": log.get("timestamp", datetime.utcnow()).isoformat(),            response_logs.append(IngestionLog(

                "metadata": log.get("metadata")                doc_id=log.get("doc_id", ""),

            })                step=log.get("step", ""),

                        status=log.get("status", ""),

        return response_logs                message=log.get("message", ""),

                        timestamp=log.get("timestamp", datetime.utcnow()).isoformat(),

    except Exception as e:                metadata=log.get("metadata")

        logger.error(f"Failed to get logs for document {doc_id}: {e}")            ))

        raise HTTPException(status_code=500, detail="Failed to retrieve document logs")        

        return response_logs

# Real-time WebSocket endpoint for admin updates        

@router.websocket("/ws/status")    except Exception as e:

async def websocket_admin_updates(        logger.error(f"Failed to get logs for document {doc_id}: {e}")

    websocket: WebSocket,        raise HTTPException(status_code=500, detail="Failed to retrieve document logs")

    redis_client: RedisClient = Depends(get_redis_client)

):# Real-time WebSocket endpoint for admin updates

    """@router.websocket("/ws/status")

    WebSocket endpoint for real-time admin updates (PUBLIC ENDPOINT - NO AUTH).async def websocket_admin_updates(

    Streams ingestion status updates and system notifications.    websocket: WebSocket,

    """    redis_client: RedisClient = Depends(get_redis_client)

    await websocket.accept()):

        """

    try:    WebSocket endpoint for real-time admin updates.

        logger.info("Admin WebSocket client connected")    Streams ingestion status updates and system notifications.

            """

        # Send initial queue status    await websocket.accept()

        queue_status = await get_queue_status()    

        await websocket.send_json({    try:

            "type": "queue_status",        logger.info("Admin WebSocket client connected")

            "data": queue_status,        

            "timestamp": datetime.utcnow().isoformat()        # Send initial queue status

        })        queue_status = await get_queue_status()

                await websocket.send_json({

        # Subscribe to Redis updates            "type": "queue_status",

        async def listen_for_updates():            "data": queue_status,

            try:            "timestamp": datetime.utcnow().isoformat()

                async for message in redis_client.subscribe_to_channel("admin_updates"):        })

                    await websocket.send_json({        

                        "type": "update",        # Subscribe to Redis updates

                        "data": message,        async def listen_for_updates():

                        "timestamp": datetime.utcnow().isoformat()            try:

                    })                async for message in redis_client.subscribe_to_channel("admin_updates"):

            except Exception as e:                    await websocket.send_json({

                logger.error(f"Error in Redis subscription: {e}")                        "type": "update",

                                "data": message,

        # Start listening for updates                        "timestamp": datetime.utcnow().isoformat()

        listen_task = asyncio.create_task(listen_for_updates())                    })

                    except Exception as e:

        # Keep connection alive and handle client messages                logger.error(f"Error in Redis subscription: {e}")

        try:        

            while True:        # Start listening for updates

                # Wait for client message or connection close        listen_task = asyncio.create_task(listen_for_updates())

                message = await websocket.receive_json()        

                        # Keep connection alive and handle client messages

                # Handle client requests        try:

                if message.get("type") == "get_queue_status":            while True:

                    queue_status = await get_queue_status()                # Wait for client message or connection close

                    await websocket.send_json({                message = await websocket.receive_json()

                        "type": "queue_status",                

                        "data": queue_status,                # Handle client requests

                        "timestamp": datetime.utcnow().isoformat()                if message.get("type") == "get_queue_status":

                    })                    queue_status = await get_queue_status()

                                    await websocket.send_json({

        except WebSocketDisconnect:                        "type": "queue_status",

            logger.info("Admin WebSocket client disconnected")                        "data": queue_status,

        finally:                        "timestamp": datetime.utcnow().isoformat()

            # Cancel the listening task                    })

            listen_task.cancel()                

            try:        except WebSocketDisconnect:

                await listen_task            logger.info("Admin WebSocket client disconnected")

            except asyncio.CancelledError:        finally:

                pass            # Cancel the listening task

                        listen_task.cancel()

            # Unsubscribe from Redis            try:

            await redis_client.unsubscribe_from_channel("admin_updates")                await listen_task

                except asyncio.CancelledError:

    except Exception as e:                pass

        logger.error(f"WebSocket error: {e}")            

        try:            # Unsubscribe from Redis

            await websocket.close(code=1000)            await redis_client.unsubscribe_from_channel("admin_updates")

        except:    

            pass    except Exception as e:

        logger.error(f"WebSocket error: {e}")

@router.get("/queue/status")        try:

async def get_ingestion_queue_status():            await websocket.close(code=1000)

    """Get current ingestion queue status - PUBLIC ENDPOINT."""        except:

    try:            pass

        return await get_queue_status()

    except Exception as e:@router.get("/queue/status")

        logger.error(f"Failed to get queue status: {e}")async def get_ingestion_queue_status(

        raise HTTPException(status_code=500, detail="Failed to get queue status")    current_user: Dict[str, Any] = Depends(require_admin_role)

):
    """Get current ingestion queue status."""
    try:
        return await get_queue_status()
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queue status")