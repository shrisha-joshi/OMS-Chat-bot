"""
Background worker for document ingestion processing.
This module handles the asynchronous processing of uploaded documents,
including parsing, chunking, embedding, and indexing operations.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
import hashlib

from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..core.db_qdrant import get_qdrant_client, QdrantDBClient
from ..core.db_arango import get_arango_client, ArangoDBClient
from ..core.cache_redis import get_redis_client, RedisClient
from ..services.ingest_service import IngestService
from ..config import settings

logger = logging.getLogger(__name__)

# Global queue for ingestion tasks
ingestion_queue: asyncio.Queue = asyncio.Queue()

class IngestWorker:
    """Background worker for document ingestion."""
    
    def __init__(self):
        self.is_running = False
        self.processed_count = 0
        self.failed_count = 0
        self.ingest_service = None
        
    async def initialize(self):
        """Initialize the worker with required services."""
        try:
            self.ingest_service = IngestService()
            await self.ingest_service.initialize()
            logger.info("Ingest worker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ingest worker: {e}")
            raise
    
    async def start(self):
        """Start the background worker loop."""
        self.is_running = True
        logger.info("Starting ingest worker...")
        
        while self.is_running:
            try:
                # Wait for ingestion tasks
                doc_id = await asyncio.wait_for(
                    ingestion_queue.get(), 
                    timeout=1.0
                )
                
                logger.info(f"Processing document: {doc_id}")
                await self._process_document(doc_id)
                
                # Mark task as done
                ingestion_queue.task_done()
                
            except asyncio.TimeoutError:
                # No tasks available, continue loop
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
    
    async def stop(self):
        """Stop the background worker."""
        self.is_running = False
        logger.info(f"Ingest worker stopped. Processed: {self.processed_count}, Failed: {self.failed_count}")
    
    async def _process_document(self, doc_id: str):
        """
        Process a single document through the ingestion pipeline.
        
        Args:
            doc_id: Document ID to process
        """
        start_time = datetime.utcnow()
        
        try:
            # Get MongoDB client
            mongo_client = await get_mongodb_client()
            redis_client = await get_redis_client()
            
            # Update status to processing
            await mongo_client.update_document_status(doc_id, "PROCESSING")
            await mongo_client.log_ingestion_step(
                doc_id, "START", "SUCCESS", "Started document processing"
            )
            
            # Publish update to Redis
            await redis_client.publish_ingestion_update(
                doc_id, "PROCESSING", "Document processing started"
            )
            
            # Process the document
            success = await self.ingest_service.process_document(doc_id)
            
            if success:
                # Update status to completed
                await mongo_client.update_document_status(doc_id, "COMPLETED")
                await mongo_client.log_ingestion_step(
                    doc_id, "COMPLETE", "SUCCESS", "Document processing completed successfully"
                )
                
                # Publish completion update
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                await redis_client.publish_ingestion_update(
                    doc_id, "COMPLETED", 
                    f"Document processed successfully in {processing_time:.2f}s",
                    {"processing_time": processing_time}
                )
                
                self.processed_count += 1
                logger.info(f"Document {doc_id} processed successfully")
            else:
                raise Exception("Document processing failed")
                
        except Exception as e:
            # Update status to failed
            error_message = str(e)
            
            try:
                mongo_client = await get_mongodb_client()
                redis_client = await get_redis_client()
                
                await mongo_client.update_document_status(
                    doc_id, "FAILED", error_message
                )
                await mongo_client.log_ingestion_step(
                    doc_id, "ERROR", "FAILED", error_message
                )
                
                # Publish failure update
                await redis_client.publish_ingestion_update(
                    doc_id, "FAILED", f"Processing failed: {error_message}"
                )
                
            except Exception as log_error:
                logger.error(f"Failed to log error for document {doc_id}: {log_error}")
            
            self.failed_count += 1
            logger.error(f"Document {doc_id} processing failed: {error_message}")


# Global worker instance
worker = IngestWorker()


async def enqueue_document(doc_id: str):
    """
    Add a document to the ingestion queue.
    
    Args:
        doc_id: Document ID to process
    """
    try:
        await ingestion_queue.put(doc_id)
        logger.info(f"Document {doc_id} added to ingestion queue")
    except Exception as e:
        logger.error(f"Failed to enqueue document {doc_id}: {e}")


async def get_queue_status() -> Dict[str, Any]:
    """Get current queue status."""
    return {
        "queue_size": ingestion_queue.qsize(),
        "is_running": worker.is_running,
        "processed_count": worker.processed_count,
        "failed_count": worker.failed_count,
        "success_rate": (
            worker.processed_count / max(worker.processed_count + worker.failed_count, 1)
        ) * 100
    }


async def start_ingest_worker():
    """Start the ingestion worker (called from main.py)."""
    try:
        await worker.initialize()
        await worker.start()
    except asyncio.CancelledError:
        logger.info("Ingest worker was cancelled")
        await worker.stop()
    except Exception as e:
        logger.error(f"Ingest worker failed: {e}")
        await worker.stop()
        raise


async def stop_ingest_worker():
    """Stop the ingestion worker."""
    await worker.stop()


# Utility functions for queue management

async def clear_queue():
    """Clear all items from the ingestion queue."""
    try:
        count = 0
        while not ingestion_queue.empty():
            try:
                ingestion_queue.get_nowait()
                ingestion_queue.task_done()
                count += 1
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"Cleared {count} items from ingestion queue")
        return count
        
    except Exception as e:
        logger.error(f"Failed to clear queue: {e}")
        return 0


async def retry_failed_documents():
    """Retry processing for all failed documents."""
    try:
        mongo_client = await get_mongodb_client()
        
        # Find all failed documents
        cursor = mongo_client.database.documents.find({"ingest_status": "FAILED"})
        failed_docs = []
        
        async for doc in cursor:
            doc_id = str(doc["_id"])
            failed_docs.append(doc_id)
            
            # Reset status and re-enqueue
            await mongo_client.update_document_status(doc_id, "PENDING")
            await enqueue_document(doc_id)
        
        logger.info(f"Re-queued {len(failed_docs)} failed documents for retry")
        return len(failed_docs)
        
    except Exception as e:
        logger.error(f"Failed to retry failed documents: {e}")
        return 0


async def get_processing_statistics() -> Dict[str, Any]:
    """Get detailed processing statistics."""
    try:
        mongo_client = await get_mongodb_client()
        
        # Get document counts by status
        pipeline = [
            {"$group": {"_id": "$ingest_status", "count": {"$sum": 1}}}
        ]
        
        cursor = mongo_client.database.documents.aggregate(pipeline)
        status_counts = {doc["_id"]: doc["count"] async for doc in cursor}
        
        # Get recent processing times
        recent_logs = mongo_client.database.ingestion_logs.find({
            "step": "COMPLETE",
            "status": "SUCCESS"
        }).sort("timestamp", -1).limit(10)
        
        processing_times = []
        async for log in recent_logs:
            if "processing_time" in log.get("metadata", {}):
                processing_times.append(log["metadata"]["processing_time"])
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "queue_status": await get_queue_status(),
            "document_counts": status_counts,
            "average_processing_time": avg_processing_time,
            "recent_processing_times": processing_times
        }
        
    except Exception as e:
        logger.error(f"Failed to get processing statistics: {e}")
        return {}