"""Script to check and process PENDING documents."""
import asyncio
import logging
from app.core.db_mongo import get_mongodb_client
from app.services.ingest_service import IngestService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_pending_documents():
    """Find and process all PENDING documents."""
    try:
        # Get MongoDB client
        mongo_client = await get_mongodb_client()
        
        # Ensure connected
        if not mongo_client.is_connected():
            await mongo_client.connect()
        
        # Find all PENDING documents
        pending_docs = []
        cursor = mongo_client.db.documents.find({"ingest_status": "PENDING"})
        async for doc in cursor:
            pending_docs.append(doc)
        
        logger.info(f"Found {len(pending_docs)} PENDING documents")
        
        if not pending_docs:
            logger.info("No PENDING documents to process")
            return
        
        # Initialize ingest service
        ingest_service = IngestService()
        await ingest_service.initialize()
        
        # Ensure Qdrant client is connected
        if not ingest_service.qdrant_client.is_connected():
            logger.info("Connecting to Qdrant...")
            await ingest_service.qdrant_client.connect()
        
        # Process each document
        for doc in pending_docs:
            doc_id = str(doc["_id"])
            filename = doc.get("filename", "unknown")
            logger.info(f"Processing: {filename} (ID: {doc_id})")
            
            try:
                success = await ingest_service.process_document(doc_id)
                if success:
                    logger.info(f"  ✅ SUCCESS: {filename}")
                else:
                    logger.warning(f"  ⚠️ PARTIAL: {filename}")
            except Exception as e:
                logger.error(f"  ❌ FAILED: {filename} - {e}")
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(process_pending_documents())
