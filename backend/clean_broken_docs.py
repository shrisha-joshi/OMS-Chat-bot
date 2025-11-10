"""Delete all broken PENDING documents that have no GridFS content."""
import asyncio
import logging
from bson import ObjectId
from app.core.db_mongo import get_mongodb_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def clean_broken_documents():
    """Delete all PENDING documents with missing GridFS files."""
    try:
        mongo_client = await get_mongodb_client()
        
        if not mongo_client.is_connected():
            await mongo_client.connect()
        
        # Find all PENDING documents
        pending_docs = []
        async for doc in mongo_client.db.documents.find({"ingest_status": "PENDING"}):
            pending_docs.append(doc)
        
        logger.info(f"Found {len(pending_docs)} PENDING documents")
        
        deleted_count = 0
        for doc in pending_docs:
            doc_id = doc["_id"]
            filename = doc.get("filename", "unknown")
            gridfs_id = doc.get("gridfs_id")
            
            if gridfs_id:
                # Check if GridFS file exists
                try:
                    exists = await mongo_client.fs_bucket.find({"_id": gridfs_id}).to_list(length=1)
                    if not exists:
                        logger.warning(f"Missing GridFS for {filename} - DELETING")
                        await mongo_client.db.documents.delete_one({"_id": doc_id})
                        deleted_count += 1
                    else:
                        logger.info(f"GridFS OK for {filename}")
                except:
                    logger.warning(f"Error checking GridFS for {filename} - DELETING")
                    await mongo_client.db.documents.delete_one({"_id": doc_id})
                    deleted_count += 1
            else:
                logger.warning(f"No GridFS ID for {filename} - DELETING")
                await mongo_client.db.documents.delete_one({"_id": doc_id})
                deleted_count += 1
        
        logger.info(f"✅ Deleted {deleted_count} broken documents")
        logger.info(f"Remaining: {len(pending_docs) - deleted_count} valid PENDING documents")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(clean_broken_documents())
