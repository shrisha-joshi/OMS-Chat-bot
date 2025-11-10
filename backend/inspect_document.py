"""Script to download and inspect a PENDING document."""
import asyncio
import logging
from app.core.db_mongo import get_mongodb_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def inspect_document():
    """Get one PENDING document and show its content."""
    try:
        mongo_client = await get_mongodb_client()
        
        if not mongo_client.is_connected():
            await mongo_client.connect()
        
        # Get one document
        doc = await mongo_client.db.documents.find_one({"ingest_status": "PENDING"})
        
        if not doc:
            logger.info("No PENDING documents found")
            return
        
        doc_id = doc["_id"]
        filename = doc["filename"]
        
        logger.info(f"Found: {filename} (ID: {doc_id})")
        
        # Get document metadata
        doc_meta = await mongo_client.get_document(str(doc_id))
        gridfs_id = doc_meta.get("gridfs_id")
        
        if not gridfs_id:
            logger.error("No GridFS ID found")
            return
        
        # Get content from GridFS
        fs_file = await mongo_client.fs_bucket.open_download_stream(gridfs_id)
        content = await fs_file.read()
        
        # Show first 500 characters
        text = content.decode('utf-8', errors='ignore')
        logger.info(f"\n=== First 500 characters ===\n{text[:500]}\n")
        
        # Save to file for inspection
        with open("d:/OMS Chat Bot/backend/sample_document.json", "wb") as f:
            f.write(content)
        
        logger.info(f"Full document saved to: backend/sample_document.json")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(inspect_document())
