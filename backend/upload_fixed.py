"""Upload the properly fixed JSON file to all PENDING documents."""
import asyncio
from bson import ObjectId
from app.core.db_mongo import get_mongodb_client

async def upload_fixed_file():
    """Upload the fixed JSON to all PENDING documents."""
    mongo_client = await get_mongodb_client()
    if not mongo_client.is_connected():
        await mongo_client.connect()
    
    # Read the fixed file
    with open("d:/OMS Chat Bot/backend/fixed_faq.json", "rb") as f:
        fixed_content = f.read()
    
    print(f"Fixed file size: {len(fixed_content)} bytes")
    
    # Get all PENDING documents
    pending_docs = []
    async for doc in mongo_client.db.documents.find({"ingest_status": "PENDING"}):
        pending_docs.append(doc)
    
    print(f"Updating {len(pending_docs)} documents...")
    
    for doc in pending_docs:
        doc_id = doc["_id"]
        filename = doc["filename"]
        old_gridfs_id = ObjectId(doc["gridfs_id"])
        
        # Delete old file
        try:
            await mongo_client.fs_bucket.delete(old_gridfs_id)
        except:
            pass
        
        # Upload new file
        new_gridfs_id = await mongo_client.fs_bucket.upload_from_stream(
            filename,
            fixed_content,
            metadata={"content_type": "application/json"}
        )
        
        # Update document
        await mongo_client.db.documents.update_one(
            {"_id": doc_id},
            {"$set": {
                "gridfs_id": str(new_gridfs_id),
                "size": len(fixed_content)
            }}
        )
        
        print(f"  ✅ Updated: {filename}")
    
    print(f"\n✅ All {len(pending_docs)} documents updated with fixed JSON!")

if __name__ == "__main__":
    asyncio.run(upload_fixed_file())
