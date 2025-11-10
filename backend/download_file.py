"""Download the actual GridFS file."""
import asyncio
from bson import ObjectId
from app.core.db_mongo import get_mongodb_client

async def download_file():
    mongo_client = await get_mongodb_client()
    if not mongo_client.is_connected():
        await mongo_client.connect()
    
    doc = await mongo_client.db.documents.find_one({"ingest_status": "PENDING"})
    gridfs_id_str = doc["gridfs_id"]
    
    # Convert string to ObjectId
    gridfs_id = ObjectId(gridfs_id_str)
    
    # Download file
    fs_file = await mongo_client.fs_bucket.open_download_stream(gridfs_id)
    content = await fs_file.read()
    
    # Save to file
    with open("d:/OMS Chat Bot/backend/downloaded_faq.json", "wb") as f:
        f.write(content)
    
    print(f"Downloaded {len(content)} bytes to: downloaded_faq.json")
    
    # Show first 300 characters
    text = content.decode('utf-8', errors='ignore')
    print(f"\n===Content preview (first 300 chars)===")
    print(text[:300])

if __name__ == "__main__":
    asyncio.run(download_file())
