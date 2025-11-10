"""Check document structure."""
import asyncio
import json
from app.core.db_mongo import get_mongodb_client

async def show_doc_structure():
    mongo_client = await get_mongodb_client()
    if not mongo_client.is_connected():
        await mongo_client.connect()
    
    doc = await mongo_client.db.documents.find_one({"ingest_status": "PENDING"})
    
    # Print document structure (without binary data)
    doc_copy = doc.copy()
    if '_id' in doc_copy:
        doc_copy['_id'] = str(doc_copy['_id'])
    if 'gridfs_id' in doc_copy:
        doc_copy['gridfs_id'] = str(doc_copy['gridfs_id'])
    
    print(json.dumps(doc_copy, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(show_doc_structure())
