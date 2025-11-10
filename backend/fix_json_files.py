"""Fix the JSON files by replacing ISODate format with plain strings."""
import asyncio
import re
from bson import ObjectId
from app.core.db_mongo import get_mongodb_client

async def fix_json_files():
    """Fix all PENDING documents with ISODate issues."""
    mongo_client = await get_mongodb_client()
    if not mongo_client.is_connected():
        await mongo_client.connect()
    
    pending_docs = []
    async for doc in mongo_client.db.documents.find({"ingest_status": "PENDING"}):
        pending_docs.append(doc)
    
    print(f"Found {len(pending_docs)} PENDING documents to fix")
    
    for doc in pending_docs:
        doc_id = doc["_id"]
        filename = doc["filename"]
        gridfs_id = ObjectId(doc["gridfs_id"])
        
        print(f"\nFixing: {filename}")
        
        # Download file
        fs_file = await mongo_client.fs_bucket.open_download_stream(gridfs_id)
        content = await fs_file.read()
        text = content.decode('utf-8')
        
        # Fix ISODate format
        # Replace all variations: ISODate("..."), ISODate('...'), "ISODate(...)", etc.
        fixed_text = text
        
        # Pattern 1: "ISODate("date")"
        fixed_text = re.sub(r'"ISODate\("([^"]+)"\)"', r'"\1"', fixed_text)
        
        # Pattern 2: "ISODate('date')"  
        fixed_text = re.sub(r'"ISODate\(\'([^\']+)\'\)"', r'"\1"', fixed_text)
        
        # Pattern 3: ISODate("date") without quotes
        fixed_text = re.sub(r'ISODate\("([^"]+)"\)', r'"\1"', fixed_text)
        
        # Pattern 4: ISODate('date') without quotes
        fixed_text = re.sub(r'ISODate\(\'([^\']+)\'\)', r'"\1"', fixed_text)
        
        # Also fix any ObjectId references
        fixed_text = re.sub(r'"ObjectId\("([^"]+)"\)"', r'"\1"', fixed_text)
        fixed_text = re.sub(r'ObjectId\("([^"]+)"\)', r'"\1"', fixed_text)
        
        changes_made = (fixed_text != text)
        
        if changes_made:
            # Count fixes
            iso_fixes = text.count('ISODate') - fixed_text.count('ISODate')
            print(f"  - Fixed {iso_fixes} ISODate occurrences")
            
            # Validate JSON before uploading
            try:
                import json
                json.loads(fixed_text)
                print(f"  - JSON validation: PASSED")
            except Exception as e:
                print(f"  - JSON validation: FAILED - {e}")
                print(f"  - Skipping this file")
                continue
            
            # Delete old GridFS file
            await mongo_client.fs_bucket.delete(gridfs_id)
            
            # Upload fixed content
            new_gridfs_id = await mongo_client.fs_bucket.upload_from_stream(
                filename,
                fixed_text.encode('utf-8'),
                metadata={"content_type": "application/json"}
            )
            
            # Update document with new GridFS ID
            await mongo_client.db.documents.update_one(
                {"_id": doc_id},
                {"$set": {"gridfs_id": str(new_gridfs_id)}}
            )
            
            print(f"  ✅ Fixed and re-uploaded {filename}")
        else:
            print(f"  - No ISODate issues found")
    
    print(f"\n✅ All documents processed!")

if __name__ == "__main__":
    asyncio.run(fix_json_files())
