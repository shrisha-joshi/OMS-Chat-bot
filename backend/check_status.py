"""Check document processing status."""
import asyncio
from app.core.db_mongo import get_mongodb_client

async def check_documents():
    """Check all document statuses."""
    mongo_client = await get_mongodb_client()
    if not mongo_client.is_connected():
        await mongo_client.connect()
    
    print("\n" + "="*80)
    print(" DOCUMENT PROCESSING STATUS")
    print("="*80 + "\n")
    
    cursor = mongo_client.db.documents.find({})
    doc_count = 0
    completed = 0
    pending = 0
    failed = 0
    
    async for doc in cursor:
        doc_count += 1
        status = doc.get("ingest_status", "UNKNOWN")
        filename = doc.get("filename", "N/A")
        chunks = doc.get("chunks_count", 0)
        uploaded = doc.get("uploaded_at", "N/A")
        
        if status == "COMPLETED":
            completed += 1
            symbol = "✅"
        elif status == "PENDING":
            pending += 1
            symbol = "⏸️"
        elif status == "FAILED":
            failed += 1
            symbol = "❌"
        else:
            symbol = "❓"
        
        print(f"{symbol} {filename}")
        print(f"   Status: {status}")
        print(f"   Chunks: {chunks}")
        print(f"   Uploaded: {uploaded}")
        print()
    
    print("="*80)
    print(f" SUMMARY: {doc_count} total | {completed} completed | {pending} pending | {failed} failed")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(check_documents())
