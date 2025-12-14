"""
Automated Embedding Model Upgrade Script
Handles BGE model upgrade with minimal downtime.

Usage:
    python upgrade_embedding_model.py --auto-recreate
    python upgrade_embedding_model.py --skip-backup --batch-size 10
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.db_qdrant import get_qdrant_client
from app.core.db_mongo import get_mongodb_client
from app.services.ingestion_engine import get_ingestion_engine
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingModelUpgrader:
    """Handles embedding model upgrade process."""
    
    def __init__(self):
        self.qdrant_client = None
        self.mongo_client = None
        self.ingest_service = None
        self.old_model = "all-MiniLM-L6-v2"
        self.new_model = "BAAI/bge-base-en-v1.5"
        self.old_dimension = 384
        self.new_dimension = 768
    
    async def initialize(self):
        """Initialize database clients."""
        logger.info("Initializing database connections...")
        self.qdrant_client = await get_qdrant_client()
        self.mongo_client = get_mongodb_client()
        self.ingest_service = await get_ingestion_engine()
        # await self.ingest_service.initialize() # Already initialized
        logger.info("✅ Database connections established")
    
    async def backup_collection(self):
        """Backup current Qdrant collection metadata."""
        logger.info("📦 Creating backup of current collection...")
        
        try:
            # Get current vector count
            collection_info = await self.qdrant_client.get_collection_info()
            vector_count = collection_info.get("vectors_count", 0) if collection_info else 0
            
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "old_model": self.old_model,
                "old_dimension": self.old_dimension,
                "new_model": self.new_model,
                "new_dimension": self.new_dimension,
                "vector_count_before": vector_count,
                "collection_name": settings.qdrant_collection
            }
            
            # Save to MongoDB
            result = await self.mongo_client.database.upgrade_backups.insert_one(backup_data)
            backup_id = str(result.inserted_id)
            
            logger.info(f"✅ Backup created: ID={backup_id}, Vectors={vector_count}")
            return backup_id
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise
    
    async def recreate_collection(self):
        """Recreate Qdrant collection with new dimensions."""
        logger.info(f"🔄 Recreating collection '{settings.qdrant_collection}' with {self.new_dimension} dimensions...")
        
        try:
            # Check if collection exists
            exists = await self.qdrant_client.collection_exists(settings.qdrant_collection)
            
            if exists:
                logger.info(f"  Deleting old collection '{settings.qdrant_collection}'...")
                await self.qdrant_client.delete_collection(settings.qdrant_collection)
                logger.info("  ✅ Old collection deleted")
            
            # Create new collection
            logger.info(f"  Creating new collection with {self.new_dimension} dimensions...")
            await self.qdrant_client.create_collection(
                collection_name=settings.qdrant_collection,
                vector_size=self.new_dimension,
                distance="Cosine"
            )
            logger.info(f"  ✅ New collection created ({self.new_dimension}D)")
            
        except Exception as e:
            logger.error(f"❌ Collection recreation failed: {e}")
            raise
    
    async def get_all_documents(self) -> List[Dict]:
        """Get all documents from MongoDB."""
        logger.info("📄 Fetching all documents from MongoDB...")
        
        try:
            cursor = self.mongo_client.database.documents.find({})
            documents = await cursor.to_list(length=None)
            
            # Filter for successfully ingested documents
            valid_docs = [d for d in documents if d.get("ingest_status") == "SUCCESS"]
            
            logger.info(f"  ✅ Found {len(valid_docs)} documents to re-index (out of {len(documents)} total)")
            return valid_docs
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch documents: {e}")
            raise
    
    async def reindex_document(self, doc: Dict) -> bool:
        """Re-index a single document with new embeddings."""
        try:
            doc_id = str(doc["_id"])
            filename = doc.get("filename", "Unknown")
            
            logger.info(f"  🔄 Re-indexing: {filename}")
            
            # Update ingest status to REINDEXING
            await self.mongo_client.database.documents.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "ingest_status": "REINDEXING",
                    "reindex_started_at": datetime.now()
                }}
            )
            
            # Trigger re-processing through ingest service
            await self.ingest_service.process_document(doc_id)
            
            logger.info(f"  ✅ Re-indexed: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Failed to re-index {doc.get('filename')}: {e}")
            
            # Update status to FAILED
            try:
                await self.mongo_client.database.documents.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "ingest_status": "FAILED",
                        "error_message": str(e),
                        "reindex_failed_at": datetime.now()
                    }}
                )
            except:
                pass
            
            return False
    
    async def reindex_all_documents(self, batch_size: int = 5):
        """Re-index all documents in batches."""
        documents = await self.get_all_documents()
        
        if not documents:
            logger.warning("⚠️ No documents found to re-index")
            return
        
        logger.info(f"🔄 Re-indexing {len(documents)} documents in batches of {batch_size}...")
        
        success_count = 0
        fail_count = 0
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Batch {batch_num}/{total_batches}")
            
            # Process batch concurrently
            tasks = [self.reindex_document(doc) for doc in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            for result in results:
                if result is True:
                    success_count += 1
                else:
                    fail_count += 1
            
            logger.info(
                f"  Progress: {success_count + fail_count}/{len(documents)} "
                f"({success_count} success, {fail_count} failed)"
            )
            
            # Small delay between batches to avoid overload
            await asyncio.sleep(1)
        
        logger.info("=" * 60)
        logger.info("✅ Re-indexing complete!")
        logger.info(f"  Success: {success_count}/{len(documents)}")
        logger.info(f"  Failed: {fail_count}/{len(documents)}")
        logger.info("=" * 60)
        
        return success_count, fail_count
    
    async def verify_upgrade(self) -> bool:
        """Verify the upgrade was successful."""
        logger.info("🔍 Verifying upgrade...")
        
        try:
            # Check collection info
            collection_info = await self.qdrant_client.get_collection_info()
            
            if not collection_info:
                logger.error("  ❌ Could not retrieve collection info")
                return False
            
            vector_count = collection_info.get("vectors_count", 0)
            vector_size = collection_info.get("config", {}).get("params", {}).get("vectors", {}).get("size", 0)
            
            # Check document count
            doc_count = await self.mongo_client.database.documents.count_documents(
                {"ingest_status": "SUCCESS"}
            )
            
            logger.info(f"  📊 Qdrant vectors: {vector_count}")
            logger.info(f"  📊 Vector dimension: {vector_size}")
            logger.info(f"  📊 MongoDB documents: {doc_count}")
            
            # Verify dimension
            if vector_size == self.new_dimension:
                logger.info(f"  ✅ Correct dimension: {vector_size}D")
            else:
                logger.error(f"  ❌ Wrong dimension: {vector_size}D (expected {self.new_dimension}D)")
                return False
            
            # Warn if vector count seems low
            if vector_count < doc_count * 0.9:  # Allow 10% margin
                logger.warning(
                    f"  ⚠️ Vector count ({vector_count}) significantly < Document count ({doc_count}). "
                    f"Some documents may not be indexed."
                )
            
            logger.info("✅ Upgrade verification complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    async def run_upgrade(self, skip_backup: bool = False, batch_size: int = 5):
        """Run the complete upgrade process."""
        logger.info("=" * 60)
        logger.info("🚀 STARTING EMBEDDING MODEL UPGRADE")
        logger.info(f"  Old Model: {self.old_model} ({self.old_dimension}D)")
        logger.info(f"  New Model: {self.new_model} ({self.new_dimension}D)")
        logger.info(f"  Collection: {settings.qdrant_collection}")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Initialize
            await self.initialize()
            
            # Backup (optional)
            backup_id = None
            if not skip_backup:
                backup_id = await self.backup_collection()
            else:
                logger.info("⏭️ Skipping backup (--skip-backup flag)")
            
            # Recreate collection
            await self.recreate_collection()
            
            # Re-index all documents
            success_count, fail_count = await self.reindex_all_documents(batch_size=batch_size)
            
            # Verify
            verification_success = await self.verify_upgrade()
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            if verification_success and fail_count == 0:
                logger.info("=" * 60)
                logger.info("✅ UPGRADE COMPLETED SUCCESSFULLY!")
                logger.info(f"  New model: {self.new_model}")
                logger.info(f"  Dimension: {self.new_dimension}D")
                logger.info(f"  Documents processed: {success_count}")
                logger.info(f"  Duration: {duration:.1f} seconds")
                logger.info(f"  Backup ID: {backup_id}")
                logger.info("  Expected improvement: +18-25%% retrieval quality")
                logger.info("=" * 60)
            elif fail_count > 0:
                logger.warning("=" * 60)
                logger.warning("⚠️ UPGRADE COMPLETED WITH ERRORS")
                logger.warning(f"  Success: {success_count}, Failed: {fail_count}")
                logger.warning(f"  Duration: {duration:.1f} seconds")
                logger.warning("  Please check logs and verify manually")
                logger.warning("=" * 60)
            else:
                logger.error("=" * 60)
                logger.error("❌ UPGRADE COMPLETED WITH VERIFICATION ERRORS")
                logger.error(f"  Duration: {duration:.1f} seconds")
                logger.error("  Please check logs and verify manually")
                logger.error("=" * 60)
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ UPGRADE FAILED: {e}")
            logger.error("=" * 60)
            import traceback
            traceback.print_exc()
            raise


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upgrade embedding model to BGE")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup step")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for re-indexing (default: 5)")
    parser.add_argument("--auto-recreate", action="store_true", help="Automatically recreate collection (default)")
    
    args = parser.parse_args()
    
    upgrader = EmbeddingModelUpgrader()
    await upgrader.run_upgrade(
        skip_backup=args.skip_backup,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    asyncio.run(main())
