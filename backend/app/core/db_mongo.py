"""
MongoDB client and GridFS utilities for document storage and metadata management.
This module provides async MongoDB operations using Motor driver with GridFS
for file storage and regular collections for document metadata.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

from ..config import settings

logger = logging.getLogger(__name__)

class MongoDBClient:
    """Async MongoDB client with GridFS support."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.fs_bucket: Optional[AsyncIOMotorGridFSBucket] = None
        self.available = False  # Flag to indicate if service is available
    
    @property
    def db(self):
        """Alias for database to maintain compatibility with service code."""
        return self.database
    
    def is_connected(self) -> bool:
        """Check if MongoDB is currently connected."""
        return self.available and self.client is not None
    
    async def connect(self):
        """Establish connection to MongoDB and initialize GridFS."""
        try:
            if not settings.mongodb_uri:
                logger.warning("MongoDB URI not configured, running without MongoDB")
                self.available = False
                return False
                
            logger.info("Attempting to connect to MongoDB Atlas...")
            self.client = AsyncIOMotorClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=5000
            )
            self.database = self.client[settings.mongodb_db]
            self.fs_bucket = AsyncIOMotorGridFSBucket(self.database, bucket_name="fs")
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"✅ Successfully connected to MongoDB: {settings.mongodb_db}")
            self.available = True
            
            # Create indexes
            await self._create_indexes()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            # Clean up failed connection
            self.client = None
            self.database = None
            self.fs_bucket = None
            self.available = False
            return False
    
    async def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create necessary indexes for optimal performance."""
        try:
            # Documents collection indexes
            await self.database.documents.create_index([("uploaded_at", -1)])
            await self.database.documents.create_index([("ingest_status", 1)])
            await self.database.documents.create_index([("uploader", 1)])
            
            # Chunks collection indexes
            await self.database.chunks.create_index([
                ("doc_id", 1), 
                ("chunk_index", 1)
            ], unique=True)
            await self.database.chunks.create_index([("doc_id", 1)])
            
            # Feedback collection indexes
            await self.database.feedback.create_index([("session_id", 1)])
            await self.database.feedback.create_index([("created_at", -1)])
            
            # Ingestion logs indexes
            await self.database.ingestion_logs.create_index([("doc_id", 1)])
            await self.database.ingestion_logs.create_index([("timestamp", -1)])
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    async def store_file(self, filename: str, content: bytes, metadata: Dict = None) -> ObjectId:
        """Store a file in GridFS and return the file ID."""
        try:
            file_id = await self.fs_bucket.upload_from_stream(
                filename=filename,
                source=content,
                metadata=metadata or {}
            )
            logger.info(f"File stored in GridFS: {filename} -> {file_id}")
            return file_id
        except Exception as e:
            logger.error(f"Failed to store file {filename}: {e}")
            raise
    
    async def retrieve_file(self, file_id: ObjectId) -> bytes:
        """Retrieve a file from GridFS by file ID."""
        try:
            # Handle both ObjectId and string file_id
            if isinstance(file_id, str):
                file_id = ObjectId(file_id)
            
            stream = await self.fs_bucket.open_download_stream(file_id)
            content = await stream.read()
            return content
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id}: {e}")
            raise
    
    async def delete_file(self, file_id: ObjectId):
        """Delete a file from GridFS."""
        try:
            await self.fs_bucket.delete(file_id)
            logger.info(f"File deleted from GridFS: {file_id}")
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            raise
    
    async def save_document(
        self, 
        filename: str, 
        content: bytes, 
        content_type: str = "application/octet-stream",
        size: int = 0
    ) -> str:
        """
        Save a document: store file in GridFS and create metadata record.
        Returns the document ID for tracking.
        """
        try:
            # Step 1: Store file in GridFS
            file_id = await self.store_file(
                filename=filename,
                content=content,
                metadata={
                    "content_type": content_type,
                    "size": size,
                    "uploaded_at": datetime.utcnow().isoformat()
                }
            )
            
            # Step 2: Create document metadata record
            doc_id = await self.create_document({
                "filename": filename,
                "gridfs_id": file_id,
                "size": size,
                "content_type": content_type,
                "uploader": "api",
                "ingest_status": "PENDING"
            })
            
            logger.info(f"✅ Document saved: {filename} (GridFS: {file_id}, Doc: {doc_id})")
            return doc_id
        
        except Exception as e:
            logger.error(f"❌ Failed to save document {filename}: {e}")
            raise
    
    async def create_document(self, document_data: Dict) -> str:
        """Create a new document record."""
        try:
            document_data["uploaded_at"] = datetime.utcnow()
            document_data["ingest_status"] = "PENDING"
            
            result = await self.database.documents.insert_one(document_data)
            doc_id = str(result.inserted_id)
            logger.info(f"Document created: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to create document: {e}")
            raise
    
    async def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID."""
        try:
            document = await self.database.documents.find_one({"_id": ObjectId(doc_id)})
            if document:
                document["_id"] = str(document["_id"])
                document["gridfs_id"] = str(document["gridfs_id"])
            return document
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    async def update_document_status(self, doc_id: str, status: str, error_message: str = None):
        """Update document ingestion status."""
        try:
            update_data = {
                "ingest_status": status,
                "updated_at": datetime.utcnow()
            }
            if error_message:
                update_data["error_message"] = error_message
            
            await self.database.documents.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": update_data}
            )
            logger.info(f"Document {doc_id} status updated to {status}")
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
    
    async def list_documents(self, skip: int = 0, limit: int = 50) -> List[Dict]:
        """List documents with pagination."""
        try:
            cursor = self.database.documents.find().sort("uploaded_at", -1).skip(skip).limit(limit)
            documents = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                doc["gridfs_id"] = str(doc["gridfs_id"])
                documents.append(doc)
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    async def store_chunks(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Store document chunks in the database."""
        try:
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "_id": f"{doc_id}_chunk_{i}",
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunk["text"],
                    "char_start": chunk.get("char_start", 0),
                    "char_end": chunk.get("char_end", len(chunk["text"])),
                    "tokens": chunk.get("tokens", 0),
                    "created_at": datetime.utcnow()
                }
                
                try:
                    await self.database.chunks.insert_one(chunk_doc)
                except DuplicateKeyError:
                    # Update existing chunk
                    await self.database.chunks.update_one(
                        {"_id": chunk_doc["_id"]},
                        {"$set": chunk_doc}
                    )
            
            logger.info(f"Stored {len(chunks)} chunks for document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            return False
    
    async def get_chunks(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a document."""
        try:
            cursor = self.database.chunks.find({"doc_id": doc_id}).sort("chunk_index", 1)
            chunks = []
            async for chunk in cursor:
                chunks.append(chunk)
            return chunks
        except Exception as e:
            logger.error(f"Failed to get chunks for {doc_id}: {e}")
            return []
    
    async def delete_chunks(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            result = await self.database.chunks.delete_many({"doc_id": doc_id})
            logger.info(f"Deleted {result.deleted_count} chunks for document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return False
    
    async def log_ingestion_step(self, doc_id: str, step: str, status: str, message: str = "", metadata: Dict = None):
        """Log an ingestion step."""
        try:
            log_entry = {
                "doc_id": doc_id,
                "step": step,
                "status": status,
                "message": message,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow()
            }
            await self.database.ingestion_logs.insert_one(log_entry)
        except Exception as e:
            logger.error(f"Failed to log ingestion step: {e}")
    
    async def get_ingestion_logs(self, doc_id: str) -> List[Dict]:
        """Get ingestion logs for a document."""
        try:
            cursor = self.database.ingestion_logs.find({"doc_id": doc_id}).sort("timestamp", 1)
            logs = []
            async for log in cursor:
                log["_id"] = str(log["_id"])
                logs.append(log)
            return logs
        except Exception as e:
            logger.error(f"Failed to get ingestion logs: {e}")
            return []
    
    async def store_feedback(self, session_id: str, query: str, response: str, 
                           rating: str, correction: str = None) -> bool:
        """Store user feedback."""
        try:
            feedback_data = {
                "session_id": session_id,
                "query": query,
                "response": response,
                "rating": rating,
                "correction": correction,
                "created_at": datetime.utcnow()
            }
            await self.database.feedback.insert_one(feedback_data)
            logger.info(f"Feedback stored for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            return False
    
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents for BM25 indexing."""
        if not self.database:
            logger.warning("MongoDB not available, returning empty document list")
            return []
            
        try:
            documents = await self.database.documents.find({
                "ingest_status": "completed"
            }, {"title": 1, "content": 1, "chunk_id": 1}).to_list(None)
            return documents
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []
    
    async def create_document_relationship(self, from_doc_id: str, to_doc_id: str, 
                                         relationship_type: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Create a graph relationship between two documents in MongoDB.
        
        Relationship types:
        - "references": Document A references Document B
        - "related_topic": Documents discuss similar topics
        - "child": Document A is a subsection of Document B
        - "similar": Documents have similar content
        """
        try:
            if not self.database:
                logger.warning("MongoDB not available for relationship storage")
                return False
            
            relationship = {
                "from_doc_id": from_doc_id,
                "to_doc_id": to_doc_id,
                "type": relationship_type,
                "metadata": metadata or {},
                "created_at": datetime.utcnow()
            }
            
            # Create composite index for fast lookups
            await self.database.document_relationships.create_index([
                ("from_doc_id", 1),
                ("type", 1)
            ])
            await self.database.document_relationships.create_index([
                ("to_doc_id", 1),
                ("type", 1)
            ])
            
            # Insert or update relationship
            await self.database.document_relationships.update_one(
                {
                    "from_doc_id": from_doc_id,
                    "to_doc_id": to_doc_id,
                    "type": relationship_type
                },
                {"$set": relationship},
                upsert=True
            )
            
            logger.info(f"Created relationship: {from_doc_id} --{relationship_type}--> {to_doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create document relationship: {e}")
            return False
    
    async def get_related_documents(self, doc_id: str, relationship_type: str = None) -> List[str]:
        """
        Get all documents related to the given document.
        
        Returns: List of related document IDs
        """
        try:
            if not self.database:
                return []
            
            query = {"from_doc_id": doc_id}
            if relationship_type:
                query["type"] = relationship_type
            
            cursor = self.database.document_relationships.find(query, {"to_doc_id": 1})
            related_docs = []
            async for rel in cursor:
                related_docs.append(rel["to_doc_id"])
            
            logger.info(f"Found {len(related_docs)} related documents for {doc_id}")
            return related_docs
            
        except Exception as e:
            logger.error(f"Failed to get related documents: {e}")
            return []
    
    async def extract_topics_and_create_relationships(self, doc_id: str, topics: List[str], 
                                                     all_docs: List[Dict[str, Any]]) -> int:
        """
        Extract topics from document and create relationships with other documents
        that share the same topics.
        
        Returns: Number of relationships created
        """
        try:
            if not self.database:
                return 0
            
            relationships_created = 0
            
            # For each extracted topic, find other documents with same topic
            for topic in topics:
                # Search other documents for this topic
                for other_doc in all_docs:
                    if other_doc.get("_id") == doc_id:
                        continue
                    
                    # Check if other document has this topic
                    other_topics = other_doc.get("topics", [])
                    if topic in other_topics:
                        # Create relationship
                        success = await self.create_document_relationship(
                            doc_id,
                            str(other_doc.get("_id")),
                            "related_topic",
                            {"topic": topic}
                        )
                        if success:
                            relationships_created += 1
            
            logger.info(f"Created {relationships_created} topic-based relationships for {doc_id}")
            return relationships_created
            
        except Exception as e:
            logger.error(f"Failed to extract topics and create relationships: {e}")
            return 0
    
    async def get_document_graph(self, doc_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get the complete relationship graph for a document up to specified depth.
        
        Returns: Graph structure with nodes and edges
        """
        try:
            if not self.database:
                return {"nodes": [], "edges": []}
            
            nodes = set()
            edges = []
            visited = set()
            queue = [(doc_id, 0)]
            
            while queue:
                current_doc, current_depth = queue.pop(0)
                
                if current_depth > depth or current_doc in visited:
                    continue
                
                visited.add(current_doc)
                nodes.add(current_doc)
                
                # Get related documents
                cursor = self.database.document_relationships.find(
                    {"from_doc_id": current_doc},
                    {"to_doc_id": 1, "type": 1}
                )
                
                async for rel in cursor:
                    related_doc = rel["to_doc_id"]
                    rel_type = rel["type"]
                    
                    nodes.add(related_doc)
                    edges.append({
                        "from": current_doc,
                        "to": related_doc,
                        "type": rel_type
                    })
                    
                    if related_doc not in visited and current_depth < depth:
                        queue.append((related_doc, current_depth + 1))
            
            logger.info(f"Generated graph with {len(nodes)} nodes and {len(edges)} edges for {doc_id}")
            return {
                "nodes": list(nodes),
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"Failed to get document graph: {e}")
            return {"nodes": [], "edges": []}


# Global MongoDB client instance
mongodb_client = MongoDBClient()


async def get_mongodb_client() -> MongoDBClient:
    """Dependency injection for MongoDB client."""
    return mongodb_client