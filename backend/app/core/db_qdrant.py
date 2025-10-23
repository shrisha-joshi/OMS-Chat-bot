"""
Qdrant vector database client for semantic search and vector storage.
This module handles all vector operations including collection management,
document indexing, and similarity search.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models
from typing import List, Dict, Any, Optional, Tuple
import logging
import uuid
import asyncio
from datetime import datetime

from ..config import settings

logger = logging.getLogger(__name__)

class QdrantDBClient:
    """Qdrant vector database client for RAG operations."""
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.collection_name = settings.qdrant_collection
    
    async def connect(self):
        """Establish connection to Qdrant and create collection if needed."""
        try:
            if not settings.qdrant_url:
                logger.warning("Qdrant URL not configured, running without Qdrant")
                return False
                
            logger.info("Attempting to connect to Qdrant Cloud...")
            
            # Initialize client (synchronous but fast)
            client_kwargs = {"url": settings.qdrant_url}
            if settings.qdrant_api_key:
                client_kwargs["api_key"] = settings.qdrant_api_key
            
            self.client = QdrantClient(**client_kwargs)
            
            # Test connection in thread to avoid blocking event loop
            def _test_connection():
                return self.client.get_collections()
            
            collections = await asyncio.to_thread(_test_connection)
            logger.info(f"✅ Successfully connected to Qdrant Cloud! Available collections: {len(collections.collections)}")
            
            # Create collection if it doesn't exist
            await self._ensure_collection_exists()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            # Clean up failed connection
            self.client = None
            return False
    
    async def disconnect(self):
        """Close Qdrant connection."""
        if self.client:
            await asyncio.to_thread(self.client.close)
            logger.info("Disconnected from Qdrant")
    
    async def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists (blocking) -> run in thread
            def _check_and_create():
                collections = self.client.get_collections()
                collection_names = [col.name for col in collections.collections]
                
                if self.collection_name not in collection_names:
                    # Create collection with appropriate vector configuration
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=settings.embedding_dimension,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created Qdrant collection: {self.collection_name}")
                    return "created"
                else:
                    logger.info(f"Qdrant collection already exists: {self.collection_name}")
                    return "exists"
            
            await asyncio.to_thread(_check_and_create)
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    async def upsert_vectors(self, doc_id: str, chunks: List[Dict], embeddings: List[List[float]]) -> bool:
        """
        Upsert document chunks as vectors into Qdrant.
        
        Args:
            doc_id: Document ID
            chunks: List of chunk data dictionaries
            embeddings: List of embedding vectors corresponding to chunks
        
        Returns:
            bool: Success status
        """
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")
            
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_chunk_{i}"
                
                point = PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "text": chunk["text"][:1000],  # Limit text for payload
                        "char_start": chunk.get("char_start", 0),
                        "char_end": chunk.get("char_end", len(chunk["text"])),
                        "tokens": chunk.get("tokens", 0),
                        "created_at": datetime.utcnow().isoformat()
                    }
                )
                points.append(point)
            
            # Batch upsert points (blocking) -> run in thread
            def _do_upsert():
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            await asyncio.to_thread(_do_upsert)
            logger.info(f"Upserted {len(points)} vectors for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return False
    
    async def search_similar(self, query_vector: List[float], top_k: int = None, 
                           doc_filter: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return (defaults to settings)
            doc_filter: Optional document ID to filter by
        
        Returns:
            List of search results with metadata
        """
        try:
            if top_k is None:
                top_k = settings.top_k_retrieval
            
            # Prepare filter if specified
            search_filter = None
            if doc_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_filter)
                        )
                    ]
                )
            
            # Perform search (blocking) -> run in thread
            def _do_search():
                return self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=search_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False
                )
            
            search_results = await asyncio.to_thread(_do_search)
            
            # Format results
            results = []
            for result in search_results:
                formatted_result = {
                    "id": result.id,
                    "score": float(result.score),
                    "doc_id": result.payload.get("doc_id"),
                    "chunk_id": result.payload.get("chunk_id"),
                    "chunk_index": result.payload.get("chunk_index"),
                    "text": result.payload.get("text", ""),
                    "char_start": result.payload.get("char_start"),
                    "char_end": result.payload.get("char_end"),
                    "tokens": result.payload.get("tokens"),
                    "metadata": {
                        "created_at": result.payload.get("created_at")
                    }
                }
                results.append(formatted_result)
            
            logger.info(f"Found {len(results)} similar vectors (top_k={top_k})")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar vectors: {e}")
            return []
    
    async def delete_document_vectors(self, doc_id: str) -> bool:
        """Delete all vectors associated with a document."""
        try:
            # Delete points by doc_id filter (blocking) -> run in thread
            def _do_delete():
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="doc_id",
                                    match=MatchValue(value=doc_id)
                                )
                            ]
                        )
                    )
                )
            
            await asyncio.to_thread(_do_delete)
            logger.info(f"Deleted vectors for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors for document {doc_id}: {e}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            # Get collection info (blocking) -> run in thread
            def _get_info():
                return self.client.get_collection(self.collection_name)
            
            collection_info = await asyncio.to_thread(_get_info)
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    async def search_with_hybrid_score(self, query_vector: List[float], 
                                     semantic_weight: float = 0.7,
                                     top_k: int = None) -> List[Dict[str, Any]]:
        """
        Advanced search with hybrid scoring (future enhancement).
        Currently implements semantic search only.
        """
        # For now, delegate to regular semantic search
        return await self.search_similar(query_vector, top_k)
    
    async def batch_search(self, query_vectors: List[List[float]], 
                          top_k: int = None) -> List[List[Dict[str, Any]]]:
        """
        Batch search for multiple query vectors.
        
        Args:
            query_vectors: List of query embedding vectors
            top_k: Number of results per query
        
        Returns:
            List of search results for each query
        """
        try:
            if top_k is None:
                top_k = settings.top_k_retrieval
            
            # Perform batch search (blocking) -> run in thread
            def _do_batch_search():
                return self.client.search_batch(
                    collection_name=self.collection_name,
                    requests=[
                        models.SearchRequest(
                            vector=vector,
                            limit=top_k,
                            with_payload=True,
                            with_vector=False
                        ) for vector in query_vectors
                    ]
                )
            
            batch_results = await asyncio.to_thread(_do_batch_search)
            
            # Format results
            formatted_results = []
            for search_results in batch_results:
                results = []
                for result in search_results:
                    formatted_result = {
                        "id": result.id,
                        "score": float(result.score),
                        "doc_id": result.payload.get("doc_id"),
                        "chunk_id": result.payload.get("chunk_id"),
                        "text": result.payload.get("text", ""),
                        "metadata": result.payload
                    }
                    results.append(formatted_result)
                formatted_results.append(results)
            
            logger.info(f"Batch search completed for {len(query_vectors)} queries")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to perform batch search: {e}")
            return [[] for _ in query_vectors]


# Global Qdrant client instance
qdrant_client = QdrantDBClient()


async def get_qdrant_client() -> QdrantDBClient:
    """Dependency injection for Qdrant client."""
    return qdrant_client