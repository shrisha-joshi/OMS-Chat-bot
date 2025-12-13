"""
Retrieval Engine
Consolidated service for RAG retrieval, hybrid search, and reranking.
Replaces: chat_service.py, hybrid_retrieval_service.py, reranking.py
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import time

from ..core.db_mongo import get_mongodb_client
from ..core.db_qdrant import get_qdrant_client
from ..core.model_manager import get_model_manager
from ..config import settings

logger = logging.getLogger(__name__)

class RetrievalEngine:
    def __init__(self):
        self.mongo_client = None
        self.qdrant_client = None
        self.model_manager = None
        
    async def initialize(self):
        """Initialize dependencies."""
        self.mongo_client = get_mongodb_client()
        self.qdrant_client = await get_qdrant_client()
        self.model_manager = await get_model_manager()
        
    async def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Perform hybrid search (Vector + Keyword) and Reranking.
        """
        if not self.model_manager:
            await self.initialize()
            
        start_time = time.time()
        
        # 1. Generate Query Embedding
        query_vector = self._generate_embedding(query)
        
        # 2. Vector Search (Qdrant)
        vector_results = await self.qdrant_client.search_similar(query_vector, top_k=top_k * 2)
        
        # 3. Keyword Search (BM25 - simplified via MongoDB text search or regex for now)
        # Note: For true BM25, we'd need a dedicated engine or use Qdrant's sparse vectors.
        # Here we'll rely primarily on vector search for the "minimal" version, 
        # but we can add a simple keyword filter if needed.
        
        # 4. Reranking (Cross-Encoder)
        reranked_results = self._rerank_results(query, vector_results, top_k)
        
        return {
            "results": reranked_results,
            "metrics": {
                "retrieval_time": time.time() - start_time,
                "total_found": len(vector_results)
            }
        }

    def _generate_embedding(self, text: str) -> List[float]:
        model = self.model_manager.get_embedding_model()
        return model.encode(text).tolist()

    def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """Rerank results using CrossEncoder."""
        if not results:
            return []
            
        reranker = self.model_manager.get_reranker_model()
        if not reranker:
            return results[:top_k]
            
        # Prepare pairs for reranking
        pairs = [(query, r["payload"]["text"]) for r in results]
        scores = reranker.predict(pairs)
        
        # Attach scores and sort
        for i, res in enumerate(results):
            res["score"] = float(scores[i])
            
        # Sort descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# Global instance
retrieval_engine = RetrievalEngine()

async def get_retrieval_engine():
    await retrieval_engine.initialize()
    return retrieval_engine
