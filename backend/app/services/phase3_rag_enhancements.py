"""
Phase 3: Advanced RAG Enhancements Service
Implements Contextual Retrieval, Hybrid Search, and other optimization techniques
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time

from ..core.db_mongo import MongoDBClient
from ..core.db_qdrant import QdrantDBClient
from ..core.cache_redis import RedisClient
from ..config import settings

logger = logging.getLogger(__name__)


class ContextualRetrievalService:
    """Implements contextual retrieval - adding surrounding context to chunks."""
    
    def __init__(self, mongo_client: Optional[MongoDBClient] = None):
        self.mongo_client = mongo_client
        self.context_cache = {}
    
    async def enhance_chunks_with_context(
        self, 
        chunks: List[Dict[str, Any]], 
        context_window: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Add surrounding context to retrieved chunks.
        
        Args:
            chunks: List of retrieved chunks
            context_window: Number of surrounding chunks to include
            
        Returns:
            Enhanced chunks with contextual information
        """
        enhanced_chunks = []
        
        for chunk in chunks:
            try:
                enhanced_chunk = {
                    **chunk,
                    "contextual_content": "",
                    "context_quality": 0.0,
                    "confidence_boost": 1.0
                }
                
                # Get surrounding context from same document
                if self.mongo_client and "document_id" in chunk:
                    context_info = await self._get_surrounding_context(
                        chunk["document_id"],
                        chunk.get("chunk_id", ""),
                        context_window
                    )
                    
                    if context_info:
                        enhanced_chunk["contextual_content"] = context_info["context"]
                        enhanced_chunk["context_quality"] = context_info["quality"]
                        # Boost confidence if high-quality context available
                        enhanced_chunk["confidence_boost"] = 1.3 if context_info["quality"] > 0.7 else 1.0
                        enhanced_chunk["context_sources"] = context_info["sources"]
                
                enhanced_chunks.append(enhanced_chunk)
                
            except Exception as e:
                logger.warning(f"Failed to enhance chunk: {e}")
                enhanced_chunks.append(chunk)
        
        logger.info(f"Enhanced {len(enhanced_chunks)} chunks with contextual information")
        return enhanced_chunks
    
    async def _get_surrounding_context(
        self,
        document_id: str,
        chunk_id: str,
        context_window: int
    ) -> Optional[Dict[str, Any]]:
        """Get surrounding chunks and context."""
        try:
            if not self.mongo_client:
                return None
            
            # Query chunks in same document
            collection = self.mongo_client.db.chunks
            
            current_chunk = await collection.find_one({
                "_id": chunk_id,
                "document_id": document_id
            })
            
            if not current_chunk:
                return None
            
            current_index = current_chunk.get("chunk_index", 0)
            
            # Get surrounding chunks
            surrounding = await collection.find({
                "document_id": document_id,
                "chunk_index": {
                    "$gte": max(0, current_index - context_window),
                    "$lte": current_index + context_window
                }
            }).to_list(None)
            
            if not surrounding:
                return None
            
            # Build context text
            context_parts = []
            for chunk in sorted(surrounding, key=lambda x: x.get("chunk_index", 0)):
                if chunk.get("_id") != chunk_id:  # Exclude current chunk
                    context_parts.append(chunk.get("content", ""))
            
            context_text = " ".join(context_parts)
            quality = min(len(surrounding) / (2 * context_window + 1), 1.0)
            
            return {
                "context": context_text[:1000],  # Limit to 1000 chars
                "quality": quality,
                "sources": [c.get("_id") for c in surrounding if c.get("_id") != chunk_id]
            }
            
        except Exception as e:
            logger.warning(f"Failed to get surrounding context: {e}")
            return None


class HybridSearchService:
    """Implements hybrid search combining BM25 + Vector search."""
    
    def __init__(self, bm25_index=None, qdrant_client: Optional[QdrantDBClient] = None):
        self.bm25_index = bm25_index
        self.qdrant_client = qdrant_client
        self.bm25_weight = 0.4  # Weight for BM25 score
        self.vector_weight = 0.6  # Weight for vector score
    
    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining keyword and semantic search.
        
        Args:
            query: User query text
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Ranked list of results from hybrid search
        """
        try:
            start_time = time.time()
            
            # Parallel search: Vector search + BM25 search
            vector_results_task = self._vector_search(query_embedding, top_k * 2)
            bm25_results_task = self._bm25_search(query, top_k * 2)
            
            vector_results, bm25_results = await asyncio.gather(
                vector_results_task,
                bm25_results_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.warning(f"Vector search failed: {vector_results}")
                vector_results = []
            
            if isinstance(bm25_results, Exception):
                logger.warning(f"BM25 search failed: {bm25_results}")
                bm25_results = []
            
            # Merge results with hybrid scoring
            merged = await self._merge_hybrid_results(
                vector_results,
                bm25_results,
                top_k
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Hybrid search completed in {elapsed:.2f}s, returned {len(merged)} results")
            
            return merged
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        if not self.qdrant_client:
            return []
        
        try:
            results = await self.qdrant_client.search_similar(
                query_vector=query_embedding,
                top_k=top_k
            )
            
            # Normalize scores to [0, 1]
            if results:
                max_score = max(r.get("score", 0) for r in results)
                for r in results:
                    r["vector_score"] = r.get("score", 0) / max_score if max_score > 0 else 0
            
            return results
            
        except Exception as e:
            logger.warning(f"Vector search error: {e}")
            return []
    
    async def _bm25_search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """BM25 keyword search."""
        if not self.bm25_index:
            return []
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k results
            if len(scores) > 0:
                max_score = max(scores)
                top_indices = sorted(
                    range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True
                )[:top_k]
                
                results = []
                for idx in top_indices:
                    if scores[idx] > 0:
                        results.append({
                            "chunk_index": idx,
                            "bm25_score": scores[idx] / max_score if max_score > 0 else 0
                        })
                
                return results
            
            return []
            
        except Exception as e:
            logger.warning(f"BM25 search error: {e}")
            return []
    
    async def _merge_hybrid_results(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Merge vector and BM25 results with hybrid scoring."""
        
        # Create score dictionary
        scores = {}
        
        # Add vector scores
        for result in vector_results:
            key = result.get("_id") or result.get("chunk_index")
            scores[key] = {
                "vector_score": result.get("vector_score", 0),
                "bm25_score": 0,
                "data": result
            }
        
        # Add BM25 scores
        for result in bm25_results:
            key = result.get("_id") or result.get("chunk_index")
            if key in scores:
                scores[key]["bm25_score"] = result.get("bm25_score", 0)
            else:
                scores[key] = {
                    "vector_score": 0,
                    "bm25_score": result.get("bm25_score", 0),
                    "data": result
                }
        
        # Calculate hybrid scores
        ranked = []
        for key, score_data in scores.items():
            hybrid_score = (
                score_data["vector_score"] * self.vector_weight +
                score_data["bm25_score"] * self.bm25_weight
            )
            
            result = score_data["data"]
            result["hybrid_score"] = hybrid_score
            result["vector_score"] = score_data["vector_score"]
            result["bm25_score"] = score_data["bm25_score"]
            
            ranked.append(result)
        
        # Sort by hybrid score and return top-k
        ranked.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        return ranked[:top_k]
    
    async def hybrid_search_with_rrf(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        rrf_k: int = 60
    ) -> Dict[str, Any]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF) algorithm.
        RRF mathematically combines rankings from multiple sources with proven effectiveness.
        
        Args:
            query: User query text
            query_embedding: Query embedding vector
            top_k: Number of results to return
            rrf_k: RRF parameter (typically 60, controls contribution of each ranker)
            
        Returns:
            Dictionary with RRF-fused results and metrics
        """
        try:
            start_time = time.time()
            
            # Parallel execution: Vector search + BM25 search
            vector_task = self._vector_search(query_embedding, top_k * 3)
            bm25_task = self._bm25_search(query, top_k * 3)
            
            vector_results, bm25_results = await asyncio.gather(
                vector_task,
                bm25_task,
                return_exceptions=True
            )
            
            # Handle exceptions gracefully
            if isinstance(vector_results, Exception):
                logger.warning(f"Vector search failed in RRF: {vector_results}")
                vector_results = []
            
            if isinstance(bm25_results, Exception):
                logger.warning(f"BM25 search failed in RRF: {bm25_results}")
                bm25_results = []
            
            # Apply RRF algorithm
            rrf_scores = {}
            
            # RRF formula: score = 1 / (k + rank)
            # where k is rrf_k parameter and rank is position in ranking
            
            # Process vector results
            for rank, result in enumerate(vector_results, 1):
                doc_id = str(result.get("_id") or result.get("chunk_index", ""))
                if doc_id:
                    rrf_score = 1.0 / (rrf_k + rank)
                    if doc_id not in rrf_scores:
                        rrf_scores[doc_id] = {"score": 0.0, "data": result, "sources": []}
                    rrf_scores[doc_id]["score"] += rrf_score
                    rrf_scores[doc_id]["sources"].append("vector")
            
            # Process BM25 results
            for rank, result in enumerate(bm25_results, 1):
                doc_id = str(result.get("_id") or result.get("chunk_index", ""))
                if doc_id:
                    rrf_score = 1.0 / (rrf_k + rank)
                    if doc_id not in rrf_scores:
                        rrf_scores[doc_id] = {"score": 0.0, "data": result, "sources": []}
                    rrf_scores[doc_id]["score"] += rrf_score
                    rrf_scores[doc_id]["sources"].append("keyword")
            
            # Sort by RRF score and prepare results
            ranked_results = []
            for doc_id, score_data in rrf_scores.items():
                result = score_data["data"]
                result["rrf_score"] = score_data["score"]
                result["fusion_sources"] = list(set(score_data["sources"]))
                result["appears_in_multiple"] = len(score_data["sources"]) > 1
                ranked_results.append(result)
            
            # Sort by RRF score (descending) and get top-k
            ranked_results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
            final_results = ranked_results[:top_k]
            
            elapsed = time.time() - start_time
            
            return {
                "merged_results": final_results,
                "vector_count": len(vector_results),
                "bm25_count": len(bm25_results),
                "rrf_count": len(rrf_scores),
                "algorithm": "reciprocal_rank_fusion",
                "rrf_k_parameter": rrf_k,
                "total_time_ms": int(elapsed * 1000),
                "deduplication_ratio": len(final_results) / max(1, len(rrf_scores))
            }
            
        except Exception as e:
            logger.error(f"RRF hybrid search failed: {e}")
            return {
                "merged_results": [],
                "vector_count": 0,
                "bm25_count": 0,
                "rrf_count": 0,
                "algorithm": "reciprocal_rank_fusion",
                "error": str(e)
            }


class QueryRewritingService:
    """Implements query rewriting for better retrieval."""
    
    def __init__(self, llm_handler=None):
        self.llm_handler = llm_handler
    
    async def rewrite_query(self, query: str) -> Dict[str, Any]:
        """
        Rewrite query using multiple strategies.
        
        Args:
            query: Original user query
            
        Returns:
            Dictionary with rewrites, strategies, and scores
        """
        try:
            rewrites = {
                "original": query,
                "variants": [],
                "strategies_used": []
            }
            
            # Strategy 1: Question expansion
            expanded = await self._expand_query(query)
            if expanded != query:
                rewrites["variants"].append({
                    "query": expanded,
                    "strategy": "expansion",
                    "score": 0.7
                })
                rewrites["strategies_used"].append("expansion")
            
            # Strategy 2: Synonym replacement
            synonymized = await self._apply_synonyms(query)
            if synonymized != query:
                rewrites["variants"].append({
                    "query": synonymized,
                    "strategy": "synonyms",
                    "score": 0.6
                })
                rewrites["strategies_used"].append("synonyms")
            
            # Strategy 3: Decomposition (break into sub-queries)
            decomposed = await self._decompose_query(query)
            if decomposed:
                rewrites["variants"].extend(decomposed)
                rewrites["strategies_used"].append("decomposition")
            
            # Strategy 4: Specification (add context)
            specified = await self._specify_context(query)
            if specified != query:
                rewrites["variants"].append({
                    "query": specified,
                    "strategy": "specification",
                    "score": 0.65
                })
                rewrites["strategies_used"].append("specification")
            
            logger.info(f"Generated {len(rewrites['variants'])} query rewrites")
            return rewrites
            
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return {
                "original": query,
                "variants": [],
                "strategies_used": []
            }
    
    async def _expand_query(self, query: str) -> str:
        """Expand query with related concepts."""
        # Simple expansion logic
        expansion_map = {
            "benefits": "advantages, pros, positive effects",
            "issues": "problems, challenges, difficulties",
            "explain": "describe, detail, elaborate on",
        }
        
        expanded = query
        for key, value in expansion_map.items():
            if key.lower() in query.lower():
                expanded = query.replace(key, f"{key} ({value})")
        
        return expanded
    
    async def _apply_synonyms(self, query: str) -> str:
        """Replace keywords with synonyms."""
        synonym_map = {
            "help": "assist, support, aid",
            "understand": "comprehend, grasp, understand",
            "use": "utilize, employ, use",
            "show": "display, demonstrate, show",
            "find": "locate, discover, find",
        }
        
        result = query
        for original, synonyms in synonym_map.items():
            if original.lower() in query.lower():
                result = result.replace(
                    original,
                    f"{original} or {synonyms.split(',')[0]}",
                    1
                )
        
        return result
    
    async def _decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """Decompose complex query into sub-queries."""
        # Detect multi-part queries
        if " and " in query.lower() or "," in query:
            parts = [p.strip() for p in query.split(" and ")]
            if len(parts) == 1:
                parts = [p.strip() for p in query.split(",")]
            
            if len(parts) > 1:
                return [
                    {
                        "query": part,
                        "strategy": "decomposition",
                        "score": 0.8,
                        "part_index": i
                    }
                    for i, part in enumerate(parts) if part
                ]
        
        return []
    
    async def _specify_context(self, query: str) -> str:
        """Add contextual information to query."""
        # Add domain context
        if not any(word in query.lower() for word in ["document", "file", "text"]):
            return f"{query} (in documents)"
        
        return query


class EmbeddingCachingService:
    """Implements caching for embeddings and frequently accessed data."""
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        self.redis_client = redis_client
        # Extended TTL for embeddings since they are deterministic (same text = same embedding)
        # 30 days = 2592000 seconds. Embeddings can be cached longer than query results.
        self.embedding_cache_ttl = 2592000  # 30 days
        self.query_cache_ttl = 86400  # 24 hours (for query-specific cache)
    
    async def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text. Returns None if cache miss or expired."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"embedding:{hash(text)}"
            cached = await self.redis_client.get(cache_key)
            
            if cached:
                logger.debug(f"Cache hit for embedding (will expire in ~30 days)")
                return cached.get("embedding")
            
            return None
            
        except Exception as e:
            logger.warning(f"Embedding cache retrieval failed: {e}")
            return None
    
    async def cache_embedding(self, text: str, embedding: List[float]) -> bool:
        """Cache embedding for text with extended TTL (30 days).
        
        This uses a longer TTL than query results because:
        1. Embeddings are deterministic (same input = same output)
        2. Model doesn't change frequently during a session
        3. Long-term cache reduces compute load
        4. Embeddings can be safely evicted after 30 days
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = f"embedding:{hash(text)}"
            # Use extended TTL (30 days) instead of short 24-hour TTL
            await self.redis_client.set(
                cache_key,
                {
                    "embedding": embedding,
                    "timestamp": datetime.now().isoformat(),
                    "ttl_days": 30
                },
                ttl=self.embedding_cache_ttl  # 30 days
            )
            
            logger.debug(f"Embedding cached with 30-day TTL")
            return True
            
        except Exception as e:
            logger.warning(f"Embedding cache storage failed: {e}")
            return False
    
    async def get_cached_chunks(self, query_hash: str) -> Optional[List[Dict]]:
        """Get cached retrieval results."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"chunks:{query_hash}"
            cached = await self.redis_client.get(cache_key)
            
            if cached:
                logger.debug(f"Cache hit for chunk retrieval")
                return cached.get("chunks")
            
            return None
            
        except Exception as e:
            logger.warning(f"Chunk cache retrieval failed: {e}")
            return None
    
    async def cache_chunks(self, query_hash: str, chunks: List[Dict]) -> bool:
        """Cache retrieval results."""
        if not self.redis_client:
            return False
        
        try:
            cache_key = f"chunks:{query_hash}"
            await self.redis_client.set(
                cache_key,
                {"chunks": chunks, "timestamp": datetime.now().isoformat()},
                ttl=self.cache_ttl
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"Chunk cache storage failed: {e}")
            return False


# Global service instances (lazy initialized)
_contextual_retrieval_service = None
_hybrid_search_service = None
_query_rewriting_service = None
_embedding_cache_service = None


async def get_contextual_retrieval_service(mongo_client: Optional[MongoDBClient] = None):
    """Get or create contextual retrieval service."""
    global _contextual_retrieval_service
    if _contextual_retrieval_service is None:
        _contextual_retrieval_service = ContextualRetrievalService(mongo_client)
    return _contextual_retrieval_service


async def get_hybrid_search_service(bm25_index=None, qdrant_client: Optional[QdrantDBClient] = None):
    """Get or create hybrid search service."""
    global _hybrid_search_service
    if _hybrid_search_service is None:
        _hybrid_search_service = HybridSearchService(bm25_index, qdrant_client)
    return _hybrid_search_service


async def get_query_rewriting_service(llm_handler=None):
    """Get or create query rewriting service."""
    global _query_rewriting_service
    if _query_rewriting_service is None:
        _query_rewriting_service = QueryRewritingService(llm_handler)
    return _query_rewriting_service


async def get_embedding_cache_service(redis_client: Optional[RedisClient] = None):
    """Get or create embedding cache service."""
    global _embedding_cache_service
    if _embedding_cache_service is None:
        _embedding_cache_service = EmbeddingCachingService(redis_client)
    return _embedding_cache_service
