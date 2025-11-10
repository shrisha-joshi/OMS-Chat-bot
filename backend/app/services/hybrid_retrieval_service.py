"""
Hybrid Retrieval Service - Combines Vector Search + Knowledge Graph
Implements Microsoft GraphRAG patterns for world-class retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

from ..core.db_qdrant import qdrant_client
from ..core.db_neo4j import neo4j_client

logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """
    Hybrid retrieval combining:
    1. Vector search (Qdrant) - semantic similarity
    2. Graph expansion (Neo4j) - knowledge graph context
    3. Reranking - combine scores for optimal results
    """
    
    def __init__(self):
        self.qdrant = qdrant_client
        self.graph = neo4j_client
        self.use_graph = self.graph.is_connected()
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        vector_k: int = 20,
        graph_hops: int = 2,
        use_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: vector search + graph expansion + reranking.
        
        Args:
            query: Search query
            top_k: Final number of results to return
            vector_k: Number of results from vector search (before graph expansion)
            graph_hops: Number of hops for graph expansion
            use_reranking: Whether to rerank results
            
        Returns:
            List of retrieval results with text, metadata, and scores
        """
        logger.info(f"Hybrid retrieval for query: '{query[:50]}...'")
        
        # Stage 1: Vector search
        vector_results = await self._vector_search(query, k=vector_k)
        
        if not vector_results:
            logger.warning("No vector results found")
            return []
        
        # Stage 2: Graph expansion (if Neo4j available)
        if self.use_graph:
            try:
                expanded_results = await self._graph_expansion(
                    vector_results,
                    hops=graph_hops
                )
            except Exception as e:
                logger.warning(f"Graph expansion failed: {e}, using vector results only")
                expanded_results = vector_results
        else:
            expanded_results = vector_results
        
        # Stage 3: Reranking
        if use_reranking and len(expanded_results) > top_k:
            final_results = await self._rerank(expanded_results, query, top_k)
        else:
            final_results = expanded_results[:top_k]
        
        logger.info(f"Retrieved {len(final_results)} results (vector: {len(vector_results)}, expanded: {len(expanded_results)})")
        return final_results
    
    async def _vector_search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """
        Perform vector search using Qdrant.
        """
        try:
            if not self.qdrant.client:
                logger.warning("Qdrant not connected")
                return []
            
            # Search in Qdrant
            results = await self.qdrant.search_similar(query, limit=k)
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "chunk_id": result.get("id") or result.get("chunk_id"),
                    "text": result.get("text", ""),
                    "doc_id": result.get("doc_id", ""),
                    "filename": result.get("filename", "Unknown"),
                    "vector_score": result.get("score", 0.0),
                    "rank": i + 1,
                    "source": "vector",
                    "metadata": result.get("metadata", {})
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return []
    
    async def _graph_expansion(
        self,
        vector_results: List[Dict[str, Any]],
        hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Expand results using knowledge graph.
        Implements Microsoft GraphRAG community-based expansion.
        """
        try:
            # Extract chunk IDs from vector results
            chunk_ids = [r["chunk_id"] for r in vector_results if r.get("chunk_id")]
            
            if not chunk_ids:
                return vector_results
            
            # Get graph context (entities, relationships, neighboring chunks)
            graph_context = await self.graph.expand_context_from_chunks(
                chunk_ids=chunk_ids,
                hops=hops
            )
            
            # Enrich vector results with graph information
            enriched_results = []
            for result in vector_results:
                chunk_id = result["chunk_id"]
                
                # Add entities mentioned in this chunk
                chunk_entities = [
                    e for e in graph_context.get("entities", [])
                    if e.get("chunk_id") == chunk_id
                ]
                result["entities"] = chunk_entities
                
                # Add relationships
                chunk_relationships = [
                    r for r in graph_context.get("relationships", [])
                    if r.get("chunk_id") == chunk_id
                ]
                result["relationships"] = chunk_relationships
                
                # Calculate graph score based on centrality
                graph_score = len(chunk_entities) * 0.1 + len(chunk_relationships) * 0.15
                result["graph_score"] = min(graph_score, 1.0)
                
                enriched_results.append(result)
            
            # Add neighboring chunks from graph (community detection pattern)
            neighbor_chunks = graph_context.get("neighbor_chunks", [])
            for neighbor in neighbor_chunks:
                if neighbor["chunk_id"] not in [r["chunk_id"] for r in enriched_results]:
                    enriched_results.append({
                        "chunk_id": neighbor["chunk_id"],
                        "text": neighbor.get("text", ""),
                        "doc_id": neighbor.get("doc_id", ""),
                        "filename": neighbor.get("filename", "Unknown"),
                        "vector_score": 0.5,  # Lower score for graph-discovered chunks
                        "graph_score": 0.8,   # High graph relevance
                        "source": "graph",
                        "entities": neighbor.get("entities", []),
                        "relationships": neighbor.get("relationships", [])
                    })
            
            logger.info(f"Graph expansion: {len(vector_results)} → {len(enriched_results)} results")
            return enriched_results
            
        except Exception as e:
            logger.error(f"Graph expansion error: {e}", exc_info=True)
            return vector_results
    
    async def _rerank(
        self,
        results: List[Dict[str, Any]],
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using combined scoring:
        - Vector similarity (0.5 weight)
        - Graph centrality (0.3 weight)
        - Freshness (0.2 weight)
        """
        try:
            for result in results:
                vector_score = result.get("vector_score", 0.0)
                graph_score = result.get("graph_score", 0.0)
                
                # Combined score (Microsoft GraphRAG pattern)
                combined_score = (
                    vector_score * 0.5 +
                    graph_score * 0.3 +
                    0.2  # Default freshness score
                )
                
                result["combined_score"] = combined_score
            
            # Sort by combined score
            reranked = sorted(results, key=lambda x: x.get("combined_score", 0), reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]
    
    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieval results for LLM.
        Includes chunks + entities + relationships (graph-aware prompting).
        """
        if not results:
            return ""
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            # Basic chunk info
            chunk_text = result.get("text", "")
            filename = result.get("filename", "Unknown")
            
            context_parts.append(f"[{i}] From {filename}:")
            context_parts.append(chunk_text)
            
            # Add entity information (graph context)
            entities = result.get("entities", [])
            if entities:
                entity_names = [e.get("name", "") for e in entities[:5]]
                context_parts.append(f"   Key entities: {', '.join(entity_names)}")
            
            # Add relationship information
            relationships = result.get("relationships", [])
            if relationships:
                rel_summaries = [
                    f"{r.get('from', '')} → {r.get('type', '')} → {r.get('to', '')}"
                    for r in relationships[:3]
                ]
                context_parts.append(f"   Relationships: {'; '.join(rel_summaries)}")
            
            context_parts.append("")  # Blank line between chunks
        
        return "\n".join(context_parts)


# Global service instance
hybrid_retrieval_service = HybridRetrievalService()
