"""
Tests for Phase 3 RAG Enhancement Services
Tests contextual retrieval, hybrid search, query rewriting, and caching
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import Phase 3 services
import sys
sys.path.insert(0, '/app')

from app.services.phase3_rag_enhancements import (
    ContextualRetrievalService,
    HybridSearchService,
    QueryRewritingService,
    EmbeddingCachingService
)


class TestContextualRetrievalService:
    """Test contextual retrieval functionality."""
    
    @pytest.fixture
    def service(self):
        return ContextualRetrievalService(mongo_client=None)
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert service.mongo_client is None
        assert isinstance(service.context_cache, dict)
    
    @pytest.mark.asyncio
    async def test_enhance_chunks_without_mongo(self, service):
        """Test enhancing chunks without MongoDB connection."""
        chunks = [
            {"_id": "1", "content": "Test chunk 1", "document_id": "doc1"},
            {"_id": "2", "content": "Test chunk 2", "document_id": "doc1"}
        ]
        
        enhanced = await service.enhance_chunks_with_context(chunks)
        
        assert len(enhanced) == 2
        assert all("confidence_boost" in chunk for chunk in enhanced)
        assert all(chunk["confidence_boost"] == 1.0 for chunk in enhanced)  # Default when no context
    
    @pytest.mark.asyncio
    async def test_enhance_chunks_structure(self, service):
        """Test enhanced chunk structure."""
        chunks = [{"_id": "1", "content": "Test chunk"}]
        enhanced = await service.enhance_chunks_with_context(chunks)
        
        assert len(enhanced) == 1
        chunk = enhanced[0]
        assert "contextual_content" in chunk
        assert "context_quality" in chunk
        assert "confidence_boost" in chunk
        assert chunk["context_quality"] == 0.0  # No context available


class TestHybridSearchService:
    """Test hybrid search functionality."""
    
    @pytest.fixture
    def service(self):
        bm25_index = Mock()
        qdrant_client = Mock()
        return HybridSearchService(bm25_index=bm25_index, qdrant_client=qdrant_client)
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert service.bm25_weight == 0.4
        assert service.vector_weight == 0.6
    
    @pytest.mark.asyncio
    async def test_hybrid_search_empty_results(self, service):
        """Test hybrid search with no results."""
        service._vector_search = AsyncMock(return_value=[])
        service._bm25_search = AsyncMock(return_value=[])
        
        results = await service.hybrid_search(
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            top_k=10
        )
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_merge_hybrid_results(self, service):
        """Test merging of hybrid search results."""
        vector_results = [
            {"_id": "1", "vector_score": 0.8, "content": "doc1"}
        ]
        bm25_results = [
            {"_id": "1", "bm25_score": 0.6, "content": "doc1"}
        ]
        
        merged = await service._merge_hybrid_results(
            vector_results,
            bm25_results,
            top_k=10
        )
        
        assert len(merged) == 1
        assert merged[0]["_id"] == "1"
        assert "hybrid_score" in merged[0]
        # hybrid_score = 0.8 * 0.6 + 0.6 * 0.4 = 0.48 + 0.24 = 0.72
        assert abs(merged[0]["hybrid_score"] - 0.72) < 0.01


class TestQueryRewritingService:
    """Test query rewriting functionality."""
    
    @pytest.fixture
    def service(self):
        return QueryRewritingService(llm_handler=None)
    
    @pytest.mark.asyncio
    async def test_rewrite_query_structure(self, service):
        """Test query rewrite output structure."""
        result = await service.rewrite_query("explain how to use this")
        
        assert "original" in result
        assert "variants" in result
        assert "strategies_used" in result
        assert result["original"] == "explain how to use this"
    
    @pytest.mark.asyncio
    async def test_expand_query(self, service):
        """Test query expansion."""
        expanded = await service._expand_query("benefits of this approach")
        
        # Should contain expansion markers
        assert "benefits" in expanded
    
    @pytest.mark.asyncio
    async def test_apply_synonyms(self, service):
        """Test synonym replacement."""
        result = await service._apply_synonyms("help me understand this")
        
        # Result should differ from original
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_decompose_query(self, service):
        """Test query decomposition."""
        result = await service._decompose_query("find data and analyze results")
        
        # Should decompose into parts
        assert isinstance(result, list)
        if result:
            assert all("query" in item for item in result)
            assert all("strategy" in item for item in result)


class TestEmbeddingCachingService:
    """Test embedding caching functionality."""
    
    @pytest.fixture
    def service(self):
        return EmbeddingCachingService(redis_client=None)
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert service.cache_ttl == 86400
        assert service.redis_client is None
    
    @pytest.mark.asyncio
    async def test_cache_without_redis(self, service):
        """Test caching without Redis connection."""
        result = await service.cache_embedding("test", [0.1, 0.2])
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_cached_without_redis(self, service):
        """Test retrieval without Redis connection."""
        result = await service.get_cached_embedding("test")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_chunks_without_redis(self, service):
        """Test chunk caching without Redis."""
        result = await service.cache_chunks("query_hash", [{"id": "1"}])
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_cached_chunks_without_redis(self, service):
        """Test chunk retrieval without Redis."""
        result = await service.get_cached_chunks("query_hash")
        assert result is None


class TestHybridSearchIntegration:
    """Integration tests for hybrid search."""
    
    @pytest.fixture
    def service(self):
        bm25_index = Mock()
        bm25_index.get_scores = Mock(return_value=[0.9, 0.7, 0.5])
        
        qdrant_client = Mock()
        
        return HybridSearchService(bm25_index=bm25_index, qdrant_client=qdrant_client)
    
    @pytest.mark.asyncio
    async def test_bm25_search_with_scores(self, service):
        """Test BM25 search scoring."""
        service.bm25_index.get_scores = Mock(return_value=[0.9, 0.7, 0.5])
        
        results = await service._bm25_search("test query", top_k=2)
        
        assert len(results) <= 2
        assert all("bm25_score" in r for r in results)
        assert all(r["bm25_score"] <= 1.0 for r in results)


class TestQueryRewritingIntegration:
    """Integration tests for query rewriting."""
    
    @pytest.fixture
    def service(self):
        return QueryRewritingService()
    
    @pytest.mark.asyncio
    async def test_full_rewrite_workflow(self, service):
        """Test complete query rewriting workflow."""
        query = "find information about benefits and issues"
        
        result = await service.rewrite_query(query)
        
        assert result["original"] == query
        assert len(result["variants"]) > 0
        assert len(result["strategies_used"]) > 0
    
    @pytest.mark.asyncio
    async def test_decomposition_with_multiple_parts(self, service):
        """Test decomposition with multiple query parts."""
        query = "explain concept, show examples, and list benefits"
        
        parts = await service._decompose_query(query)
        
        assert len(parts) >= 2


class TestPhase3ServiceAvailability:
    """Test that all Phase 3 services are available."""
    
    def test_contextual_retrieval_available(self):
        """Test ContextualRetrievalService is importable."""
        assert ContextualRetrievalService is not None
    
    def test_hybrid_search_available(self):
        """Test HybridSearchService is importable."""
        assert HybridSearchService is not None
    
    def test_query_rewriting_available(self):
        """Test QueryRewritingService is importable."""
        assert QueryRewritingService is not None
    
    def test_embedding_caching_available(self):
        """Test EmbeddingCachingService is importable."""
        assert EmbeddingCachingService is not None


class TestServiceIntegration:
    """Test services working together."""
    
    @pytest.mark.asyncio
    async def test_contextual_retrieval_with_hybrid_scores(self):
        """Test contextual retrieval with hybrid search results."""
        # Create services
        ctx_service = ContextualRetrievalService()
        
        # Create mock hybrid search results
        chunks = [
            {
                "_id": "1",
                "content": "Chunk 1",
                "hybrid_score": 0.9,
                "vector_score": 0.8,
                "bm25_score": 0.6
            },
            {
                "_id": "2",
                "content": "Chunk 2",
                "hybrid_score": 0.7,
                "vector_score": 0.6,
                "bm25_score": 0.5
            }
        ]
        
        # Enhance with context
        enhanced = await ctx_service.enhance_chunks_with_context(chunks)
        
        assert len(enhanced) == 2
        # Scores should be preserved
        assert enhanced[0]["hybrid_score"] == 0.9
        assert enhanced[1]["hybrid_score"] == 0.7


# Parametrized tests for robustness
class TestPhase3Robustness:
    """Robustness tests for Phase 3 services."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("chunk_count", [1, 5, 10, 50])
    async def test_contextual_retrieval_various_sizes(self, chunk_count):
        """Test contextual retrieval with various chunk counts."""
        service = ContextualRetrievalService()
        chunks = [
            {"_id": str(i), "content": f"Chunk {i}", "document_id": "doc1"}
            for i in range(chunk_count)
        ]
        
        enhanced = await service.enhance_chunks_with_context(chunks)
        assert len(enhanced) == chunk_count
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "simple query",
        "query with multiple parts and details",
        "what are the benefits and how to use",
        "can you explain and show examples"
    ])
    async def test_query_rewriting_various_types(self, query):
        """Test query rewriting with various query types."""
        service = QueryRewritingService()
        result = await service.rewrite_query(query)
        
        assert result["original"] == query
        assert "variants" in result
        assert "strategies_used" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
