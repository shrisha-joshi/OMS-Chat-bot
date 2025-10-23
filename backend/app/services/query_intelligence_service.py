"""
Advanced Query Intelligence Service for Enhanced RAG Capabilities.
This module implements query rewriting, decomposition, and HyDE (Hypothetical Document Embedding)
features to improve retrieval quality and response accuracy.
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import httpx

from sentence_transformers import SentenceTransformer

from ..core.cache_redis import get_redis_client, RedisClient
from ..config import settings

logger = logging.getLogger(__name__)

class QueryIntelligenceService:
    """Service for advanced query understanding and enhancement."""
    
    def __init__(self):
        self.embedding_model = None
        self.redis_client = None
        self.http_client = None
    
    async def initialize(self):
        """Initialize the query intelligence service."""
        try:
            logger.info("Initializing query intelligence service...")
            
            # Get Redis client for caching
            self.redis_client = await get_redis_client()
            
            # Initialize HTTP client for LMStudio
            self.http_client = httpx.AsyncClient(timeout=60.0)
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(settings.embedding_model_name)
            
            logger.info("Query intelligence service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize query intelligence service: {e}")
            raise
    
    async def enhance_query(self, original_query: str) -> Dict[str, Any]:
        """
        Enhance a query using multiple advanced techniques.
        
        Args:
            original_query: The original user query
            
        Returns:
            Dictionary containing enhanced query information
        """
        try:
            # Step 1: Query classification
            query_type = self._classify_query(original_query)
            
            # Step 2: Query rewriting
            rewritten_queries = await self._rewrite_query(original_query)
            
            # Step 3: Query decomposition (for complex queries)
            sub_queries = await self._decompose_query(original_query)
            
            # Step 4: Generate HyDE (Hypothetical Document Embedding)
            hyde_embedding = await self._generate_hyde_embedding(original_query)
            
            # Step 5: Extract intent and entities
            intent = self._extract_intent(original_query)
            entities = self._extract_key_terms(original_query)
            
            return {
                "original_query": original_query,
                "query_type": query_type,
                "rewritten_queries": rewritten_queries,
                "sub_queries": sub_queries,
                "hyde_embedding": hyde_embedding,
                "intent": intent,
                "entities": entities,
                "processing_strategy": self._determine_processing_strategy(query_type, intent)
            }
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return {
                "original_query": original_query,
                "query_type": "unknown",
                "rewritten_queries": [original_query],
                "sub_queries": [original_query],
                "hyde_embedding": None,
                "intent": "information_seeking",
                "entities": [],
                "processing_strategy": "standard"
            }
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the query type for adaptive retrieval.
        
        Args:
            query: User query
            
        Returns:
            Query type classification
        """
        query_lower = query.lower()
        
        # Question patterns
        if any(query_lower.startswith(word) for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return "factual"
        
        # Comparison patterns
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']):
            return "comparative"
        
        # Temporal patterns
        if any(word in query_lower for word in ['latest', 'recent', 'new', 'current', 'today', 'yesterday']):
            return "temporal"
        
        # Procedure patterns
        if any(word in query_lower for word in ['how to', 'steps', 'process', 'procedure', 'tutorial']):
            return "procedural"
        
        # Definition patterns
        if any(word in query_lower for word in ['what is', 'define', 'meaning', 'definition']):
            return "definitional"
        
        # List/enumeration patterns
        if any(word in query_lower for word in ['list', 'types', 'kinds', 'examples', 'all']):
            return "enumeration"
        
        # Troubleshooting patterns
        if any(word in query_lower for word in ['error', 'problem', 'issue', 'fix', 'troubleshoot']):
            return "troubleshooting"
        
        return "exploratory"
    
    async def _rewrite_query(self, query: str) -> List[str]:
        """
        Rewrite the query for better retrieval using LLM.
        
        Args:
            query: Original query
            
        Returns:
            List of rewritten queries
        """
        try:
            # Check cache first
            cache_key = f"rewrite:{hash(query)}"
            cached_result = await self.redis_client.get_json(cache_key)
            if cached_result:
                return cached_result
            
            prompt = f"""Rewrite the following query to make it more specific and retrievable from a knowledge base. 
Generate 3 different versions that capture the same intent but use different wording and focus on different aspects.

Original query: "{query}"

Provide 3 rewritten versions, one per line:
1. 
2. 
3. """

            # Call LMStudio for query rewriting
            response = await self._call_lmstudio(prompt, max_tokens=200)
            
            if response:
                # Parse the numbered list
                lines = response.strip().split('\n')
                rewritten_queries = []
                
                for line in lines:
                    # Extract text after numbers (1., 2., 3.)
                    match = re.match(r'\d+\.\s*(.+)', line.strip())
                    if match:
                        rewritten_query = match.group(1).strip()
                        if rewritten_query and rewritten_query != query:
                            rewritten_queries.append(rewritten_query)
                
                # Add original query if we don't have enough rewrites
                if not rewritten_queries:
                    rewritten_queries = [query]
                elif query not in rewritten_queries:
                    rewritten_queries.insert(0, query)
                
                # Cache result
                await self.redis_client.set_json(cache_key, rewritten_queries, expire_seconds=3600)
                
                return rewritten_queries[:3]  # Limit to 3 variations
            
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
        
        return [query]  # Fallback to original query
    
    async def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex queries into simpler sub-queries.
        
        Args:
            query: Original query
            
        Returns:
            List of sub-queries
        """
        try:
            # Simple decomposition for obvious multi-part queries
            if ' and ' in query.lower() or ' or ' in query.lower():
                # Handle conjunctions
                sub_queries = []
                
                # Split on 'and' and 'or'
                parts = re.split(r'\s+(?:and|or)\s+', query, flags=re.IGNORECASE)
                
                for part in parts:
                    part = part.strip()
                    if len(part) > 10:  # Only include substantial parts
                        sub_queries.append(part)
                
                if len(sub_queries) > 1:
                    return sub_queries
            
            # For complex questions, use LLM decomposition
            if len(query.split()) > 10:  # Only decompose longer queries
                cache_key = f"decompose:{hash(query)}"
                cached_result = await self.redis_client.get_json(cache_key)
                if cached_result:
                    return cached_result
                
                prompt = f"""Break down this complex query into 2-3 simpler, focused sub-questions that together would provide a complete answer.

Complex query: "{query}"

Sub-questions (one per line):
1. 
2. 
3. """

                response = await self._call_lmstudio(prompt, max_tokens=150)
                
                if response:
                    lines = response.strip().split('\n')
                    sub_queries = []
                    
                    for line in lines:
                        match = re.match(r'\d+\.\s*(.+)', line.strip())
                        if match:
                            sub_query = match.group(1).strip()
                            if len(sub_query) > 10:  # Only substantial questions
                                sub_queries.append(sub_query)
                    
                    if len(sub_queries) > 1:
                        # Cache result
                        await self.redis_client.set_json(cache_key, sub_queries, expire_seconds=3600)
                        return sub_queries
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
        
        return [query]  # Fallback to original query
    
    async def _generate_hyde_embedding(self, query: str) -> Optional[List[float]]:
        """
        Generate HyDE (Hypothetical Document Embedding) for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Embedding vector for hypothetical answer
        """
        try:
            cache_key = f"hyde:{hash(query)}"
            cached_embedding = await self.redis_client.get_json(cache_key)
            if cached_embedding:
                return cached_embedding
            
            # Generate hypothetical answer using LLM
            prompt = f"""Generate a comprehensive, factual answer to this question as if you were writing documentation or a knowledge base entry. Be specific and detailed.

Question: {query}

Answer:"""

            hypothetical_answer = await self._call_lmstudio(prompt, max_tokens=300)
            
            if hypothetical_answer and len(hypothetical_answer.strip()) > 50:
                # Generate embedding for the hypothetical answer
                hyde_embedding = self.embedding_model.encode(hypothetical_answer).tolist()
                
                # Cache the embedding
                await self.redis_client.set_json(cache_key, hyde_embedding, expire_seconds=7200)
                
                return hyde_embedding
            
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
        
        return None
    
    def _extract_intent(self, query: str) -> str:
        """Extract the user's intent from the query."""
        query_lower = query.lower()
        
        # Information seeking
        if any(word in query_lower for word in ['what', 'who', 'where', 'when', 'which', 'explain', 'describe']):
            return "information_seeking"
        
        # How-to/procedural
        if any(word in query_lower for word in ['how', 'steps', 'process', 'tutorial', 'guide']):
            return "how_to"
        
        # Problem solving
        if any(word in query_lower for word in ['fix', 'solve', 'error', 'problem', 'issue', 'troubleshoot']):
            return "problem_solving"
        
        # Comparison
        if any(word in query_lower for word in ['compare', 'difference', 'better', 'versus', 'vs']):
            return "comparison"
        
        # Recommendation
        if any(word in query_lower for word in ['recommend', 'suggest', 'best', 'should', 'advice']):
            return "recommendation"
        
        return "general"
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms and potential entities from the query."""
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b[A-Za-z]{3,}\b', query)
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 
            'did', 'she', 'use', 'way', 'what', 'when', 'where', 'why', 'will', 
            'with', 'have', 'this', 'that', 'they', 'from', 'been', 'said', 'each', 
            'which', 'there', 'would', 'make', 'like', 'into', 'time', 'very'
        }
        
        key_terms = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        
        return key_terms[:10]  # Limit to top 10 terms
    
    def _determine_processing_strategy(self, query_type: str, intent: str) -> str:
        """Determine the best processing strategy based on query analysis."""
        
        if query_type == "factual" and intent == "information_seeking":
            return "high_precision"
        elif query_type == "comparative":
            return "multi_perspective"
        elif query_type == "temporal":
            return "time_weighted"
        elif query_type == "procedural" or intent == "how_to":
            return "step_by_step"
        elif query_type == "exploratory":
            return "broad_search"
        elif intent == "problem_solving":
            return "solution_focused"
        
        return "balanced"
    
    async def _call_lmstudio(self, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """Call LMStudio API for text generation."""
        try:
            data = {
                "model": settings.lmstudio_model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.3,  # Lower temperature for more focused responses
                "top_p": 0.9,
                "stop": ["\\n\\n", "User:", "Question:"]
            }
            
            response = await self.http_client.post(
                f"{settings.lmstudio_api_url}/completions",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("choices", [{}])[0].get("text", "").strip()
                return text if text else None
            else:
                logger.warning(f"LMStudio API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"LMStudio call failed: {e}")
            return None

# Global instance
query_intelligence_service = QueryIntelligenceService()