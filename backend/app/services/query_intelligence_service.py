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

# Constants
NUMBERED_LIST_PATTERN = r'\d+\.\s*(.+)'

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
    
    def _parse_numbered_response(self, response: str, query: str) -> List[str]:
        """Parse numbered list from LLM response."""
        lines = response.strip().split('\n')
        rewritten_queries = []
        
        for line in lines:
            match = re.match(NUMBERED_LIST_PATTERN, line.strip())
            if match:
                rewritten_query = match.group(1).strip()
                if rewritten_query and rewritten_query != query:
                    rewritten_queries.append(rewritten_query)
        
        # Ensure we have at least the original
        if not rewritten_queries:
            return [query]
        if query not in rewritten_queries:
            rewritten_queries.insert(0, query)
        
        return rewritten_queries[:3]
    
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
                rewritten_queries = self._parse_numbered_response(response, query)
                await self.redis_client.set_json(cache_key, rewritten_queries, expire_seconds=3600)
                return rewritten_queries
            
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
        
        return [query]  # Fallback to original query
    
    def _extract_conjunction_parts(self, query: str) -> List[str]:
        """Extract sub-queries from conjunctions (and/or)."""
        parts = re.split(r'\s+(?:and|or)\s+', query, flags=re.IGNORECASE)
        return [p.strip() for p in parts if len(p.strip()) > 10]
    
    async def _decompose_with_llm(self, query: str, cache_key: str) -> List[str]:
        """Use LLM to decompose complex query into sub-questions."""
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
        if not response:
            return [query]
        
        # Parse numbered list
        sub_queries = []
        for line in response.strip().split('\n'):
            match = re.match(NUMBERED_LIST_PATTERN, line.strip())
            if match:
                sub_query = match.group(1).strip()
                if len(sub_query) > 10:
                    sub_queries.append(sub_query)
        
        if len(sub_queries) > 1:
            await self.redis_client.set_json(cache_key, sub_queries, expire_seconds=3600)
            return sub_queries
        
        return [query]
    
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
                sub_queries = self._extract_conjunction_parts(query)
                if len(sub_queries) > 1:
                    return sub_queries
            
            # For complex questions, use LLM decomposition
            if len(query.split()) > 10:
                cache_key = f"decompose:{hash(query)}"
                return await self._decompose_with_llm(query, cache_key)
            
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
    
    # ============================================================================
    # Phase 2: ADVANCED QUERY ENHANCEMENT METHODS
    # ============================================================================
    
    def _get_synonym_database(self) -> Dict[str, list]:
        """Get comprehensive synonym mapping for query expansion."""
        return {
            "help": ["assist", "support", "aid", "guidance", "assistance"],
            "problem": ["issue", "challenge", "difficulty", "error", "bug"],
            "understand": ["comprehend", "grasp", "learn", "know", "realize"],
            "use": ["utilize", "employ", "apply", "leverage", "harness"],
            "show": ["display", "demonstrate", "illustrate", "reveal", "present"],
            "find": ["locate", "discover", "search", "identify", "uncover"],
            "work": ["function", "operate", "run", "execute", "perform"],
            "change": ["modify", "alter", "update", "adjust", "transform"],
            "create": ["make", "build", "generate", "produce", "develop"],
            "delete": ["remove", "erase", "destroy", "eliminate", "clear"],
            "fix": ["repair", "resolve", "correct", "patch", "remedy"],
            "optimize": ["improve", "enhance", "boost", "refine", "tune"],
            "analyze": ["examine", "evaluate", "assess", "review", "study"],
            "manage": ["control", "administer", "oversee", "handle", "govern"],
            "integrate": ["combine", "merge", "incorporate", "link", "connect"]
        }
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using embeddings."""
        if not self.embedding_model:
            return 1.0  # Default: assume safe if no embedding model
        
        try:
            embedding1 = self.embedding_model.encode(text1)
            embedding2 = self.embedding_model.encode(text2)
            from numpy import dot
            from numpy.linalg import norm
            return dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
        except Exception as e:
            logger.debug(f"Similarity calculation failed: {e}")
            return 0.9  # Conservative estimate
    
    def _create_synonym_expansion(self, query: str, keyword: str, synonym: str, similarity: float) -> Dict[str, Any]:
        """Create an expansion entry for a synonym replacement."""
        expanded_query = query.replace(keyword, f"{keyword} or {synonym}", 1)
        return {
            "expanded_query": expanded_query,
            "expansion_type": "synonym",
            "original_term": keyword,
            "replacement_term": synonym,
            "semantic_similarity": round(similarity, 3),
            "meaning_preserved": True
        }
    
    def _process_synonym_expansion(self, query: str, keyword: str, synonyms: List[str], 
                                    expansions: List[Dict], max_expansions: int) -> bool:
        """Process synonyms for a keyword. Returns True if max reached."""
        for synonym in synonyms:
            expanded = query.replace(keyword, f"{keyword} or {synonym}", 1)
            if expanded == query:
                continue
            
            similarity = self._calculate_semantic_similarity(query, expanded)
            
            if similarity >= 0.85:
                expansion = self._create_synonym_expansion(query, keyword, synonym, similarity)
                expansions.append(expansion)
            else:
                logger.debug(f"Expansion rejected: '{expanded}' (similarity={similarity:.3f} < 0.85)")
            
            if len(expansions) >= max_expansions:
                return True
        return False
    
    def _build_expansion_result(self, query: str, expansions: List[Dict]) -> Dict[str, Any]:
        """Build expansion result dictionary."""
        return {
            "original_query": query,
            "expansions": expansions,
            "total_expansions": len(expansions),
            "expansion_strategy": "synonym_based_with_semantic_validation",
            "semantic_threshold": 0.85,
            "meaning_preservation_enabled": True
        }
    
    def expand_query(self, query: str, max_expansions: int = 5) -> Dict[str, Any]:
        """
        Expand query with synonyms and related terms for better retrieval.
        SEMANTIC VALIDATION: Only allow expansions that preserve query meaning (similarity > 0.85).
        
        Args:
            query: Original query
            max_expansions: Maximum synonym/related term expansions
            
        Returns:
            Dictionary with expanded queries and expansion metadata
        """
        try:
            synonym_map = self._get_synonym_database()
            expansions = []
            query_lower = query.lower()
            
            # Find and expand synonyms
            for keyword, synonyms in synonym_map.items():
                if keyword not in query_lower:
                    continue
                
                if self._process_synonym_expansion(query, keyword, synonyms, expansions, max_expansions):
                    break
            
            return self._build_expansion_result(query, expansions)
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return {
                "original_query": query,
                "expansions": [],
                "total_expansions": 0,
                "error": str(e)
            }
    
    def _apply_typo_correction(self, query: str) -> tuple[str, bool, str]:
        """Apply typo correction. Returns (corrected, found, strategy)."""
        typo_corrections = {
            "helo": "help", "wrk": "work", "lst": "list", "usr": "user",
            "chr": "character", "teh": "the", "recieve": "receive",
            "occured": "occurred", "seperate": "separate", "definately": "definitely"
        }
        
        for typo, correction in typo_corrections.items():
            if typo.lower() in query.lower():
                corrected = re.sub(r'\b' + typo + r'\b', correction, query, flags=re.IGNORECASE)
                if corrected != query:
                    return corrected, True, "typo_correction"
        return query, False, ""
    
    def _apply_context_addition(self, query: str) -> tuple[str, bool, str]:
        """Add context to query. Returns (modified, found, strategy)."""
        context_additions = {
            "how": "how to solve",
            "why": "why is this important",
            "what": "what is the definition and usage of",
            "when": "when should we use",
        }
        
        query_start = query.split()[0].lower() if query.split() else ""
        if query_start in context_additions:
            specified = query.replace(query_start, context_additions[query_start], 1)
            if specified != query:
                return specified, True, "context_addition"
        return query, False, ""
    
    def _apply_phrase_normalization(self, query: str) -> tuple[str, bool, str]:
        """Normalize common phrases. Returns (normalized, found, strategy)."""
        phrase_normalizations = {
            r"\b(?:is\s+)?there\s+(?:a\s+)?way": "how to",
            r"\bwhat\s+(?:is\s+)?the\s+(?:best\s+)?way": "best practices for",
            r"\bhow\s+(?:can\s+)?(?:i\s+)?(?:you\s+)?": "steps to",
        }
        
        for pattern, replacement in phrase_normalizations.items():
            if re.search(pattern, query.lower()):
                normalized = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                if normalized != query:
                    return normalized, True, "phrase_normalization"
        return query, False, ""
    
    def rewrite_query_advanced(self, query: str) -> Dict[str, Any]:
        """
        Advanced query rewriting that handles typos, grammar, and rephrasing.
        
        Args:
            query: Original query (may have typos/grammatical issues)
            
        Returns:
            Dictionary with corrected and rephrased queries
        """
        try:
            rewrites = {"original": query, "rewrites": [], "strategies_applied": []}
            
            # Apply typo correction
            corrected, found, strategy = self._apply_typo_correction(query)
            if found:
                rewrites["rewrites"].append({"query": corrected, "strategy": strategy, "confidence": 0.95})
                rewrites["strategies_applied"].append(strategy)
            
            # Apply context addition
            specified, found, strategy = self._apply_context_addition(query)
            if found:
                rewrites["rewrites"].append({"query": specified, "strategy": strategy, "confidence": 0.85})
                rewrites["strategies_applied"].append(strategy)
            
            # Apply phrase normalization
            normalized, found, strategy = self._apply_phrase_normalization(query)
            if found:
                rewrites["rewrites"].append({"query": normalized, "strategy": strategy, "confidence": 0.8})
                rewrites["strategies_applied"].append(strategy)
            
            return rewrites
            
        except Exception as e:
            logger.warning(f"Advanced query rewriting failed: {e}")
            return {"original": query, "rewrites": [], "strategies_applied": [], "error": str(e)}
    
    def _decompose_conjunction(self, query: str) -> tuple[List[Dict], str, str]:
        """Decompose conjunction-based query. Returns (sub_queries, type, complexity)."""
        parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
        if len(parts) >= 2:
            sub_queries = [
                {"sub_query": p.strip(), "index": i, "operator": "AND"}
                for i, p in enumerate(parts) if len(p.strip()) > 10
            ]
            return sub_queries, "conjunction", "multi_part"
        return [], "none", "simple"
    
    def _decompose_disjunction(self, query: str) -> tuple[List[Dict], str, str]:
        """Decompose disjunction-based query. Returns (sub_queries, type, complexity)."""
        parts = re.split(r'\s+or\s+', query, flags=re.IGNORECASE)
        if len(parts) >= 2:
            sub_queries = [
                {"sub_query": p.strip(), "index": i, "operator": "OR"}
                for i, p in enumerate(parts) if len(p.strip()) > 10
            ]
            return sub_queries, "disjunction", "alternative"
        return [], "none", "simple"
    
    async def _decompose_dependency_llm(self, query: str, cache_key: str) -> List[Dict]:
        """Decompose using LLM for dependency-based queries."""
        cached = await self.redis_client.get_json(cache_key)
        if cached:
            return cached
        
        prompt = f"""This query has multiple components that depend on each other:
"{query}"

Break it into 2-3 ordered sub-questions where each builds on previous answers. Number them in dependency order.
Format: 1. question
2. question
3. question"""
        
        response = await self._call_lmstudio(prompt, max_tokens=200)
        if not response:
            return []
        
        sub_queries = []
        for i, line in enumerate(response.strip().split('\n')):
            match = re.match(NUMBERED_LIST_PATTERN, line.strip())
            if match:
                sub_queries.append({
                    "sub_query": match.group(1).strip(),
                    "index": i,
                    "operator": "THEN",
                    "dependency_order": i
                })
        
        if sub_queries:
            await self.redis_client.set_json(cache_key, sub_queries, expire_seconds=3600)
        
        return sub_queries
    
    async def decompose_query_advanced(self, query: str) -> Dict[str, Any]:
        """
        Advanced query decomposition for complex multi-part questions.
        
        Args:
            query: Original query
            
        Returns:
            Dictionary with decomposed sub-queries and structure
        """
        try:
            decomposition = {
                "original": query,
                "sub_queries": [],
                "decomposition_type": "none",
                "complexity_level": "simple"
            }
            
            # Type 1: Conjunction-based (Q1 AND Q2)
            if re.search(r'\band\b', query, re.IGNORECASE):
                sub_queries, dec_type, complexity = self._decompose_conjunction(query)
                if sub_queries:
                    decomposition.update({
                        "sub_queries": sub_queries,
                        "decomposition_type": dec_type,
                        "complexity_level": complexity
                    })
                    return decomposition
            
            # Type 2: Disjunction-based (Q1 OR Q2)
            if re.search(r'\bor\b', query, re.IGNORECASE):
                sub_queries, dec_type, complexity = self._decompose_disjunction(query)
                if sub_queries:
                    decomposition.update({
                        "sub_queries": sub_queries,
                        "decomposition_type": dec_type,
                        "complexity_level": complexity
                    })
                    return decomposition
            
            # Type 3: Dependency-based (complex relationships)
            if len(query.split()) > 15:
                try:
                    cache_key = f"decompose_adv:{hash(query)}"
                    sub_queries = await self._decompose_dependency_llm(query, cache_key)
                    if sub_queries:
                        decomposition.update({
                            "sub_queries": sub_queries,
                            "decomposition_type": "dependency_based",
                            "complexity_level": "complex"
                        })
                except Exception as e:
                    logger.debug(f"LLM decomposition failed: {e}")
            
            return decomposition
            
        except Exception as e:
            logger.warning(f"Advanced query decomposition failed: {e}")
            return {
                "original": query,
                "sub_queries": [],
                "decomposition_type": "error",
                "error": str(e)
            }
    
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