"""
Context Optimization Service for Advanced RAG.
This module implements context compression, autocut filtering, and chain-of-thought prompting
to improve response quality and reduce token usage.
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import httpx

from transformers import pipeline, AutoTokenizer
import tiktoken

from ..core.cache_redis import get_redis_client, RedisClient
from ..config import settings

logger = logging.getLogger(__name__)

class ContextOptimizationService:
    """Service for optimizing context before LLM generation."""
    
    def __init__(self):
        self.summarizer = None
        self.tokenizer = None
        self.redis_client = None
        self.http_client = None
    
    async def initialize(self):
        """Initialize the context optimization service."""
        try:
            logger.info("Initializing context optimization service...")
            
            # Get Redis client for caching
            self.redis_client = await get_redis_client()
            
            # Initialize HTTP client for LMStudio
            self.http_client = httpx.AsyncClient(timeout=60.0)
            
            # Initialize tokenizer for token counting
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception:
                # Fallback to simple word-based counting
                self.tokenizer = None
                logger.warning("tiktoken not available, using word-based token estimation")
            
            # Initialize summarizer model (lightweight)
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # CPU
                )
            except Exception as e:
                logger.warning(f"Could not load summarizer model: {e}")
                self.summarizer = None
            
            logger.info("Context optimization service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize context optimization service: {e}")
            raise
    
    async def optimize_context(self, chunks: List[Dict], query: str, 
                             max_tokens: int = 2000, strategy: str = "balanced") -> Dict[str, Any]:
        """
        Optimize context for LLM generation.
        
        Args:
            chunks: Retrieved document chunks
            query: Original user query
            max_tokens: Maximum token limit for context
            strategy: Optimization strategy
            
        Returns:
            Optimized context data
        """
        try:
            # Step 1: Filter and rank chunks by relevance
            filtered_chunks = await self._autocut_filter(chunks, query, strategy)
            
            # Step 2: Compress context if needed
            compressed_context = await self._compress_context(filtered_chunks, query, max_tokens)
            
            # Step 3: Structure context for chain-of-thought
            structured_context = self._structure_for_cot(compressed_context, query, strategy)
            
            # Step 4: Calculate final metrics
            token_count = self._count_tokens(structured_context["formatted_context"])
            
            return {
                "formatted_context": structured_context["formatted_context"],
                "reasoning_template": structured_context["reasoning_template"],
                "sources_used": structured_context["sources_used"],
                "original_chunks": len(chunks),
                "filtered_chunks": len(filtered_chunks),
                "final_token_count": token_count,
                "compression_ratio": len(chunks) / len(filtered_chunks) if filtered_chunks else 1.0,
                "optimization_strategy": strategy
            }
            
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            # Fallback to simple concatenation
            fallback_text = "\n\n".join([chunk.get("text", "") for chunk in chunks[:5]])
            return {
                "formatted_context": fallback_text[:max_tokens * 4],  # Rough char limit
                "reasoning_template": "Answer based on the provided information.",
                "sources_used": chunks[:5],
                "original_chunks": len(chunks),
                "filtered_chunks": min(5, len(chunks)),
                "final_token_count": self._count_tokens(fallback_text),
                "compression_ratio": 1.0,
                "optimization_strategy": "fallback"
            }
    
    async def _autocut_filter(self, chunks: List[Dict], query: str, strategy: str) -> List[Dict]:
        """
        Filter chunks based on relevance thresholds and strategy.
        
        Args:
            chunks: List of retrieved chunks
            query: User query
            strategy: Filtering strategy
            
        Returns:
            Filtered chunks
        """
        if not chunks:
            return []
        
        # Define relevance thresholds based on strategy
        thresholds = {
            "high_precision": 0.7,
            "multi_perspective": 0.4,
            "time_weighted": 0.5,
            "step_by_step": 0.6,
            "broad_search": 0.3,
            "solution_focused": 0.5,
            "balanced": 0.5
        }
        
        threshold = thresholds.get(strategy, 0.5)
        
        # Filter by score threshold
        filtered = []
        for chunk in chunks:
            score = chunk.get("score", 0)
            combined_score = chunk.get("combined_score", score)
            
            if combined_score >= threshold:
                filtered.append(chunk)
        
        # Ensure minimum number of chunks for broad strategies
        if strategy in ["broad_search", "multi_perspective"] and len(filtered) < 3:
            # Include top 3 chunks regardless of threshold
            sorted_chunks = sorted(chunks, key=lambda x: x.get("combined_score", x.get("score", 0)), reverse=True)
            filtered = sorted_chunks[:3]
        
        # Limit maximum chunks based on strategy
        max_chunks = {
            "high_precision": 3,
            "multi_perspective": 8,
            "time_weighted": 5,
            "step_by_step": 6,
            "broad_search": 10,
            "solution_focused": 4,
            "balanced": 6
        }
        
        limit = max_chunks.get(strategy, 6)
        
        return filtered[:limit]
    
    async def _compress_context(self, chunks: List[Dict], query: str, max_tokens: int) -> List[Dict]:
        """
        Compress context using summarization and deduplication.
        
        Args:
            chunks: Filtered chunks
            query: User query
            max_tokens: Token limit
            
        Returns:
            Compressed chunks
        """
        if not chunks:
            return []
        
        # Calculate current token usage
        total_text = "\n\n".join([chunk.get("text", "") for chunk in chunks])
        current_tokens = self._count_tokens(total_text)
        
        if current_tokens <= max_tokens:
            return chunks  # No compression needed
        
        # Try content deduplication first
        deduplicated = self._deduplicate_content(chunks)
        
        dedup_text = "\n\n".join([chunk.get("text", "") for chunk in deduplicated])
        dedup_tokens = self._count_tokens(dedup_text)
        
        if dedup_tokens <= max_tokens:
            return deduplicated
        
        # Apply summarization if available and needed
        if self.summarizer and len(chunks) > 3:
            try:
                compressed_chunks = await self._summarize_chunks(deduplicated, query, max_tokens)
                return compressed_chunks
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
        
        # Fallback: truncate to fit token limit
        truncated = []
        current_tokens = 0
        
        for chunk in deduplicated:
            chunk_tokens = self._count_tokens(chunk.get("text", ""))
            if current_tokens + chunk_tokens <= max_tokens:
                truncated.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Try to fit partial content
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only if we have reasonable space left
                    partial_text = self._truncate_to_tokens(chunk.get("text", ""), remaining_tokens)
                    if partial_text:
                        partial_chunk = chunk.copy()
                        partial_chunk["text"] = partial_text
                        partial_chunk["truncated"] = True
                        truncated.append(partial_chunk)
                break
        
        return truncated
    
    def _deduplicate_content(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate or highly similar content."""
        if len(chunks) <= 1:
            return chunks
        
        deduplicated = []
        seen_content = set()
        
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            
            # Create a signature for similarity checking
            words = text.lower().split()
            if len(words) < 5:
                signature = text.lower()
            else:
                # Use first and last few words as signature
                signature = " ".join(words[:3] + words[-2:])
            
            # Check for exact or very similar content
            is_duplicate = False
            for seen_sig in seen_content:
                if self._text_similarity(signature, seen_sig) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(chunk)
                seen_content.add(signature)
        
        return deduplicated
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _summarize_chunks(self, chunks: List[Dict], query: str, max_tokens: int) -> List[Dict]:
        """Summarize chunks using the summarization model."""
        if not self.summarizer:
            return chunks
        
        try:
            # Group similar chunks for summarization
            grouped_chunks = self._group_chunks_by_topic(chunks)
            summarized_chunks = []
            
            for group in grouped_chunks:
                if len(group) == 1:
                    summarized_chunks.extend(group)
                else:
                    # Combine texts from group
                    combined_text = "\n\n".join([chunk.get("text", "") for chunk in group])
                    
                    # Summarize if the combined text is long enough
                    if len(combined_text.split()) > 100:
                        try:
                            summary = self.summarizer(
                                combined_text,
                                max_length=150,
                                min_length=50,
                                do_sample=False
                            )[0]["summary_text"]
                            
                            # Create summarized chunk
                            summarized_chunk = {
                                "text": summary,
                                "score": max(chunk.get("score", 0) for chunk in group),
                                "combined_score": max(chunk.get("combined_score", 0) for chunk in group),
                                "doc_id": group[0].get("doc_id"),
                                "metadata": {
                                    "summarized": True,
                                    "original_chunks": len(group),
                                    "sources": [chunk.get("doc_id") for chunk in group]
                                },
                                "sources": ["summarization"]
                            }
                            summarized_chunks.append(summarized_chunk)
                            
                        except Exception:
                            # If summarization fails, use original chunks
                            summarized_chunks.extend(group)
                    else:
                        summarized_chunks.extend(group)
            
            return summarized_chunks
            
        except Exception as e:
            logger.error(f"Chunk summarization failed: {e}")
            return chunks
    
    def _group_chunks_by_topic(self, chunks: List[Dict]) -> List[List[Dict]]:
        """Group chunks by topic similarity for better summarization."""
        if len(chunks) <= 2:
            return [chunks]
        
        groups = []
        remaining_chunks = chunks.copy()
        
        while remaining_chunks:
            # Start new group with first remaining chunk
            current_group = [remaining_chunks.pop(0)]
            
            # Find similar chunks to add to group
            to_remove = []
            for i, chunk in enumerate(remaining_chunks):
                # Simple similarity check based on document source and text overlap
                if self._chunks_are_similar(current_group[0], chunk):
                    current_group.append(chunk)
                    to_remove.append(i)
            
            # Remove chunks added to current group
            for i in reversed(to_remove):
                remaining_chunks.pop(i)
            
            groups.append(current_group)
        
        return groups
    
    def _chunks_are_similar(self, chunk1: Dict, chunk2: Dict) -> bool:
        """Check if two chunks are similar enough to group together."""
        # Same document
        if chunk1.get("doc_id") == chunk2.get("doc_id"):
            return True
        
        # Similar metadata
        meta1 = chunk1.get("metadata", {})
        meta2 = chunk2.get("metadata", {})
        
        if meta1.get("document_type") == meta2.get("document_type"):
            return True
        
        # Text similarity
        text1 = chunk1.get("text", "")
        text2 = chunk2.get("text", "")
        
        if self._text_similarity(text1[:200], text2[:200]) > 0.3:
            return True
        
        return False
    
    def _structure_for_cot(self, chunks: List[Dict], query: str, strategy: str) -> Dict[str, Any]:
        """Structure context for chain-of-thought reasoning."""
        
        # Build the formatted context
        context_parts = []
        sources_used = []
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "").strip()
            if text:
                # Add source information
                doc_id = chunk.get("doc_id", "unknown")
                metadata = chunk.get("metadata", {})
                source_info = f"Source {i}"
                
                if metadata.get("filename"):
                    source_info += f" ({metadata['filename']})"
                elif metadata.get("title"):
                    source_info += f" ({metadata['title']})"
                
                context_parts.append(f"{source_info}:\n{text}")
                sources_used.append(chunk)
        
        formatted_context = "\n\n".join(context_parts)
        
        # Create reasoning template based on strategy
        reasoning_template = self._get_reasoning_template(strategy, query)
        
        return {
            "formatted_context": formatted_context,
            "reasoning_template": reasoning_template,
            "sources_used": sources_used
        }
    
    def _get_reasoning_template(self, strategy: str, query: str) -> str:
        """Generate chain-of-thought reasoning template based on strategy."""
        
        base_template = """Based on the provided information, I will analyze this step by step:

1. **Key Information Analysis**: Let me identify the most relevant facts from the sources.

2. **Context Integration**: I'll connect the information from different sources to build a comprehensive understanding.

3. **Direct Response**: Now I'll provide a clear, accurate answer to your question.

4. **Source Verification**: I'll note which sources support my response."""
        
        strategy_templates = {
            "high_precision": """I will provide a precise, fact-based response by:

1. **Fact Verification**: Checking the reliability and consistency of information across sources.
2. **Precision Focus**: Identifying the exact information that directly answers your question.
3. **Accurate Response**: Providing a definitive answer based only on verified information.
4. **Source Citation**: Clearly indicating which sources support each claim.""",
            
            "multi_perspective": """I will analyze this from multiple angles:

1. **Perspective Analysis**: Examining different viewpoints presented in the sources.
2. **Comparison**: Identifying similarities and differences between perspectives.
3. **Balanced Synthesis**: Integrating multiple viewpoints into a comprehensive response.
4. **Complete Picture**: Providing a well-rounded answer that considers all relevant aspects.""",
            
            "step_by_step": """I will break this down into clear steps:

1. **Process Identification**: Determining the logical sequence of steps or components.
2. **Step-by-Step Breakdown**: Organizing information in a clear, sequential manner.
3. **Practical Application**: Explaining how to apply or understand each step.
4. **Complete Procedure**: Providing a thorough, actionable response.""",
            
            "solution_focused": """I will focus on providing actionable solutions:

1. **Problem Analysis**: Understanding the specific issue or challenge.
2. **Solution Identification**: Finding relevant solutions from the available information.
3. **Implementation Guidance**: Explaining how to apply the solutions practically.
4. **Expected Outcomes**: Describing what results to expect."""
        }
        
        return strategy_templates.get(strategy, base_template)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback to word-based estimation (rough approximation)
        words = len(text.split())
        return int(words * 1.3)  # Rough conversion factor
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if self._count_tokens(text) <= max_tokens:
            return text
        
        # Simple truncation by words
        words = text.split()
        estimated_words = int(max_tokens / 1.3)  # Reverse of estimation factor
        
        if estimated_words >= len(words):
            return text
        
        # Truncate and add ellipsis
        truncated_words = words[:estimated_words]
        return " ".join(truncated_words) + "..."

# Global instance
context_optimization_service = ContextOptimizationService()