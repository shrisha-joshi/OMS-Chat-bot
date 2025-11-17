"""
Context Window Optimizer - Dynamic Context Management
Based on research from arXiv:2407.19794 (Context Window Utilization)

This service optimizes the utilization of the LLM's context window by dynamically
adjusting chunk selection to target 70-85% context utilization, maximizing both
efficiency and response quality.
"""

import logging
from typing import List, Dict, Any, Tuple
import tiktoken
from ..config import settings

logger = logging.getLogger(__name__)


class ContextWindowOptimizer:
    """
    Optimizes context window utilization for LLM prompts.
    
    Key Features:
    - Dynamic chunk selection based on available context
    - Target utilization of 70-85% (research-backed optimal range)
    - Token-aware truncation and prioritization
    - Metrics tracking for monitoring
    """
    
    def __init__(
        self,
        target_utilization: float = 0.75,
        min_utilization: float = 0.50,
        max_utilization: float = 0.90
    ):
        """
        Initialize the context window optimizer.
        
        Args:
            target_utilization: Target context window usage (default: 75%)
            min_utilization: Minimum acceptable utilization (default: 50%)
            max_utilization: Maximum acceptable utilization (default: 90%)
        """
        self.target_utilization = target_utilization
        self.min_utilization = min_utilization
        self.max_utilization = max_utilization
        
        # LLM context limits
        self.llm_context_limit = settings.max_context_tokens
        self.response_reserve = settings.max_llm_output_tokens
        
        # Initialize tokenizer (cl100k_base for most modern models)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}, using fallback")
            self.tokenizer = None
        
        # Metrics
        self.optimization_count = 0
        self.total_utilization = 0.0
        
        logger.info(
            f"Context Window Optimizer initialized: "
            f"target={target_utilization:.0%}, "
            f"limit={self.llm_context_limit} tokens"
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or fallback estimation.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Tokenization failed: {e}, using fallback")
        
        # Fallback: approximate token count (1 token ≈ 4 characters)
        return len(text) // 4
    
    def optimize_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        system_prompt: str = "",
        metadata_tokens: int = 100
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Optimize chunk selection to target context window utilization.
        
        Args:
            query: User query
            chunks: Retrieved chunks (sorted by relevance)
            system_prompt: System prompt text
            metadata_tokens: Estimated tokens for metadata (default: 100)
            
        Returns:
            Tuple of (optimized_chunks, metrics_dict)
        """
        # Count fixed tokens (query + system prompt + response reserve + overhead)
        query_tokens = self.count_tokens(query)
        system_tokens = self.count_tokens(system_prompt)
        fixed_tokens = query_tokens + system_tokens + self.response_reserve + metadata_tokens
        
        # Calculate available tokens for chunks
        available_tokens = self.llm_context_limit - fixed_tokens
        
        if available_tokens <= 0:
            logger.warning(
                f"No tokens available for chunks! "
                f"Fixed tokens ({fixed_tokens}) exceed limit ({self.llm_context_limit})"
            )
            return [], {
                "available_tokens": 0,
                "utilized_tokens": 0,
                "utilization": 0.0,
                "chunks_selected": 0,
                "chunks_dropped": len(chunks),
                "efficiency": "poor"
            }
        
        # Target token count for chunks
        target_chunk_tokens = int(available_tokens * self.target_utilization)
        
        # Select chunks to fit target utilization
        selected_chunks = []
        current_tokens = 0
        chunk_token_counts = []
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            chunk_tokens = self.count_tokens(chunk_text)
            chunk_token_counts.append(chunk_tokens)
            
            # Check if adding this chunk would exceed max utilization
            potential_tokens = current_tokens + chunk_tokens
            max_allowed_tokens = int(available_tokens * self.max_utilization)
            
            if potential_tokens <= max_allowed_tokens:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
                
                # Stop if we've reached target (with some buffer)
                if current_tokens >= target_chunk_tokens * 0.95:
                    break
            else:
                # Truncate last chunk if it would exceed limit
                if current_tokens < target_chunk_tokens * 0.80:
                    # We haven't reached target yet, try to truncate this chunk
                    remaining_tokens = max_allowed_tokens - current_tokens
                    if remaining_tokens > 50:  # Only truncate if meaningful space remains
                        truncated_chunk = self._truncate_chunk(chunk, remaining_tokens)
                        selected_chunks.append(truncated_chunk)
                        current_tokens += self.count_tokens(truncated_chunk["text"])
                break
        
        # Calculate actual utilization
        actual_utilization = current_tokens / available_tokens if available_tokens > 0 else 0.0
        
        # Determine efficiency rating
        if actual_utilization >= self.min_utilization and actual_utilization <= self.max_utilization:
            efficiency = "optimal"
        elif actual_utilization < self.min_utilization:
            efficiency = "underutilized"
        else:
            efficiency = "overutilized"
        
        # Update metrics
        self.optimization_count += 1
        self.total_utilization += actual_utilization
        
        # Build metrics dictionary
        metrics = {
            "query_tokens": query_tokens,
            "system_tokens": system_tokens,
            "fixed_tokens": fixed_tokens,
            "available_tokens": available_tokens,
            "target_tokens": target_chunk_tokens,
            "utilized_tokens": current_tokens,
            "utilization": actual_utilization,
            "chunks_total": len(chunks),
            "chunks_selected": len(selected_chunks),
            "chunks_dropped": len(chunks) - len(selected_chunks),
            "efficiency": efficiency,
            "avg_chunk_tokens": sum(chunk_token_counts[:len(selected_chunks)]) / len(selected_chunks) if selected_chunks else 0,
            "target_utilization": self.target_utilization
        }
        
        # Log optimization results
        logger.info(
            f"Context optimization: {actual_utilization:.1%} utilization "
            f"({current_tokens}/{available_tokens} tokens), "
            f"{len(selected_chunks)}/{len(chunks)} chunks, "
            f"efficiency={efficiency}"
        )
        
        if actual_utilization < self.min_utilization:
            logger.warning(
                f"Context underutilized: {actual_utilization:.1%} < {self.min_utilization:.1%}. "
                f"Consider retrieving more chunks or increasing chunk size."
            )
        elif actual_utilization > self.max_utilization:
            logger.warning(
                f"Context overutilized: {actual_utilization:.1%} > {self.max_utilization:.1%}. "
                f"Some content may be truncated."
            )
        
        return selected_chunks, metrics
    
    def _truncate_chunk(
        self,
        chunk: Dict[str, Any],
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Truncate a chunk to fit within token limit.
        
        Args:
            chunk: Chunk to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated chunk
        """
        text = chunk.get("text", "")
        
        if self.tokenizer:
            # Use tokenizer for accurate truncation
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                truncated_text = self.tokenizer.decode(truncated_tokens)
                truncated_text += "... [truncated]"
            else:
                truncated_text = text
        else:
            # Fallback: character-based truncation
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                truncated_text = text[:max_chars] + "... [truncated]"
            else:
                truncated_text = text
        
        # Create new chunk with truncated text
        truncated_chunk = chunk.copy()
        truncated_chunk["text"] = truncated_text
        truncated_chunk["truncated"] = True
        
        return truncated_chunk
    
    def get_average_utilization(self) -> float:
        """
        Get average context window utilization across all optimizations.
        
        Returns:
            Average utilization as a float (0.0 to 1.0)
        """
        if self.optimization_count == 0:
            return 0.0
        return self.total_utilization / self.optimization_count
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimization metrics.
        
        Returns:
            Dictionary with optimization statistics
        """
        avg_utilization = self.get_average_utilization()
        
        return {
            "total_optimizations": self.optimization_count,
            "average_utilization": avg_utilization,
            "target_utilization": self.target_utilization,
            "llm_context_limit": self.llm_context_limit,
            "response_reserve": self.response_reserve,
            "efficiency_status": self._get_efficiency_status(avg_utilization)
        }
    
    def _get_efficiency_status(self, utilization: float) -> str:
        """Determine efficiency status based on utilization."""
        if utilization >= self.min_utilization and utilization <= self.max_utilization:
            return "optimal"
        elif utilization < self.min_utilization:
            return "underutilized"
        else:
            return "overutilized"
    
    def adjust_target_utilization(self, new_target: float):
        """
        Dynamically adjust target utilization.
        
        Args:
            new_target: New target utilization (0.0 to 1.0)
        """
        if not 0.0 < new_target < 1.0:
            logger.warning(f"Invalid target utilization: {new_target}")
            return
        
        old_target = self.target_utilization
        self.target_utilization = new_target
        
        logger.info(f"Target utilization adjusted: {old_target:.1%} → {new_target:.1%}")


# Global instance
context_optimizer = ContextWindowOptimizer(
    target_utilization=0.75,  # 75% target (research-backed optimal)
    min_utilization=0.50,     # 50% minimum
    max_utilization=0.90      # 90% maximum
)
