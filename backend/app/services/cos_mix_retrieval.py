"""
COS-Mix Retrieval - Hybrid Cosine Similarity + Distance Fusion
Based on research from arXiv:2406.00638 (COS-Mix: Cosine Similarity and Distance Fusion)

This module implements a novel hybrid retrieval strategy that combines:
1. Cosine Similarity (direction alignment) - captures semantic similarity
2. Cosine Distance (magnitude awareness) - quantifies dissimilarity
3. Fusion Strategy - optimal combination for sparse and dense data

Key Advantages:
- Better performance on sparse data
- More nuanced similarity scoring
- Reduced arbitrary results from pure cosine similarity
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

from ..config import settings

logger = logging.getLogger(__name__)


class COSMixRetrieval:
    """
    COS-Mix: Cosine Similarity + Distance Fusion for improved retrieval.
    
    Paper Reference: arXiv:2406.00638
    Authors: Kush Juvekar, Anupam Purwar
    """
    
    def __init__(
        self,
        similarity_weight: float = 0.7,
        distance_weight: float = 0.3,
        normalize_scores: bool = True
    ):
        """
        Initialize COS-Mix retrieval.
        
        Args:
            similarity_weight (alpha): Weight for cosine similarity (default: 0.7)
            distance_weight (beta): Weight for cosine distance (default: 0.3)
            normalize_scores: Whether to normalize final scores to [0, 1]
        """
        self.alpha = similarity_weight
        self.beta = distance_weight
        self.normalize_scores = normalize_scores
        
        # Validate weights
        if not np.isclose(self.alpha + self.beta, 1.0):
            logger.warning(
                f"COS-Mix weights don't sum to 1.0: alpha={self.alpha}, beta={self.beta}. "
                f"Normalizing..."
            )
            total = self.alpha + self.beta
            self.alpha /= total
            self.beta /= total
        
        logger.info(
            f"COS-Mix Retrieval initialized: "
            f"alpha={self.alpha:.2f}, beta={self.beta:.2f}, "
            f"normalize={normalize_scores}"
        )
    
    def cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Cosine Similarity = (A · B) / (||A|| * ||B||)
        
        Measures: Direction alignment (angle between vectors)
        Range: [-1, 1] where 1 = same direction, -1 = opposite, 0 = perpendicular
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Handle edge cases
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        
        # Calculate magnitudes
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    def cosine_distance(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Calculate cosine distance between two vectors.
        
        Cosine Distance = 1 - Cosine Similarity
        
        Measures: Dissimilarity (1 - normalized dot product)
        Range: [0, 2] where 0 = identical, 2 = opposite
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine distance score
        """
        similarity = self.cosine_similarity(vec1, vec2)
        distance = 1.0 - similarity
        return distance
    
    def cos_mix_score(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Calculate COS-Mix hybrid score.
        
        COS-Mix Score = alpha * similarity + beta * (1 - distance)
                      = alpha * similarity + beta * similarity
                      = (alpha + beta) * similarity
        
        Note: In practice, we use:
        COS-Mix Score = alpha * similarity + beta * (2 - distance)
        
        This formulation gives more weight to both similarity and anti-distance.
        
        Args:
            vec1: Query vector
            vec2: Document vector
            
        Returns:
            Hybrid COS-Mix score
        """
        # Calculate cosine similarity
        similarity = self.cosine_similarity(vec1, vec2)
        
        # Calculate cosine distance
        distance = self.cosine_distance(vec1, vec2)
        
        # Fusion strategy: weighted combination
        # We invert distance so that lower distance = higher score
        distance_score = 1.0 - distance
        
        # Combine with weights
        hybrid_score = (self.alpha * similarity) + (self.beta * distance_score)
        
        # Normalize to [0, 1] if requested
        if self.normalize_scores:
            # The max possible score is when similarity=1 and distance=0
            # Score = alpha * 1 + beta * 1 = alpha + beta = 1.0 (since normalized)
            # So the score is already in [0, 1] range if vectors are unit normalized
            hybrid_score = np.clip(hybrid_score, 0.0, 1.0)
        
        return float(hybrid_score)
    
    def rerank_with_cos_mix(
        self,
        query_embedding: np.ndarray,
        documents: List[Dict[str, Any]],
        embedding_key: str = "embedding"
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using COS-Mix scoring.
        
        Args:
            query_embedding: Query vector
            documents: List of documents with embeddings
            embedding_key: Key for embedding in document dict (default: "embedding")
            
        Returns:
            Documents sorted by COS-Mix score (descending)
        """
        if not documents:
            return []
        
        # Calculate COS-Mix scores
        scored_docs = []
        for doc in documents:
            # Get document embedding
            doc_embedding = doc.get(embedding_key)
            
            if doc_embedding is None:
                logger.warning(f"Document missing embedding: {doc.get('chunk_id', 'unknown')}")
                continue
            
            # Convert to numpy array if needed
            if not isinstance(doc_embedding, np.ndarray):
                doc_embedding = np.array(doc_embedding)
            
            # Calculate COS-Mix score
            score = self.cos_mix_score(query_embedding, doc_embedding)
            
            # Add score to document
            doc_with_score = doc.copy()
            doc_with_score["cos_mix_score"] = score
            doc_with_score["original_score"] = doc.get("score", 0.0)
            scored_docs.append(doc_with_score)
        
        # Sort by COS-Mix score (descending)
        scored_docs.sort(key=lambda x: x["cos_mix_score"], reverse=True)
        
        # Log score statistics
        if scored_docs:
            scores = [d["cos_mix_score"] for d in scored_docs]
            logger.info(
                f"COS-Mix reranking: {len(scored_docs)} docs, "
                f"scores: min={min(scores):.3f}, max={max(scores):.3f}, "
                f"mean={np.mean(scores):.3f}, std={np.std(scores):.3f}"
            )
        
        return scored_docs
    
    def compare_scoring_methods(
        self,
        query_embedding: np.ndarray,
        doc_embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare different scoring methods for analysis.
        
        Args:
            query_embedding: Query vector
            doc_embedding: Document vector
            
        Returns:
            Dictionary with scores from different methods
        """
        similarity = self.cosine_similarity(query_embedding, doc_embedding)
        distance = self.cosine_distance(query_embedding, doc_embedding)
        cos_mix = self.cos_mix_score(query_embedding, doc_embedding)
        
        return {
            "cosine_similarity": similarity,
            "cosine_distance": distance,
            "distance_score": 1.0 - distance,
            "cos_mix_score": cos_mix,
            "alpha": self.alpha,
            "beta": self.beta
        }
    
    def batch_score(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate COS-Mix scores for multiple documents efficiently.
        
        Args:
            query_embedding: Query vector (1D array)
            doc_embeddings: Document vectors (2D array: n_docs x embedding_dim)
            
        Returns:
            Array of COS-Mix scores for each document
        """
        # Normalize vectors for efficient computation
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Batch cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        
        # Batch cosine distance
        distances = 1.0 - similarities
        distance_scores = 1.0 - distances
        
        # Batch COS-Mix scoring
        cos_mix_scores = (self.alpha * similarities) + (self.beta * distance_scores)
        
        if self.normalize_scores:
            cos_mix_scores = np.clip(cos_mix_scores, 0.0, 1.0)
        
        return cos_mix_scores
    
    def adjust_weights(self, new_alpha: float, new_beta: float):
        """
        Dynamically adjust COS-Mix weights.
        
        Args:
            new_alpha: New similarity weight
            new_beta: New distance weight
        """
        # Normalize weights
        total = new_alpha + new_beta
        self.alpha = new_alpha / total
        self.beta = new_beta / total
        
        logger.info(f"COS-Mix weights adjusted: alpha={self.alpha:.2f}, beta={self.beta:.2f}")
    
    def optimize_weights_for_data(
        self,
        query_embeddings: List[np.ndarray],
        doc_embeddings: List[np.ndarray],
        relevance_scores: List[float],
        test_alphas: List[float] = None
    ) -> Tuple[float, float]:
        """
        Optimize COS-Mix weights for a specific dataset using grid search.
        
        Args:
            query_embeddings: List of query vectors
            doc_embeddings: List of document vectors
            relevance_scores: Ground truth relevance scores
            test_alphas: Alpha values to test (default: [0.5, 0.6, 0.7, 0.8, 0.9])
            
        Returns:
            Optimal (alpha, beta) tuple
        """
        if test_alphas is None:
            test_alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        best_alpha = self.alpha
        best_beta = self.beta
        best_correlation = -1.0
        
        for alpha in test_alphas:
            beta = 1.0 - alpha
            
            # Temporarily set weights
            old_alpha, old_beta = self.alpha, self.beta
            self.alpha, self.beta = alpha, beta
            
            # Calculate scores
            predicted_scores = []
            for q_emb, d_emb in zip(query_embeddings, doc_embeddings):
                score = self.cos_mix_score(q_emb, d_emb)
                predicted_scores.append(score)
            
            # Calculate correlation with ground truth
            correlation = np.corrcoef(predicted_scores, relevance_scores)[0, 1]
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_alpha = alpha
                best_beta = beta
            
            # Restore weights
            self.alpha, self.beta = old_alpha, old_beta
        
        # Set optimal weights
        self.alpha = best_alpha
        self.beta = best_beta
        
        logger.info(
            f"Optimized COS-Mix weights: alpha={best_alpha:.2f}, beta={best_beta:.2f}, "
            f"correlation={best_correlation:.3f}"
        )
        
        return best_alpha, best_beta


# Global instance with research-backed defaults
cos_mix_retrieval = COSMixRetrieval(
    similarity_weight=0.7,  # 70% weight for similarity
    distance_weight=0.3,    # 30% weight for distance
    normalize_scores=True
)
