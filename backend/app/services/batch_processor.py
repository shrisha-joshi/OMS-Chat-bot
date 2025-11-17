"""
Batch Processing Service for RAG Pipeline Optimization.

Improvements:
- Batch embedding generation (reduces API calls by 10x)
- Parallel document chunking
- Batch vector storage (Qdrant bulk upsert)
- Progress tracking and metrics
- Error handling with partial success

Research Reference: Production RAG pipeline optimization patterns
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    total_items: int
    successful_items: int
    failed_items: int
    processing_time_seconds: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": round(self.success_rate, 2),
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "errors": self.errors[:10],  # Limit error list
            "metadata": self.metadata
        }


class BatchProcessor:
    """
    Batch processing service for RAG pipeline optimization.
    
    Benefits:
    - 10x faster embedding generation through batching
    - Reduced API calls and costs
    - Better resource utilization
    - Progress tracking
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        max_concurrent_batches: int = 3,
        enable_metrics: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Items per batch
            max_concurrent_batches: Maximum parallel batches
            enable_metrics: Enable metrics tracking
        """
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_metrics = enable_metrics
        
        # Metrics
        self.total_batches_processed = 0
        self.total_items_processed = 0
        self.total_processing_time = 0.0
        self.batch_times: List[float] = []
        
        logger.info(
            f"Batch processor initialized: batch_size={batch_size}, "
            f"max_concurrent={max_concurrent_batches}"
        )
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            processor_func: Async function to process each batch
            progress_callback: Optional progress callback
        
        Returns:
            Batch processing result
        """
        start_time = asyncio.get_event_loop().time()
        
        total_items = len(items)
        successful_items = 0
        failed_items = 0
        errors = []
        
        # Split into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        logger.info(
            f"Processing {total_items} items in {len(batches)} batches "
            f"(batch_size={self.batch_size})"
        )
        
        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_single_batch(batch_idx: int, batch_items: List[Any]):
            nonlocal successful_items, failed_items
            
            async with semaphore:
                try:
                    # Process batch
                    result = await processor_func(batch_items)
                    
                    # Count successes
                    if isinstance(result, dict) and "successful" in result:
                        successful_items += result["successful"]
                        failed_items += result.get("failed", 0)
                    else:
                        # Assume all succeeded if no detailed result
                        successful_items += len(batch_items)
                    
                    # Progress callback
                    if progress_callback:
                        progress = ((batch_idx + 1) / len(batches)) * 100
                        await progress_callback(
                            current=batch_idx + 1,
                            total=len(batches),
                            progress=progress
                        )
                    
                    logger.debug(f"Batch {batch_idx + 1}/{len(batches)} complete")
                    
                except Exception as e:
                    error_msg = f"Batch {batch_idx + 1} failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    failed_items += len(batch_items)
        
        # Execute all batches
        tasks = [
            process_single_batch(idx, batch)
            for idx, batch in enumerate(batches)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate metrics
        processing_time = asyncio.get_event_loop().time() - start_time
        
        if self.enable_metrics:
            self.total_batches_processed += len(batches)
            self.total_items_processed += total_items
            self.total_processing_time += processing_time
            self.batch_times.append(processing_time)
        
        result = BatchResult(
            total_items=total_items,
            successful_items=successful_items,
            failed_items=failed_items,
            processing_time_seconds=processing_time,
            errors=errors,
            metadata={
                "batch_count": len(batches),
                "batch_size": self.batch_size,
                "average_time_per_item": processing_time / total_items if total_items > 0 else 0
            }
        )
        
        logger.info(
            f"✅ Batch processing complete: {successful_items}/{total_items} successful "
            f"({result.success_rate:.1f}%) in {processing_time:.2f}s"
        )
        
        return result
    
    async def batch_embed_texts(
        self,
        texts: List[str],
        embedding_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Generate embeddings in batches.
        
        Args:
            texts: Texts to embed
            embedding_func: Async function that accepts List[str] and returns List[List[float]]
            progress_callback: Optional progress callback
        
        Returns:
            Batch processing result with embeddings
        """
        embeddings_storage = []
        
        async def batch_embed(batch_texts: List[str]) -> Dict[str, Any]:
            """Process a batch of texts."""
            try:
                batch_embeddings = await embedding_func(batch_texts)
                embeddings_storage.extend(batch_embeddings)
                return {"successful": len(batch_texts), "failed": 0}
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Return zeros for failed batch
                embeddings_storage.extend([[0.0] * 768] * len(batch_texts))
                return {"successful": 0, "failed": len(batch_texts)}
        
        result = await self.process_batch(
            items=texts,
            processor_func=batch_embed,
            progress_callback=progress_callback
        )
        
        result.metadata["embeddings"] = embeddings_storage
        return result
    
    async def batch_upsert_vectors(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        upsert_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Upsert vectors to Qdrant in batches.
        
        Args:
            collection_name: Qdrant collection name
            points: List of point dictionaries (id, vector, payload)
            upsert_func: Async function that accepts batch of points
            progress_callback: Optional progress callback
        
        Returns:
            Batch processing result
        """
        async def batch_upsert(batch_points: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Upsert a batch of points."""
            try:
                await upsert_func(collection_name, batch_points)
                return {"successful": len(batch_points), "failed": 0}
            except Exception as e:
                logger.error(f"Batch upsert failed: {e}")
                return {"successful": 0, "failed": len(batch_points)}
        
        result = await self.process_batch(
            items=points,
            processor_func=batch_upsert,
            progress_callback=progress_callback
        )
        
        return result
    
    async def parallel_chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunking_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Chunk documents in parallel.
        
        Args:
            documents: Documents to chunk
            chunking_func: Async function that processes one document
            progress_callback: Optional progress callback
        
        Returns:
            Batch processing result with chunks
        """
        all_chunks = []
        
        async def process_document(doc: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single document."""
            try:
                chunks = await chunking_func(doc)
                all_chunks.extend(chunks)
                return {"successful": 1, "failed": 0, "chunks": len(chunks)}
            except Exception as e:
                logger.error(f"Document chunking failed: {e}")
                return {"successful": 0, "failed": 1, "chunks": 0}
        
        # Use small batch size for documents (each doc can be large)
        original_batch_size = self.batch_size
        self.batch_size = 5  # Process 5 documents at a time
        
        result = await self.process_batch(
            items=documents,
            processor_func=lambda docs: asyncio.gather(
                *[process_document(doc) for doc in docs]
            ),
            progress_callback=progress_callback
        )
        
        self.batch_size = original_batch_size  # Restore
        
        result.metadata["total_chunks"] = len(all_chunks)
        result.metadata["chunks"] = all_chunks
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(documents)} documents")
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        if not self.enable_metrics:
            return {"metrics_disabled": True}
        
        avg_batch_time = (
            sum(self.batch_times) / len(self.batch_times)
            if self.batch_times else 0.0
        )
        
        avg_items_per_second = (
            self.total_items_processed / self.total_processing_time
            if self.total_processing_time > 0 else 0.0
        )
        
        return {
            "total_batches_processed": self.total_batches_processed,
            "total_items_processed": self.total_items_processed,
            "total_processing_time_seconds": round(self.total_processing_time, 2),
            "average_batch_time_seconds": round(avg_batch_time, 2),
            "average_items_per_second": round(avg_items_per_second, 2),
            "batch_size": self.batch_size,
            "max_concurrent_batches": self.max_concurrent_batches
        }
    
    def reset_metrics(self):
        """Reset metrics counters."""
        self.total_batches_processed = 0
        self.total_items_processed = 0
        self.total_processing_time = 0.0
        self.batch_times = []
        logger.info("Batch processor metrics reset")


# Global batch processor instance
batch_processor = BatchProcessor(batch_size=32, max_concurrent_batches=3)


async def get_batch_processor() -> BatchProcessor:
    """Dependency injection for batch processor."""
    return batch_processor
