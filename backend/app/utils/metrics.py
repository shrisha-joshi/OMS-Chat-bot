"""
Performance monitoring and metrics collection.
Tracks request latency, document processing, and system health.
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates application metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        
    def record_request(self, endpoint: str, duration: float, status_code: int):
        """Record API request metrics."""
        self.metrics['request_duration'].append(Metric(
            name='request_duration',
            value=duration,
            labels={'endpoint': endpoint, 'status': str(status_code)}
        ))
        self.counters[f'requests_{status_code}'] += 1
        self.counters['total_requests'] += 1
        
    def record_document_processing(self, doc_id: str, duration: float, chunks: int, success: bool):
        """Record document processing metrics."""
        self.metrics['processing_duration'].append(Metric(
            name='processing_duration',
            value=duration,
            labels={'doc_id': doc_id, 'success': str(success)}
        ))
        self.metrics['chunk_count'].append(Metric(
            name='chunk_count',
            value=chunks,
            labels={'doc_id': doc_id}
        ))
        self.counters['documents_processed'] += 1
        if success:
            self.counters['documents_success'] += 1
        else:
            self.counters['documents_failed'] += 1
    
    def record_embedding_generation(self, chunk_count: int, duration: float):
        """Record embedding generation metrics."""
        self.metrics['embedding_duration'].append(Metric(
            name='embedding_duration',
            value=duration,
            labels={'chunks': str(chunk_count)}
        ))
        self.gauges['avg_embedding_time'] = duration / chunk_count if chunk_count > 0 else 0
    
    def record_query(self, query: str, duration: float, chunks_retrieved: int):
        """Record RAG query metrics."""
        self.metrics['query_duration'].append(Metric(
            name='query_duration',
            value=duration,
            labels={'chunks': str(chunks_retrieved)}
        ))
        self.counters['total_queries'] += 1
    
    def get_stats(self, metric_name: str, minutes: int = 5) -> Dict:
        """Get statistics for a metric over time window."""
        if metric_name not in self.metrics:
            return {}
        
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = [
            m for m in self.metrics[metric_name]
            if m.timestamp > cutoff
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'p50': sorted(values)[len(values) // 2] if values else 0,
            'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else 0,
            'p99': sorted(values)[int(len(values) * 0.99)] if len(values) > 1 else 0,
        }
    
    def get_all_counters(self) -> Dict[str, int]:
        """Get all counter values."""
        return dict(self.counters)
    
    def get_all_gauges(self) -> Dict[str, float]:
        """Get all gauge values."""
        return dict(self.gauges)
    
    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        return {
            'counters': self.get_all_counters(),
            'gauges': self.get_all_gauges(),
            'request_stats': self.get_stats('request_duration', minutes=5),
            'processing_stats': self.get_stats('processing_duration', minutes=30),
            'query_stats': self.get_stats('query_duration', minutes=5),
        }


# Global metrics collector
metrics_collector = MetricsCollector()


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation: str, log_threshold: float = 1.0):
        self.operation = operation
        self.log_threshold = log_threshold
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        
        if self.duration > self.log_threshold:
            logger.warning(f"⏱️ SLOW: {self.operation} took {self.duration:.2f}s")
        else:
            logger.debug(f"⏱️ {self.operation} took {self.duration:.2f}s")
        
        return False
