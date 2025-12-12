"""
Prometheus Metrics Endpoint for Monitoring
Exposes application metrics for Prometheus scraping
"""

from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
import time

router = APIRouter()

# Create custom registry
registry = CollectorRegistry()

# Metrics
request_count = Counter(
    'oms_chatbot_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status'],
    registry=registry
)

request_duration = Histogram(
    'oms_chatbot_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint'],
    registry=registry
)

query_count = Counter(
    'oms_chatbot_queries_total',
    'Total number of chat queries',
    ['status'],
    registry=registry
)

document_uploads = Counter(
    'oms_chatbot_documents_uploaded_total',
    'Total number of documents uploaded',
    ['status'],
    registry=registry
)

active_connections = Gauge(
    'oms_chatbot_active_connections',
    'Number of active WebSocket connections',
    registry=registry
)

llm_generation_duration = Histogram(
    'oms_chatbot_llm_generation_seconds',
    'LLM generation duration in seconds',
    registry=registry
)

retrieval_duration = Histogram(
    'oms_chatbot_retrieval_duration_seconds',
    'Document retrieval duration in seconds',
    registry=registry
)

cache_hits = Counter(
    'oms_chatbot_cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses = Counter(
    'oms_chatbot_cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=registry
)

error_count = Counter(
    'oms_chatbot_errors_total',
    'Total number of errors',
    ['error_type'],
    registry=registry
)

database_connections = Gauge(
    'oms_chatbot_database_connections',
    'Number of active database connections',
    ['database'],
    registry=registry
)

tokens_generated = Counter(
    'oms_chatbot_tokens_generated_total',
    'Total number of LLM tokens generated',
    registry=registry
)

embedding_generation_duration = Histogram(
    'oms_chatbot_embedding_generation_seconds',
    'Embedding generation duration in seconds',
    registry=registry
)


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


# Helper functions to update metrics
def track_request(endpoint: str, method: str, status: int):
    """Track HTTP request"""
    request_count.labels(endpoint=endpoint, method=method, status=status).inc()


def track_query(status: str):
    """Track chat query (success/error)"""
    query_count.labels(status=status).inc()


def track_document_upload(status: str):
    """Track document upload (success/error)"""
    document_uploads.labels(status=status).inc()


def track_cache_operation(cache_type: str, hit: bool):
    """Track cache hit/miss"""
    if hit:
        cache_hits.labels(cache_type=cache_type).inc()
    else:
        cache_misses.labels(cache_type=cache_type).inc()


def track_error(error_type: str):
    """Track error occurrence"""
    error_count.labels(error_type=error_type).inc()


def set_active_connections(count: int):
    """Set number of active WebSocket connections"""
    active_connections.set(count)


def set_database_connections(database: str, count: int):
    """Set number of active database connections"""
    database_connections.labels(database=database).set(count)


def track_tokens(count: int):
    """Track tokens generated"""
    tokens_generated.inc(count)


# Context manager for tracking durations
class track_duration:
    """Context manager for tracking operation duration"""
    
    def __init__(self, metric: Histogram):
        self.metric = metric
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metric.observe(duration)
