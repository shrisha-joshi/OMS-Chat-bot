"""
Real-time monitoring and progress tracking API.
Provides WebSocket endpoints for real-time updates on document ingestion and chat processing.
"""

import asyncio
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from datetime import datetime
import json

from ..core.db_mongo import get_mongodb_client, MongoDBClient

# Import enhanced services
from ..services.websocket_manager import get_websocket_manager, EnhancedWebSocketManager
from ..services.health_monitor import get_health_monitor, ConnectionHealthMonitor
from ..services.batch_processor import get_batch_processor, BatchProcessor
from ..utils.resilience import get_all_circuit_breaker_metrics

logger = logging.getLogger(__name__)
router = APIRouter()

class ConnectionManager:
    """WebSocket connection manager for broadcasting updates."""
    def __init__(self):
        self.active_connections: dict = {}  # session_id -> [WebSocket]
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        logger.info(f"Client connected to session {session_id}")
    
    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"Client disconnected from session {session_id}")
    
    async def broadcast(self, session_id: str, message: dict):
        """Broadcast message to all clients in session."""
        if session_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                self.active_connections[session_id].remove(conn)


manager = ConnectionManager()


def _create_progress_message(doc_id: str, log: Dict[str, Any]) -> Dict[str, Any]:
    """Create progress message from log entry."""
    return {
        "type": "ingestion_progress",
        "doc_id": doc_id,
        "step": log.get("step"),
        "status": log.get("status"),
        "message": log.get("message"),
        "metadata": log.get("metadata", {}),
        "timestamp": log.get("timestamp").isoformat() if log.get("timestamp") else None
    }


def _create_completion_message(doc_id: str, status: str) -> Dict[str, Any]:
    """Create completion message."""
    return {
        "type": "ingestion_complete",
        "doc_id": doc_id,
        "status": status
    }


async def _send_new_logs(websocket: WebSocket, doc_id: str, new_logs: List[Dict], last_timestamp):
    """Send new log entries to websocket."""
    for log in new_logs:
        message = _create_progress_message(doc_id, log)
        await websocket.send_json(message)
        last_timestamp = log.get("timestamp")
    return last_timestamp


async def _check_completion_status(websocket: WebSocket, doc_id: str, document: Dict) -> bool:
    """Check if ingestion is complete and send completion message. Returns True if complete."""
    if not document:
        return False
    
    status = document.get("ingest_status")
    if status in ("SUCCESS", "FAILED"):
        await websocket.send_json(_create_completion_message(doc_id, status))
        return True
    return False


@router.websocket("/ws/ingestion/{doc_id}")
async def websocket_ingestion_progress(
    websocket: WebSocket,
    doc_id: str,
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    WebSocket endpoint for real-time document ingestion progress.
    Client connects and receives updates as document is processed.
    """
    await websocket.accept()
    
    try:
        last_timestamp = None
        
        while True:
            # Poll for new ingestion logs
            logs = await mongo_client.get_ingestion_logs(doc_id)
            new_logs = [log for log in logs if not last_timestamp or log.get("timestamp") > last_timestamp]
            
            # Send new logs
            last_timestamp = await _send_new_logs(websocket, doc_id, new_logs, last_timestamp)
            
            # Check completion status
            document = await mongo_client.get_document(doc_id)
            if await _check_completion_status(websocket, doc_id, document):
                break
            
            # Wait before polling again
            await asyncio.sleep(0.5)
    
    except WebSocketDisconnect:
        logger.info("Client disconnected from ingestion monitoring")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@router.get("/documents/{doc_id}/ingestion-status")
async def get_ingestion_status(
    doc_id: str,
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """Get current ingestion status and logs for a document."""
    try:
        document = await mongo_client.get_document(doc_id)
        logs = await mongo_client.get_ingestion_logs(doc_id)
        
        if not document:
            return {"error": "Document not found"}
        
        return {
            "doc_id": doc_id,
            "filename": document.get("filename"),
            "ingest_status": document.get("ingest_status"),
            "size": document.get("size"),
            "uploaded_at": document.get("uploaded_at").isoformat() if document.get("uploaded_at") else None,
            "logs": logs,
            "stages": {
                "extract": next((l for l in logs if l["step"] == "EXTRACT"), None),
                "chunk": next((l for l in logs if l["step"] == "CHUNK"), None),
                "embed": next((l for l in logs if l["step"] == "EMBED"), None),
                "store_chunks": next((l for l in logs if l["step"] == "STORE_CHUNKS"), None),
                "index_vectors": next((l for l in logs if l["step"] == "INDEX_VECTORS"), None),
                "extract_entities": next((l for l in logs if l["step"] == "EXTRACT_ENTITIES"), None),
            }
        }
    except Exception as e:
        logger.error(f"Failed to get ingestion status: {e}")
        return {"error": str(e)}


# ========== Enhanced Monitoring Endpoints ==========

@router.get("/websocket/metrics")
async def get_websocket_metrics(
    ws_manager: EnhancedWebSocketManager = Depends(get_websocket_manager)
):
    """Get WebSocket connection metrics."""
    try:
        return {
            "overall": ws_manager.get_metrics(),
            "connections": ws_manager.get_all_connection_metrics()
        }
    except Exception as e:
        logger.error(f"Failed to get WebSocket metrics: {e}")
        return {"error": str(e)}


@router.get("/health/services")
async def get_services_health(
    health_monitor: ConnectionHealthMonitor = Depends(get_health_monitor)
):
    """Get health status of all registered services."""
    try:
        return health_monitor.get_overall_health()
    except Exception as e:
        logger.error(f"Failed to get service health: {e}")
        return {"error": str(e)}


@router.get("/health/service/{service_name}")
async def get_service_health(
    service_name: str,
    health_monitor: ConnectionHealthMonitor = Depends(get_health_monitor)
):
    """Get health status for specific service."""
    try:
        health = health_monitor.get_service_health(service_name)
        if not health:
            return {"error": f"Service '{service_name}' not registered"}
        return health
    except Exception as e:
        logger.error(f"Failed to get health for {service_name}: {e}")
        return {"error": str(e)}


@router.get("/circuit-breakers")
async def get_circuit_breaker_status():
    """Get status of all circuit breakers."""
    try:
        return {
            "circuit_breakers": get_all_circuit_breaker_metrics(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get circuit breaker status: {e}")
        return {"error": str(e)}


@router.get("/batch-processor/metrics")
async def get_batch_processor_metrics(
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """Get batch processing metrics."""
    try:
        return batch_processor.get_metrics()
    except Exception as e:
        logger.error(f"Failed to get batch processor metrics: {e}")
        return {"error": str(e)}


@router.get("/system/comprehensive")
async def get_comprehensive_system_status(
    ws_manager: EnhancedWebSocketManager = Depends(get_websocket_manager),
    health_monitor: ConnectionHealthMonitor = Depends(get_health_monitor),
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """Get comprehensive system status including all enhanced services."""
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "websocket": {
                "metrics": ws_manager.get_metrics(),
                "active_connections": ws_manager.get_connection_count()
            },
            "health": health_monitor.get_overall_health(),
            "circuit_breakers": get_all_circuit_breaker_metrics(),
            "batch_processor": batch_processor.get_metrics()
        }
    except Exception as e:
        logger.error(f"Failed to get comprehensive status: {e}")
        return {"error": str(e)}
