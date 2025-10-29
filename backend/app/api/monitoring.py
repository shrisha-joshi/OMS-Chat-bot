"""
Real-time monitoring and progress tracking API.
Provides WebSocket endpoints for real-time updates on document ingestion and chat processing.
"""

import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from datetime import datetime
import json

from ..core.db_mongo import get_mongodb_client, MongoDBClient

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
            # Poll for new ingestion logs every 0.5 seconds
            query = {"doc_id": doc_id}
            if last_timestamp:
                query["timestamp"] = {"$gt": last_timestamp}
            
            logs = await mongo_client.get_ingestion_logs(doc_id)
            
            # Find new logs since last check
            new_logs = [log for log in logs if not last_timestamp or log.get("timestamp") > last_timestamp]
            
            for log in new_logs:
                message = {
                    "type": "ingestion_progress",
                    "doc_id": doc_id,
                    "step": log.get("step"),
                    "status": log.get("status"),
                    "message": log.get("message"),
                    "metadata": log.get("metadata", {}),
                    "timestamp": log.get("timestamp").isoformat() if log.get("timestamp") else None
                }
                
                await websocket.send_json(message)
                last_timestamp = log.get("timestamp")
            
            # Check if processing is complete
            document = await mongo_client.get_document(doc_id)
            if document and document.get("ingest_status") == "SUCCESS":
                await websocket.send_json({
                    "type": "ingestion_complete",
                    "doc_id": doc_id,
                    "status": "SUCCESS"
                })
                break
            elif document and document.get("ingest_status") == "FAILED":
                await websocket.send_json({
                    "type": "ingestion_complete",
                    "doc_id": doc_id,
                    "status": "FAILED"
                })
                break
            
            # Wait before polling again
            await asyncio.sleep(0.5)
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from ingestion monitoring")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
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
