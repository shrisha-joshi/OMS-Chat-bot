"""
WebSocket endpoint for real-time document processing updates.
Clients connect and receive live progress updates for their documents.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import logging
import asyncio
import json

logger = logging.getLogger(__name__)
router = APIRouter()

# Store active WebSocket connections by document ID
active_connections: Dict[str, Set[WebSocket]] = {}


class ConnectionManager:
    """Manage WebSocket connections for document processing updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, doc_id: str):
        """Accept new WebSocket connection for a document."""
        await websocket.accept()
        if doc_id not in self.active_connections:
            self.active_connections[doc_id] = set()
        self.active_connections[doc_id].add(websocket)
        logger.info(f"📡 WebSocket connected for document {doc_id}")
    
    def disconnect(self, websocket: WebSocket, doc_id: str):
        """Remove WebSocket connection."""
        if doc_id in self.active_connections:
            self.active_connections[doc_id].discard(websocket)
            if not self.active_connections[doc_id]:
                del self.active_connections[doc_id]
        logger.info(f"📡 WebSocket disconnected for document {doc_id}")
    
    async def send_update(self, doc_id: str, message: dict):
        """Send update to all connections watching this document."""
        if doc_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[doc_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")
                    disconnected.add(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.active_connections[doc_id].discard(conn)
    
    async def broadcast_progress(self, doc_id: str, stage: str, status: str, 
                                 progress: int, message: str = ""):
        """
        Broadcast processing progress update.
        
        Args:
            doc_id: Document ID
            stage: Current stage (e.g., "EXTRACT", "CHUNK", "EMBED")
            status: Status (e.g., "PROCESSING", "SUCCESS", "FAILED")
            progress: Progress percentage (0-100)
            message: Optional message
        """
        update = {
            "type": "progress",
            "doc_id": doc_id,
            "stage": stage,
            "status": status,
            "progress": progress,
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.send_update(doc_id, update)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/document/{doc_id}")
async def websocket_document_updates(websocket: WebSocket, doc_id: str):
    """
    WebSocket endpoint for real-time document processing updates.
    
    Client receives JSON messages with this format:
    {
        "type": "progress",
        "doc_id": "...",
        "stage": "EXTRACT|CHUNK|EMBED|INDEX",
        "status": "PROCESSING|SUCCESS|FAILED",
        "progress": 0-100,
        "message": "..."
    }
    """
    await manager.connect(websocket, doc_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "doc_id": doc_id,
            "message": "Connected to document processing updates"
        })
        
        # Keep connection alive and listen for client messages
        while True:
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_json({"type": "pong", "data": data})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, doc_id)
        logger.info(f"Client disconnected from document {doc_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, doc_id)


async def notify_progress(doc_id: str, stage: str, status: str, 
                         progress: int, message: str = ""):
    """
    Convenience function to notify progress from anywhere in the codebase.
    
    Usage:
        from app.api.websocket import notify_progress
        await notify_progress(doc_id, "EXTRACT", "SUCCESS", 25, "Text extracted")
    """
    await manager.broadcast_progress(doc_id, stage, status, progress, message)
