"""
Chat API routes for conversational interactions with the RAG system.
This module provides endpoints for chat queries, streaming responses,
and WebSocket-based real-time communication.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import json
import asyncio
import time
import hashlib
from datetime import datetime, timezone

from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..core.db_qdrant import get_qdrant_client, QdrantDBClient
from ..core.cache_redis import get_redis_client, RedisClient
from ..services.chat_service import ChatService
from ..config import settings
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Simple test endpoint to verify LMStudio connectivity
@router.post("/test-llm")
async def test_llm_direct(request: dict):
    """
    Direct LLM test endpoint - bypasses entire RAG pipeline.
    Use this to verify LMStudio is working without database/embedding dependencies.
    
    Test with: POST /chat/test-llm {"query": "Say hello"}
    """
    try:
        logger.info("=" * 80)
        logger.info("🧪 DIRECT LLM TEST (bypassing RAG pipeline)")
        logger.info("=" * 80)
        logger.info(f"Query: {request.get('query', 'NO QUERY')}")
        
        from ..services.llm_handler import LLMHandler
        
        llm = LLMHandler()
        logger.info("Initializing LLM handler...")
        await llm.initialize()
        logger.info("✅ LLM handler initialized")
        
        logger.info("Generating response...")
        response = await llm.generate_response(
            prompt=request.get("query", "Say hello"),
            max_tokens=100,
            temperature=0.7
        )
        
        logger.info(f"✅ Response received: {len(response)} characters")
        logger.info("=" * 80)
        
        return {
            "response": response,
            "status": "success",
            "message": "Direct LLM test successful - LMStudio is working!"
        }
    except Exception as e:
        logger.error(f"❌ Direct LLM test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "response": "",
            "status": "error",
            "message": str(e)
        }

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    sources: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[Dict[str, Any]]] = None

@router.post("/test-rag-debug")
async def test_rag_debug(query: str, service: ChatService = Depends(lambda: ChatService)):
    """
    DEBUG endpoint - Shows REAL RAG error without suppression.
    Use this to diagnose what's failing in the RAG pipeline.
    """
    logger.info("=== RAG DEBUG TEST (NO ERROR SUPPRESSION) ===")
    logger.info(f"Query: {query}")
    
    # Initialize service
    if not hasattr(service, 'embedding_model') or service.embedding_model is None:
        logger.info("Initializing chat service...")
        await service.initialize()
    
    # Call process_query WITHOUT try/catch so we see the real error
    result = await service.process_query(
        query=query,
        session_id="debug-session",
        context=[]
    )
    
    return result

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[List[ChatMessage]] = None
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    attachments: List[Dict[str, Any]]
    session_id: str
    processing_time: float
    tokens_generated: int
    # Phase 2: Media enrichment and validation
    media_suggestions: Optional[List[Dict[str, Any]]] = None
    validation_details: Optional[Dict[str, Any]] = None
    # Phase 3: Advanced RAG metrics
    phase3_metrics: Optional[Dict[str, Any]] = None

class StreamingChatResponse(BaseModel):
    type: str  # "token", "sources", "complete"
    content: str
    sources: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None

# Initialize chat service
chat_service = None

async def get_chat_service() -> ChatService:
    """Initialize and return chat service."""
    global chat_service
    if chat_service is None:
        chat_service = ChatService()
        await chat_service.initialize()
    return chat_service

@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    redis_client: RedisClient = Depends(get_redis_client),
    mongo_client: MongoDBClient = Depends(get_mongodb_client)
):
    """
    Process a chat query and return a complete response.
    
    Args:
        request: Chat request with query and context
        service: Chat service instance
        redis_client: Redis client for caching
        mongo_client: MongoDB client for storing messages
    
    Returns:
        Complete chat response with sources and attachments
    """
    try:
        start_time = time.time()
        
        # DEBUG: Log incoming request
        logger.info("=" * 80)
        logger.info("📥 INCOMING CHAT REQUEST")
        logger.info("=" * 80)
        logger.info(f"Query: {request.query[:200]}")
        logger.info(f"Session ID: {request.session_id}")
        logger.info(f"Stream: {request.stream}")
        logger.info(f"Context messages: {len(request.context) if request.context else 0}")
        logger.info("=" * 80)
        
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = _generate_session_id()
            logger.info(f"Generated new session ID: {request.session_id}")
        
        # Check cache for recent identical queries
        query_hash = hashlib.md5(f"{request.query}:{request.session_id}".encode()).hexdigest()
        cached_response = await redis_client.get_cached_query_result(query_hash)
        
        if cached_response and not request.stream:
            logger.info(f"✅ Returning cached response for query: {request.query[:50]}...")
            return ChatResponse(**cached_response)
        
        # Process the query
        logger.info(f"🔄 Processing chat query: {request.query[:100]}...")
        
        result = await service.process_query(
            query=request.query,
            session_id=request.session_id,
            context=request.context or []
        )
        
        processing_time = time.time() - start_time
        
        # Create response
        response = ChatResponse(
            response=result["response"],
            sources=result["sources"],
            attachments=result["attachments"],
            media_suggestions=result.get("media_suggestions", []),  # Phase 2
            validation_details=result.get("validation_details"),  # Phase 2
            session_id=request.session_id,
            processing_time=processing_time,
            tokens_generated=result.get("tokens_generated", 0),
            phase3_metrics=result.get("phase3_metrics")
        )
        
        # Save messages to MongoDB for persistence
        try:
            if mongo_client.is_connected():
                # Get or create session
                session_doc = await mongo_client.database.chat_sessions.find_one(
                    {"session_id": request.session_id}
                )
                
                now = datetime.now(timezone.utc).isoformat()
                
                if session_doc:
                    # Append messages to existing session
                    await mongo_client.database.chat_sessions.update_one(
                        {"session_id": request.session_id},
                        {
                            "$push": {
                                "messages": {
                                    "$each": [
                                        {
                                            "role": "user",
                                            "content": request.query,
                                            "timestamp": now
                                        },
                                        {
                                            "role": "assistant",
                                            "content": response.response,
                                            "timestamp": now,
                                            "sources": response.sources,
                                            "tokens_generated": response.tokens_generated
                                        }
                                    ]
                                }
                            },
                            "$set": {"updated_at": now}
                        }
                    )
                else:
                    # Create new session
                    await mongo_client.database.chat_sessions.insert_one({
                        "session_id": request.session_id,
                        "messages": [
                            {
                                "role": "user",
                                "content": request.query,
                                "timestamp": now
                            },
                            {
                                "role": "assistant",
                                "content": response.response,
                                "timestamp": now,
                                "sources": response.sources,
                                "tokens_generated": response.tokens_generated
                            }
                        ],
                        "created_at": now,
                        "updated_at": now
                    })
                
                logger.info(f"✅ Messages saved to MongoDB for session: {request.session_id}")
        except Exception as e:
            logger.warning(f"Failed to save messages to MongoDB: {e}")
            # Don't fail the request - just log the warning
        
        # Cache the response
        await redis_client.cache_query_result(
            query_hash, 
            response.dict(), 
            expiry_minutes=30
        )
        
        # Also cache session data in Redis for faster retrieval
        try:
            if redis_client.is_connected():
                await redis_client.set_session_data(
                    request.session_id,
                    {"messages": [
                        {
                            "role": "user",
                            "content": request.query,
                            "timestamp": response.response  # Will be properly timestamped in next version
                        },
                        {
                            "role": "assistant",
                            "content": response.response,
                            "sources": response.sources,
                            "tokens_generated": response.tokens_generated
                        }
                    ]},
                    expiry_minutes=1440  # 24 hours
                )
        except Exception as e:
            logger.warning(f"Failed to cache session in Redis: {e}")
        
        # Publish metrics
        await redis_client.publish_chat_metrics(
            request.session_id,
            processing_time,
            result.get("tokens_generated", 0),
            len(result["sources"])
        )
        
        logger.info("=" * 80)
        logger.info("📤 SENDING RESPONSE TO FRONTEND")
        logger.info("=" * 80)
        logger.info(f"Response length: {len(response.response)} characters")
        logger.info(f"Sources: {len(response.sources)}")
        logger.info(f"Attachments: {len(response.attachments)}")
        logger.info(f"Tokens generated: {response.tokens_generated}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Response preview: {response.response[:200]}...")
        logger.info("=" * 80)
        
        return response
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ CHAT QUERY FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.error("=" * 80)
        raise HTTPException(status_code=500, detail=f"Failed to process chat query: {str(e)}")

@router.websocket("/ws")
    # WebSocket complexity acceptable for real-time chat handling
    # pylint: disable=too-many-branches
async def websocket_chat(websocket: WebSocket):  # noqa: python:S3776
    """
    WebSocket endpoint for real-time streaming chat.
    Supports token-by-token streaming and real-time responses.
    """
    await websocket.accept()
    session_id = _generate_session_id()
    
    try:
        logger.info(f"Chat WebSocket client connected: {session_id}")
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to RAG chatbot"
        })
        
        service = await get_chat_service()
        _ = await get_redis_client()
        
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()
                
                if data.get("type") == "chat":
                    query = data.get("query", "")
                    if not query.strip():
                        await websocket.send_json({
                            "type": "error",
                            "message": "Empty query received"
                        })
                        continue
                    
                    logger.info(f"WebSocket query: {query[:100]}...")
                    start_time = time.time()
                    
                    # Send typing indicator
                    await websocket.send_json({
                        "type": "typing",
                        "session_id": session_id
                    })
                    
                    try:
                        # Stream the response
                        async for chunk in service.stream_query(query, session_id):
                            await websocket.send_json({
                                "type": chunk["type"],
                                "content": chunk.get("content", ""),
                                "sources": chunk.get("sources"),
                                "attachments": chunk.get("attachments"),
                                "session_id": session_id
                            })
                        
                        # Send completion message
                        processing_time = time.time() - start_time
                        await websocket.send_json({
                            "type": "complete",
                            "session_id": session_id,
                            "processing_time": processing_time
                        })
                        
                    except Exception as e:
                        logger.error(f"Query processing failed: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to process query",
                            "session_id": session_id
                        })
                
                elif data.get("type") == "ping":
                    # Respond to ping for connection keepalive
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Message handling failed"
                })
    
    except WebSocketDisconnect:
        logger.info(f"Chat WebSocket client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

@router.get("/sessions/{session_id}/history")
async def get_chat_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
    redis_client: RedisClient = Depends(get_redis_client)
):
    """
    Get chat history for a session.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of messages to return
        redis_client: Redis client
    
    Returns:
        List of chat messages
    """
    try:
        # Get session data from Redis
        session_data = await redis_client.get_session_data(session_id)
        
        if not session_data:
            return {
                "session_id": session_id,
                "messages": [],
                "message": "No history found for this session"
            }
        
        messages = session_data.get("messages", [])
        
        # Return most recent messages
        recent_messages = messages[-limit:] if len(messages) > limit else messages
        
        return {
            "session_id": session_id,
            "messages": recent_messages,
            "total_messages": len(messages)
        }
        
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@router.delete("/sessions/{session_id}")
async def clear_chat_session(
    session_id: str,
    redis_client: RedisClient = Depends(get_redis_client)
):
    """Clear chat session data."""
    try:
        success = await redis_client.delete_cache(f"session:{session_id}")
        
        return {
            "success": success,
            "session_id": session_id,
            "message": "Session cleared successfully" if success else "Session not found"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session")

@router.get("/suggestions")
async def get_query_suggestions(
    query: str = Query(..., min_length=2),
    limit: int = Query(5, ge=1, le=20),
    service: ChatService = Depends(get_chat_service)
):
    """
    Get query suggestions based on input.
    
    Args:
        query: Partial query for suggestions
        limit: Maximum number of suggestions
        service: Chat service instance
    
    Returns:
        List of query suggestions
    """
    try:
        suggestions = await service.get_query_suggestions(query, limit)
        
        return {
            "query": query,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get query suggestions")

@router.post("/sessions/{session_id}/context")
async def update_session_context(
    session_id: str,
    context: List[ChatMessage],
    redis_client: RedisClient = Depends(get_redis_client)
):
    """Update session context with conversation history."""
    try:
        # Get existing session data
        session_data = await redis_client.get_session_data(session_id) or {}
        
        # Update context
        session_data["messages"] = [msg.dict() for msg in context]
        session_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Store updated session data
        await redis_client.store_session_data(session_id, session_data)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Context updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to update session context: {e}")
        raise HTTPException(status_code=500, detail="Failed to update session context")

@router.get("/health")
async def chat_health_check():
    """Health check for chat service."""
    try:
        service = await get_chat_service()
        
        # Test basic functionality
        test_result = await service.health_check()
        
        return {
            "status": "healthy" if test_result else "degraded",
            "service": "chat",
            "embedding_model": settings.embedding_model_name,
            "llm_endpoint": settings.lmstudio_api_url,
            "features": {
                "vector_search": True,
                "graph_search": settings.use_graph_search,
                "reranking": settings.use_reranker,
                "streaming": True
            }
        }
        
    except Exception as e:
        logger.error(f"Chat health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "chat",
            "error": str(e)
        }

# Utility functions

def _generate_session_id() -> str:
    """Generate a unique session ID."""
    import uuid
    return str(uuid.uuid4())

def _extract_attachments(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract multimedia attachments from sources."""
    attachments = []
    
    for source in sources:
        filename = source.get("filename", "")
        
        # Check for different media types
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gi', '.webp')):
            attachments.append({
                "type": "image",
                "url": f"/api/files/{source.get('doc_id')}/view",
                "filename": filename,
                "title": filename
            })
        elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv')):
            attachments.append({
                "type": "video",
                "url": f"/api/files/{source.get('doc_id')}/view",
                "filename": filename,
                "title": filename
            })
        elif filename.lower().endswith('.pdf'):
            attachments.append({
                "type": "pd",
                "url": f"/api/files/{source.get('doc_id')}/view",
                "filename": filename,
                "title": filename
            })
        elif "youtube.com" in source.get("text", "") or "youtu.be" in source.get("text", ""):
            # Extract YouTube video ID
            import re
            youtube_regex = r'(?:youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)([a-zA-Z0-9_-]{11})'
            match = re.search(youtube_regex, source.get("text", ""))
            if match:
                video_id = match.group(1)
                attachments.append({
                    "type": "youtube",
                    "videoId": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": f"YouTube Video: {video_id}"
                })
    
    return attachments


@router.get("/history/{session_id}")
async def get_session_history(
    session_id: str,
    mongo_client: MongoDBClient = Depends(get_mongodb_client),
    redis_client: RedisClient = Depends(get_redis_client)
) -> Dict[str, Any]:
    """
    Retrieve chat history for a session.
    
    **Path Parameters:**
    - session_id: Unique session identifier
    
    **Returns:**
    - session_id: The session identifier
    - messages: List of messages in the conversation
    - source: Where data came from (cache, database, or new)
    - total: Total number of messages
    """
    try:
        logger.info(f"Fetching history for session: {session_id}")
        
        # Try Redis cache first (fastest)
        if redis_client.is_connected():
            try:
                session_data = await redis_client.get_session_data(session_id)
                if session_data and "messages" in session_data:
                    logger.info(f"✅ Session history found in Redis: {len(session_data['messages'])} messages")
                    return {
                        "session_id": session_id,
                        "messages": session_data["messages"],
                        "source": "cache",
                        "total": len(session_data["messages"])
                    }
            except Exception as e:
                logger.warning(f"Redis lookup failed, trying MongoDB: {e}")
        
        # Fall back to MongoDB (persistent storage)
        if mongo_client.is_connected():
            try:
                history_doc = await mongo_client.database.chat_sessions.find_one(
                    {"session_id": session_id}
                )
                
                if history_doc:
                    messages = history_doc.get("messages", [])
                    logger.info(f"✅ Session history found in MongoDB: {len(messages)} messages")
                    return {
                        "session_id": session_id,
                        "messages": messages,
                        "source": "database",
                        "total": len(messages)
                    }
            except Exception as e:
                logger.warning(f"MongoDB lookup failed: {e}")
        
        # No history found - return empty
        logger.info(f"No history found for session: {session_id}")
        return {
            "session_id": session_id,
            "messages": [],
            "source": "new",
            "total": 0
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve session history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@router.get("/sessions/list")
async def list_sessions(
    mongo_client: MongoDBClient = Depends(get_mongodb_client),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=50)
) -> Dict[str, Any]:
    """
    List all chat sessions.
    
    **Query Parameters:**
    - skip: Number of sessions to skip (pagination)
    - limit: Maximum sessions to return (1-50)
    
    **Returns:**
    - sessions: List of session summaries
    - total: Total number of sessions
    """
    try:
        if not mongo_client.is_connected():
            raise HTTPException(status_code=503, detail="MongoDB not connected")
        
        # Get total sessions
        total = await mongo_client.database.chat_sessions.count_documents({})
        
        # Fetch sessions
        cursor = mongo_client.database.chat_sessions.find({})\
            .sort("created_at", -1)\
            .skip(skip)\
            .limit(limit)
        
        sessions = await cursor.to_list(length=limit)
        
        formatted_sessions = [
            {
                "session_id": session.get("session_id", ""),
                "created_at": session.get("created_at", ""),
                "message_count": len(session.get("messages", [])),
                "last_message_at": session.get("last_message_at", "")
            }
            for session in sessions
        ]
        
        logger.info(f"Listed {len(formatted_sessions)} sessions (total: {total})")
        
        return {
            "sessions": formatted_sessions,
            "total": total,
            "page": (skip // limit) + 1 if limit > 0 else 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")