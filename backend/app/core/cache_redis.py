"""
Redis client for caching and real-time messaging operations.
This module provides caching functionality and pub/sub capabilities
for real-time updates in the admin dashboard.
"""

import redis.asyncio as redis
import json
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta

from ..config import settings

logger = logging.getLogger(__name__)

class RedisClient:
    """Redis client for caching and pub/sub operations."""
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.pubsub = None
        self.available = False  # Flag to indicate if service is available
    
    def is_connected(self) -> bool:
        """Check if Redis is currently connected."""
        return self.available and self.client is not None
    
    async def connect(self):
        """Establish connection to Redis."""
        try:
            if not settings.redis_url:
                logger.warning("Redis URL not configured, running without Redis cache")
                self.available = False
                return False
                
            logger.info("Attempting to connect to Redis Cloud...")
            self.client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            # The redis client exposes an async ping; await it to avoid runtime warnings
            await self.client.ping()
            logger.info("✅ Successfully connected to Redis Cloud!")
            self.available = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            # Clean up failed connection
            self.client = None
            self.available = False
            return False
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Redis")
    
    async def set_cache(self, key: str, value: Any, expiry_seconds: int = 3600) -> bool:
        """
        Set a cached value with expiration.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            expiry_seconds: Expiration time in seconds
        
        Returns:
            bool: Success status
        """
        if not self.client:
            logger.debug("Redis not available, skipping cache set")
            return False
            
        try:
            serialized_value = json.dumps(value, default=str)
            await self.client.setex(key, expiry_seconds, serialized_value)
            logger.debug(f"Cached value for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to set cache for key {key}: {e}")
            return False
    
    async def get_cache(self, key: str) -> Any:
        """
        Get a cached value.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        if not self.client:
            logger.debug("Redis not available, skipping cache get")
            return None
            
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get cache for key {key}: {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete a cached value."""
        if not self.client:
            logger.debug("Redis not available, skipping cache delete")
            return False
            
        try:
            result = await self.client.delete(key)
            logger.debug(f"Deleted cache key: {key}")
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete cache for key {key}: {e}")
            return False
    
    async def cache_exists(self, key: str) -> bool:
        """Check if a cache key exists."""
        try:
            result = await self.client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to check cache existence for key {key}: {e}")
            return False
    
    async def set_hash(self, hash_key: str, field: str, value: Any) -> bool:
        """Set a field in a Redis hash."""
        try:
            serialized_value = json.dumps(value, default=str)
            await self.client.hset(hash_key, field, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Failed to set hash {hash_key}.{field}: {e}")
            return False
    
    async def get_hash(self, hash_key: str, field: str) -> Any:
        """Get a field from a Redis hash."""
        try:
            value = await self.client.hget(hash_key, field)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get hash {hash_key}.{field}: {e}")
            return None
    
    async def get_all_hash(self, hash_key: str) -> Dict[str, Any]:
        """Get all fields from a Redis hash."""
        try:
            hash_data = await self.client.hgetall(hash_key)
            result = {}
            for field, value in hash_data.items():
                try:
                    result[field] = json.loads(value)
                except json.JSONDecodeError:
                    result[field] = value
            return result
        except Exception as e:
            logger.error(f"Failed to get all hash {hash_key}: {e}")
            return {}
    
    async def publish_message(self, channel: str, message: Dict[str, Any]) -> bool:
        """
        Publish a message to a Redis channel.
        
        Args:
            channel: Channel name
            message: Message data (will be JSON serialized)
        
        Returns:
            bool: Success status
        """
        try:
            serialized_message = json.dumps(message, default=str)
            await self.client.publish(channel, serialized_message)
            logger.debug(f"Published message to channel {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            return False
    
    async def subscribe_to_channel(self, channel: str):
        """
        Subscribe to a Redis channel for real-time updates.
        
        Args:
            channel: Channel name
        
        Returns:
            AsyncIterator of messages
        """
        try:
            self.pubsub = self.client.pubsub()
            await self.pubsub.subscribe(channel)
            
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        yield data
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode message: {message['data']}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to subscribe to channel {channel}: {e}")
    
    async def unsubscribe_from_channel(self, channel: str = None):
        """Unsubscribe from Redis channels."""
        try:
            if self.pubsub:
                if channel:
                    await self.pubsub.unsubscribe(channel)
                else:
                    await self.pubsub.unsubscribe()
                await self.pubsub.close()
                self.pubsub = None
                logger.debug(f"Unsubscribed from channel: {channel or 'all'}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")
    
    # Specialized caching methods for the RAG system
    
    async def cache_embedding(self, text_hash: str, embedding: List[float], 
                            expiry_hours: int = 24) -> bool:
        """Cache an embedding vector."""
        key = f"embedding:{text_hash}"
        return await self.set_cache(key, embedding, expiry_hours * 3600)
    
    async def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get a cached embedding vector."""
        key = f"embedding:{text_hash}"
        return await self.get_cache(key)
    
    async def cache_query_result(self, query_hash: str, result: Dict[str, Any], 
                               expiry_minutes: int = 30) -> bool:
        """Cache a query result."""
        key = f"query_result:{query_hash}"
        return await self.set_cache(key, result, expiry_minutes * 60)
    
    async def get_cached_query_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get a cached query result."""
        key = f"query_result:{query_hash}"
        return await self.get_cache(key)
    
    async def cache_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Cache document metadata."""
        key = f"doc_metadata:{doc_id}"
        return await self.set_cache(key, metadata, 86400)  # 24 hours
    
    async def get_cached_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document metadata."""
        key = f"doc_metadata:{doc_id}"
        return await self.get_cache(key)
    
    async def store_session_data(self, session_id: str, data: Dict[str, Any], 
                               expiry_hours: int = 8) -> bool:
        """Store session-specific data."""
        key = f"session:{session_id}"
        success = await self.set_cache(key, data, expiry_hours * 3600)
        if success:
            logger.info(f"✅ Cached session data: {session_id} (TTL: {expiry_hours}h)")
        return success
    
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session-specific data."""
        key = f"session:{session_id}"
        data = await self.get_cache(key)
        if data:
            logger.info(f"✅ Retrieved cached session data: {session_id}")
        return data
    
    async def set_session_data(self, session_id: str, data: Dict[str, Any], 
                              expiry_minutes: int = 1440) -> bool:
        """Alias for store_session_data with different naming."""
        return await self.store_session_data(session_id, data, expiry_minutes // 60)
    
    async def increment_counter(self, key: str, expiry_seconds: int = 86400) -> int:
        """Increment a counter and return the new value."""
        try:
            # Use a pipeline to ensure atomicity
            pipe = self.client.pipeline()
            pipe.incr(key)
            pipe.expire(key, expiry_seconds)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.error(f"Failed to increment counter {key}: {e}")
            return 0
    
    async def get_counter(self, key: str) -> int:
        """Get the current value of a counter."""
        try:
            value = await self.client.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Failed to get counter {key}: {e}")
            return 0
    
    # Real-time admin updates
    
    async def publish_ingestion_update(self, doc_id: str, status: str, 
                                     message: str = "", metadata: Dict = None) -> bool:
        """Publish ingestion status update to admin channel."""
        update = {
            "type": "ingestion_update",
            "doc_id": doc_id,
            "status": status,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        return await self.publish_message("admin_updates", update)
    
    async def publish_system_status(self, component: str, status: str, 
                                  details: Dict = None) -> bool:
        """Publish system status update."""
        update = {
            "type": "system_status",
            "component": component,
            "status": status,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        return await self.publish_message("admin_updates", update)
    
    async def publish_chat_metrics(self, session_id: str, query_time: float, 
                                 tokens_generated: int, sources_count: int) -> bool:
        """Publish chat performance metrics."""
        metrics = {
            "type": "chat_metrics",
            "session_id": session_id,
            "query_time": query_time,
            "tokens_generated": tokens_generated,
            "sources_count": sources_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        return await self.publish_message("admin_updates", metrics)
    
    # Utility methods
    
    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        try:
            info = await self.client.info()
            return {
                "version": info.get("redis_version"),
                "memory_used": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses")
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        try:
            keys = await self.client.keys(pattern)
            if keys:
                return await self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear pattern {pattern}: {e}")
            return 0


    async def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self.client is not None

    async def get_json(self, key: str) -> Any:
        """Get a JSON value from cache (alias for get_cache)."""
        return await self.get_cache(key)
    
    async def set_json(self, key: str, value: Any, expiry_seconds: int = 3600, expire_seconds: int = None) -> bool:
        """Set a JSON value in cache (alias for set_cache)."""
        # Support both parameter names for backward compatibility
        expiry = expire_seconds if expire_seconds is not None else expiry_seconds
        return await self.set_cache(key, value, expiry)

# Global Redis client instance
redis_client = RedisClient()


async def get_redis_client() -> RedisClient:
    """Dependency injection for Redis client."""
    return redis_client