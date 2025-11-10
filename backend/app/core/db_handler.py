"""
Simplified startup configuration for testing without full database stack.
This allows the application to run with minimal dependencies.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DatabaseConnectionHandler:
    """Handles database connections gracefully with fallbacks."""
    
    def __init__(self):
        self.mongodb_available = False
        self.qdrant_available = False
        self.arango_available = False
        self.redis_available = False
        self.neo4j_available = False
        
    async def connect_mongodb(self) -> bool:
        """Try to connect to MongoDB, return success status."""
        try:
            if os.getenv('MONGODB_URI') and not os.getenv('QDRANT_DISABLED'):
                from .db_mongo import mongodb_client
                await mongodb_client.connect()
                self.mongodb_available = True
                logger.info("✅ MongoDB connected")
                return True
        except Exception as e:
            logger.warning(f"⚠️ MongoDB not available: {e}")
        return False
    
    async def connect_qdrant(self) -> bool:
        """Try to connect to Qdrant, return success status."""
        try:
            if os.getenv('QDRANT_URL') and not os.getenv('QDRANT_DISABLED'):
                from .db_qdrant import qdrant_client
                await qdrant_client.connect()
                self.qdrant_available = True
                logger.info("✅ Qdrant connected")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Qdrant not available: {e}")
        return False
    
    async def connect_arango(self) -> bool:
        """Try to connect to ArangoDB, return success status."""
        try:
            if os.getenv('ARANGODB_URL') and not os.getenv('ARANGODB_DISABLED'):
                from .db_arango import arango_client
                await arango_client.connect()
                self.arango_available = True
                logger.info("✅ ArangoDB connected")
                return True
        except Exception as e:
            logger.warning(f"⚠️ ArangoDB not available: {e}")
        return False
    
    async def connect_redis(self) -> bool:
        """Try to connect to Redis, return success status."""
        try:
            if os.getenv('REDIS_URL') and not os.getenv('REDIS_DISABLED'):
                from .cache_redis import redis_client
                await redis_client.connect()
                self.redis_available = True
                logger.info("✅ Redis connected")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
        return False
    
    async def connect_neo4j(self) -> bool:
        """Try to connect to Neo4j, return success status."""
        try:
            if os.getenv('NEO4J_URI') and not os.getenv('NEO4J_DISABLED'):
                from .db_neo4j import neo4j_client
                success = await neo4j_client.connect()
                if success:
                    self.neo4j_available = True
                    logger.info("✅ Neo4j connected and schema initialized")
                    return True
                else:
                    logger.warning("⚠️ Neo4j connection failed")
        except Exception as e:
            logger.warning(f"⚠️ Neo4j not available: {e}")
        return False
    
    async def connect_all(self):
        """Attempt to connect to all databases."""
        logger.info("🔌 Attempting database connections...")
        
        await self.connect_mongodb()
        await self.connect_qdrant()
        await self.connect_arango()
        await self.connect_redis()
        await self.connect_neo4j()  # Add Neo4j connection
        
        # Log summary
        connected = sum([
            self.mongodb_available,
            self.qdrant_available, 
            self.arango_available,
            self.redis_available,
            self.neo4j_available
        ])
        
        logger.info(f"📊 Database Status: {connected}/5 services connected")
        if connected == 0:
            logger.warning("⚠️ No databases connected - running in minimal mode")
        
        return {
            'mongodb': self.mongodb_available,
            'qdrant': self.qdrant_available,
            'arango': self.arango_available,
            'redis': self.redis_available,
            'neo4j': self.neo4j_available,
            'total_connected': connected
        }

# Global database handler
db_handler = DatabaseConnectionHandler()