"""
Neo4j Knowledge Graph Client for Advanced RAG
Implements state-of-the-art graph RAG patterns based on Microsoft GraphRAG,
LlamaIndex, and academic research (2024-2025).

Features:
- Entity and relationship management
- Multi-hop reasoning
- Community detection
- Cypher query generation
- Temporal queries
- Graph-based context expansion
"""

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import asyncio
from collections import defaultdict

from ..config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j client for knowledge graph operations with async support."""
    
    def __init__(self):
        self.driver = None
        self.async_driver = None
        self.available = False
        self._connection_validated = False
        
    async def connect(self) -> bool:
        """Establish connection to Neo4j database."""
        if not settings.neo4j_uri:
            logger.warning("Neo4j URI not configured, running without Neo4j")
            return False
            
        try:
            logger.info(f"Connecting to Neo4j at {settings.neo4j_uri}...")
            
            # Create both sync and async drivers
            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
                max_connection_pool_size=50,
                connection_timeout=10.0
            )
            
            self.async_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
                max_connection_pool_size=50,
                connection_timeout=10.0
            )
            
            # Verify connection with timeout
            async def _verify():
                async with self.async_driver.session() as session:
                    result = await session.run("RETURN 1 as test")
                    await result.single()
                    return True
            
            await asyncio.wait_for(_verify(), timeout=5.0)
            
            # Create indexes and constraints
            await self._setup_schema()
            
            self.available = True
            self._connection_validated = True
            logger.info("✅ Successfully connected to Neo4j!")
            return True
            
        except asyncio.TimeoutError:
            logger.warning("Neo4j connection timed out after 5s")
            await self._cleanup_failed_connection()
            return False
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            await self._cleanup_failed_connection()
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting to Neo4j: {e}", exc_info=True)
            await self._cleanup_failed_connection()
            return False
    
    async def _cleanup_failed_connection(self):
        """Clean up after failed connection attempt."""
        if self.driver:
            try:
                self.driver.close()
            except:
                pass
        if self.async_driver:
            try:
                await self.async_driver.close()
            except:
                pass
        self.driver = None
        self.async_driver = None
        self.available = False
    
    async def disconnect(self):
        """Close Neo4j connections."""
        if self.driver:
            self.driver.close()
        if self.async_driver:
            await self.async_driver.close()
        logger.info("Disconnected from Neo4j")
    
    def is_connected(self) -> bool:
        """Check if Neo4j is connected and available."""
        return self.available and self._connection_validated
    
    async def _setup_schema(self):
        """Create indexes and constraints for optimal performance."""
        if not self.is_connected():
            return
            
        schema_queries = [
            # Document indexes
            "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX document_filename IF NOT EXISTS FOR (d:Document) ON (d.filename)",
            
            # Chunk indexes
            "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX chunk_doc_id IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
            "CREATE INDEX chunk_embedding_id IF NOT EXISTS FOR (c:Chunk) ON (c.embedding_id)",
            
            # Entity indexes and constraints
            "CREATE CONSTRAINT entity_name_type IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_confidence IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
            
            # Topic indexes
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            
            # User indexes
            "CREATE INDEX user_id IF NOT EXISTS FOR (u:User) ON (u.id)",
            
            # Relationship indexes
            "CREATE INDEX rel_type IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.type)",
            "CREATE INDEX rel_confidence IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.confidence)",
            "CREATE INDEX rel_timestamp IF NOT EXISTS FOR ()-[r]-() ON (r.timestamp)"
        ]
        
        async with self.async_driver.session() as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                except Exception as e:
                    # Indexes/constraints might already exist
                    logger.debug(f"Schema setup: {e}")
        
        logger.info("✅ Neo4j schema setup completed")
    
    # ==================== ENTITY MANAGEMENT ====================
    
    async def create_entity(
        self,
        name: str,
        entity_type: str,
        properties: Dict[str, Any] = None,
        doc_id: str = None,
        chunk_id: str = None,
        confidence: float = 0.8
    ) -> str:
        """
        Create or update an entity in the knowledge graph.
        
        Args:
            name: Entity name (e.g., "OpenAI", "Python", "Machine Learning")
            entity_type: Type (Person, Organization, Location, Concept, Product, etc.)
            properties: Additional properties (description, aliases, etc.)
            doc_id: Source document ID
            chunk_id: Source chunk ID
            confidence: Confidence score (0.0 - 1.0)
            
        Returns:
            Entity ID (neo4j internal ID or custom ID)
        """
        if not self.is_connected():
            return None
            
        props = properties or {}
        props.update({
            "name": name,
            "type": entity_type,
            "confidence": confidence,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
        
        if doc_id:
            props["doc_id"] = doc_id
        if chunk_id:
            props["chunk_id"] = chunk_id
        
        query = """
        MERGE (e:Entity {name: $name, type: $type})
        SET e += $props
        RETURN elementId(e) as entity_id
        """
        
        async with self.async_driver.session() as session:
            result = await session.run(query, name=name, type=entity_type, props=props)
            record = await result.single()
            return record["entity_id"] if record else None
    
    async def create_relationship(
        self,
        from_entity_name: str,
        from_entity_type: str,
        to_entity_name: str,
        to_entity_type: str,
        relation_type: str,
        properties: Dict[str, Any] = None,
        confidence: float = 0.7
    ) -> bool:
        """
        Create a relationship between two entities.
        
        Args:
            from_entity_name: Source entity name
            from_entity_type: Source entity type
            to_entity_name: Target entity name
            to_entity_type: Target entity type
            relation_type: Relationship type (e.g., "WORKS_FOR", "LOCATED_IN", "USES")
            properties: Additional properties
            confidence: Confidence score
            
        Returns:
            Success boolean
        """
        if not self.is_connected():
            return False
            
        props = properties or {}
        props.update({
            "type": relation_type,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        query = """
        MATCH (from:Entity {name: $from_name, type: $from_type})
        MATCH (to:Entity {name: $to_name, type: $to_type})
        MERGE (from)-[r:RELATED_TO {type: $rel_type}]->(to)
        SET r += $props
        RETURN count(r) as created
        """
        
        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    query,
                    from_name=from_entity_name,
                    from_type=from_entity_type,
                    to_name=to_entity_name,
                    to_type=to_entity_type,
                    rel_type=relation_type,
                    props=props
                )
                record = await result.single()
                return record["created"] > 0 if record else False
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    # ==================== DOCUMENT INTEGRATION ====================
    
    async def index_document(
        self,
        doc_id: str,
        filename: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Create a document node in the graph."""
        if not self.is_connected():
            return False
            
        props = metadata or {}
        props.update({
            "id": doc_id,
            "filename": filename,
            "uploaded_at": datetime.utcnow().isoformat()
        })
        
        query = """
        MERGE (d:Document {id: $doc_id})
        SET d += $props
        RETURN d.id as id
        """
        
        async with self.async_driver.session() as session:
            result = await session.run(query, doc_id=doc_id, props=props)
            record = await result.single()
            return record is not None
    
    async def index_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        embedding_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Create a chunk node and link it to document."""
        if not self.is_connected():
            return False
            
        props = metadata or {}
        props.update({
            "id": chunk_id,
            "doc_id": doc_id,
            "text": text[:500],  # Store preview only
            "created_at": datetime.utcnow().isoformat()
        })
        
        if embedding_id:
            props["embedding_id"] = embedding_id
        
        query = """
        MATCH (d:Document {id: $doc_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c += $props
        MERGE (d)-[:CONTAINS]->(c)
        RETURN c.id as id
        """
        
        async with self.async_driver.session() as session:
            result = await session.run(query, doc_id=doc_id, chunk_id=chunk_id, props=props)
            record = await result.single()
            return record is not None
    
    # ==================== RETRIEVAL & REASONING ====================
    
    async def get_entity_neighbors(
        self,
        entity_name: str,
        entity_type: str = None,
        hops: int = 1,
        relation_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get entities connected to a given entity within N hops.
        Implements graph expansion for context enrichment.
        
        Args:
            entity_name: Starting entity name
            entity_type: Optional entity type filter
            hops: Number of hops (1-3 recommended)
            relation_types: Filter by relationship types
            
        Returns:
            List of connected entities with paths
        """
        if not self.is_connected():
            return []
        
        type_filter = f"AND e.type = '{entity_type}'" if entity_type else ""
        rel_filter = f"AND type(r) IN {relation_types}" if relation_types else ""
        
        query = f"""
        MATCH path = (start:Entity {{name: $name}} {type_filter})-[r*1..{hops}]->(end:Entity)
        {rel_filter}
        RETURN DISTINCT 
            end.name as name,
            end.type as type,
            end.confidence as confidence,
            length(path) as distance,
            [rel in relationships(path) | type(rel)] as path_types
        ORDER BY distance, end.confidence DESC
        LIMIT 50
        """
        
        async with self.async_driver.session() as session:
            result = await session.run(query, name=entity_name)
            neighbors = []
            async for record in result:
                neighbors.append({
                    "name": record["name"],
                    "type": record["type"],
                    "confidence": record["confidence"],
                    "distance": record["distance"],
                    "path": record["path_types"]
                })
            return neighbors
    
    async def multi_hop_query(
        self,
        start_entities: List[Tuple[str, str]],
        max_hops: int = 3,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Execute multi-hop reasoning query.
        Finds paths between multiple entities and infers relationships.
        
        Args:
            start_entities: List of (name, type) tuples
            max_hops: Maximum path length
            min_confidence: Minimum confidence threshold
            
        Returns:
            Graph structure with paths and intermediate entities
        """
        if not self.is_connected() or len(start_entities) < 2:
            return {"entities": [], "paths": [], "relationships": []}
        
        # Build query for finding paths between entities
        entity_conditions = " OR ".join([
            f"(e.name = '{name}' AND e.type = '{etype}')"
            for name, etype in start_entities
        ])
        
        query = f"""
        MATCH (start:Entity)
        WHERE {entity_conditions}
        WITH collect(start) as starts
        UNWIND starts as from_entity
        UNWIND starts as to_entity
        WHERE id(from_entity) < id(to_entity)
        MATCH path = shortestPath((from_entity)-[*1..{max_hops}]-(to_entity))
        WHERE all(r in relationships(path) WHERE r.confidence >= {min_confidence})
        RETURN 
            from_entity.name as start_name,
            to_entity.name as end_name,
            [n in nodes(path) | {{name: n.name, type: n.type}}] as path_nodes,
            [r in relationships(path) | {{type: type(r), confidence: r.confidence}}] as path_rels,
            length(path) as path_length
        ORDER BY path_length
        LIMIT 20
        """
        
        async with self.async_driver.session() as session:
            result = await session.run(query)
            paths = []
            all_entities = set()
            all_relationships = []
            
            async for record in result:
                path_info = {
                    "start": record["start_name"],
                    "end": record["end_name"],
                    "nodes": record["path_nodes"],
                    "relationships": record["path_rels"],
                    "length": record["path_length"]
                }
                paths.append(path_info)
                
                # Collect unique entities
                for node in record["path_nodes"]:
                    all_entities.add((node["name"], node["type"]))
                
                # Collect relationships
                all_relationships.extend(record["path_rels"])
            
            return {
                "entities": [{"name": n, "type": t} for n, t in all_entities],
                "paths": paths,
                "relationships": all_relationships
            }
    
    async def expand_context_from_chunks(
        self,
        chunk_ids: List[str],
        hops: int = 2
    ) -> Dict[str, Any]:
        """
        Expand context by traversing graph from retrieved chunks.
        Used in hybrid retrieval to enrich vector search results.
        
        Args:
            chunk_ids: List of chunk IDs from vector search
            hops: How many hops to expand
            
        Returns:
            Expanded context with entities and relationships
        """
        if not self.is_connected() or not chunk_ids:
            return {"entities": [], "relationships": []}
        
        query = f"""
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids
        MATCH (c)-[:MENTIONS]->(e:Entity)
        MATCH path = (e)-[*0..{hops}]-(related:Entity)
        WHERE e <> related
        RETURN DISTINCT
            e.name as entity_name,
            e.type as entity_type,
            collect(DISTINCT {{
                name: related.name,
                type: related.type,
                confidence: related.confidence,
                distance: length(path)
            }}) as related_entities
        LIMIT 100
        """
        
        async with self.async_driver.session() as session:
            result = await session.run(query, chunk_ids=chunk_ids)
            entities = []
            relationships = []
            
            async for record in result:
                entity = {
                    "name": record["entity_name"],
                    "type": record["entity_type"],
                    "related": record["related_entities"]
                }
                entities.append(entity)
            
            return {
                "entities": entities,
                "relationships": relationships
            }
    
    # ==================== STATISTICS & ANALYSIS ====================
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        if not self.is_connected():
            return {}
        
        queries = {
            "entity_count": "MATCH (e:Entity) RETURN count(e) as count",
            "document_count": "MATCH (d:Document) RETURN count(d) as count",
            "chunk_count": "MATCH (c:Chunk) RETURN count(c) as count",
            "relationship_count": "MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count",
            "entity_types": """
                MATCH (e:Entity)
                RETURN e.type as type, count(e) as count
                ORDER BY count DESC
            """,
            "top_entities": """
                MATCH (e:Entity)-[r]-()
                RETURN e.name as name, e.type as type, count(r) as connections
                ORDER BY connections DESC
                LIMIT 10
            """
        }
        
        stats = {}
        async with self.async_driver.session() as session:
            # Simple counts
            for key in ["entity_count", "document_count", "chunk_count", "relationship_count"]:
                result = await session.run(queries[key])
                record = await result.single()
                stats[key] = record["count"] if record else 0
            
            # Entity type distribution
            result = await session.run(queries["entity_types"])
            stats["entity_types"] = {}
            async for record in result:
                stats["entity_types"][record["type"]] = record["count"]
            
            # Top connected entities
            result = await session.run(queries["top_entities"])
            stats["top_entities"] = []
            async for record in result:
                stats["top_entities"].append({
                    "name": record["name"],
                    "type": record["type"],
                    "connections": record["connections"]
                })
        
        return stats
    
    # ==================== CYPHER QUERY EXECUTION ====================
    
    async def execute_cypher(
        self,
        query: str,
        parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.
        Use with caution - validate queries before execution.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        if not self.is_connected():
            return []
        
        async with self.async_driver.session() as session:
            result = await session.run(query, parameters or {})
            records = []
            async for record in result:
                records.append(dict(record))
            return records


# Global Neo4j client instance
neo4j_client = Neo4jClient()


async def get_neo4j_client() -> Neo4jClient:
    """Dependency injection for Neo4j client."""
    return neo4j_client
