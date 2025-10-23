"""
ArangoDB client for knowledge graph operations.
This module manages graph relationships, entities, and provides graph-based
context enhancement for the RAG system.
"""

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.graph import Graph
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import hashlib
import asyncio

from ..config import settings

logger = logging.getLogger(__name__)

class ArangoDBClient:
    """ArangoDB client for knowledge graph operations."""
    
    def __init__(self):
        self.client: Optional[ArangoClient] = None
        self.database: Optional[StandardDatabase] = None
        self.graph: Optional[Graph] = None
        self.graph_name = "knowledge_graph"
        self.entities_collection = "entities"
        self.relations_collection = "relations"
    
    async def connect(self):
        """Establish connection to ArangoDB and setup graph structure.

        This method runs the blocking arango-python driver calls inside threads
        and enforces a short overall timeout so backend startup doesn't hang
        if ArangoDB is unreachable.
        """
        if not settings.arangodb_url:
            logger.warning("ArangoDB URL not configured, running without ArangoDB")
            return False

        timeout_seconds = getattr(settings, "arangodb_connect_timeout", 5.0)
        logger.info(f"Attempting to connect to ArangoDB at {settings.arangodb_url} with {timeout_seconds}s timeout...")

        async def _do_connect():
            # Initialize client (fast) and run blocking network/auth calls in a thread
            self.client = ArangoClient(hosts=settings.arangodb_url)

            # Connect to system database first to create our database
            sys_db = await asyncio.to_thread(
                lambda: self.client.db(
                    "_system",
                    username=settings.arangodb_user,
                    password=settings.arangodb_password
                )
            )

            # Test connection (blocking) -> offload to thread
            try:
                await asyncio.to_thread(sys_db.properties)
                logger.info("✅ Successfully connected to ArangoDB!")
            except Exception as auth_error:
                logger.error(f"❌ ArangoDB authentication failed: {auth_error}")
                self.client = None
                return False

            # Create database if it doesn't exist (blocking calls -> thread)
            def _ensure_database(sys_db_local):
                if not sys_db_local.has_database(settings.arangodb_db):
                    sys_db_local.create_database(
                        name=settings.arangodb_db,
                        users=[{
                            "username": settings.arangodb_user,
                            "password": settings.arangodb_password,
                            "active": True
                        }]
                    )
                    logger.info(f"Created ArangoDB database: {settings.arangodb_db}")

                return self.client.db(
                    settings.arangodb_db,
                    username=settings.arangodb_user,
                    password=settings.arangodb_password
                )

            # Ensure database and get database object in thread
            self.database = await asyncio.to_thread(_ensure_database, sys_db)

            # Setup graph structure (blocking operations) in a thread
            await asyncio.to_thread(self._setup_graph)

            logger.info(f"ArangoDB graph setup completed successfully")
            return True

        try:
            return await asyncio.wait_for(_do_connect(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(f"ArangoDB connection timed out after {timeout_seconds}s - skipping ArangoDB for now")
            # Clean up partial state
            self.client = None
            self.database = None
            self.graph = None
            return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to ArangoDB: {e}")
            # Clean up failed connection
            self.client = None
            self.database = None
            self.graph = None
            return False
    
    async def disconnect(self):
        """Close ArangoDB connection."""
        # ArangoDB client doesn't need explicit disconnection
        logger.info("Disconnected from ArangoDB")
    
    def _setup_graph(self):
        """Setup graph collections and relationships.

        This method is synchronous and intended to be executed inside a
        thread via `asyncio.to_thread` because the underlying arango driver
        is blocking.
        """
        try:
            # Create vertex collection for entities
            if not self.database.has_collection(self.entities_collection):
                entities_col = self.database.create_collection(
                    self.entities_collection,
                    vertex=True
                )
                logger.info(f"Created entities collection: {self.entities_collection}")

            # Create edge collection for relations
            if not self.database.has_collection(self.relations_collection):
                relations_col = self.database.create_collection(
                    self.relations_collection,
                    edge=True
                )
                logger.info(f"Created relations collection: {self.relations_collection}")

            # Create or get graph
            if not self.database.has_graph(self.graph_name):
                self.graph = self.database.create_graph(
                    self.graph_name,
                    edge_definitions=[{
                        "edge_collection": self.relations_collection,
                        "from_vertex_collections": [self.entities_collection],
                        "to_vertex_collections": [self.entities_collection]
                    }]
                )
                logger.info(f"Created knowledge graph: {self.graph_name}")
            else:
                self.graph = self.database.graph(self.graph_name)

            # Create indexes for better performance
            self._create_indexes()

        except Exception as e:
            logger.error(f"Failed to setup graph: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for optimal graph performance.

        This is synchronous and runs inside the same thread as _setup_graph.
        """
        try:
            entities_col = self.database.collection(self.entities_collection)
            relations_col = self.database.collection(self.relations_collection)

            # Entity indexes
            entities_col.add_hash_index(fields=["name"], unique=False)
            entities_col.add_hash_index(fields=["type"], unique=False)
            entities_col.add_hash_index(fields=["doc_id"], unique=False)

            # Relation indexes
            relations_col.add_hash_index(fields=["relation_type"], unique=False)
            relations_col.add_hash_index(fields=["confidence"], unique=False)

            logger.info("Created ArangoDB indexes")

        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    def _generate_entity_key(self, name: str, entity_type: str) -> str:
        """Generate a consistent key for entities."""
        content = f"{entity_type}:{name}".lower()
        return hashlib.md5(content.encode()).hexdigest()
    
    async def create_entity(self, name: str, entity_type: str, doc_id: str = None, 
                          chunk_id: str = None, metadata: Dict = None) -> str:
        """
        Create or update an entity in the knowledge graph.
        
        Args:
            name: Entity name
            entity_type: Type of entity (PERSON, ORG, CONCEPT, etc.)
            doc_id: Source document ID
            chunk_id: Source chunk ID
            metadata: Additional entity metadata
        
        Returns:
            str: Entity key
        """
        try:
            entity_key = self._generate_entity_key(name, entity_type)
            entities_col = self.database.collection(self.entities_collection)
            
            entity_data = {
                "_key": entity_key,
                "name": name.strip(),
                "type": entity_type.upper(),
                "doc_ids": [doc_id] if doc_id else [],
                "chunk_ids": [chunk_id] if chunk_id else [],
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "mention_count": 1
            }
            
            # Try to insert, if exists then update
            try:
                entities_col.insert(entity_data)
                logger.debug(f"Created new entity: {name} ({entity_type})")
            except Exception:
                # Entity exists, update it
                existing = entities_col.get(entity_key)
                if existing:
                    # Merge doc_ids and chunk_ids
                    if doc_id and doc_id not in existing.get("doc_ids", []):
                        existing["doc_ids"].append(doc_id)
                    if chunk_id and chunk_id not in existing.get("chunk_ids", []):
                        existing["chunk_ids"].append(chunk_id)
                    
                    existing["mention_count"] = existing.get("mention_count", 1) + 1
                    existing["updated_at"] = datetime.utcnow().isoformat()
                    
                    entities_col.update(entity_key, existing)
                    logger.debug(f"Updated existing entity: {name} ({entity_type})")
            
            return entity_key
            
        except Exception as e:
            logger.error(f"Failed to create entity {name}: {e}")
            return ""
    
    async def create_relation(self, from_entity_key: str, to_entity_key: str, 
                            relation_type: str, confidence: float = 0.5,
                            doc_id: str = None, chunk_id: str = None) -> bool:
        """
        Create a relationship between two entities.
        
        Args:
            from_entity_key: Source entity key
            to_entity_key: Target entity key
            relation_type: Type of relationship
            confidence: Confidence score (0-1)
            doc_id: Source document ID
            chunk_id: Source chunk ID
        
        Returns:
            bool: Success status
        """
        try:
            relations_col = self.database.collection(self.relations_collection)
            
            # Create unique key for the relation
            relation_key = hashlib.md5(
                f"{from_entity_key}:{to_entity_key}:{relation_type}".encode()
            ).hexdigest()
            
            relation_data = {
                "_key": relation_key,
                "_from": f"{self.entities_collection}/{from_entity_key}",
                "_to": f"{self.entities_collection}/{to_entity_key}",
                "relation_type": relation_type.upper(),
                "confidence": confidence,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Insert or update relation
            try:
                relations_col.insert(relation_data)
                logger.debug(f"Created relation: {from_entity_key} -{relation_type}-> {to_entity_key}")
            except Exception:
                # Relation exists, update confidence if higher
                existing = relations_col.get(relation_key)
                if existing and confidence > existing.get("confidence", 0):
                    existing["confidence"] = confidence
                    existing["updated_at"] = datetime.utcnow().isoformat()
                    relations_col.update(relation_key, existing)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create relation: {e}")
            return False
    
    async def find_related_entities(self, entity_names: List[str], 
                                  max_depth: int = 2, limit: int = 10) -> List[Dict]:
        """
        Find entities related to the given entity names within max_depth.
        
        Args:
            entity_names: List of entity names to start traversal from
            max_depth: Maximum traversal depth
            limit: Maximum number of results
        
        Returns:
            List of related entities with path information
        """
        try:
            # Build entity keys from names
            entity_keys = []
            for name in entity_names:
                # Try to find entity by name
                query = f"""
                FOR entity IN {self.entities_collection}
                FILTER LOWER(entity.name) == LOWER(@name)
                RETURN entity._key
                """
                cursor = self.database.aql.execute(query, bind_vars={"name": name})
                entity_keys.extend([doc for doc in cursor])
            
            if not entity_keys:
                return []
            
            # Traverse graph to find related entities
            query = f"""
            FOR vertex, edge, path IN 1..@max_depth ANY @start_vertices
            GRAPH @graph_name
            OPTIONS {{uniqueVertices: "global", uniqueEdges: "global"}}
            LIMIT @limit
            RETURN {{
                entity: vertex,
                relation: edge,
                path_length: LENGTH(path.edges),
                confidence: edge ? edge.confidence : 1.0
            }}
            """
            
            bind_vars = {
                "start_vertices": [f"{self.entities_collection}/{key}" for key in entity_keys],
                "graph_name": self.graph_name,
                "max_depth": max_depth,
                "limit": limit
            }
            
            cursor = self.database.aql.execute(query, bind_vars=bind_vars)
            results = []
            
            for doc in cursor:
                if doc["entity"]:  # Exclude starting vertices
                    results.append({
                        "entity_id": doc["entity"]["_key"],
                        "entity_name": doc["entity"]["name"],
                        "entity_type": doc["entity"]["type"],
                        "relation_type": doc["relation"]["relation_type"] if doc["relation"] else None,
                        "confidence": doc["confidence"],
                        "path_length": doc["path_length"],
                        "doc_ids": doc["entity"].get("doc_ids", [])
                    })
            
            logger.info(f"Found {len(results)} related entities for {entity_names}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find related entities: {e}")
            return []
    
    async def get_entity_context(self, entity_names: List[str]) -> Dict[str, Any]:
        """
        Get contextual information about entities including their relationships.
        
        Args:
            entity_names: List of entity names to get context for
        
        Returns:
            Dictionary with entity context information
        """
        try:
            context = {
                "entities": [],
                "relationships": [],
                "related_documents": set(),
                "entity_types": set()
            }
            
            for name in entity_names:
                # Find entity
                query = f"""
                FOR entity IN {self.entities_collection}
                FILTER LOWER(entity.name) == LOWER(@name)
                RETURN entity
                """
                cursor = self.database.aql.execute(query, bind_vars={"name": name})
                entities = [doc for doc in cursor]
                
                for entity in entities:
                    context["entities"].append({
                        "name": entity["name"],
                        "type": entity["type"],
                        "doc_ids": entity.get("doc_ids", []),
                        "mention_count": entity.get("mention_count", 1)
                    })
                    
                    context["entity_types"].add(entity["type"])
                    context["related_documents"].update(entity.get("doc_ids", []))
                    
                    # Get relationships for this entity
                    rel_query = f"""
                    FOR rel IN {self.relations_collection}
                    FILTER rel._from == @entity_id OR rel._to == @entity_id
                    RETURN rel
                    """
                    
                    entity_id = f"{self.entities_collection}/{entity['_key']}"
                    rel_cursor = self.database.aql.execute(
                        rel_query, 
                        bind_vars={"entity_id": entity_id}
                    )
                    
                    for rel in rel_cursor:
                        context["relationships"].append({
                            "from": rel["_from"],
                            "to": rel["_to"],
                            "type": rel["relation_type"],
                            "confidence": rel["confidence"]
                        })
            
            # Convert sets to lists for JSON serialization
            context["related_documents"] = list(context["related_documents"])
            context["entity_types"] = list(context["entity_types"])
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get entity context: {e}")
            return {"entities": [], "relationships": [], "related_documents": [], "entity_types": []}
    
    async def delete_document_graph_data(self, doc_id: str) -> bool:
        """Delete all graph data associated with a document."""
        try:
            # Delete relations associated with the document
            rel_query = f"""
            FOR rel IN {self.relations_collection}
            FILTER rel.doc_id == @doc_id
            REMOVE rel IN {self.relations_collection}
            RETURN OLD
            """
            
            rel_cursor = self.database.aql.execute(rel_query, bind_vars={"doc_id": doc_id})
            deleted_relations = len([doc for doc in rel_cursor])
            
            # Update entities to remove doc_id references
            entity_query = f"""
            FOR entity IN {self.entities_collection}
            FILTER @doc_id IN entity.doc_ids
            UPDATE entity WITH {{
                doc_ids: REMOVE_VALUE(entity.doc_ids, @doc_id),
                chunk_ids: (
                    FOR chunk_id IN entity.chunk_ids
                    FILTER NOT STARTS_WITH(chunk_id, CONCAT(@doc_id, "_"))
                    RETURN chunk_id
                )
            }} IN {self.entities_collection}
            RETURN NEW
            """
            
            entity_cursor = self.database.aql.execute(entity_query, bind_vars={"doc_id": doc_id})
            updated_entities = len([doc for doc in entity_cursor])
            
            logger.info(f"Deleted {deleted_relations} relations and updated {updated_entities} entities for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete graph data for document {doc_id}: {e}")
            return False
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        try:
            entities_col = self.database.collection(self.entities_collection)
            relations_col = self.database.collection(self.relations_collection)
            
            # Basic counts
            entity_count = entities_col.count()
            relation_count = relations_col.count()
            
            # Entity type distribution
            type_query = f"""
            FOR entity IN {self.entities_collection}
            COLLECT entity_type = entity.type WITH COUNT INTO type_count
            RETURN {{type: entity_type, count: type_count}}
            """
            type_cursor = self.database.aql.execute(type_query)
            entity_types = {doc["type"]: doc["count"] for doc in type_cursor}
            
            # Relation type distribution
            rel_type_query = f"""
            FOR rel IN {self.relations_collection}
            COLLECT rel_type = rel.relation_type WITH COUNT INTO type_count
            RETURN {{type: rel_type, count: type_count}}
            """
            rel_type_cursor = self.database.aql.execute(rel_type_query)
            relation_types = {doc["type"]: doc["count"] for doc in rel_type_cursor}
            
            return {
                "entity_count": entity_count,
                "relation_count": relation_count,
                "entity_types": entity_types,
                "relation_types": relation_types,
                "graph_density": relation_count / max(entity_count, 1)
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {}


# Global ArangoDB client instance
arango_client = ArangoDBClient()


async def get_arango_client() -> ArangoDBClient:
    """Dependency injection for ArangoDB client."""
    return arango_client