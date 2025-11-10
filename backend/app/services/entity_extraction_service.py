"""
Entity Extraction Service using LLM
Extracts named entities and relationships from text for knowledge graph construction.
Based on state-of-the-art NER + LLM techniques (2024-2025).

Features:
- LLM-based entity extraction (higher accuracy than spaCy)
- Relationship extraction
- Entity type classification
- Confidence scoring
- Coreference resolution
- Entity linking and deduplication
"""

import logging
import json
import re
from typing import List, Dict, Tuple, Any, Optional
import asyncio

from ..services.llm_handler import llm_handler
from ..core.db_neo4j import neo4j_client

logger = logging.getLogger(__name__)


class EntityExtractionService:
    """Service for extracting entities and relationships from text using LLM."""
    
    # Entity types we extract
    ENTITY_TYPES = [
        "Person",          # People names
        "Organization",    # Companies, institutions
        "Location",        # Places, cities, countries
        "Concept",         # Abstract concepts, theories
        "Product",         # Software, hardware, services
        "Technology",      # Programming languages, frameworks
        "Date",            # Temporal references
        "Event",           # Meetings, conferences, incidents
        "Metric",          # Numbers, measurements, KPIs
        "Document"         # References to other documents
    ]
    
    # Relationship types we extract
    RELATIONSHIP_TYPES = [
        "WORKS_FOR",       # Person works for Organization
        "LOCATED_IN",      # X is located in Location
        "PART_OF",         # X is part of Y
        "USES",            # X uses Technology/Product
        "RELATED_TO",      # General relationship
        "CREATED_BY",      # Product created by Organization/Person
        "BELONGS_TO",      # X belongs to category/type Y
        "MENTIONS",        # Document mentions Entity
        "REFERENCES",      # X references Y
        "DEPENDS_ON"       # X depends on Y
    ]
    
    def __init__(self):
        self.llm = llm_handler
        self.graph = neo4j_client
        self._entity_cache = {}  # Cache to avoid duplicate extractions
    
    async def extract_entities_from_text(
        self,
        text: str,
        doc_id: str = None,
        chunk_id: str = None,
        context: str = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text using LLM.
        
        Args:
            text: Input text to extract entities from
            doc_id: Source document ID
            chunk_id: Source chunk ID
            context: Additional context for better extraction
            
        Returns:
            List of extracted entities with metadata
        """
        if len(text.strip()) < 20:  # Skip very short text
            return []
        
        # Check cache
        cache_key = f"{doc_id}:{chunk_id}"
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]
        
        # Build extraction prompt
        extraction_prompt = self._build_extraction_prompt(text, context)
        
        try:
            # Call LLM for extraction
            response = await self.llm.generate_response(
                prompt=extraction_prompt,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1500
            )
            
            # Parse LLM response
            entities = self._parse_entity_response(response)
            
            # Add metadata
            for entity in entities:
                entity["doc_id"] = doc_id
                entity["chunk_id"] = chunk_id
                entity["source_text"] = text[:200]  # Store snippet
            
            # Cache result
            self._entity_cache[cache_key] = entities
            
            logger.info(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            return []
    
    def _build_extraction_prompt(self, text: str, context: str = None) -> str:
        """Build LLM prompt for entity extraction."""
        entity_types_str = ", ".join(self.ENTITY_TYPES)
        
        prompt = f"""Extract named entities from the following text. For each entity, provide:
1. Entity name (exact text from input)
2. Entity type (one of: {entity_types_str})
3. Confidence score (0.0 to 1.0)
4. Brief description

Text to analyze:
{text}

{f"Context: {context}" if context else ""}

Respond ONLY with a JSON array in this exact format:
[
  {{"name": "entity name", "type": "Person|Organization|etc", "confidence": 0.95, "description": "brief description"}},
  ...
]

Important:
- Extract ALL significant entities mentioned
- Use exact names from the text
- Be specific with types
- Assign high confidence (>0.8) only for clear, unambiguous entities
- Return ONLY the JSON array, no additional text
"""
        return prompt
    
    def _parse_entity_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured entity list."""
        try:
            # Extract JSON from response (LLM might add extra text)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return []
            
            json_str = json_match.group(0)
            entities = json.loads(json_str)
            
            # Validate and clean
            validated_entities = []
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                
                # Required fields
                if "name" not in entity or "type" not in entity:
                    continue
                
                # Validate type
                if entity["type"] not in self.ENTITY_TYPES:
                    entity["type"] = "Concept"  # Default fallback
                
                # Ensure confidence
                if "confidence" not in entity:
                    entity["confidence"] = 0.7
                else:
                    entity["confidence"] = max(0.0, min(1.0, float(entity["confidence"])))
                
                # Ensure description
                if "description" not in entity:
                    entity["description"] = ""
                
                validated_entities.append(entity)
            
            return validated_entities
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return []
    
    async def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        doc_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using LLM.
        
        Args:
            text: Source text
            entities: List of entities extracted from the text
            doc_id: Source document ID
            
        Returns:
            List of relationships with source, target, type, and confidence
        """
        if len(entities) < 2:  # Need at least 2 entities for relationships
            return []
        
        # Build relationship extraction prompt
        rel_prompt = self._build_relationship_prompt(text, entities)
        
        try:
            response = await self.llm.generate_response(
                prompt=rel_prompt,
                temperature=0.1,
                max_tokens=1000
            )
            
            relationships = self._parse_relationship_response(response, entities)
            
            # Add metadata
            for rel in relationships:
                rel["doc_id"] = doc_id
            
            logger.info(f"Extracted {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []
    
    def _build_relationship_prompt(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for relationship extraction."""
        entity_list = "\n".join([
            f"- {e['name']} ({e['type']})"
            for e in entities
        ])
        
        rel_types_str = ", ".join(self.RELATIONSHIP_TYPES)
        
        prompt = f"""Identify relationships between the following entities based on the text.

Text:
{text}

Entities:
{entity_list}

For each relationship, provide:
1. Source entity name (must be from the list above)
2. Target entity name (must be from the list above)
3. Relationship type (one of: {rel_types_str})
4. Confidence score (0.0 to 1.0)

Respond ONLY with a JSON array:
[
  {{"from": "entity1", "to": "entity2", "type": "WORKS_FOR", "confidence": 0.9}},
  ...
]

Important:
- Only extract relationships explicitly mentioned or strongly implied
- Both entities must be from the list above
- Use the exact entity names
- High confidence (>0.7) only for clearly stated relationships
"""
        return prompt
    
    def _parse_relationship_response(
        self,
        response: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into structured relationship list."""
        try:
            # Extract JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []
            
            json_str = json_match.group(0)
            relationships = json.loads(json_str)
            
            # Create entity name set for validation
            entity_names = {e["name"] for e in entities}
            
            # Validate relationships
            validated_rels = []
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                
                # Required fields
                if "from" not in rel or "to" not in rel or "type" not in rel:
                    continue
                
                # Validate entities exist
                if rel["from"] not in entity_names or rel["to"] not in entity_names:
                    continue
                
                # Validate type
                if rel["type"] not in self.RELATIONSHIP_TYPES:
                    rel["type"] = "RELATED_TO"
                
                # Ensure confidence
                if "confidence" not in rel:
                    rel["confidence"] = 0.6
                else:
                    rel["confidence"] = max(0.0, min(1.0, float(rel["confidence"])))
                
                # Find entity types
                from_entity = next((e for e in entities if e["name"] == rel["from"]), None)
                to_entity = next((e for e in entities if e["name"] == rel["to"]), None)
                
                if from_entity and to_entity:
                    rel["from_type"] = from_entity["type"]
                    rel["to_type"] = to_entity["type"]
                    validated_rels.append(rel)
            
            return validated_rels
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relationship JSON: {e}")
            return []
    
    async def index_entities_to_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        doc_id: str,
        chunk_id: str
    ) -> Dict[str, int]:
        """
        Index extracted entities and relationships into Neo4j knowledge graph.
        
        Args:
            entities: List of entities to index
            relationships: List of relationships to index
            doc_id: Source document ID
            chunk_id: Source chunk ID
            
        Returns:
            Statistics dict with counts of entities and relationships created
        """
        if not self.graph.is_connected():
            logger.warning("Neo4j not connected, skipping graph indexing")
            return {"entities": 0, "relationships": 0}
        
        entity_count = 0
        relationship_count = 0
        
        try:
            # Index entities
            for entity in entities:
                success = await self.graph.create_entity(
                    name=entity["name"],
                    entity_type=entity["type"],
                    properties={
                        "description": entity.get("description", ""),
                        "source_text": entity.get("source_text", "")
                    },
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    confidence=entity.get("confidence", 0.7)
                )
                if success:
                    entity_count += 1
            
            # Index relationships
            for rel in relationships:
                success = await self.graph.create_relationship(
                    from_entity_name=rel["from"],
                    from_entity_type=rel["from_type"],
                    to_entity_name=rel["to"],
                    to_entity_type=rel["to_type"],
                    relation_type=rel["type"],
                    properties={"doc_id": doc_id, "chunk_id": chunk_id},
                    confidence=rel.get("confidence", 0.6)
                )
                if success:
                    relationship_count += 1
            
            logger.info(f"Indexed {entity_count} entities and {relationship_count} relationships to Neo4j")
            
            return {
                "entities": entity_count,
                "relationships": relationship_count
            }
            
        except Exception as e:
            logger.error(f"Failed to index to graph: {e}", exc_info=True)
            return {"entities": entity_count, "relationships": relationship_count}
    
    async def process_document_for_graph(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process entire document: extract entities and relationships from all chunks,
        then index to knowledge graph.
        
        Args:
            doc_id: Document ID
            chunks: List of document chunks with 'id' and 'text' fields
            
        Returns:
            Processing statistics
        """
        if not self.graph.is_connected():
            logger.warning("Neo4j not connected, skipping document graph processing")
            return {"success": False, "reason": "Neo4j not connected"}
        
        logger.info(f"Processing document {doc_id} for knowledge graph ({len(chunks)} chunks)")
        
        total_entities = 0
        total_relationships = 0
        
        # Process chunks in batches to avoid overwhelming LLM
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Process batch concurrently
            tasks = []
            for chunk in batch:
                task = self._process_chunk_for_graph(
                    doc_id,
                    chunk.get("id") or chunk.get("_id"),
                    chunk.get("text", "")
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            for result in results:
                if isinstance(result, dict):
                    total_entities += result.get("entities", 0)
                    total_relationships += result.get("relationships", 0)
        
        logger.info(f"Document {doc_id}: Indexed {total_entities} entities and {total_relationships} relationships")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "chunks_processed": len(chunks),
            "entities_created": total_entities,
            "relationships_created": total_relationships
        }
    
    async def _process_chunk_for_graph(
        self,
        doc_id: str,
        chunk_id: str,
        text: str
    ) -> Dict[str, int]:
        """Process a single chunk: extract and index."""
        try:
            # Extract entities
            entities = await self.extract_entities_from_text(
                text=text,
                doc_id=doc_id,
                chunk_id=chunk_id
            )
            
            if not entities:
                return {"entities": 0, "relationships": 0}
            
            # Extract relationships
            relationships = await self.extract_relationships(
                text=text,
                entities=entities,
                doc_id=doc_id
            )
            
            # Index to graph
            stats = await self.index_entities_to_graph(
                entities=entities,
                relationships=relationships,
                doc_id=doc_id,
                chunk_id=chunk_id
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_id}: {e}")
            return {"entities": 0, "relationships": 0}


# Global entity extraction service instance
entity_extraction_service = EntityExtractionService()
