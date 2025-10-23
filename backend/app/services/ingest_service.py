"""
Document ingestion service for processing uploaded files.
This module handles the complete ingestion pipeline: file parsing, text extraction,
chunking, embedding generation, vector storage, and knowledge graph population.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import mimetypes
from pathlib import Path
import json
import re
from bson import ObjectId

# Document processing imports
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import spacy
from sentence_transformers import SentenceTransformer
import tiktoken

from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..core.db_qdrant import get_qdrant_client, QdrantDBClient
from ..core.db_arango import get_arango_client, ArangoDBClient
from ..core.cache_redis import get_redis_client, RedisClient
from .hierarchical_indexing_service import hierarchical_indexing_service
from .json_processor_service import JSONProcessorService
from ..config import settings

logger = logging.getLogger(__name__)

class IngestService:
    """Service for processing and indexing documents."""
    
    def __init__(self):
        self.embedding_model = None
        self.nlp_model = None
        self.tokenizer = None
        self.mongo_client = None
        self.qdrant_client = None
        self.arango_client = None
        self.redis_client = None
        self.json_processor = None
    
    async def initialize(self):
        """Initialize the service with required models and clients."""
        try:
            logger.info("Initializing ingest service...")
            
            # Get database clients
            self.mongo_client = await get_mongodb_client()
            self.qdrant_client = await get_qdrant_client()
            self.arango_client = await get_arango_client()
            self.redis_client = await get_redis_client()
            
            # Initialize embedding model
            logger.info(f"Loading embedding model: {settings.embedding_model_name}")
            self.embedding_model = SentenceTransformer(settings.embedding_model_name)
            
            # Initialize NLP model for entity extraction
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except IOError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Entity extraction will be limited.")
                self.nlp_model = None
            
            # Initialize tokenizer for chunk size calculation
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Initialize JSON processor
            self.json_processor = JSONProcessorService()
            
            # Initialize hierarchical indexing service
            await hierarchical_indexing_service.initialize()
            
            logger.info("Ingest service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ingest service: {e}")
            raise
    
    async def process_document(self, doc_id: str) -> bool:
        """
        Process a document through the complete ingestion pipeline.
        
        Args:
            doc_id: Document ID to process
        
        Returns:
            bool: Success status
        """
        try:
            # Get document metadata
            document = await self.mongo_client.get_document(doc_id)
            if not document:
                raise ValueError(f"Document {doc_id} not found")
            
            filename = document.get("filename", "")
            gridfs_id = document.get("gridfs_id")
            
            logger.info(f"Processing document: {filename}")
            
            # Check if this is a JSON file for special processing
            file_ext = Path(filename).suffix.lower()
            is_json_file = file_ext == ".json"
            
            # Step 1: Extract file content
            await self.mongo_client.log_ingestion_step(
                doc_id, "EXTRACT", "PROCESSING", "Extracting content from file"
            )
            
            file_content = await self.mongo_client.retrieve_file(gridfs_id)
            
            # Handle JSON files specially
            if is_json_file:
                await self._process_json_document(doc_id, file_content, filename)
                return True
            
            text_content = await self._extract_text(file_content, filename)
            
            if not text_content or len(text_content.strip()) < 10:
                raise ValueError("No meaningful text content extracted from file")
            
            await self.mongo_client.log_ingestion_step(
                doc_id, "EXTRACT", "SUCCESS", 
                f"Extracted {len(text_content)} characters of text",
                {"text_length": len(text_content)}
            )
            
            # Step 2: Create hierarchical chunks with enhanced metadata
            await self.mongo_client.log_ingestion_step(
                doc_id, "CHUNK", "PROCESSING", "Creating hierarchical chunks with metadata enrichment"
            )
            
            # Extract document type from filename
            document_type = self._get_document_type(filename)
            
            # Use hierarchical chunking for better structure
            chunks = await hierarchical_indexing_service.create_hierarchical_chunks(
                text_content, doc_id, filename, document_type
            )
            
            await self.mongo_client.log_ingestion_step(
                doc_id, "CHUNK", "SUCCESS", 
                f"Created {len(chunks)} chunks",
                {"chunk_count": len(chunks)}
            )
            
            # Step 3: Generate embeddings
            await self.mongo_client.log_ingestion_step(
                doc_id, "EMBED", "PROCESSING", "Generating embeddings for chunks"
            )
            
            embeddings = await self._generate_embeddings([chunk["text"] for chunk in chunks])
            
            await self.mongo_client.log_ingestion_step(
                doc_id, "EMBED", "SUCCESS", 
                f"Generated embeddings for {len(embeddings)} chunks",
                {"embedding_count": len(embeddings)}
            )
            
            # Step 4: Store chunks in MongoDB
            await self.mongo_client.log_ingestion_step(
                doc_id, "STORE_CHUNKS", "PROCESSING", "Storing chunks in database"
            )
            
            chunk_success = await self.mongo_client.store_chunks(doc_id, chunks)
            if not chunk_success:
                raise ValueError("Failed to store chunks in database")
            
            await self.mongo_client.log_ingestion_step(
                doc_id, "STORE_CHUNKS", "SUCCESS", "Chunks stored successfully"
            )
            
            # Step 5: Index vectors in Qdrant
            await self.mongo_client.log_ingestion_step(
                doc_id, "INDEX_VECTORS", "PROCESSING", "Indexing vectors in Qdrant"
            )
            
            vector_success = await self.qdrant_client.upsert_vectors(doc_id, chunks, embeddings)
            if not vector_success:
                raise ValueError("Failed to index vectors in Qdrant")
            
            await self.mongo_client.log_ingestion_step(
                doc_id, "INDEX_VECTORS", "SUCCESS", "Vectors indexed successfully"
            )
            
            # Step 6: Extract entities and build graph (if enabled)
            if settings.use_graph_search and self.nlp_model:
                await self.mongo_client.log_ingestion_step(
                    doc_id, "EXTRACT_ENTITIES", "PROCESSING", "Extracting entities for knowledge graph"
                )
                
                entity_count = await self._extract_and_store_entities(text_content, doc_id, chunks)
                
                await self.mongo_client.log_ingestion_step(
                    doc_id, "EXTRACT_ENTITIES", "SUCCESS", 
                    f"Extracted {entity_count} entities",
                    {"entity_count": entity_count}
                )
            
            # Step 7: Cache document for faster retrieval
            await self._cache_document_metadata(doc_id, document, len(chunks))
            
            logger.info(f"Document {doc_id} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            return False
    
    async def _process_json_document(self, doc_id: str, file_content: bytes, filename: str) -> bool:
        """
        Process a JSON document with special handling.
        
        Args:
            doc_id: Document ID
            file_content: Raw file content
            filename: File name
            
        Returns:
            bool: Success status
        """
        try:
            await self.mongo_client.log_ingestion_step(
                doc_id, "JSON_PARSE", "PROCESSING", "Parsing JSON file structure"
            )
            
            # Parse JSON and extract data
            json_data = json.loads(file_content.decode("utf-8", errors="ignore"))
            
            # Use JSON processor to extract structured data
            schema_type = self.json_processor.detect_schema_type(json_data)
            extracted_data = self.json_processor.extract_structured_data(json_data)
            
            await self.mongo_client.log_ingestion_step(
                doc_id, "JSON_PARSE", "SUCCESS", 
                f"Detected schema type: {schema_type}, Extracted {len(extracted_data.get('records', []))} records"
            )
            
            # Generate Q&A pairs from JSON data
            await self.mongo_client.log_ingestion_step(
                doc_id, "JSON_QA", "PROCESSING", "Generating Q&A pairs from JSON data"
            )
            
            qa_pairs = self.json_processor.generate_qa_pairs(json_data, extracted_data)
            
            await self.mongo_client.log_ingestion_step(
                doc_id, "JSON_QA", "SUCCESS", 
                f"Generated {len(qa_pairs)} Q&A pairs from JSON data",
                {"qa_count": len(qa_pairs)}
            )
            
            # Create chunks from Q&A pairs for embedding
            chunks = []
            for idx, qa_pair in enumerate(qa_pairs):
                chunk = {
                    "text": f"Q: {qa_pair['question']}\nA: {qa_pair['answer']}",
                    "type": "qa",
                    "question": qa_pair['question'],
                    "answer": qa_pair['answer'],
                    "json_field": qa_pair.get('json_field', ''),
                    "position": idx,
                    "doc_id": doc_id,
                    "filename": filename
                }
                chunks.append(chunk)
            
            # Generate embeddings for Q&A pairs
            await self.mongo_client.log_ingestion_step(
                doc_id, "EMBED", "PROCESSING", "Generating embeddings for Q&A pairs"
            )
            
            qa_texts = [qa['question'] + " " + qa['answer'] for qa in qa_pairs]
            embeddings = await self._generate_embeddings(qa_texts)
            
            await self.mongo_client.log_ingestion_step(
                doc_id, "EMBED", "SUCCESS", 
                f"Generated {len(embeddings)} embeddings for Q&A pairs"
            )
            
            # Store chunks in MongoDB
            await self.mongo_client.log_ingestion_step(
                doc_id, "STORE_CHUNKS", "PROCESSING", "Storing JSON Q&A chunks in database"
            )
            
            chunk_success = await self.mongo_client.store_chunks(doc_id, chunks)
            if not chunk_success:
                raise ValueError("Failed to store chunks in database")
            
            await self.mongo_client.log_ingestion_step(
                doc_id, "STORE_CHUNKS", "SUCCESS", f"Stored {len(chunks)} chunks"
            )
            
            # Store embeddings in Qdrant
            await self.mongo_client.log_ingestion_step(
                doc_id, "STORE_VECTORS", "PROCESSING", "Storing embeddings in vector database"
            )
            
            vector_success = await self.qdrant_client.store_chunks(
                chunks, embeddings, doc_id, filename
            )
            if not vector_success:
                raise ValueError("Failed to store embeddings in vector database")
            
            await self.mongo_client.log_ingestion_step(
                doc_id, "STORE_VECTORS", "SUCCESS", "Embeddings stored in vector database"
            )
            
            # Store JSON metadata for reference
            await self.mongo_client.update_document_status(doc_id, "COMPLETED")
            
            # Store additional JSON metadata in document
            await self.mongo_client.db["documents"].update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {
                    "json_metadata": {
                        "schema_type": schema_type,
                        "field_count": len(extracted_data.get('fields', {})),
                        "record_count": len(extracted_data.get('records', [])),
                        "qa_count": len(qa_pairs),
                        "fields": extracted_data.get('fields', {})
                    }
                }}
            )
            
            logger.info(f"JSON document {doc_id} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process JSON document {doc_id}: {e}")
            return False
    
    async def _extract_text(self, file_content: bytes, filename: str) -> str:
        """
        Extract text from various file formats.
        
        Args:
            file_content: Raw file content
            filename: Original filename for type detection
        
        Returns:
            Extracted text content
        """
        try:
            # Determine file type
            mime_type, _ = mimetypes.guess_type(filename)
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == ".pdf" or mime_type == "application/pdf":
                return self._extract_text_from_pdf(file_content)
            elif file_ext == ".docx" or mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self._extract_text_from_docx(file_content)
            elif file_ext == ".html" or mime_type == "text/html":
                return self._extract_text_from_html(file_content)
            elif file_ext == ".json" or mime_type == "application/json":
                return self._extract_text_from_json(file_content)
            elif file_ext == ".txt" or mime_type == "text/plain":
                return file_content.decode("utf-8", errors="ignore")
            else:
                # Try to decode as text
                return file_content.decode("utf-8", errors="ignore")
                
        except Exception as e:
            logger.error(f"Text extraction failed for {filename}: {e}")
            # Fallback: try to decode as text
            try:
                return file_content.decode("utf-8", errors="ignore")
            except:
                return ""
    
    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file."""
        try:
            import io
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return ""
    
    def _extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file."""
        try:
            import io
            docx_file = io.BytesIO(file_content)
            doc = Document(docx_file)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                text_parts.append(paragraph.text)
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"DOCX text extraction failed: {e}")
            return ""
    
    def _extract_text_from_html(self, file_content: bytes) -> str:
        """Extract text from HTML file."""
        try:
            html_content = file_content.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"HTML text extraction failed: {e}")
            return ""
    
    def _extract_text_from_json(self, file_content: bytes) -> str:
        """Extract text from JSON file."""
        try:
            json_content = file_content.decode("utf-8", errors="ignore")
            data = json.loads(json_content)
            
            def extract_strings(obj, depth=0):
                """Recursively extract all string values from JSON."""
                if depth > 10:  # Prevent infinite recursion
                    return []
                
                strings = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, str) and len(value.strip()) > 2:
                            strings.append(f"{key}: {value}")
                        else:
                            strings.extend(extract_strings(value, depth + 1))
                elif isinstance(obj, list):
                    for item in obj:
                        strings.extend(extract_strings(item, depth + 1))
                elif isinstance(obj, str) and len(obj.strip()) > 2:
                    strings.append(obj)
                
                return strings
            
            text_parts = extract_strings(data)
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"JSON text extraction failed: {e}")
            return ""
    
    async def _chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with proper token management.
        
        Args:
            text: Text to chunk
            doc_id: Document ID for chunk identification
        
        Returns:
            List of chunk dictionaries
        """
        try:
            # Clean text
            text = self._clean_text(text)
            
            chunks = []
            chunk_size = settings.chunk_size
            overlap = settings.chunk_overlap
            
            # Use tiktoken for accurate token counting
            tokens = self.tokenizer.encode(text)
            
            start_idx = 0
            chunk_index = 0
            
            while start_idx < len(tokens):
                # Calculate end index
                end_idx = min(start_idx + chunk_size, len(tokens))
                
                # Extract chunk tokens
                chunk_tokens = tokens[start_idx:end_idx]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                # Skip very short chunks
                if len(chunk_text.strip()) < 10:
                    start_idx = end_idx
                    continue
                
                # Calculate character positions (approximate)
                char_start = int((start_idx / len(tokens)) * len(text))
                char_end = int((end_idx / len(tokens)) * len(text))
                
                chunk_data = {
                    "text": chunk_text.strip(),
                    "char_start": char_start,
                    "char_end": char_end,
                    "tokens": len(chunk_tokens),
                    "chunk_index": chunk_index
                }
                
                chunks.append(chunk_data)
                
                # Move start index with overlap
                start_idx = end_idx - overlap if end_idx < len(tokens) else end_idx
                chunk_index += 1
            
            logger.info(f"Created {len(chunks)} chunks from {len(tokens)} tokens")
            return chunks
            
        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        try:
            # Check cache first
            embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                cached_embedding = await self.redis_client.get_cached_embedding(text_hash)
                
                if cached_embedding:
                    embeddings.append(cached_embedding)
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                logger.info(f"Generating embeddings for {len(uncached_texts)} texts")
                new_embeddings = self.embedding_model.encode(uncached_texts).tolist()
                
                # Cache new embeddings and fill results
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    embeddings[idx] = embedding
                    
                    # Cache the embedding
                    text_hash = hashlib.md5(texts[idx].encode()).hexdigest()
                    await self.redis_client.cache_embedding(text_hash, embedding)
            
            logger.info(f"Generated/retrieved {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    async def _extract_and_store_entities(self, text: str, doc_id: str, chunks: List[Dict]) -> int:
        """
        Extract entities from text and store in knowledge graph.
        
        Args:
            text: Full document text
            doc_id: Document ID
            chunks: List of text chunks
        
        Returns:
            Number of entities extracted
        """
        try:
            if not self.nlp_model:
                return 0
            
            entity_count = 0
            
            # Process full text for global entities
            doc = self.nlp_model(text[:1000000])  # Limit text length for processing
            
            # Extract entities
            entities_by_chunk = {}
            
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                    # Find which chunk contains this entity
                    chunk_id = self._find_entity_chunk(ent.start_char, chunks)
                    
                    if chunk_id:
                        entity_key = await self.arango_client.create_entity(
                            name=ent.text,
                            entity_type=ent.label_,
                            doc_id=doc_id,
                            chunk_id=chunk_id
                        )
                        
                        if entity_key:
                            entity_count += 1
                            
                            # Track entities per chunk for relationship building
                            if chunk_id not in entities_by_chunk:
                                entities_by_chunk[chunk_id] = []
                            entities_by_chunk[chunk_id].append({
                                "key": entity_key,
                                "name": ent.text,
                                "type": ent.label_
                            })
            
            # Create relationships between entities in the same chunk
            for chunk_id, chunk_entities in entities_by_chunk.items():
                if len(chunk_entities) > 1:
                    for i, entity1 in enumerate(chunk_entities):
                        for entity2 in chunk_entities[i+1:]:
                            await self.arango_client.create_relation(
                                entity1["key"],
                                entity2["key"],
                                "CO_OCCURS",
                                confidence=0.6,
                                doc_id=doc_id,
                                chunk_id=chunk_id
                            )
            
            logger.info(f"Extracted {entity_count} entities from document {doc_id}")
            return entity_count
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return 0
    
    def _find_entity_chunk(self, char_position: int, chunks: List[Dict]) -> Optional[str]:
        """Find which chunk contains the entity at given character position."""
        for chunk in chunks:
            if chunk["char_start"] <= char_position <= chunk["char_end"]:
                return f"{chunk.get('doc_id', 'unknown')}_chunk_{chunk['chunk_index']}"
        return None
    
    def _get_document_type(self, filename: str) -> str:
        """Extract document type from filename."""
        extension = Path(filename).suffix.lower().lstrip('.')
        
        # Map file extensions to document types
        type_mapping = {
            'pdf': 'pdf',
            'doc': 'docx',
            'docx': 'docx',
            'txt': 'txt',
            'md': 'markdown',
            'markdown': 'markdown',
            'html': 'html',
            'htm': 'html',
            'xml': 'xml',
            'json': 'json',
            'csv': 'csv',
            'xlsx': 'xlsx',
            'xls': 'xlsx'
        }
        
        return type_mapping.get(extension, 'unknown')
    
    async def _cache_document_metadata(self, doc_id: str, document: Dict, chunk_count: int):
        """Cache document metadata for faster access."""
        try:
            metadata = {
                "doc_id": doc_id,
                "filename": document.get("filename", ""),
                "file_type": document.get("file_type", ""),
                "uploaded_at": document.get("uploaded_at", "").isoformat() if document.get("uploaded_at") else "",
                "chunk_count": chunk_count,
                "ingest_status": "COMPLETED",
                "processed_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.cache_document_metadata(doc_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to cache document metadata: {e}")