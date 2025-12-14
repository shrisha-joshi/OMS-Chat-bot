"""
Ingestion Engine
Consolidated service for document processing, chunking, and indexing.
Replaces: ingest_service.py, chunking_service.py, json_processor.py
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import re
from pathlib import Path

# Document parsers
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup

from ..core.db_mongo import get_mongodb_client
from ..core.db_qdrant import get_qdrant_client
from ..core.model_manager import get_model_manager
from ..config import settings

logger = logging.getLogger(__name__)

class IngestionEngine:
    def __init__(self):
        self.mongo_client = None
        self.qdrant_client = None
        self.model_manager = None
        
    async def initialize(self):
        """Initialize dependencies."""
        self.mongo_client = get_mongodb_client()
        self.qdrant_client = await get_qdrant_client()
        self.model_manager = await get_model_manager()
        
    async def process_document(self, doc_id: str) -> bool:
        """
        Main entry point: Process a document from MongoDB GridFS.
        1. Retrieve file
        2. Extract text
        3. Chunk text
        4. Generate embeddings
        5. Store in Qdrant & MongoDB
        """
        try:
            if not self.mongo_client:
                await self.initialize()
                
            # 1. Get Document Metadata
            doc = await self.mongo_client.get_document(doc_id)
            if not doc:
                logger.error(f"Document {doc_id} not found")
                return False
                
            filename = doc.get("filename", "unknown")
            logger.info(f"🚀 Starting ingestion for: {filename} ({doc_id})")
            await self.mongo_client.log_ingestion_step(doc_id, "init", "STARTED", f"Starting ingestion for {filename}")
            
            # 2. Retrieve File Content
            file_content = await self.mongo_client.retrieve_file(doc.get("gridfs_id"))
            
            # 3. Extract Text
            text = self._extract_text(filename, file_content)
            if not text:
                await self.mongo_client.log_ingestion_step(doc_id, "extraction", "FAILED", "No text extracted")
                raise ValueError("No text extracted from document")
            await self.mongo_client.log_ingestion_step(doc_id, "extraction", "COMPLETED", f"Extracted {len(text)} chars")
                
            # 4. Create Chunks
            chunks = self._create_chunks(text, doc_id)
            logger.info(f"📄 Created {len(chunks)} chunks")
            await self.mongo_client.log_ingestion_step(doc_id, "chunking", "COMPLETED", f"Created {len(chunks)} chunks")
            
            # 5. Generate Embeddings
            embeddings = self._generate_embeddings([c["text"] for c in chunks])
            await self.mongo_client.log_ingestion_step(doc_id, "embedding", "COMPLETED", "Generated embeddings")
            
            # 6. Store in Qdrant
            await self.qdrant_client.upsert_vectors(doc_id, chunks, embeddings)
            
            # 7. Store Chunks in MongoDB (for hybrid search/retrieval)
            await self.mongo_client.store_chunks(doc_id, chunks)
            
            # 8. Update Status
            await self.mongo_client.update_document_status(doc_id, "PROCESSED")
            await self.mongo_client.log_ingestion_step(doc_id, "complete", "COMPLETED", "Ingestion finished successfully")
            
            logger.info(f"✅ Ingestion complete for {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed for {doc_id}: {e}", exc_info=True)
            if self.mongo_client:
                await self.mongo_client.update_document_status(doc_id, "FAILED", str(e))
            return False

    def _extract_text(self, filename: str, content: bytes) -> str:
        """Extract text based on file extension."""
        ext = Path(filename).suffix.lower()
        
        try:
            if ext == '.pdf':
                return self._parse_pdf(content)
            elif ext in ['.docx', '.doc']:
                return self._parse_docx(content)
            elif ext == '.txt':
                return content.decode('utf-8', errors='ignore')
            elif ext == '.json':
                return self._parse_json(content)
            elif ext == '.html':
                return self._parse_html(content)
            else:
                logger.warning(f"Unsupported file type: {ext}, treating as text")
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Text extraction failed for {filename}: {e}")
            return ""

    def _parse_pdf(self, content: bytes) -> str:
        import io
        text = []
        with io.BytesIO(content) as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    def _parse_docx(self, content: bytes) -> str:
        import io
        with io.BytesIO(content) as f:
            doc = Document(f)
            return "\n".join([p.text for p in doc.paragraphs])

    def _parse_html(self, content: bytes) -> str:
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text(separator="\n")

    def _parse_json(self, content: bytes) -> str:
        """Flatten JSON to text, optimized for large lists."""
        try:
            data = json.loads(content)
            
            # Optimization for list of objects (common in datasets)
            if isinstance(data, list):
                text_parts = []
                for item in data:
                    # Compact dump for each item to save space/time
                    text_parts.append(json.dumps(item, ensure_ascii=False))
                return "\n\n".join(text_parts)
            
            return json.dumps(data, indent=2, ensure_ascii=False)
        except:
            return content.decode('utf-8', errors='ignore')

    def _create_chunks(self, text: str, doc_id: str) -> List[Dict]:
        """Split text into overlapping chunks. Optimized for large texts."""
        chunk_size = settings.chunk_size
        overlap = settings.chunk_overlap
        
        tokenizer = self.model_manager.get_tokenizer()
        
        # Optimization: For very large texts, split by lines first to avoid 
        # tokenizing the entire string at once (which can spike memory)
        MAX_TEXT_SEGMENT = 1_000_000  # 1MB segments
        
        if len(text) > MAX_TEXT_SEGMENT:
            logger.info(f"📦 Large text detected ({len(text)} chars). Using segmented tokenization.")
            all_tokens = []
            
            # Split by double newlines (paragraphs) or just slice
            # Slicing is safer for memory
            for i in range(0, len(text), MAX_TEXT_SEGMENT):
                segment = text[i:i + MAX_TEXT_SEGMENT]
                segment_tokens = tokenizer.encode(segment)
                all_tokens.extend(segment_tokens)
                
            tokens = all_tokens
        else:
            tokens = tokenizer.encode(text)
        
        chunks = []
        total_tokens = len(tokens)
        start = 0
        
        chunk_idx = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "_id": f"{doc_id}_chunk_{chunk_idx}",
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "tokens": len(chunk_tokens)
            })
            
            chunk_idx += 1
            start += (chunk_size - overlap)
            
        return chunks

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the singleton model."""
        model = self.model_manager.get_embedding_model()
        return model.encode(texts, convert_to_tensor=False).tolist()

# Global instance
ingestion_engine = IngestionEngine()

async def get_ingestion_engine():
    await ingestion_engine.initialize()
    return ingestion_engine
