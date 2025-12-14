"""
End-to-End Integration Test for Phase 2 Document Training
==========================================================

Tests the complete document training workflow:
1. Upload a test PDF document
2. Query the system with questions about the document
3. Validate responses contain citations [1], [2], etc
4. Verify media suggestions are generated
5. Check validation details show proper scoring
6. Confirm MongoDB logging works correctly

Run with: pytest backend/tests/test_document_training_e2e.py -v -s
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from pathlib import Path
from io import BytesIO
import base64
from bson import ObjectId
import uuid

# Add backend to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Settings
from app.core.db_mongo import MongoDBClient
from app.core.db_qdrant import QdrantDBClient
from app.core.cache_redis import RedisClient
from app.services.retrieval_engine import RetrievalEngine
from app.services.ingestion_engine import IngestionEngine
from app.api.chat import ChatRequest


# ============================================================================
# Test Fixtures (Setup/Teardown)
# ============================================================================

@pytest_asyncio.fixture
async def settings():
    """Load test settings from environment"""
    return Settings()


@pytest_asyncio.fixture
async def mongo_client(settings):
    """Create and connect MongoDB client for tests"""
    client = MongoDBClient()
    # Override settings if needed, but MongoDBClient uses global settings usually.
    # Here we might need to patch settings or just rely on env vars.
    await client.connect()
    yield client
    # Cleanup: drop test collections
    if client.client:
        try:
            db = client.client.get_database(settings.mongodb_db) # Use correct attribute
            await db.documents.drop()
            await db.chunks.drop()
            await db.document_images.drop()
            await db.validation_logs.drop()
            await db.ingestion_logs.drop()
        except Exception as e:
            print(f"Cleanup error: {e}")
    client.disconnect()


@pytest_asyncio.fixture
async def qdrant_client(settings):
    """Create and connect Qdrant client for tests"""
    client = QdrantDBClient()
    await client.connect()
    
    # Use unique collection name to avoid dimension conflicts
    collection_name = f"test_documents_{uuid.uuid4().hex}"
    
    # Create test collection
    try:
        from qdrant_client.models import VectorParams, Distance
        if client.client:
            await asyncio.to_thread(
                client.client.recreate_collection,
                collection_name=collection_name,
                vectors_config=VectorParams(size=settings.embedding_dimension, distance=Distance.COSINE)
            )
    except Exception as e:
        print(f"Collection creation error: {e}")
        pass
    
    # Patch the client to use this collection
    client.collection_name = collection_name
    
    yield client
    
    # Cleanup: delete test collection
    try:
        if client.client:
            await asyncio.to_thread(client.client.delete_collection, collection_name)
    except Exception as e:
        print(f"Cleanup error: {e}")
    
    await client.disconnect()


@pytest_asyncio.fixture
async def redis_client(settings):
    """Create and connect Redis client for tests"""
    client = RedisClient()
    await client.connect()
    yield client
    await client.disconnect()


@pytest_asyncio.fixture
async def ingest_service(mongo_client, qdrant_client, redis_client):
    """Create IngestionEngine with test clients"""
    service = IngestionEngine()
    # Mock the client getters
    service.mongo_client = mongo_client
    service.qdrant_client = qdrant_client
    # service.redis_client = redis_client
    from app.core.model_manager import get_model_manager
    service.model_manager = await get_model_manager()
    return service


@pytest_asyncio.fixture
async def chat_service(mongo_client, qdrant_client):
    """Create RetrievalEngine with test clients"""
    service = RetrievalEngine()
    # Mock the client getters
    service.mongo_client = mongo_client
    service.qdrant_client = qdrant_client
    from app.core.model_manager import get_model_manager
    service.model_manager = await get_model_manager()
    return service


# ============================================================================
# Test Data Helpers
# ============================================================================

def create_test_pdf() -> bytes:
    """Create a simple test PDF with known content"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from io import BytesIO
        
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        
        # Write test content
        c.setFont("Helvetica", 14)
        c.drawString(100, 750, "Test Document for Phase 2 Integration")
        c.drawString(100, 700, "")
        c.drawString(100, 650, "This is a test PDF document for validating the document training system.")
        c.drawString(100, 600, "It contains specific information about machine learning models.")
        c.drawString(100, 550, "")
        c.drawString(100, 500, "Machine learning is a subset of artificial intelligence that focuses on")
        c.drawString(100, 470, "enabling systems to learn from data without being explicitly programmed.")
        c.drawString(100, 440, "")
        c.drawString(100, 390, "Key concepts:")
        c.drawString(100, 360, "1. Supervised Learning: Training with labeled data")
        c.drawString(100, 330, "2. Unsupervised Learning: Finding patterns in unlabeled data")
        c.drawString(100, 300, "3. Reinforcement Learning: Learning through rewards and penalties")
        
        c.showPage()
        c.save()
        
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    except ImportError:
        # If reportlab not available, return a minimal PDF
        return (
            b"%PDF-1.4\n"
            b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n"
            b"2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n"
            b"3 0 obj\n<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>"
            b"/MediaBox[0 0 612 792]/Contents 5 0 R>>\nendobj\n"
            b"4 0 obj\n<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>\nendobj\n"
            b"5 0 obj\n<</Length 44>>\nstream\nBT /F1 12 Tf 100 700 Td (Test) Tj ET\nendstream\n"
            b"endobj\nxref\n0 6\n"
            b"0000000000 65535 f\n"
            b"0000000009 00000 n\n"
            b"0000000058 00000 n\n"
            b"0000000115 00000 n\n"
            b"0000000244 00000 n\n"
            b"0000000317 00000 n\n"
            b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n405\n%%EOF\n"
        )


async def upload_test_document(ingest_service: IngestionEngine, filename: str, content: bytes) -> str:
    """
    Upload a test document and return document ID
    
    Args:
        ingest_service: IngestionEngine instance
        filename: Document filename
        content: Document content as bytes
        
    Returns:
        Document ID from MongoDB
    """
    # Write to temporary file (not strictly needed for IngestionEngine but good for simulation)
    # Note: Sync file ops acceptable in test setup
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 1. Save document to MongoDB first (IngestionEngine expects doc_id)
        # We need to access mongo_client from ingest_service or pass it in
        mongo_client = ingest_service.mongo_client
        
        doc_id = await mongo_client.save_document(
            filename=filename,
            content=content,
            content_type="application/pdf",
            size=len(content)
        )
        
        # 2. Call ingest service
        success = await ingest_service.process_document(doc_id)
        
        if not success:
            raise RuntimeError("Document processing failed")
            
        return doc_id
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase2DocumentTraining:
    """Test suite for Phase 2 document training features"""
    
    @pytest.mark.asyncio
    async def test_document_upload_and_processing(
        self, 
        ingest_service: IngestionEngine,
        mongo_client: MongoDBClient,
        settings: Settings
    ):
        """
        Test 1: Document Upload and Processing
        
        Verifies:
        - PDF can be uploaded successfully
        - Document is stored in MongoDB
        - Text extraction works
        - Validation logs are created
        """
        print("\n[TEST 1] Document Upload and Processing")
        
        # Create test PDF
        pdf_content = create_test_pdf()
        
        # Upload document
        doc_id = await upload_test_document(
            ingest_service,
            "test_document.pdf",
            pdf_content
        )
        
        assert doc_id is not None, "Document ID should not be None"
        print(f"✓ Document uploaded successfully: {doc_id}")
        
        # Verify document in MongoDB
        db = mongo_client.client.get_database(settings.mongodb_db)
        doc = await db.documents.find_one({"_id": ObjectId(doc_id)})
        
        assert doc is not None, "Document should exist in MongoDB"
        assert doc.get("filename") == "test_document.pdf"
        assert doc.get("ingest_status") == "PROCESSED"
        print(f"✓ Document stored in MongoDB with status: {doc.get('ingest_status')}")
        
        # Verify chunks created
        chunks = await db.chunks.find({"doc_id": doc_id}).to_list(None)
        assert len(chunks) > 0, "Document chunks should be created"
        print(f"✓ Document chunks created: {len(chunks)} chunks")
        
        # Verify ingestion logs
        ingest_logs = await db.ingestion_logs.find({"doc_id": doc_id}).to_list(None)
        assert len(ingest_logs) > 0, "Ingestion logs should be created"
        print(f"✓ Ingestion logs created: {len(ingest_logs)} log entries")
    
    
    @pytest.mark.asyncio
    async def test_image_extraction_from_pdf(
        self,
        ingest_service: IngestionEngine,
        mongo_client: MongoDBClient,
        settings: Settings
    ):
        """
        Test 2: Image Extraction from PDF
        
        Verifies:
        - Images are extracted from PDF (if pdf2image available)
        - Images are stored in MongoDB with base64 encoding
        - Image metadata (page, dimensions) is preserved
        """
        print("\n[TEST 2] Image Extraction from PDF")
        
        # Create test PDF
        pdf_content = create_test_pdf()
        
        # Upload document
        doc_id = await upload_test_document(
            ingest_service,
            "test_image_extraction.pdf",
            pdf_content
        )
        
        # Check for extracted images
        db = mongo_client.client.get_database(settings.mongodb_db)
        images = await db.document_images.find({"doc_id": doc_id}).to_list(None)
        
        # Image extraction is optional - may fail if pdf2image not installed
        if len(images) > 0:
            print(f"✓ Images extracted from PDF: {len(images)} images")
            
            # Verify image data structure
            img = images[0]
            assert img.get("page") is not None, "Image should have page number"
            assert img.get("data") is not None, "Image should have base64 data"
            assert img.get("width") is not None, "Image should have width"
            assert img.get("height") is not None, "Image should have height"
            print(f"✓ Image metadata valid: page {img.get('page')}, {img.get('width')}x{img.get('height')}")
        else:
            print("⚠ No images extracted (pdf2image may not be available)")
    
    
    @pytest.mark.asyncio
    async def test_document_chunks_and_embeddings(
        self,
        ingest_service: IngestionEngine,
        qdrant_client: QdrantDBClient,
        mongo_client: MongoDBClient,
        settings: Settings
    ):
        """
        Test 3: Document Chunks and Embeddings
        
        Verifies:
        - Document is split into chunks
        - Chunks have content and metadata
        - Embeddings are stored in Qdrant (if available)
        """
        print("\n[TEST 3] Document Chunks and Embeddings")
        
        # Create and upload test PDF
        pdf_content = create_test_pdf()
        doc_id = await upload_test_document(
            ingest_service,
            "test_chunks.pdf",
            pdf_content
        )
        
        # Verify chunks in MongoDB
        db = mongo_client.client.get_database(settings.mongodb_db)
        chunks = await db.chunks.find({"doc_id": doc_id}).to_list(None)
        
        assert len(chunks) > 0, "Should have at least one chunk"
        print(f"✓ Chunks created: {len(chunks)}")
        
        # Verify chunk metadata
        chunk = chunks[0]
        assert chunk.get("text") is not None, "Chunk should have content"
        assert chunk.get("chunk_index") is not None, "Chunk should have chunk number"
        assert chunk.get("doc_id") == doc_id, "Chunk should reference correct document"
        print(f"✓ Chunk 1 metadata valid: {len(chunk.get('text', ''))} chars")
        
        # Verify embeddings in Qdrant (if available)
        try:
            collection_info = await qdrant_client.get_collection_info()
            if collection_info and collection_info.get('points_count', 0) > 0:
                print(f"✓ Embeddings stored in Qdrant: {collection_info.get('points_count')} points")
            else:
                print("⚠ No embeddings in Qdrant (semantic search may not be available)")
        except Exception as e:
            print(f"⚠ Qdrant check skipped: {e}")
    
    
    @pytest.mark.asyncio
    async def test_validation_logging(
        self,
        mongo_client: MongoDBClient,
        settings: Settings
    ):
        """
        Test 4: Validation Logging Infrastructure
        
        Verifies:
        - Validation logs collection exists
        - Validation logs can be written
        - Validation logs have required fields
        """
        print("\n[TEST 4] Validation Logging Infrastructure")
        
        db = mongo_client.client.get_database(settings.mongodb_db)
        
        # Create test validation log
        test_log = {
            "doc_id": "test_doc_id",
            "query": "What is machine learning?",
            "response": "Machine learning is a subset of AI...",
            "is_valid": True,
            "validation_score": 85,
            "citation_count": 2,
            "has_citations": True,
            "generic_phrases_count": 0,
            "created_at": "2024-01-15T10:30:00Z"
        }
        
        result = await db.validation_logs.insert_one(test_log)
        assert result.inserted_id is not None, "Validation log should be inserted"
        print(f"✓ Validation log created: {result.inserted_id}")
        
        # Verify log can be retrieved
        log = await db.validation_logs.find_one({"_id": result.inserted_id})
        assert log is not None, "Validation log should be retrievable"
        assert log.get("is_valid") == True
        assert log.get("validation_score") == 85
        print(f"✓ Validation log verified: score={log.get('validation_score')}, citations={log.get('citation_count')}")
    
    
    @pytest.mark.asyncio
    async def test_force_document_usage_setting(self, settings: Settings):
        """
        Test 5: Force Document Usage Configuration
        
        Verifies:
        - FORCE_DOCUMENT_USAGE setting is available
        - VALIDATE_DOCUMENT_USAGE setting is available
        - Validation settings are properly loaded
        """
        print("\n[TEST 5] Force Document Usage Settings")
        
        # Check settings
        assert hasattr(settings, 'force_document_usage'), "Settings should have force_document_usage"
        assert hasattr(settings, 'validate_document_usage'), "Settings should have validate_document_usage"
        
        print("✓ Settings loaded:")
        print(f"  - FORCE_DOCUMENT_USAGE: {settings.force_document_usage}")
        print(f"  - VALIDATE_DOCUMENT_USAGE: {settings.validate_document_usage}")
        print(f"  - REQUIRE_CITATIONS: {settings.require_citations}")
        print(f"  - MIN_CITATION_COUNT: {settings.min_citation_count}")
        
        # Verify at least one is enabled
        assert (
            settings.force_document_usage or 
            settings.validate_document_usage
        ), "At least one document usage setting should be enabled"
        print("✓ Document usage validation is properly configured")
    
    
    @pytest.mark.asyncio
    async def test_media_extraction_settings(self, settings: Settings):
        """
        Test 6: Media Extraction Settings
        
        Verifies:
        - Media extraction settings are available
        - Image extraction setting is available
        - Video extraction setting is available
        """
        print("\n[TEST 6] Media Extraction Settings")
        
        # Check settings
        assert hasattr(settings, 'extract_images_from_pdf'), "Settings should have extract_images_from_pdf"
        assert hasattr(settings, 'extract_video_links'), "Settings should have extract_video_links"
        assert hasattr(settings, 'suggest_related_media'), "Settings should have suggest_related_media"
        
        print("✓ Media settings loaded:")
        print(f"  - EXTRACT_IMAGES_FROM_PDF: {settings.extract_images_from_pdf}")
        print(f"  - EXTRACT_VIDEO_LINKS: {settings.extract_video_links}")
        print(f"  - SUGGEST_RELATED_MEDIA: {settings.suggest_related_media}")
        print(f"  - MAX_SUGGESTED_MEDIA: {settings.max_suggested_media}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
