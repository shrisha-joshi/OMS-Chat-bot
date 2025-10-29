"""
MongoDB Schema Migrations for Document Training and Media Enhancement.
Run these migrations to set up the required collections and indexes.
"""

import logging
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


async def create_collections_and_indexes(db: AsyncIOMotorDatabase):
    """
    Create all required collections and indexes for media and document tracking.
    
    Args:
        db: Motor AsyncIOMotorDatabase instance
    """
    try:
        # Check if collections exist
        existing_collections = await db.list_collection_names()
        
        # 1. Create document_images collection
        if "document_images" not in existing_collections:
            await db.create_collection("document_images")
            logger.info("✅ Created document_images collection")
        
        # Add indexes for document_images
        await db.document_images.create_indexes([
            [("doc_id", 1), ("page", 1)],  # Compound index for efficient retrieval
            [("description", "text"), ("alt_text", "text")],  # Text search for images
            [("created_at", -1)],  # For sorting by creation date
            [("doc_id", 1)],  # For document lookups
        ])
        logger.info("✅ Created indexes for document_images")
        
        # 2. Create media_attachments collection
        if "media_attachments" not in existing_collections:
            await db.create_collection("media_attachments")
            logger.info("✅ Created media_attachments collection")
        
        await db.media_attachments.create_indexes([
            [("doc_id", 1), ("type", 1)],  # Compound index for queries
            [("type", 1)],  # For filtering by media type
            [("created_at", -1)],  # For sorting
            [("tags", 1)],  # For tag-based filtering
        ])
        logger.info("✅ Created indexes for media_attachments")
        
        # 3. Create query_source_mappings collection
        if "query_source_mappings" not in existing_collections:
            await db.create_collection("query_source_mappings")
            logger.info("✅ Created query_source_mappings collection")
        
        await db.query_source_mappings.create_indexes([
            [("query_id", 1), ("source_id", 1)],  # Compound index
            [("query_id", 1)],  # Query lookups
            [("source_id", 1)],  # Source lookups
            [("created_at", -1)],  # For sorting
            [("relevance_score", -1)],  # For sorting by relevance
        ])
        logger.info("✅ Created indexes for query_source_mappings")
        
        # 4. Create document_validation_logs collection
        if "document_validation_logs" not in existing_collections:
            await db.create_collection("document_validation_logs")
            logger.info("✅ Created document_validation_logs collection")
        
        await db.document_validation_logs.create_indexes([
            [("query_id", 1)],  # Query lookups
            [("is_valid", 1), ("created_at", -1)],  # Filter by validity
            [("created_at", -1)],  # For sorting
            [("validation_score", -1)],  # For sorting by score
        ])
        logger.info("✅ Created indexes for document_validation_logs")
        
        # 5. Extend documents collection with media-related fields
        # Note: This just ensures document structure, doesn't recreate collection
        if "documents" in existing_collections:
            await db.documents.create_indexes([
                [("has_images", 1)],  # Quick filter for image-containing docs
                [("has_videos", 1)],  # Quick filter for video-containing docs
                [("media_count", -1)],  # Sort by media count
            ])
            logger.info("✅ Created media-related indexes for documents")
        
        logger.info("✅ All collections and indexes created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating collections and indexes: {e}")
        raise


async def create_document_image_schema(db: AsyncIOMotorDatabase):
    """
    Create validator schema for document_images collection.
    
    Structure:
    {
        doc_id: ObjectId,
        page: int,
        data: str (base64 encoded),
        description: str,
        alt_text: str,
        width: int,
        height: int,
        format: str (jpg, png, etc),
        size: int (bytes),
        created_at: datetime,
        updated_at: datetime
    }
    """
    try:
        await db.command(
            "collMod",
            "document_images",
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["doc_id", "page", "data"],
                    "properties": {
                        "_id": {"bsonType": "objectId"},
                        "doc_id": {"bsonType": "objectId"},
                        "page": {"bsonType": "int", "minimum": 0},
                        "data": {"bsonType": "string", "description": "Base64 encoded image"},
                        "description": {"bsonType": "string"},
                        "alt_text": {"bsonType": "string"},
                        "width": {"bsonType": "int"},
                        "height": {"bsonType": "int"},
                        "format": {"bsonType": "string", "enum": ["jpg", "png", "gif", "webp"]},
                        "size": {"bsonType": "int"},
                        "created_at": {"bsonType": "date"},
                        "updated_at": {"bsonType": "date"}
                    }
                }
            }
        )
        logger.info("✅ Applied schema validation to document_images")
    except Exception as e:
        logger.warning(f"Schema validation not applied (collection may not exist yet): {e}")


async def create_media_attachment_schema(db: AsyncIOMotorDatabase):
    """
    Create validator schema for media_attachments collection.
    
    Structure:
    {
        doc_id: ObjectId,
        type: str (youtube, video, pdf, link),
        title: str,
        url: str,
        video_id: str (for YouTube),
        duration: int (seconds),
        tags: [str],
        created_at: datetime
    }
    """
    try:
        await db.command(
            "collMod",
            "media_attachments",
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["doc_id", "type"],
                    "properties": {
                        "_id": {"bsonType": "objectId"},
                        "doc_id": {"bsonType": "objectId"},
                        "type": {"bsonType": "string", "enum": ["youtube", "video", "pdf", "link", "image"]},
                        "title": {"bsonType": "string"},
                        "url": {"bsonType": "string"},
                        "video_id": {"bsonType": "string"},
                        "duration": {"bsonType": "int"},
                        "tags": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "created_at": {"bsonType": "date"}
                    }
                }
            }
        )
        logger.info("✅ Applied schema validation to media_attachments")
    except Exception as e:
        logger.warning(f"Schema validation not applied (collection may not exist yet): {e}")


async def create_query_mapping_schema(db: AsyncIOMotorDatabase):
    """
    Create validator schema for query_source_mappings collection.
    
    Structure:
    {
        query_id: str,
        source_id: ObjectId,
        relevance_score: float,
        used_in_response: bool,
        citation_index: int,
        created_at: datetime
    }
    """
    try:
        await db.command(
            "collMod",
            "query_source_mappings",
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["query_id", "source_id"],
                    "properties": {
                        "_id": {"bsonType": "objectId"},
                        "query_id": {"bsonType": "string"},
                        "source_id": {"bsonType": "objectId"},
                        "relevance_score": {"bsonType": "double", "minimum": 0, "maximum": 1},
                        "used_in_response": {"bsonType": "bool"},
                        "citation_index": {"bsonType": "int"},
                        "created_at": {"bsonType": "date"}
                    }
                }
            }
        )
        logger.info("✅ Applied schema validation to query_source_mappings")
    except Exception as e:
        logger.warning(f"Schema validation not applied (collection may not exist yet): {e}")


async def create_validation_log_schema(db: AsyncIOMotorDatabase):
    """
    Create validator schema for document_validation_logs collection.
    
    Structure:
    {
        query_id: str,
        response: str,
        is_valid: bool,
        validation_score: float,
        has_citations: bool,
        citation_count: int,
        has_generic_phrases: bool,
        generic_phrase_count: int,
        validation_details: object,
        created_at: datetime
    }
    """
    try:
        await db.command(
            "collMod",
            "document_validation_logs",
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["query_id", "is_valid"],
                    "properties": {
                        "_id": {"bsonType": "objectId"},
                        "query_id": {"bsonType": "string"},
                        "response": {"bsonType": "string"},
                        "is_valid": {"bsonType": "bool"},
                        "validation_score": {"bsonType": "double", "minimum": 0, "maximum": 1},
                        "has_citations": {"bsonType": "bool"},
                        "citation_count": {"bsonType": "int"},
                        "has_generic_phrases": {"bsonType": "bool"},
                        "generic_phrase_count": {"bsonType": "int"},
                        "validation_details": {"bsonType": "object"},
                        "created_at": {"bsonType": "date"}
                    }
                }
            }
        )
        logger.info("✅ Applied schema validation to document_validation_logs")
    except Exception as e:
        logger.warning(f"Schema validation not applied (collection may not exist yet): {e}")


async def run_all_migrations(db: AsyncIOMotorDatabase):
    """Run all migrations to set up the database schema."""
    try:
        logger.info("🚀 Starting database migrations...")
        
        await create_collections_and_indexes(db)
        await create_document_image_schema(db)
        await create_media_attachment_schema(db)
        await create_query_mapping_schema(db)
        await create_validation_log_schema(db)
        
        logger.info("✅ All database migrations completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise
