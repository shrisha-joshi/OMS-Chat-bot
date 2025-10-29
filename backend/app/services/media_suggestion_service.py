"""
Media suggestion service for enriching chat responses with visual content.
Extracts and suggests images, videos, and PDFs from documents.
"""

import logging
import re
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
from io import BytesIO

from ..config import settings
from ..core.db_mongo import get_mongodb_client, MongoDBClient

logger = logging.getLogger(__name__)


class MediaSuggestionService:
    """Service for suggesting media attachments based on context."""
    
    def __init__(self):
        self.mongo_client: Optional[MongoDBClient] = None
        self.youtube_regex = r'(?:youtube\.com|youtu\.be)\/[^\s]+'
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        self.pdf_extensions = ['.pdf']
    
    async def initialize(self):
        """Initialize the media suggestion service."""
        try:
            self.mongo_client = await get_mongodb_client()
            logger.info("✅ Media suggestion service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize media suggestion service: {e}")
    
    async def suggest_media_for_response(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Intelligently suggest media attachments based on query and response.
        
        Args:
            query: Original user query
            response: LLM-generated response
            sources: List of source documents used
        
        Returns:
            List of suggested media items
        """
        if not settings.suggest_related_media:
            logger.debug("Media suggestion disabled in config")
            return []
        
        try:
            suggestions = []
            
            # 1. Extract images from referenced documents
            if settings.extract_images_from_pdf:
                images = await self._extract_images_from_sources(sources)
                suggestions.extend(images)
            
            # 2. Extract YouTube videos from response or documents
            if settings.extract_video_links:
                videos = await self._extract_youtube_videos(response, sources)
                suggestions.extend(videos)
            
            # 3. Suggest relevant PDF documents
            pdfs = await self._suggest_pdf_documents(sources)
            suggestions.extend(pdfs)
            
            # 4. Suggest related images by query
            related_images = await self._find_related_images(query, sources)
            suggestions.extend(related_images)
            
            # Limit suggestions
            suggestions = suggestions[:settings.max_suggested_media]
            
            logger.info(f"💾 Suggested {len(suggestions)} media items for response")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting media: {e}")
            return []
    
    async def _extract_images_from_sources(
        self,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract images from source documents."""
        images = []
        
        try:
            for source in sources:
                doc_id = source.get('doc_id')
                if not doc_id:
                    continue
                
                # Query stored images
                try:
                    image_docs = await self.mongo_client.database.document_images.find(
                        {"doc_id": doc_id}
                    ).to_list(3)  # Get up to 3 images per source
                    
                    for img_doc in image_docs:
                        images.append({
                            "type": "image",
                            "data": img_doc.get("data"),  # Base64 encoded
                            "source": source.get("filename"),
                            "page": img_doc.get("page"),
                            "doc_id": str(doc_id),
                            "relevance": source.get("similarity", 0)
                        })
                        logger.debug(f"Added image from {source.get('filename')} page {img_doc.get('page')}")
                except Exception as e:
                    logger.debug(f"Could not extract images from {doc_id}: {e}")
        
        except Exception as e:
            logger.warning(f"Error extracting images from sources: {e}")
        
        return images
    
    async def _extract_youtube_videos(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract YouTube videos from response and document content."""
        videos = []
        
        try:
            # Extract from response text
            youtube_urls = re.findall(self.youtube_regex, response)
            for url in youtube_urls:
                video_id = self._extract_video_id(url)
                if video_id:
                    videos.append({
                        "type": "youtube",
                        "videoId": video_id,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "title": "Related YouTube Video",
                        "source": "response_context"
                    })
                    logger.debug(f"Added YouTube video: {video_id}")
            
            # Extract from document metadata
            for source in sources:
                doc_id = source.get('doc_id')
                if not doc_id:
                    continue
                
                try:
                    # Check for stored video links
                    media_docs = await self.mongo_client.database.media_attachments.find(
                        {"doc_id": doc_id, "type": "youtube"}
                    ).to_list(2)
                    
                    for media in media_docs:
                        videos.append({
                            "type": "youtube",
                            "videoId": media.get("video_id"),
                            "url": media.get("url"),
                            "title": media.get("title", "Related Video"),
                            "source": source.get("filename")
                        })
                except Exception as e:
                    logger.debug(f"Could not extract videos from {doc_id}: {e}")
        
        except Exception as e:
            logger.warning(f"Error extracting YouTube videos: {e}")
        
        return videos
    
    async def _suggest_pdf_documents(
        self,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest PDF documents from sources."""
        pdfs = []
        
        try:
            for source in sources:
                filename = source.get("filename", "")
                if filename.lower().endswith(".pdf"):
                    pdfs.append({
                        "type": "pdf",
                        "filename": filename,
                        "doc_id": source.get("doc_id"),
                        "size": source.get("size", 0),
                        "relevance": source.get("similarity", 0),
                        "source": "document"
                    })
                    logger.debug(f"Suggested PDF: {filename}")
        
        except Exception as e:
            logger.warning(f"Error suggesting PDFs: {e}")
        
        return pdfs
    
    async def _find_related_images(
        self,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find images related to query keywords."""
        images = []
        
        try:
            if not self.mongo_client:
                return images
            
            # Extract keywords from query
            keywords = set(query.lower().split())
            keywords = {kw for kw in keywords if len(kw) > 3}  # Filter short words
            
            # Find images with matching descriptions
            if keywords:
                query_pattern = "|".join(keywords)
                image_docs = await self.mongo_client.database.document_images.find(
                    {
                        "$or": [
                            {"description": {"$regex": query_pattern, "$options": "i"}},
                            {"alt_text": {"$regex": query_pattern, "$options": "i"}}
                        ]
                    }
                ).to_list(settings.max_suggested_media)
                
                for img in image_docs:
                    images.append({
                        "type": "image",
                        "data": img.get("data"),
                        "source": "semantic_search",
                        "description": img.get("description", ""),
                        "doc_id": str(img.get("doc_id")),
                        "relevance": 0.7
                    })
        
        except Exception as e:
            logger.debug(f"Error finding related images: {e}")
        
        return images
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        try:
            # Standard youtube.com URL
            match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})', url)
            if match:
                return match.group(1)
            
            # Short URL format
            if 'youtu.be/' in url:
                return url.split('youtu.be/')[-1].split('?')[0]
            
            # Standard format
            if 'v=' in url:
                return url.split('v=')[-1].split('&')[0]
        except Exception as e:
            logger.debug(f"Error extracting video ID from {url}: {e}")
        
        return None
    
    async def validate_media_suggestion(
        self,
        media: Dict[str, Any]
    ) -> bool:
        """Validate that media suggestion is valid."""
        try:
            media_type = media.get("type")
            
            if media_type == "youtube":
                return bool(media.get("videoId") and len(media.get("videoId", "")) == 11)
            elif media_type == "image":
                return bool(media.get("data"))
            elif media_type == "pdf":
                return bool(media.get("filename") or media.get("doc_id"))
            
            return True
        except Exception as e:
            logger.warning(f"Error validating media: {e}")
            return False


# Global media suggestion service instance
media_suggestion_service = MediaSuggestionService()


async def get_media_suggestion_service() -> MediaSuggestionService:
    """Dependency injection for media suggestion service."""
    return media_suggestion_service
