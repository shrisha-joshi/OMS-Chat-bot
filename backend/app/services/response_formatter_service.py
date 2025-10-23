"""
Response formatter service for chat responses.
Formats responses with media embeds, citations, and proper markdown.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FormattedResponse:
    """Formatted response with media and citations."""
    text: str
    attachments: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ResponseFormatterService:
    """Service for formatting chat responses with media and citations."""
    
    def __init__(self):
        self.youtube_regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        self.pdf_extensions = ['.pdf']
        self.link_regex = r'https?://[^\s\)"\]]+|www\.[^\s\)"\]]+'
    
    async def format_response(
        self,
        response_text: str,
        sources: List[Dict[str, Any]],
        original_query: str = None
    ) -> FormattedResponse:
        """
        Format a response with media embeds and citations.
        
        Args:
            response_text: Raw response text from LLM
            sources: List of source documents
            original_query: Original user query
            
        Returns:
            FormattedResponse with formatted text, attachments, and citations
        """
        try:
            # Extract and format media
            attachments = await self._extract_media(response_text, sources)
            
            # Create citations
            citations = await self._create_citations(sources)
            
            # Format text with inline citations
            formatted_text = await self._add_inline_citations(response_text, sources)
            
            # Metadata
            metadata = {
                'has_attachments': len(attachments) > 0,
                'attachment_count': len(attachments),
                'citation_count': len(citations),
                'formatted_at': self._get_timestamp()
            }
            
            return FormattedResponse(
                text=formatted_text,
                attachments=attachments,
                citations=citations,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Response formatting error: {e}")
            # Return raw response if formatting fails
            return FormattedResponse(
                text=response_text,
                attachments=[],
                citations=await self._create_citations(sources),
                metadata={'error': str(e)}
            )
    
    async def _extract_media(
        self,
        text: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract media references from text and sources."""
        attachments = []
        processed_urls = set()
        
        # Extract YouTube videos
        youtube_videos = re.findall(self.youtube_regex, text)
        for video_id in set(youtube_videos):
            if video_id not in processed_urls:
                attachments.append({
                    'type': 'youtube',
                    'videoId': video_id,
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'title': f'YouTube Video: {video_id[:8]}...'
                })
                processed_urls.add(video_id)
        
        # Extract links from text
        links = re.findall(self.link_regex, text)
        for link in set(links):
            if link not in processed_urls and link not in [v.get('url') for v in attachments]:
                attachments.append({
                    'type': 'link',
                    'url': link,
                    'title': self._shorten_url(link)
                })
                processed_urls.add(link)
        
        # Extract media from sources
        for source in sources:
            filename = source.get('filename', '').lower()
            
            # Check for images
            if any(filename.endswith(ext) for ext in self.image_extensions):
                doc_id = source.get('doc_id', '')
                if doc_id and doc_id not in processed_urls:
                    attachments.append({
                        'type': 'image',
                        'url': f'/api/documents/{doc_id}/download',
                        'filename': source.get('filename', ''),
                        'title': source.get('filename', '')
                    })
                    processed_urls.add(doc_id)
            
            # Check for PDFs
            elif any(filename.endswith(ext) for ext in self.pdf_extensions):
                doc_id = source.get('doc_id', '')
                if doc_id and doc_id not in processed_urls:
                    attachments.append({
                        'type': 'pdf',
                        'url': f'/api/documents/{doc_id}/download',
                        'filename': source.get('filename', ''),
                        'title': source.get('filename', '')
                    })
                    processed_urls.add(doc_id)
        
        return attachments
    
    async def _create_citations(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create formatted citations from sources."""
        citations = []
        
        for idx, source in enumerate(sources, 1):
            citation = {
                'id': f'cite-{idx}',
                'number': idx,
                'filename': source.get('filename', 'Unknown'),
                'similarity': round(source.get('similarity', 0) * 100, 1),
                'text_preview': source.get('text', '')[:100] + '...' if source.get('text') else '',
                'page': source.get('page', None),
                'doc_id': source.get('doc_id', '')
            }
            citations.append(citation)
        
        return citations
    
    async def _add_inline_citations(
        self,
        text: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """Add inline citations to response text."""
        if not sources:
            return text
        
        # Create citation mapping
        citation_map = {}
        for idx, source in enumerate(sources, 1):
            citation_map[idx] = {
                'filename': source.get('filename', 'Unknown'),
                'similarity': round(source.get('similarity', 0) * 100, 1)
            }
        
        # Add footer with citations
        citation_footer = self._create_citation_footer(citation_map)
        formatted_text = f"{text}\n\n{citation_footer}"
        
        return formatted_text
    
    def _create_citation_footer(self, citation_map: Dict[int, Dict[str, Any]]) -> str:
        """Create formatted citation footer."""
        footer_lines = ["---", "**Sources:**"]
        
        for idx, citation in citation_map.items():
            line = f"[{idx}] {citation['filename']} (Similarity: {citation['similarity']}%)"
            footer_lines.append(line)
        
        return "\n".join(footer_lines)
    
    def _shorten_url(self, url: str, max_length: int = 50) -> str:
        """Shorten URL for display."""
        if len(url) > max_length:
            # Try to extract domain
            match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if match:
                return match.group(1) + '/...'
        return url
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    async def format_json_response(
        self,
        query: str,
        json_data: Dict[str, Any],
        extracted_data: Dict[str, Any]
    ) -> str:
        """
        Format a response for JSON document queries.
        
        Args:
            query: User query
            json_data: Raw JSON data
            extracted_data: Extracted structured data
            
        Returns:
            Formatted response text
        """
        response_parts = []
        
        # Add query acknowledgement
        response_parts.append(f"Based on your query: '{query}'\n")
        
        # Add relevant information
        if 'records' in extracted_data:
            records = extracted_data['records']
            response_parts.append(f"Found {len(records)} relevant record(s):\n")
            
            # Show first few records
            for idx, record in enumerate(records[:3]):
                response_parts.append(f"**Record {idx + 1}:**")
                if isinstance(record, dict):
                    for key, value in record.items():
                        response_parts.append(f"  • {key}: {value}")
                else:
                    response_parts.append(f"  {record}")
        
        # Add fields information
        if 'fields' in extracted_data:
            response_parts.append(f"\n**Available Fields:**")
            for field, field_type in extracted_data['fields'].items():
                response_parts.append(f"  • {field} ({field_type})")
        
        return "\n".join(response_parts)
    
    async def format_streaming_response(
        self,
        response_generator
    ):
        """
        Format streaming response token by token.
        
        Args:
            response_generator: Async generator yielding tokens
            
        Yields:
            Formatted tokens with metadata
        """
        buffer = ""
        attachment_buffer = []
        
        async for token in response_generator:
            buffer += token
            
            # Check for media patterns
            if '```' in buffer or '![' in buffer or '[link](' in buffer:
                # Process buffer
                processed, attachments = await self._process_buffer(buffer)
                if processed:
                    yield processed
                    buffer = ""
                attachment_buffer.extend(attachments)
            
            # Yield regular tokens
            yield token


# Singleton instance
_response_formatter = None


async def get_response_formatter() -> ResponseFormatterService:
    """Get or create response formatter instance."""
    global _response_formatter
    if _response_formatter is None:
        _response_formatter = ResponseFormatterService()
    return _response_formatter
