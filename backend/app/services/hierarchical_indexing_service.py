"""
Hierarchical Document Indexing Service for Advanced RAG.
This module implements multi-level chunking (document → section → paragraph → sentence)
and enriched metadata tagging for improved retrieval quality.
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

import spacy
from sentence_transformers import SentenceTransformer
import tiktoken
from bs4 import BeautifulSoup

from ..core.cache_redis import get_redis_client, RedisClient
from ..config import settings

logger = logging.getLogger(__name__)

class HierarchicalIndexingService:
    """Service for hierarchical document indexing and metadata enrichment."""
    
    def __init__(self):
        self.nlp_model = None
        self.tokenizer = None
        self.redis_client = None
    
    async def initialize(self):
        """Initialize the hierarchical indexing service."""
        try:
            logger.info("Initializing hierarchical indexing service...")
            
            # Get Redis client for caching
            self.redis_client = await get_redis_client()
            
            # Initialize NLP model
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except IOError:
                logger.warning("spaCy model not found. Using fallback methods.")
                self.nlp_model = None
            
            # Initialize tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            logger.info("Hierarchical indexing service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hierarchical indexing service: {e}")
            raise
    
    async def create_hierarchical_chunks(self, text: str, doc_id: str, filename: str, 
                                       document_type: str) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks with multi-level indexing.
        
        Args:
            text: Document text content
            doc_id: Document identifier
            filename: Original filename
            document_type: Type of document (pdf, docx, etc.)
        
        Returns:
            List of hierarchically structured chunks
        """
        try:
            # Step 1: Document-level analysis
            doc_metadata = await self._analyze_document_structure(text, filename, document_type)
            
            # Step 2: Extract hierarchical structure
            hierarchy = await self._extract_document_hierarchy(text, document_type)
            
            # Step 3: Create multi-level chunks
            hierarchical_chunks = []
            
            for section_idx, section in enumerate(hierarchy):
                section_chunks = await self._process_section(
                    section, doc_id, section_idx, doc_metadata
                )
                hierarchical_chunks.extend(section_chunks)
            
            # Step 4: Add document-level summary chunks
            if len(hierarchical_chunks) > 5:  # Only for substantial documents
                summary_chunks = await self._create_summary_chunks(
                    text, doc_id, doc_metadata, hierarchical_chunks
                )
                hierarchical_chunks.extend(summary_chunks)
            
            logger.info(f"Created {len(hierarchical_chunks)} hierarchical chunks from {len(hierarchy)} sections")
            return hierarchical_chunks
            
        except Exception as e:
            logger.error(f"Hierarchical chunking failed: {e}")
            # Fallback to simple chunking
            return await self._fallback_chunking(text, doc_id)
    
    async def _analyze_document_structure(self, text: str, filename: str, 
                                        document_type: str) -> Dict[str, Any]:
        """Analyze overall document structure and characteristics."""
        
        # Basic document analysis
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]+', text))
        paragraph_count = len([p for p in text.split('\n\n') if len(p.strip()) > 10])
        
        # Detect document characteristics
        has_headings = bool(re.search(r'^#{1,6}\s+.+|^[A-Z][A-Za-z\s]{2,50}$', text, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*•]\s+|^\s*\d+\.\s+', text, re.MULTILINE))
        has_code = bool(re.search(r'```|`[^`]+`|def\s+\w+|class\s+\w+', text))
        
        # Complexity analysis
        avg_sentence_length = word_count / max(sentence_count, 1)
        complexity_score = self._calculate_complexity_score(text, avg_sentence_length)
        
        # Extract topics/themes
        topics = self._extract_document_topics(text)
        
        return {
            "filename": filename,
            "document_type": document_type,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "has_headings": has_headings,
            "has_lists": has_lists,
            "has_code": has_code,
            "avg_sentence_length": avg_sentence_length,
            "complexity_score": complexity_score,
            "topics": topics,
            "processing_date": datetime.utcnow().isoformat(),
            "chunk_strategy": self._determine_chunk_strategy(
                document_type, has_headings, has_lists, has_code, complexity_score
            )
        }
    
    async def _extract_document_hierarchy(self, text: str, document_type: str) -> List[Dict[str, Any]]:
        """Extract hierarchical document structure (sections, subsections, etc.)."""
        
        if document_type in ['html', 'xml']:
            return self._extract_html_hierarchy(text)
        elif document_type in ['md', 'markdown']:
            return self._extract_markdown_hierarchy(text)
        else:
            return self._extract_text_hierarchy(text)
    
    def _extract_html_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """Extract hierarchy from HTML content."""
        try:
            soup = BeautifulSoup(text, 'html.parser')
            sections = []
            
            # Find all heading elements
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            for i, heading in enumerate(headings):
                # Extract content until next heading
                content_elements = []
                current = heading.next_sibling
                
                while current and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if hasattr(current, 'get_text'):
                        content_elements.append(current.get_text())
                    elif isinstance(current, str) and current.strip():
                        content_elements.append(current)
                    current = current.next_sibling
                
                section_text = ' '.join(content_elements).strip()
                
                if section_text:
                    sections.append({
                        "title": heading.get_text().strip(),
                        "text": section_text,
                        "level": int(heading.name[1]),  # h1 -> 1, h2 -> 2, etc.
                        "type": "heading_section",
                        "position": i
                    })
            
            # If no headings found, split by paragraphs
            if not sections:
                paragraphs = soup.find_all('p')
                for i, p in enumerate(paragraphs):
                    text = p.get_text().strip()
                    if len(text) > 50:  # Only substantial paragraphs
                        sections.append({
                            "title": f"Paragraph {i+1}",
                            "text": text,
                            "level": 1,
                            "type": "paragraph",
                            "position": i
                        })
            
            return sections
            
        except Exception as e:
            logger.warning(f"HTML hierarchy extraction failed: {e}")
            return self._extract_text_hierarchy(text)
    
    def _extract_markdown_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """Extract hierarchy from Markdown content."""
        sections = []
        lines = text.split('\n')
        current_section = {"title": "", "text": [], "level": 1, "type": "section", "position": 0}
        
        for line_num, line in enumerate(lines):
            # Check for markdown headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if heading_match:
                # Save previous section if it has content
                if current_section["text"] and current_section["title"]:
                    current_section["text"] = '\n'.join(current_section["text"]).strip()
                    if current_section["text"]:
                        sections.append(current_section.copy())
                
                # Start new section
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                current_section = {
                    "title": title,
                    "text": [],
                    "level": level,
                    "type": "heading_section",
                    "position": len(sections)
                }
            else:
                # Add content to current section
                if line.strip():
                    current_section["text"].append(line)
        
        # Add final section
        if current_section["text"]:
            current_section["text"] = '\n'.join(current_section["text"]).strip()
            if current_section["text"]:
                sections.append(current_section)
        
        return sections if sections else self._extract_text_hierarchy(text)
    
    def _extract_text_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """Extract hierarchy from plain text using heuristics."""
        sections = []
        
        # First, try to split by obvious section markers
        section_patterns = [
            r'\n\s*(?:[A-Z][A-Z\s]{5,50})\s*\n',  # ALL CAPS headings
            r'\n\s*(?:\d+\.?\s+[A-Z][^.!?]*[.!?]?)\s*\n',  # Numbered sections
            r'\n\s*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:)\s*\n',  # Title Case with colon
        ]
        
        # Try each pattern
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            if len(matches) >= 2:  # At least 2 sections found
                return self._split_by_pattern_matches(text, matches)
        
        # Fallback: split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        
        sections = []
        for i, paragraph in enumerate(paragraphs):
            # Extract potential title from first sentence
            sentences = re.split(r'[.!?]+', paragraph)
            title = sentences[0][:50] + "..." if len(sentences[0]) > 50 else sentences[0]
            
            sections.append({
                "title": title or f"Section {i+1}",
                "text": paragraph,
                "level": 1,
                "type": "paragraph_section",
                "position": i
            })
        
        return sections
    
    def _split_by_pattern_matches(self, text: str, matches: List) -> List[Dict[str, Any]]:
        """Split text by regex pattern matches."""
        sections = []
        
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            title = match.group(0).strip()
            content = text[start:end].strip()
            
            if content:
                sections.append({
                    "title": title,
                    "text": content,
                    "level": 1,
                    "type": "pattern_section",
                    "position": i
                })
        
        return sections
    
    async def _process_section(self, section: Dict[str, Any], doc_id: str, 
                             section_idx: int, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single section into multiple chunk levels."""
        
        section_text = section["text"]
        section_title = section["title"]
        
        # Create chunks based on the document's chunk strategy
        strategy = doc_metadata.get("chunk_strategy", "balanced")
        
        if strategy == "semantic":
            return await self._create_semantic_chunks(section, doc_id, section_idx, doc_metadata)
        elif strategy == "sentence":
            return await self._create_sentence_chunks(section, doc_id, section_idx, doc_metadata)
        elif strategy == "paragraph":
            return await self._create_paragraph_chunks(section, doc_id, section_idx, doc_metadata)
        else:
            return await self._create_balanced_chunks(section, doc_id, section_idx, doc_metadata)
    
    async def _create_balanced_chunks(self, section: Dict[str, Any], doc_id: str,
                                    section_idx: int, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create balanced chunks with multiple granularities."""
        
        chunks = []
        section_text = section["text"]
        section_title = section["title"]
        
        # Split into paragraphs first
        paragraphs = [p.strip() for p in section_text.split('\n\n') if len(p.strip()) > 20]
        
        paragraph_idx = 0
        for paragraph in paragraphs:
            # Check if paragraph is small enough to be one chunk
            token_count = len(self.tokenizer.encode(paragraph))
            
            if token_count <= settings.chunk_size:
                # Single paragraph chunk
                chunk_metadata = self._create_chunk_metadata(
                    doc_metadata, section, paragraph_idx, "paragraph", paragraph
                )
                
                chunks.append({
                    "text": paragraph,
                    "tokens": token_count,
                    "chunk_index": len(chunks),
                    "section_index": section_idx,
                    "paragraph_index": paragraph_idx,
                    "chunk_type": "paragraph",
                    "metadata": chunk_metadata
                })
            else:
                # Split paragraph into sentences
                sentences = self._split_into_sentences(paragraph)
                sentence_chunks = self._group_sentences_into_chunks(
                    sentences, section, section_idx, paragraph_idx, doc_metadata
                )
                chunks.extend(sentence_chunks)
            
            paragraph_idx += 1
        
        return chunks
    
    async def _create_semantic_chunks(self, section: Dict[str, Any], doc_id: str,
                                    section_idx: int, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks based on semantic boundaries."""
        
        if not self.nlp_model:
            return await self._create_balanced_chunks(section, doc_id, section_idx, doc_metadata)
        
        chunks = []
        section_text = section["text"]
        
        # Use spaCy to find semantic boundaries
        doc = self.nlp_model(section_text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        # Group semantically related sentences
        semantic_chunks = self._group_semantic_sentences(sentences)
        
        for chunk_idx, semantic_chunk in enumerate(semantic_chunks):
            chunk_text = ' '.join(semantic_chunk)
            token_count = len(self.tokenizer.encode(chunk_text))
            
            chunk_metadata = self._create_chunk_metadata(
                doc_metadata, section, chunk_idx, "semantic", chunk_text
            )
            
            chunks.append({
                "text": chunk_text,
                "tokens": token_count,
                "chunk_index": len(chunks),
                "section_index": section_idx,
                "chunk_type": "semantic",
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _group_semantic_sentences(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences by semantic similarity."""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = [sentences[0]]
        current_tokens = len(self.tokenizer.encode(sentences[0]))
        
        for sentence in sentences[1:]:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # Check if adding this sentence would exceed token limit
            if current_tokens + sentence_tokens > settings.chunk_size:
                chunks.append(current_chunk)
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_chunk_metadata(self, doc_metadata: Dict[str, Any], section: Dict[str, Any],
                             chunk_idx: int, chunk_type: str, text: str) -> Dict[str, Any]:
        """Create enriched metadata for a chunk."""
        
        # Extract entities if NLP model is available
        entities = []
        if self.nlp_model:
            doc = self.nlp_model(text[:500])  # Limit text for entity extraction
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        
        # Extract keywords
        keywords = self._extract_keywords(text)
        
        # Calculate readability metrics
        readability = self._calculate_readability(text)
        
        return {
            **doc_metadata,
            "section_title": section.get("title", ""),
            "section_level": section.get("level", 1),
            "section_type": section.get("type", "unknown"),
            "chunk_type": chunk_type,
            "chunk_index": chunk_idx,
            "entities": entities,
            "keywords": keywords,
            "readability_score": readability,
            "language": "en",  # Could be detected dynamically
            "has_code": bool(re.search(r'```|`[^`]+`|def\s+\w+|class\s+\w+', text)),
            "has_urls": bool(re.search(r'https?://\S+', text)),
            "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            "word_count": len(text.split()),
            "char_count": len(text)
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction (could be enhanced with TF-IDF or other methods)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 
            'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 
            'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 
            'way', 'what', 'when', 'where', 'why', 'will', 'with', 'have', 'this', 'that',
            'they', 'from', 'been', 'said', 'each', 'which', 'there', 'would', 'make'
        }
        
        keywords = [word for word in set(words) if word not in stop_words and len(word) > 3]
        
        # Return most frequent keywords (simple frequency count)
        word_freq = {word: words.count(word) for word in keywords}
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_keywords[:10]]
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate a simple readability score."""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        characters = len(re.sub(r'[^a-zA-Z]', '', text))
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease approximation
        avg_sentence_length = words / sentences
        avg_syllables = characters / words  # Rough syllable approximation
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0.0, min(100.0, score))  # Clamp between 0-100
    
    def _calculate_complexity_score(self, text: str, avg_sentence_length: float) -> float:
        """Calculate document complexity score."""
        
        # Factors that increase complexity
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', text))  # Acronyms
        long_words = len(re.findall(r'\b\w{10,}\b', text))  # Long words
        numbers = len(re.findall(r'\b\d+\b', text))  # Numbers
        
        # Normalize by text length
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        complexity = (
            (technical_terms / word_count) * 0.3 +
            (long_words / word_count) * 0.3 +
            (numbers / word_count) * 0.2 +
            min(avg_sentence_length / 20.0, 1.0) * 0.2  # Sentence length factor
        )
        
        return min(complexity, 1.0)  # Cap at 1.0
    
    def _extract_document_topics(self, text: str) -> List[str]:
        """Extract main topics/themes from document."""
        # Simple topic extraction using keyword analysis
        # Could be enhanced with more sophisticated NLP techniques
        
        # Extract potential topic words (nouns, proper nouns)
        if self.nlp_model:
            doc = self.nlp_model(text[:2000])  # Limit for performance
            topic_words = [token.lemma_.lower() for token in doc 
                          if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2]
        else:
            # Fallback: extract capitalized words and longer words
            topic_words = re.findall(r'\b[A-Z][a-z]{2,}\b|\b[a-z]{5,}\b', text)
            topic_words = [word.lower() for word in topic_words]
        
        # Count frequency and return top topics
        from collections import Counter
        word_freq = Counter(topic_words)
        return [word for word, count in word_freq.most_common(5)]
    
    def _determine_chunk_strategy(self, document_type: str, has_headings: bool, 
                                has_lists: bool, has_code: bool, complexity_score: float) -> str:
        """Determine the best chunking strategy based on document characteristics."""
        
        if has_code:
            return "semantic"  # Preserve code blocks
        elif document_type in ['html', 'xml', 'md']:
            return "semantic"  # Use document structure
        elif has_headings:
            return "paragraph"  # Respect heading structure
        elif complexity_score > 0.7:
            return "sentence"  # Break down complex content
        else:
            return "balanced"  # Default balanced approach
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self.nlp_model:
            doc = self.nlp_model(text)
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        else:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def _group_sentences_into_chunks(self, sentences: List[str], section: Dict[str, Any],
                                   section_idx: int, paragraph_idx: int, 
                                   doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group sentences into appropriately sized chunks."""
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens > settings.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = self._create_chunk_metadata(
                    doc_metadata, section, len(chunks), "sentence_group", chunk_text
                )
                
                chunks.append({
                    "text": chunk_text,
                    "tokens": current_tokens,
                    "chunk_index": len(chunks),
                    "section_index": section_idx,
                    "paragraph_index": paragraph_idx,
                    "chunk_type": "sentence_group",
                    "metadata": chunk_metadata
                })
                
                # Start new chunk
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = self._create_chunk_metadata(
                doc_metadata, section, len(chunks), "sentence_group", chunk_text
            )
            
            chunks.append({
                "text": chunk_text,
                "tokens": current_tokens,
                "chunk_index": len(chunks),
                "section_index": section_idx,
                "paragraph_index": paragraph_idx,
                "chunk_type": "sentence_group",
                "metadata": chunk_metadata
            })
        
        return chunks
    
    async def _create_summary_chunks(self, text: str, doc_id: str, doc_metadata: Dict[str, Any],
                                   existing_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create document-level summary chunks."""
        
        # Create a document summary (first and last parts)
        words = text.split()
        
        # Beginning summary (first 200 words)
        beginning_summary = ' '.join(words[:200])
        
        # Ending summary (last 100 words)  
        ending_summary = ' '.join(words[-100:]) if len(words) > 100 else ""
        
        summary_chunks = []
        
        # Beginning chunk
        if beginning_summary:
            summary_metadata = {
                **doc_metadata,
                "chunk_type": "document_summary",
                "summary_type": "beginning",
                "original_chunk_count": len(existing_chunks)
            }
            
            summary_chunks.append({
                "text": beginning_summary,
                "tokens": len(self.tokenizer.encode(beginning_summary)),
                "chunk_index": -1,  # Special index for summaries
                "chunk_type": "document_summary",
                "metadata": summary_metadata
            })
        
        # Ending chunk
        if ending_summary and len(words) > 300:  # Only for longer documents
            summary_metadata = {
                **doc_metadata,
                "chunk_type": "document_summary",
                "summary_type": "ending",
                "original_chunk_count": len(existing_chunks)
            }
            
            summary_chunks.append({
                "text": ending_summary,
                "tokens": len(self.tokenizer.encode(ending_summary)),
                "chunk_index": -2,  # Special index for summaries
                "chunk_type": "document_summary", 
                "metadata": summary_metadata
            })
        
        return summary_chunks
    
    async def _fallback_chunking(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Simple fallback chunking when hierarchical processing fails."""
        
        chunks = []
        chunk_size = settings.chunk_size
        tokens = self.tokenizer.encode(text)
        
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if len(chunk_text.strip()) > 10:
                chunks.append({
                    "text": chunk_text.strip(),
                    "tokens": len(chunk_tokens),
                    "chunk_index": len(chunks),
                    "chunk_type": "fallback",
                    "metadata": {
                        "document_type": "unknown",
                        "chunk_type": "fallback",
                        "processing_date": datetime.utcnow().isoformat()
                    }
                })
            
            start_idx = end_idx - settings.chunk_overlap
        
        return chunks

# Global instance
hierarchical_indexing_service = HierarchicalIndexingService()