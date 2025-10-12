"""
Chat service for processing queries through the RAG pipeline.
This module orchestrates retrieval, graph queries, reranking, and LLM generation
to provide accurate and contextual responses.
"""

import asyncio
import logging
import httpx
import json
import hashlib
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import time
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy

from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..core.db_qdrant import get_qdrant_client, QdrantDBClient
from ..core.db_arango import get_arango_client, ArangoDBClient
from ..core.cache_redis import get_redis_client, RedisClient
from ..config import settings

logger = logging.getLogger(__name__)

class ChatService:
    """Service for processing chat queries through RAG pipeline."""
    
    def __init__(self):
        self.embedding_model = None
        self.reranker_model = None
        self.nlp_model = None
        self.mongo_client = None
        self.qdrant_client = None
        self.arango_client = None
        self.redis_client = None
        self.http_client = None
    
    async def initialize(self):
        """Initialize the chat service with required models and clients."""
        try:
            logger.info("Initializing chat service...")
            
            # Get database clients
            self.mongo_client = await get_mongodb_client()
            self.qdrant_client = await get_qdrant_client()
            self.arango_client = await get_arango_client()
            self.redis_client = await get_redis_client()
            
            # Initialize HTTP client for LMStudio
            self.http_client = httpx.AsyncClient(timeout=120.0)
            
            # Initialize embedding model
            logger.info(f"Loading embedding model: {settings.embedding_model_name}")
            self.embedding_model = SentenceTransformer(settings.embedding_model_name)
            
            # Initialize reranker if enabled
            if settings.use_reranker:
                logger.info(f"Loading reranker model: {settings.reranker_model_name}")
                self.reranker_model = CrossEncoder(settings.reranker_model_name)
            
            # Initialize NLP model for entity extraction
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except IOError:
                logger.warning("spaCy model not found. Entity extraction will be limited.")
                self.nlp_model = None
            
            logger.info("Chat service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat service: {e}")
            raise
    
    async def process_query(self, query: str, session_id: str, context: List[Dict] = None) -> Dict[str, Any]:
        """
        Process a chat query through the complete RAG pipeline.
        
        Args:
            query: User's query
            session_id: Session identifier
            context: Previous conversation context
        
        Returns:
            Dictionary with response, sources, and attachments
        """
        try:
            start_time = time.time()
            
            # Step 1: Query preprocessing and enhancement
            processed_query = await self._preprocess_query(query)
            logger.info(f"Processing query: {processed_query[:100]}...")
            
            # Step 2: Generate query embedding
            query_embedding = await self._generate_query_embedding(processed_query)
            
            # Step 3: Retrieve relevant chunks from vector database
            vector_results = await self.qdrant_client.search_similar(
                query_vector=query_embedding,
                top_k=settings.top_k_retrieval * 2  # Get more for reranking
            )
            
            # Step 4: Extract entities from query for graph search
            graph_results = []
            if settings.use_graph_search and self.nlp_model:
                entities = self._extract_entities_from_query(processed_query)
                if entities:
                    graph_results = await self.arango_client.find_related_entities(
                        entities, max_depth=2, limit=5
                    )
            
            # Step 5: Merge and rerank results
            merged_results = await self._merge_and_rerank_results(
                vector_results, graph_results, processed_query
            )
            
            # Step 6: Build context for LLM
            context_text, sources = await self._build_llm_context(merged_results, processed_query)
            
            # Step 7: Generate response using LMStudio
            response = await self._generate_llm_response(processed_query, context_text, context)
            
            # Step 8: Extract attachments from sources
            attachments = self._extract_attachments(sources)
            
            # Step 9: Store conversation in session
            await self._store_conversation_turn(session_id, query, response, sources)
            
            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f}s")
            
            return {
                "response": response,
                "sources": sources,
                "attachments": attachments,
                "processing_time": processing_time,
                "tokens_generated": len(response.split())  # Rough token estimate
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your query. Please try again.",
                "sources": [],
                "attachments": [],
                "processing_time": 0,
                "tokens_generated": 0
            }
    
    async def stream_query(self, query: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream query processing with real-time updates.
        
        Args:
            query: User's query
            session_id: Session identifier
        
        Yields:
            Dictionary chunks with streaming data
        """
        try:
            # Step 1: Preprocessing
            yield {"type": "status", "content": "Processing query..."}
            processed_query = await self._preprocess_query(query)
            
            # Step 2: Retrieval
            yield {"type": "status", "content": "Searching knowledge base..."}
            query_embedding = await self._generate_query_embedding(processed_query)
            vector_results = await self.qdrant_client.search_similar(
                query_vector=query_embedding,
                top_k=settings.top_k_retrieval * 2
            )
            
            # Step 3: Graph search (if enabled)
            graph_results = []
            if settings.use_graph_search and self.nlp_model:
                yield {"type": "status", "content": "Analyzing relationships..."}
                entities = self._extract_entities_from_query(processed_query)
                if entities:
                    graph_results = await self.arango_client.find_related_entities(entities)
            
            # Step 4: Context building
            yield {"type": "status", "content": "Building context..."}
            merged_results = await self._merge_and_rerank_results(
                vector_results, graph_results, processed_query
            )
            context_text, sources = await self._build_llm_context(merged_results, processed_query)
            
            # Step 5: Generate streaming response
            yield {"type": "status", "content": "Generating response..."}
            
            full_response = ""
            async for token in self._stream_llm_response(processed_query, context_text):
                full_response += token
                yield {"type": "token", "content": token}
            
            # Step 6: Send final data
            attachments = self._extract_attachments(sources)
            await self._store_conversation_turn(session_id, query, full_response, sources)
            
            yield {
                "type": "sources",
                "sources": sources,
                "attachments": attachments
            }
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield {
                "type": "error",
                "content": "I encountered an error while processing your query."
            }
    
    async def _preprocess_query(self, query: str) -> str:
        """Preprocess and enhance the user query."""
        # Clean the query
        query = query.strip()
        
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Add question mark if missing for question-type queries
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(query.lower().startswith(word) for word in question_words) and not query.endswith('?'):
            query += '?'
        
        return query
    
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query with caching."""
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cached_embedding = await self.redis_client.get_cached_embedding(query_hash)
        
        if cached_embedding:
            return cached_embedding
        
        # Generate new embedding
        embedding = self.embedding_model.encode(query).tolist()
        
        # Cache for future use
        await self.redis_client.cache_embedding(query_hash, embedding, 24)
        
        return embedding
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract named entities from the query."""
        if not self.nlp_model:
            return []
        
        doc = self.nlp_model(query)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                entities.append(ent.text)
        
        return entities
    
    async def _merge_and_rerank_results(self, vector_results: List[Dict], 
                                       graph_results: List[Dict], 
                                       query: str) -> List[Dict]:
        """Merge vector and graph results and rerank them."""
        try:
            # Start with vector results
            merged_results = vector_results.copy()
            
            # Add graph results with adjusted scores
            for graph_result in graph_results:
                # Convert graph result to similar format
                doc_ids = graph_result.get("doc_ids", [])
                for doc_id in doc_ids:
                    # Check if we already have chunks from this document
                    existing = next((r for r in merged_results if r.get("doc_id") == doc_id), None)
                    if not existing:
                        # Add a placeholder result for graph-discovered documents
                        merged_results.append({
                            "id": f"graph_{doc_id}",
                            "score": 0.5,  # Medium confidence for graph results
                            "doc_id": doc_id,
                            "text": f"Related to: {graph_result.get('entity_name', 'Unknown')}",
                            "metadata": {
                                "source": "graph",
                                "entity_name": graph_result.get("entity_name"),
                                "relation_type": graph_result.get("relation_type")
                            }
                        })
            
            # Rerank if reranker is available
            if self.reranker_model and settings.use_reranker:
                reranked_results = await self._rerank_results(merged_results, query)
                return reranked_results[:settings.top_k_retrieval]
            
            # Sort by score and return top results
            merged_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return merged_results[:settings.top_k_retrieval]
            
        except Exception as e:
            logger.error(f"Result merging failed: {e}")
            return vector_results[:settings.top_k_retrieval]
    
    async def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Rerank results using cross-encoder model."""
        try:
            if not results:
                return results
            
            # Prepare query-document pairs for reranking
            pairs = []
            for result in results:
                text = result.get("text", "")
                pairs.append([query, text])
            
            # Get reranking scores
            scores = self.reranker_model.predict(pairs)
            
            # Update scores and sort
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])
                result["original_score"] = result.get("score", 0)
                # Combine original and rerank scores
                result["score"] = 0.7 * float(scores[i]) + 0.3 * result.get("score", 0)
            
            # Sort by new combined score
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            logger.info(f"Reranked {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results
    
    async def _build_llm_context(self, results: List[Dict], query: str) -> tuple[str, List[Dict]]:
        """Build context for LLM and prepare source metadata."""
        if not results:
            return "No relevant information found.", []
        
        context_parts = []
        sources = []
        
        for i, result in enumerate(results):
            # Get document metadata
            doc_id = result.get("doc_id", "")
            chunk_text = result.get("text", "")
            
            if doc_id and chunk_text:
                # Get document info from cache or database
                doc_metadata = await self.redis_client.get_cached_document_metadata(doc_id)
                if not doc_metadata:
                    document = await self.mongo_client.get_document(doc_id)
                    if document:
                        doc_metadata = {
                            "filename": document.get("filename", "Unknown"),
                            "file_type": document.get("file_type", ""),
                            "uploaded_at": document.get("uploaded_at", "").isoformat() if document.get("uploaded_at") else ""
                        }
                        await self.redis_client.cache_document_metadata(doc_id, doc_metadata)
                
                # Add to context
                source_label = f"[Source {i+1}]"
                context_parts.append(f"{source_label} {chunk_text}")
                
                # Add to sources list
                sources.append({
                    "id": result.get("id", ""),
                    "doc_id": doc_id,
                    "chunk_id": result.get("chunk_id", ""),
                    "filename": doc_metadata.get("filename", "Unknown") if doc_metadata else "Unknown",
                    "text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    "score": result.get("score", 0),
                    "source_label": source_label
                })
        
        context_text = "\n\n".join(context_parts)
        return context_text, sources
    
    async def _generate_llm_response(self, query: str, context: str, conversation_context: List[Dict] = None) -> str:
        """Generate response using LMStudio."""
        try:
            # Build the prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, context, conversation_context)
            
            # Prepare the request
            payload = {
                "model": "mistral-3b",  # This should match your loaded model
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": settings.default_temperature,
                "top_p": settings.default_top_p,
                "max_tokens": 512,
                "stream": False
            }
            
            # Add API key if available
            headers = {"Content-Type": "application/json"}
            if settings.lmstudio_api_key:
                headers["Authorization"] = f"Bearer {settings.lmstudio_api_key}"
            
            # Make the request
            response = await self.http_client.post(
                settings.lmstudio_api_url,
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the response text
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError("Invalid response format from LMStudio")
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    async def _stream_llm_response(self, query: str, context: str) -> AsyncGenerator[str, None]:
        """Stream response from LMStudio token by token."""
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, context)
            
            payload = {
                "model": "mistral-3b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": settings.default_temperature,
                "top_p": settings.default_top_p,
                "max_tokens": 512,
                "stream": True
            }
            
            headers = {"Content-Type": "application/json"}
            if settings.lmstudio_api_key:
                headers["Authorization"] = f"Bearer {settings.lmstudio_api_key}"
            
            async with self.http_client.stream(
                "POST",
                settings.lmstudio_api_url,
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming response failed: {e}")
            yield "I apologize, but I'm having trouble generating a response right now."
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return """You are a helpful AI assistant that answers questions based on the provided context. 

IMPORTANT INSTRUCTIONS:
1. Always base your answers on the information provided in the context
2. If the context doesn't contain relevant information, say "I don't have enough information to answer that question"
3. When referencing information, mention the source (e.g., "According to Source 1...")
4. Be concise but thorough in your responses
5. If asked about something not in the context, politely redirect to what you can help with
6. Always maintain a professional and helpful tone

Remember: Your knowledge is limited to the context provided. Do not make up information."""
    
    def _build_user_prompt(self, query: str, context: str, conversation_context: List[Dict] = None) -> str:
        """Build the user prompt with context."""
        prompt_parts = []
        
        # Add conversation context if available
        if conversation_context:
            prompt_parts.append("Previous conversation:")
            for msg in conversation_context[-3:]:  # Last 3 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.title()}: {content}")
            prompt_parts.append("")
        
        # Add the context
        prompt_parts.append("Context information:")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Add the current query
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("")
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)
    
    def _extract_attachments(self, sources: List[Dict]) -> List[Dict]:
        """Extract multimedia attachments from sources."""
        attachments = []
        
        for source in sources:
            filename = source.get("filename", "").lower()
            doc_id = source.get("doc_id", "")
            
            # Check for different media types
            if any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                attachments.append({
                    "type": "image",
                    "url": f"/admin/documents/{doc_id}/download",
                    "filename": source.get("filename", ""),
                    "title": source.get("filename", "")
                })
            elif any(filename.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.wmv']):
                attachments.append({
                    "type": "video",
                    "url": f"/admin/documents/{doc_id}/download",
                    "filename": source.get("filename", ""),
                    "title": source.get("filename", "")
                })
            elif filename.endswith('.pdf'):
                attachments.append({
                    "type": "pdf",
                    "url": f"/admin/documents/{doc_id}/download",
                    "filename": source.get("filename", ""),
                    "title": source.get("filename", "")
                })
            
            # Check for YouTube links in text
            text = source.get("text", "")
            youtube_regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
            matches = re.findall(youtube_regex, text)
            
            for video_id in matches:
                attachments.append({
                    "type": "youtube",
                    "videoId": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": f"YouTube Video: {video_id}"
                })
        
        return attachments
    
    async def _store_conversation_turn(self, session_id: str, query: str, response: str, sources: List[Dict]):
        """Store the conversation turn in session data."""
        try:
            # Get existing session data
            session_data = await self.redis_client.get_session_data(session_id) or {"messages": []}
            
            # Add new messages
            session_data["messages"].extend([
                {
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.utcnow().isoformat(),
                    "sources": sources
                }
            ])
            
            # Keep only last 20 messages to prevent memory issues
            if len(session_data["messages"]) > 20:
                session_data["messages"] = session_data["messages"][-20:]
            
            session_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Store back in Redis
            await self.redis_client.store_session_data(session_id, session_data, 8)
            
        except Exception as e:
            logger.error(f"Failed to store conversation turn: {e}")
    
    async def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions based on partial input."""
        try:
            # Simple suggestions based on common patterns
            suggestions = []
            
            lower_query = partial_query.lower()
            
            # Common question patterns
            if lower_query.startswith("what"):
                suggestions.extend([
                    "What is the refund policy?",
                    "What are the requirements?",
                    "What is the process for?"
                ])
            elif lower_query.startswith("how"):
                suggestions.extend([
                    "How do I apply for?",
                    "How can I contact support?",
                    "How long does it take?"
                ])
            elif lower_query.startswith("when"):
                suggestions.extend([
                    "When is the deadline?",
                    "When will I receive?"
                ])
            else:
                # General suggestions
                suggestions.extend([
                    "What is your refund policy?",
                    "How do I contact customer support?",
                    "What are your business hours?",
                    "How can I track my order?",
                    "What payment methods do you accept?"
                ])
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Perform health check of chat service components."""
        try:
            # Test embedding model
            test_embedding = self.embedding_model.encode("test query")
            if len(test_embedding) != settings.embedding_dimension:
                return False
            
            # Test LMStudio connection
            try:
                response = await self.http_client.get(settings.lmstudio_api_url.replace("/v1/chat/completions", "/health"))
                # If health endpoint doesn't exist, try a simple completion
                if response.status_code == 404:
                    # Test with minimal request
                    test_payload = {
                        "model": "mistral-3b",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1
                    }
                    test_response = await self.http_client.post(
                        settings.lmstudio_api_url,
                        json=test_payload,
                        timeout=10.0
                    )
                    return test_response.status_code == 200
                else:
                    return response.status_code == 200
            except:
                # LMStudio connection failed
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()