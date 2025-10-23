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
from rank_bm25 import BM25Okapi

from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..core.db_qdrant import get_qdrant_client, QdrantDBClient
from ..core.db_arango import get_arango_client, ArangoDBClient
from ..core.cache_redis import get_redis_client, RedisClient
from .query_intelligence_service import query_intelligence_service
from .context_optimization_service import context_optimization_service
from .evaluation_service import evaluation_service
from .llm_handler import llm_handler
from .prompt_service import prompt_service
from .response_formatter_service import get_response_formatter
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
        self.bm25_index = None
        self.document_texts = []
    
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
            
            # Initialize query intelligence service
            await query_intelligence_service.initialize()
            
            # Initialize context optimization service
            await context_optimization_service.initialize()
            
            # Initialize evaluation service
            await evaluation_service.initialize()
            
            # Initialize LLM handler
            await llm_handler.initialize()

            # Prompt service is synchronous (templates loaded at import); do not await
            # Keep interface consistent: prompt_service is ready to use

            # Initialize response formatter
            self.response_formatter = await get_response_formatter()
            
            # Initialize BM25 index for keyword search
            await self._initialize_bm25_index()
            
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
            context: Previous conversation context (optional, auto-retrieved if not provided)
        
        Returns:
            Dictionary with response, sources, and attachments
        """
        try:
            start_time = time.time()
            
            # Retrieve conversation context if not provided
            if context is None:
                context = await self._get_conversation_context(session_id)
            
            # Check cache for similar queries first
            cached_response = await self._check_query_cache(query)
            if cached_response:
                logger.info("Returning cached response")
                return cached_response
            
            # Step 1: Advanced query understanding and enhancement
            query_enhancement = await query_intelligence_service.enhance_query(query)
            processed_query = query_enhancement["rewritten_queries"][0]  # Use best rewrite
            logger.info(f"Processing query: {processed_query[:100]}... (Type: {query_enhancement['query_type']})")
            
            # Step 2: Generate multiple query embeddings for hybrid approach
            query_embedding = await self._generate_query_embedding(processed_query)
            hyde_embedding = query_enhancement.get("hyde_embedding")
            
            # Step 3: Hybrid retrieval - Vector + Keyword + HyDE (with caching)
            
            # Check for cached retrieval results
            vector_results = await self._get_cached_retrieval_results(query_embedding)
            
            if not vector_results:
                vector_results = await self.qdrant_client.search_similar(
                    query_vector=query_embedding,
                    top_k=settings.top_k_retrieval * 2  # Get more for reranking
                )
                # Cache the results
                await self._cache_retrieval_results(query_embedding, vector_results)
            else:
                logger.info("Using cached vector search results")
            
            # Additional HyDE search if available
            hyde_results = []
            if hyde_embedding:
                hyde_results = await self.qdrant_client.search_similar(
                    query_vector=hyde_embedding,
                    top_k=settings.top_k_retrieval
                )
            
            # BM25 keyword search
            keyword_results = await self._bm25_search(processed_query)
            
            # Step 4: Extract entities from query for graph search
            graph_results = []
            if settings.use_graph_search and self.nlp_model:
                entities = self._extract_entities_from_query(processed_query)
                if entities:
                    graph_results = await self.arango_client.find_related_entities(
                        entities, max_depth=2, limit=5
                    )
            
            # Step 5: Merge and rerank results with adaptive strategy
            merged_results = await self._advanced_merge_and_rerank(
                vector_results, hyde_results, keyword_results, graph_results, 
                processed_query, query_enhancement["processing_strategy"]
            )
            
            # Step 6: Optimize context using advanced compression and CoT
            optimization_result = await context_optimization_service.optimize_context(
                merged_results, processed_query, 
                max_tokens=settings.max_context_tokens,
                strategy=query_enhancement["processing_strategy"]
            )
            
            context_text = optimization_result["formatted_context"]
            reasoning_template = optimization_result["reasoning_template"]
            sources = optimization_result["sources_used"]
            
            # Step 7: Generate response using LMStudio with CoT reasoning
            response = await self._generate_llm_response_with_cot(
                processed_query, context_text, reasoning_template, context
            )
            
            # Step 8: Extract attachments from sources
            attachments = self._extract_attachments(sources)
            
            # Step 9: Store conversation in session
            await self._store_conversation_turn(session_id, query, response, sources)
            
            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f}s")
            
            # Step 9: Evaluate response quality
            try:
                evaluation_metrics = await evaluation_service.evaluate_query_response(
                    query, response, sources, processing_time, context_text, session_id
                )
                logger.info(f"Query evaluation - Accuracy: {evaluation_metrics.answer_accuracy:.2f}, "
                          f"Relevance: {evaluation_metrics.response_relevance:.2f}")
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                evaluation_metrics = None
            
            result = {
                "response": response,
                "sources": sources,
                "attachments": attachments,
                "processing_time": processing_time,
                "tokens_generated": len(response.split()),  # Rough token estimate
                "evaluation_metrics": evaluation_metrics.__dict__ if evaluation_metrics else None
            }
            
            # Cache the result if quality is good
            if evaluation_metrics and evaluation_metrics.answer_accuracy > 0.7:
                await self._cache_query_result(query, result)
            
            return result
            
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
            
            # Step 6: Send final data with formatting
            formatted_response = await self.response_formatter.format_response(
                response_text=full_response,
                sources=sources,
                original_query=query
            )
            
            await self._store_conversation_turn(session_id, query, full_response, sources)
            
            yield {
                "type": "response",
                "text": formatted_response.text,
                "attachments": formatted_response.attachments,
                "citations": formatted_response.citations,
                "metadata": formatted_response.metadata
            }
            
            yield {
                "type": "sources",
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield {
                "type": "error",
                "content": "I encountered an error while processing your query."
            }
    
    async def _get_conversation_context(self, session_id: str, limit: int = 3) -> List[Dict]:
        """Retrieve conversation history from session."""
        try:
            session_data = await self.redis_client.get_session_data(session_id)
            if not session_data or "messages" not in session_data:
                return []
            
            messages = session_data["messages"]
            # Return last N messages (limit parameter)
            return messages[-limit:] if len(messages) > limit else messages
            
        except Exception as e:
            logger.warning(f"Failed to get conversation context: {e}")
            return []
    
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
        """Generate response using LLM handler with prompt service."""
        try:
            # Detect task type from query (synchronous)
            task_type = prompt_service.detect_task_type(query)
            logger.info(f"Detected task type: {task_type}")

            # Get task-optimized system prompt (synchronous)
            system_prompt = prompt_service.get_system_prompt(task_type)

            # Build complete prompt with enhanced formatting. build_complete_prompt
            # returns (system_prompt, user_prompt). If it fails (e.g., unexpected
            # context type), fall back to a simple enhanced query string.
            try:
                built_system, user_prompt = prompt_service.build_complete_prompt(
                    query=query,
                    context=[],  # pass empty chunks to avoid format errors here
                    conversation_context=conversation_context[-3:] if conversation_context else None,
                    task_type=task_type,
                    include_few_shot=True
                )
                # Prefer the built system prompt if it provides more context
                if built_system:
                    system_prompt = built_system
            except Exception as e:
                logger.debug(f"Prompt building failed, falling back: {e}")
                user_prompt = prompt_service.enhance_query(query, task_type)
            
            # Count tokens before generation
            input_tokens = llm_handler.count_tokens(f"{system_prompt}\n{user_prompt}")
            available_tokens = settings.max_llm_output_tokens - input_tokens
            
            if available_tokens < 100:
                logger.warning(f"Low available tokens: {available_tokens}. Input might be too long.")
            
            # Generate response with LLM handler
            response = await llm_handler.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=min(available_tokens, 512),
                temperature=settings.default_temperature,
                top_p=settings.default_top_p,
                stream=False
            )
            
            logger.info(f"Generated response ({len(response)} chars)")
            return response
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback response
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    async def _generate_llm_response_with_cot(self, query: str, context: str, reasoning_template: str, 
                                            conversation_context: List[Dict] = None) -> str:
        """Generate response using LLM handler with Chain-of-Thought reasoning and prompt service."""
        try:
            # Get analyze task prompt for CoT reasoning (synchronous)
            system_prompt = prompt_service.get_system_prompt("analyze")
            
            # Build CoT-enhanced user prompt
            user_prompt = f"""{reasoning_template}

Context information:
{context}

Question: {query}

Please provide your reasoning step by step, then give your final answer:"""
            
            # Count tokens
            input_tokens = llm_handler.count_tokens(f"{system_prompt}\n{user_prompt}")
            available_tokens = settings.max_llm_output_tokens - input_tokens
            
            # Generate response with CoT (using more tokens for reasoning)
            response = await llm_handler.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=min(available_tokens, 768),
                temperature=settings.default_temperature * 0.8,  # Slightly lower for focused reasoning
                top_p=settings.default_top_p,
                stream=False
            )
            
            # Extract final answer from CoT response
            final_answer = self._extract_final_answer(response)
            logger.info(f"Generated CoT response ({len(final_answer)} chars)")
            return final_answer
                
        except Exception as e:
            logger.error(f"CoT LLM generation failed: {e}")
            # Fallback to regular generation
            return await self._generate_llm_response(query, context, conversation_context)
    
    def _build_cot_system_prompt(self, reasoning_template: str) -> str:
        """Build the system prompt for Chain-of-Thought reasoning."""
        return f"""You are a helpful AI assistant that answers questions using structured reasoning. 

REASONING PROCESS:
{reasoning_template}

RESPONSE FORMAT:
Always structure your response as follows:

## Analysis
[Your step-by-step reasoning process]

## Answer
[Your clear, concise final answer to the question]

IMPORTANT GUIDELINES:
1. Always follow the reasoning process outlined above
2. Base all analysis on the provided context only
3. Be explicit about your reasoning steps
4. Clearly separate your analysis from your final answer
5. If information is insufficient, state this in your analysis
6. Reference specific sources when making claims
7. Keep your final answer focused and actionable

Remember: Show your thinking process, then provide a clear answer."""
    
    def _build_cot_user_prompt(self, query: str, context: str, conversation_context: List[Dict] = None) -> str:
        """Build the user prompt for Chain-of-Thought reasoning."""
        prompt_parts = []
        
        # Add conversation context if available
        if conversation_context:
            prompt_parts.append("CONVERSATION HISTORY:")
            for msg in conversation_context[-3:]:  # Last 3 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.title()}: {content}")
            prompt_parts.append("")
        
        # Add the context
        prompt_parts.append("CONTEXT INFORMATION:")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Add the current query
        prompt_parts.append(f"QUESTION: {query}")
        prompt_parts.append("")
        prompt_parts.append("Please analyze this question using the reasoning process and provide a structured response:")
        
        return "\n".join(prompt_parts)
    
    def _extract_final_answer(self, full_response: str) -> str:
        """Extract the final answer from a Chain-of-Thought response."""
        try:
            # Look for the "## Answer" section
            answer_match = re.search(r'##\s*Answer\s*\n(.*?)(?:\n##|\Z)', full_response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
                if answer:
                    return answer
            
            # Fallback: look for "Answer:" pattern
            answer_match = re.search(r'Answer:\s*(.*?)(?:\n\n|\Z)', full_response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
                if answer:
                    return answer
            
            # If no clear answer section found, return the full response
            return full_response.strip()
            
        except Exception as e:
            logger.warning(f"Failed to extract final answer: {e}")
            return full_response.strip()
    
    async def _stream_llm_response(self, query: str, context: str) -> AsyncGenerator[str, None]:
        """Stream response from LLM handler token by token."""
        try:
            # Detect task type (synchronous)
            task_type = prompt_service.detect_task_type(query)

            # Get optimized prompts (synchronous)
            system_prompt = prompt_service.get_system_prompt(task_type)
            try:
                built_system, user_prompt = prompt_service.build_complete_prompt(
                    query=query,
                    context=[],
                    task_type=task_type,
                    include_few_shot=True
                )
                if built_system:
                    system_prompt = built_system
            except Exception:
                user_prompt = prompt_service.enhance_query(query, task_type)
            
            # Count tokens
            input_tokens = llm_handler.count_tokens(f"{system_prompt}\n{user_prompt}")
            available_tokens = settings.max_llm_output_tokens - input_tokens
            
            # Stream from LLM handler
            async for token in llm_handler._stream_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=min(available_tokens, 512),
                temperature=settings.default_temperature,
                top_p=settings.default_top_p
            ):
                yield token
                
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
    
    async def _initialize_bm25_index(self):
        """Initialize BM25 index for keyword search."""
        try:
            # Get all document texts for BM25 indexing
            documents = await self.mongo_client.get_all_documents()
            self.document_texts = []
            
            for doc in documents:
                # Get chunks for this document
                chunks = await self.mongo_client.get_document_chunks(doc["_id"])
                for chunk in chunks:
                    self.document_texts.append({
                        "text": chunk.get("text", ""),
                        "doc_id": doc["_id"],
                        "chunk_id": chunk.get("_id"),
                        "metadata": chunk.get("metadata", {})
                    })
            
            # Create BM25 index
            if self.document_texts:
                tokenized_docs = [doc["text"].lower().split() for doc in self.document_texts]
                self.bm25_index = BM25Okapi(tokenized_docs)
                logger.info(f"BM25 index initialized with {len(self.document_texts)} documents")
            else:
                logger.warning("No documents found for BM25 indexing")
                
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            self.bm25_index = None
    
    async def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform BM25 keyword search."""
        if not self.bm25_index or not self.document_texts:
            return []
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Create results with scores
            results = []
            for idx, score in enumerate(scores):
                if score > 0:  # Only include results with positive scores
                    doc = self.document_texts[idx]
                    results.append({
                        "id": f"bm25_{doc['chunk_id']}",
                        "score": float(score),
                        "doc_id": doc["doc_id"],
                        "text": doc["text"],
                        "metadata": {**doc["metadata"], "source": "bm25"}
                    })
            
            # Sort by score and return top results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    async def _advanced_merge_and_rerank(self, vector_results: List[Dict], hyde_results: List[Dict],
                                       keyword_results: List[Dict], graph_results: List[Dict], 
                                       query: str, processing_strategy: str) -> List[Dict]:
        """Advanced merging and reranking with adaptive strategies."""
        try:
            # Combine all results with weighted scores based on strategy
            all_results = {}
            
            # Vector results (semantic similarity)
            for result in vector_results:
                result_id = result.get("id", "")
                weight = self._get_vector_weight(processing_strategy)
                all_results[result_id] = {
                    **result,
                    "combined_score": result.get("score", 0) * weight,
                    "sources": ["vector"]
                }
            
            # HyDE results (hypothetical document similarity)
            for result in hyde_results:
                result_id = result.get("id", "")
                weight = self._get_hyde_weight(processing_strategy)
                if result_id in all_results:
                    all_results[result_id]["combined_score"] += result.get("score", 0) * weight
                    all_results[result_id]["sources"].append("hyde")
                else:
                    all_results[result_id] = {
                        **result,
                        "combined_score": result.get("score", 0) * weight,
                        "sources": ["hyde"]
                    }
            
            # Keyword results (BM25)
            for result in keyword_results:
                result_id = result.get("id", "")
                weight = self._get_keyword_weight(processing_strategy)
                normalized_score = min(result.get("score", 0) / 10.0, 1.0)  # Normalize BM25 scores
                if result_id in all_results:
                    all_results[result_id]["combined_score"] += normalized_score * weight
                    all_results[result_id]["sources"].append("keyword")
                else:
                    all_results[result_id] = {
                        **result,
                        "combined_score": normalized_score * weight,
                        "sources": ["keyword"]
                    }
            
            # Graph results (relationship-based)
            for result in graph_results:
                result_id = result.get("id", "")
                weight = self._get_graph_weight(processing_strategy)
                if result_id in all_results:
                    all_results[result_id]["combined_score"] += result.get("score", 0) * weight
                    all_results[result_id]["sources"].append("graph")
                else:
                    all_results[result_id] = {
                        **result,
                        "combined_score": result.get("score", 0) * weight,
                        "sources": ["graph"]
                    }
            
            # Convert back to list and apply final reranking
            merged_results = list(all_results.values())
            
            # Apply cross-encoder reranking if available
            if self.reranker_model and settings.use_reranker and len(merged_results) > 1:
                merged_results = await self._cross_encoder_rerank(merged_results, query)
            
            # Sort by combined score
            merged_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            
            return merged_results[:settings.top_k_retrieval]
            
        except Exception as e:
            logger.error(f"Advanced merge and rerank failed: {e}")
            # Fallback to simple merging
            return (vector_results + hyde_results + keyword_results + graph_results)[:settings.top_k_retrieval]
    
    def _get_vector_weight(self, strategy: str) -> float:
        """Get vector search weight based on processing strategy."""
        weights = {
            "high_precision": 0.7,
            "multi_perspective": 0.4,
            "time_weighted": 0.5,
            "step_by_step": 0.6,
            "broad_search": 0.3,
            "solution_focused": 0.5,
            "balanced": 0.5
        }
        return weights.get(strategy, 0.5)
    
    def _get_hyde_weight(self, strategy: str) -> float:
        """Get HyDE search weight based on processing strategy."""
        weights = {
            "high_precision": 0.2,
            "multi_perspective": 0.3,
            "time_weighted": 0.1,
            "step_by_step": 0.2,
            "broad_search": 0.4,
            "solution_focused": 0.3,
            "balanced": 0.25
        }
        return weights.get(strategy, 0.25)
    
    def _get_keyword_weight(self, strategy: str) -> float:
        """Get keyword search weight based on processing strategy."""
        weights = {
            "high_precision": 0.1,
            "multi_perspective": 0.2,
            "time_weighted": 0.3,
            "step_by_step": 0.1,
            "broad_search": 0.2,
            "solution_focused": 0.1,
            "balanced": 0.15
        }
        return weights.get(strategy, 0.15)
    
    def _get_graph_weight(self, strategy: str) -> float:
        """Get graph search weight based on processing strategy."""
        weights = {
            "high_precision": 0.0,
            "multi_perspective": 0.1,
            "time_weighted": 0.1,
            "step_by_step": 0.1,
            "broad_search": 0.1,
            "solution_focused": 0.1,
            "balanced": 0.1
        }
        return weights.get(strategy, 0.1)
    
    async def _cross_encoder_rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """Apply cross-encoder reranking to results."""
        try:
            if len(results) <= 1:
                return results
            
            # Prepare query-document pairs
            pairs = [(query, result.get("text", "")) for result in results]
            
            # Get cross-encoder scores
            cross_scores = self.reranker_model.predict(pairs)
            
            # Update scores and sort
            for i, result in enumerate(results):
                # Combine original score with cross-encoder score
                original_score = result.get("combined_score", 0)
                cross_score = float(cross_scores[i])
                result["combined_score"] = (original_score * 0.3) + (cross_score * 0.7)
                result["cross_encoder_score"] = cross_score
            
            return results
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return results
    
    async def _check_query_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if we have a cached response for this query."""
        try:
            # Create query hash for caching
            query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
            cache_key = f"query_response:{query_hash}"
            
            cached_result = await self.redis_client.get_json(cache_key)
            
            if cached_result:
                # Update processing time to reflect cache hit
                cached_result["processing_time"] = 0.1
                cached_result["cached"] = True
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
            return None
    
    async def _cache_query_result(self, query: str, result: Dict[str, Any]):
        """Cache query result for future use."""
        try:
            # Create query hash for caching
            query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
            cache_key = f"query_response:{query_hash}"
            
            # Prepare cacheable result (remove large objects)
            cacheable_result = result.copy()
            
            # Limit sources for caching
            if "sources" in cacheable_result:
                cacheable_result["sources"] = cacheable_result["sources"][:3]
            
            # Cache for 1 hour for high-quality responses
            await self.redis_client.set_json(
                cache_key, 
                cacheable_result,
                expire_seconds=3600
            )
            
            logger.info(f"Cached high-quality response for query: {query[:50]}...")
            
        except Exception as e:
            logger.warning(f"Failed to cache query result: {e}")
    
    async def _cache_retrieval_results(self, query_embedding: List[float], 
                                     results: List[Dict]) -> str:
        """Cache retrieval results for reuse."""
        try:
            # Create embedding hash for caching
            embedding_str = ",".join([f"{x:.4f}" for x in query_embedding[:10]])  # First 10 dims
            embedding_hash = hashlib.md5(embedding_str.encode()).hexdigest()
            cache_key = f"retrieval:{embedding_hash}"
            
            # Cache retrieval results for 30 minutes
            await self.redis_client.set_json(
                cache_key,
                {"results": results, "timestamp": time.time()},
                expire_seconds=1800
            )
            
            return cache_key
            
        except Exception as e:
            logger.warning(f"Failed to cache retrieval results: {e}")
            return ""
    
    async def _get_cached_retrieval_results(self, query_embedding: List[float]) -> Optional[List[Dict]]:
        """Get cached retrieval results."""
        try:
            # Create embedding hash for lookup
            embedding_str = ",".join([f"{x:.4f}" for x in query_embedding[:10]])
            embedding_hash = hashlib.md5(embedding_str.encode()).hexdigest()
            cache_key = f"retrieval:{embedding_hash}"
            
            cached_data = await self.redis_client.get_json(cache_key)
            
            if cached_data:
                # Check if cache is still fresh (within 30 minutes)
                if time.time() - cached_data.get("timestamp", 0) < 1800:
                    return cached_data.get("results", [])
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get cached retrieval results: {e}")
            return None

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