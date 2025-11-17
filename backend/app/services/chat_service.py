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
from datetime import datetime, timezone
import time
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy
from rank_bm25 import BM25Okapi

from ..core.db_mongo import get_mongodb_client, MongoDBClient
from ..core.db_qdrant import get_qdrant_client, QdrantDBClient
from ..core.cache_redis import get_redis_client, RedisClient
from .query_intelligence_service import query_intelligence_service
from .context_optimization_service import context_optimization_service
from .evaluation_service import evaluation_service
from .llm_handler import llm_handler
from .prompt_service import prompt_service
from .response_formatter_service import get_response_formatter
from .hybrid_retrieval_service import hybrid_retrieval_service
from .media_suggestion_service import media_suggestion_service
from .response_validation_service import response_validation_service
from .context_window_optimizer import context_optimizer
from .cos_mix_retrieval import cos_mix_retrieval
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
        self.redis_client = None
        self.http_client = None
        self.bm25_index = None
        self.document_texts = []
        
        # Research-backed enhancements
        self.context_optimizer = context_optimizer
        self.cos_mix_retrieval = cos_mix_retrieval
    
    async def initialize(self):
        """Initialize the chat service with required models and clients."""
        try:
            logger.info("Initializing chat service...")
            
            # Get database clients (mongo is sync, qdrant and redis are async)
            self.mongo_client = get_mongodb_client()
            self.qdrant_client = await get_qdrant_client()
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
            self.response_formatter = get_response_formatter()
            
            # Initialize media services (Phase 2)
            await media_suggestion_service.initialize()
            logger.info("✅ Media suggestion service initialized")
            
            # Initialize BM25 index for keyword search
            await self._initialize_bm25_index()
            
            logger.info("Chat service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat service: {e}")
            raise
    
    # ========== MAIN PIPELINE ORCHESTRATOR ==========
    
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
            logger.info("🔍 ===== SIMPLE RAG PIPELINE START =====")
            logger.info(f"📌 Session: {session_id}")
            logger.info(f"❓ Query: {query[:200]}")

            # Fast-path for simple conversational queries (no retrieval needed)
            simple_patterns = ["hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"]
            query_lower = query.lower().strip()
            if len(query_lower.split()) <= 3 and any(p in query_lower for p in simple_patterns):
                logger.info("⚡ Fast-path: Simple conversational query detected")
                from .llm_handler import llm_handler
                try:
                    response_text = await llm_handler.generate_response(
                        prompt=query,
                        system_prompt="You are a helpful assistant. Respond briefly and naturally.",
                        max_tokens=100
                    )
                    return {
                        "response": response_text if isinstance(response_text, str) else "Hello! How can I help you?",
                        "sources": [],
                        "attachments": [],
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "tokens_generated": len(response_text.split()) if isinstance(response_text, str) else 0
                    }
                except Exception as e:
                    logger.warning(f"Fast-path LLM failed: {e}, using fallback")
                    return {
                        "response": "Hello! How can I help you today?",
                        "sources": [],
                        "attachments": [],
                        "session_id": session_id,
                        "processing_time": time.time() - start_time,
                        "tokens_generated": 0
                    }

            # Retrieve conversation context if not provided
            if context is None:
                context = await self._get_conversation_context(session_id)

            # Check cache for similar queries first
            cached_response = await self._check_query_cache(query)
            if cached_response:
                logger.info("✅ SIMPLE RAG: Returning cached response")
                return cached_response

            # --- Simple RAG path: preprocess -> embed -> vector search -> LLM ---
            processed_query = self._preprocess_query(query)
            logger.info(f"📝 Processed query: {processed_query[:200]}")

            # Embedding
            query_embedding = await self._generate_query_embedding(processed_query)
            logger.info(f"🔢 Generated embedding: {len(query_embedding)} dimensions")

            # Vector search only (no hybrid / graph)
            vector_results = await self.qdrant_client.search_similar(
                query_vector=query_embedding,
                top_k=settings.top_k_retrieval
            )
            logger.info(f"📚 Retrieved {len(vector_results) if vector_results else 0} vector results")

            # Build context and sources
            context_text, sources = await self._build_llm_context(vector_results)

            # Generate LLM response using existing helper
            response_text = await self._generate_llm_response(
                query=processed_query,
                context=context_text,
                conversation_context=context,
            )

            processing_time = time.time() - start_time

            result = {
                "response": response_text,
                "sources": sources,
                "attachments": [],
                "session_id": session_id,
                "processing_time": processing_time,
                "tokens_generated": len(response_text.split()) if isinstance(response_text, str) else 0,
                "media_suggestions": [],
                "validation_details": None,
                "phase3_metrics": None,
            }

            # Store in cache for identical queries
            await self._cache_query_result(query, result)

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Simple RAG query processing failed: {error_msg}", exc_info=True)

            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

            return {
                "response": f"Error: {error_msg}. Check backend logs for details.",
                "sources": [],
                "attachments": [],
                "processing_time": 0,
                "tokens_generated": 0,
                "error": error_msg,
                "error_type": type(e).__name__
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
            processed_query = self._preprocess_query(query)
            
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
                    # Query MongoDB for related entities
                    try:
                        related_docs = await self.mongo_client.database.entities.find(
                            {"name": {"$in": entities}}
                        ).to_list(5)
                        graph_results = related_docs if related_docs else []
                    except Exception as e:
                        logger.warning(f"Failed to query entities: {e}")
                        graph_results = []
            
            # Step 4: Context building
            yield {"type": "status", "content": "Building context..."}
            merged_results = await self._merge_and_rerank_results(
                vector_results, graph_results, processed_query
            )
            context_text, sources = await self._build_llm_context(merged_results)
            
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
    
    def _preprocess_query(self, query: str) -> str:
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
        """Rerank results using COS-Mix + cross-encoder model (research-backed)."""
        await asyncio.sleep(0)  # Use async feature
        try:
            if not results:
                return results
            
            # Stage 1: COS-Mix Reranking (if embeddings available)
            logger.info(f"🔬 Research Enhancement: COS-Mix Reranking")
            query_embedding = None
            has_embeddings = any(r.get("embedding") is not None for r in results)
            
            if has_embeddings:
                # Generate query embedding for COS-Mix
                query_embedding = self.embedding_model.encode(query)
                
                # Apply COS-Mix scoring
                cos_mix_results = self.cos_mix_retrieval.rerank_with_cos_mix(
                    query_embedding=query_embedding,
                    documents=results,
                    embedding_key="embedding"
                )
                
                logger.info(
                    f"  📊 COS-Mix scores: "
                    f"min={min(r['cos_mix_score'] for r in cos_mix_results):.3f}, "
                    f"max={max(r['cos_mix_score'] for r in cos_mix_results):.3f}"
                )
                
                results = cos_mix_results
            
            # Stage 2: Cross-Encoder Reranking (if model available)
            if self.reranker_model:
                logger.info(f"🔬 Cross-Encoder Reranking")
                
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
                    
                    # Combine COS-Mix and Cross-Encoder scores if both available
                    if "cos_mix_score" in result:
                        # Hybrid fusion: 60% cross-encoder, 40% COS-Mix (research-backed)
                        combined_score = (0.6 * float(scores[i])) + (0.4 * result["cos_mix_score"])
                        result["score"] = combined_score
                        result["ranking_method"] = "hybrid_cos_mix_crossencoder"
                    else:
                        # Cross-encoder only: combine with original score
                        result["score"] = 0.7 * float(scores[i]) + 0.3 * result.get("original_score", 0)
                        result["ranking_method"] = "crossencoder_only"
                
                # Sort by new combined score
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                
                logger.info(
                    f"  ✅ Reranked {len(results)} results using "
                    f"{'Hybrid COS-Mix + Cross-Encoder' if has_embeddings else 'Cross-Encoder only'}"
                )
            else:
                # No cross-encoder, just use COS-Mix scores
                if has_embeddings:
                    results.sort(key=lambda x: x.get("cos_mix_score", 0), reverse=True)
                    for r in results:
                        r["ranking_method"] = "cos_mix_only"
                        r["score"] = r.get("cos_mix_score", r.get("score", 0))
            
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            return results
    
    async def _get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata from cache or database."""
        # Try cache first
        doc_metadata = await self.redis_client.get_cached_document_metadata(doc_id)
        if doc_metadata:
            return doc_metadata
        
        # Fetch from database
        document = await self.mongo_client.get_document(doc_id)
        if not document:
            return None
        
        doc_metadata = {
            "filename": document.get("filename", "Unknown"),
            "file_type": document.get("file_type", ""),
            "uploaded_at": document.get("uploaded_at", "").isoformat() if document.get("uploaded_at") else ""
        }
        await self.redis_client.cache_document_metadata(doc_id, doc_metadata)
        return doc_metadata
    
    def _format_source_entry(self, result: Dict, doc_metadata: Optional[Dict], index: int) -> tuple[str, Dict]:
        """Format a single source entry for context and metadata."""
        chunk_text = result.get("text", "")
        source_label = f"[Source {index+1}]"
        context_part = f"{source_label} {chunk_text}"
        
        source_dict = {
            "id": result.get("id", ""),
            "doc_id": result.get("doc_id", ""),
            "chunk_id": result.get("chunk_id", ""),
            "filename": doc_metadata.get("filename", "Unknown") if doc_metadata else "Unknown",
            "text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
            "score": result.get("score", 0),
            "source_label": source_label
        }
        
        return context_part, source_dict
    
    async def _build_llm_context(self, results: List[Dict]) -> tuple[str, List[Dict]]:
        """Build context for LLM and prepare source metadata."""
        if not results:
            return "No relevant information found.", []
        
        context_parts = []
        sources = []
        
        for i, result in enumerate(results):
            doc_id = result.get("doc_id", "")
            chunk_text = result.get("text", "")
            
            if not (doc_id and chunk_text):
                continue
            
            # Get document metadata
            doc_metadata = await self._get_document_metadata(doc_id)
            
            # Format source entry
            context_part, source_dict = self._format_source_entry(result, doc_metadata, i)
            context_parts.append(context_part)
            sources.append(source_dict)
        
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

            # Build complete prompt with enhanced formatting
            try:
                built_system, user_prompt = prompt_service.build_complete_prompt(
                    query=query,
                    context=[],
                    conversation_context=conversation_context[-3:] if conversation_context else None,
                    task_type=task_type,
                    include_few_shot=True
                )
                if built_system:
                    system_prompt = built_system
            except Exception as e:
                logger.debug(f"Prompt building failed, falling back: {e}")
                user_prompt = prompt_service.enhance_query(query, task_type)
            
            # Count tokens before generation
            input_tokens = llm_handler.count_tokens(f"{system_prompt}\n{user_prompt}")
            available_tokens = settings.max_llm_output_tokens - input_tokens
            
            logger.info(f"Token info - Input: {input_tokens}, Available: {available_tokens}, Max Output: {settings.max_llm_output_tokens}")
            
            if available_tokens < 100:
                logger.warning(f"Low available tokens: {available_tokens}. Reducing context or query.")
                # Reduce input to fit better
                user_prompt = user_prompt[:len(user_prompt)//2] if len(user_prompt) > 500 else user_prompt
                input_tokens = llm_handler.count_tokens(f"{system_prompt}\n{user_prompt}")
                available_tokens = settings.max_llm_output_tokens - input_tokens
            
            max_output_tokens = min(available_tokens, 512)
            logger.debug(f"Requesting LLM with max_tokens={max_output_tokens}")
            
            # Generate response with LLM handler
            try:
                response = await llm_handler.generate_response(
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    max_tokens=max_output_tokens,
                    temperature=settings.default_temperature,
                    top_p=settings.default_top_p,
                    stream=False
                )
                
                # Ensure response is string (not dict)
                if isinstance(response, dict):
                    logger.warning(f"LLM returned dict instead of string: {type(response)}")
                    response = response.get("response", str(response))
                
                if not response or not isinstance(response, str):
                    logger.error(f"Invalid response type: {type(response)}, value: {response}")
                    raise TypeError(f"Invalid LLM response type: {type(response)}")
                
                logger.info(f"✅ Generated response ({len(response)} chars)")
                return response
                
            except Exception as e:
                logger.error(f"LLM generation error: {type(e).__name__}: {e}", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"LLM generation failed: {type(e).__name__}: {e}", exc_info=True)
            # Fallback response
            fallback = "I apologize, but I'm having trouble generating a response right now. Please try again or check if the LLM service is running."
            logger.info("Returning fallback response")
            return fallback
    
    async def _generate_llm_response_with_cot(self, query: str, context: str, _reasoning_template: str, 
                                            conversation_context: List[Dict] = None) -> str:
        """Generate response using LLM handler with Chain-of-Thought reasoning and prompt service."""
        try:
            # Get analyze task prompt for CoT reasoning (synchronous)
            system_prompt = prompt_service.get_system_prompt("analyze")
            
            # Phase 2: Enforce document usage if configured
            if settings.force_document_usage and context:
                logger.info("🔒 Document usage enforcement ENABLED for CoT - Forcing LLM to use provided documents")
                force_doc_clause = """
IMPORTANT - DOCUMENT-BASED RESPONSE REQUIREMENT:
You MUST base your answer primarily on the provided documents/context. Always:
1. Use information from the context to support your reasoning
2. Cite specific parts of the documents when making claims  
3. Do NOT generate generic responses or rely on general knowledge
4. Prefix key claims with citations like [1], [2], etc.
5. If context doesn't answer the question, state this explicitly
Example: [1] According to the document, the key fact is..."""
                system_prompt = system_prompt + "\n" + force_doc_clause
                logger.debug("✅ Added document usage enforcement to CoT system prompt")
            
            # Build CoT-enhanced user prompt
            user_prompt = """{reasoning_template}

Context information:
{context}

Question: {query}

Please provide your reasoning step by step, then give your final answer:"""
            
            # Count tokens
            input_tokens = llm_handler.count_tokens(f"{system_prompt}\n{user_prompt}")
            available_tokens = settings.max_llm_output_tokens - input_tokens
            
            logger.info(f"CoT Token info - Input: {input_tokens}, Available: {available_tokens}")
            
            if available_tokens < 100:
                logger.warning(f"Low tokens for CoT: {available_tokens}, reducing context")
                context = context[:len(context)//2]
                user_prompt = """{reasoning_template}

Context: {context}

Question: {query}

Final answer:"""
                input_tokens = llm_handler.count_tokens(f"{system_prompt}\n{user_prompt}")
                available_tokens = settings.max_llm_output_tokens - input_tokens
            
            max_output_tokens = min(available_tokens, 512)
            
            # Generate response with CoT (using more tokens for reasoning)
            try:
                response = await llm_handler.generate_response(
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    max_tokens=max_output_tokens,
                    temperature=settings.default_temperature * 0.8,
                    top_p=settings.default_top_p,
                    stream=False
                )
                
                # Ensure response is string
                if isinstance(response, dict):
                    logger.warning("CoT: LLM returned dict, extracting content")
                    response = response.get("response", str(response))
                
                # Extract final answer from CoT response
                final_answer = self._extract_final_answer(response)
                logger.info(f"✅ Generated CoT response ({len(final_answer)} chars)")
                return final_answer
                
            except Exception as e:
                logger.error(f"CoT LLM generation error: {type(e).__name__}: {e}", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"CoT LLM generation failed: {type(e).__name__}: {e}", exc_info=True)
            # Fallback to regular generation
            logger.info("Falling back to regular LLM generation")
            return await self._generate_llm_response(query, context, conversation_context)
    
    def _build_cot_system_prompt(self, _reasoning_template: str) -> str:
        """Build the system prompt for Chain-of-Thought reasoning."""
        return """You are a friendly, conversational AI assistant like ChatGPT, Gemini, or Claude. Your goal is to help users while showing your reasoning in a natural, understandable way.

CONVERSATIONAL TONE:
- Write naturally, as if explaining to a friend
- Use "I" and "you" to make it personal
- Break down your thinking in simple terms
- No formal headings or rigid structure
- Flow naturally from thought to conclusion

YOUR THINKING PROCESS:
{reasoning_template}

HOW TO RESPOND:
1. Start by briefly acknowledging the question
2. Share your thinking process conversationally (e.g., "Let me think about this...", "From what I can see...", "Here's how I'm approaching this...")
3. Walk through relevant points from the documents naturally
4. Arrive at your answer smoothly
5. Offer to clarify or expand if helpful

IMPORTANT:
- Base everything on the provided context
- Be honest about limitations: "I don't have enough information about..."
- Reference sources naturally: "According to [document]..." or "I found that..."
- Keep it conversational, not academic
- No formal citations like [1] or [Source: filename]
- If uncertain, say so: "I'm not entirely sure, but it seems..."

Remember: You're having a helpful conversation, not writing a report."""
    
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
            answer_match = re.search(r'##\s*Answer\s*\n(.+)(?:\n##|\Z)', full_response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
                if answer:
                    return answer
            
            # Fallback: look for "Answer:" pattern
            answer_match = re.search(r'Answer:\s*(.+)(?:\n\n|\Z)', full_response, re.DOTALL | re.IGNORECASE)
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
            stream_generator = await llm_handler.generate_response(
                system_prompt=system_prompt,
                prompt=user_prompt,
                max_tokens=min(available_tokens, 512),
                temperature=settings.default_temperature,
                top_p=settings.default_top_p,
                stream=True
            )
            
            async for token in stream_generator:
                yield token
                
        except Exception as e:
            logger.error(f"Streaming response failed: {e}")
            yield "I apologize, but I'm having trouble generating a response right now."
    
    def _build_system_prompt(self, query_intent: Optional[str] = None) -> str:
        """Build intent-specific system prompt for the LLM.
        
        Args:
            query_intent: Detected intent type (e.g., 'factual', 'procedural', 'comparative')
                         If None, uses generic helpful assistant prompt.
        
        Returns:
            System prompt tailored to the query intent
        """
        # Intent-specific system prompts
        intent_prompts = {
            "factual": """You are an expert assistant specializing in factual, accurate information retrieval.

INSTRUCTIONS FOR FACTUAL QUERIES:
1. Provide precise, verifiable facts from the context
2. Include specific details, numbers, dates, and sources when available
3. Clearly distinguish between facts and interpretations
4. If information is incomplete or ambiguous, explicitly state what is missing
5. Always cite sources when referencing specific information
6. Correct any misconceptions with supporting evidence from context

RESPONSE FORMAT:
- Lead with the most important fact
- Provide supporting details and evidence
- Include source citations
- Be specific and avoid vague language""",

            "procedural": """You are a step-by-step procedural guide expert.

INSTRUCTIONS FOR HOW-TO QUERIES:
1. Organize your answer as clear, numbered steps
2. Include prerequisites and required materials/information
3. Explain the "why" behind each step when relevant
4. Provide context for each step (what to expect, common issues)
5. Include tips and best practices
6. Warn about potential pitfalls or mistakes

RESPONSE FORMAT:
- Start with prerequisites (if any)
- Numbered steps with clear, concise instructions
- Expected outcomes for each step
- Tips/warnings in separate sections
- Verification step at the end""",

            "comparative": """You are an expert at providing balanced comparative analysis.

INSTRUCTIONS FOR COMPARISON QUERIES:
1. Present multiple perspectives fairly
2. Use structured comparison (e.g., pros/cons, advantages/disadvantages)
3. Highlight key differences and similarities
4. Include specific examples for each comparison point
5. Be neutral and avoid bias toward one option
6. Help users make informed decisions

RESPONSE FORMAT:
- Brief introduction of items being compared
- Organized comparison table or structured list
- Strengths and weaknesses of each option
- Use case recommendations
- Conclusion without prescribing "the best" choice""",

            "analytical": """You are a deep-dive analytical expert.

INSTRUCTIONS FOR ANALYTICAL QUERIES:
1. Break down complex topics into understandable components
2. Explain cause-effect relationships
3. Provide supporting evidence and reasoning
4. Identify patterns and trends in the information
5. Consider multiple perspectives and implications
6. Connect related concepts

RESPONSE FORMAT:
- Clear problem/topic statement
- Key factors or components
- Analysis of relationships and impacts
- Patterns and trends
- Implications and conclusions""",

            "troubleshooting": """You are a systematic troubleshooting expert.

INSTRUCTIONS FOR PROBLEM-SOLVING QUERIES:
1. Start by clarifying the problem scope
2. Suggest diagnostic steps to identify root causes
3. Provide solutions in order of likelihood
4. Include step-by-step fix instructions
5. Explain what each solution addresses
6. Suggest preventive measures

RESPONSE FORMAT:
- Problem diagnosis checklist
- Ordered solutions (most likely first)
- Step-by-step fix for each solution
- Verification steps
- Prevention tips""",

            "definition": """You are a definition and explanation expert.

INSTRUCTIONS FOR DEFINITION QUERIES:
1. Start with a clear, concise definition
2. Provide context and background
3. Include examples or use cases
4. Explain related concepts and connections
5. Distinguish from similar terms
6. Include practical applications

RESPONSE FORMAT:
- Definition (1-2 sentences)
- Etymology or origin (if relevant)
- Key characteristics or components
- Examples and use cases
- Related concepts
- Practical applications""",

            "creative": """You are a creative thinking facilitator.

INSTRUCTIONS FOR CREATIVE QUERIES:
1. Generate diverse ideas and perspectives
2. Use the context as inspiration, not limitation
3. Provide multiple options or approaches
4. Include practical and unconventional ideas
5. Explain the rationale behind suggestions
6. Encourage exploration

RESPONSE FORMAT:
- Diverse suggestion options (at least 3)
- Rationale for each suggestion
- How to implement or explore each idea
- Potential challenges and benefits
- Encouragement for further exploration"""
        }
        
        # Get appropriate prompt based on intent, fallback to generic
        if query_intent and query_intent in intent_prompts:
            logger.debug(f"Using intent-specific prompt for: {query_intent}")
            return intent_prompts[query_intent]
        
        # Generic fallback prompt
        logger.debug("Using generic system prompt (no specific intent detected)")
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
            if any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gi', '.webp']):
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
            # Simplified YouTube URL patterns
            youtube_regex = r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
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
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sources": sources
                }
            ])
            
            # Keep only last 20 messages to prevent memory issues
            if len(session_data["messages"]) > 20:
                session_data["messages"] = session_data["messages"][-20:]
            
            session_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            # Store back in Redis
            await self.redis_client.store_session_data(session_id, session_data, 8)
            
        except Exception as e:
            logger.error(f"Failed to store conversation turn: {e}")
    
    async def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions based on partial input."""
        await asyncio.sleep(0)  # Use async feature
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
    
    async def _reload_bm25_index(self):
        """Reload BM25 index after new document ingestion."""
        logger.info("🔄 Reloading BM25 index with new documents...")
        await self._initialize_bm25_index()
        logger.info("✅ BM25 index reloaded successfully")
    
    async def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform BM25 keyword search."""
        await asyncio.sleep(0)  # Use async feature
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
    
    def _get_vector_weight(self, strategy: str) -> float:
        """Get weight for vector results based on strategy."""
        weights = {"semantic": 0.5, "keyword": 0.2, "balanced": 0.35, "hybrid": 0.3}
        return weights.get(strategy, 0.35)
    
    def _get_hyde_weight(self, strategy: str) -> float:
        """Get weight for HyDE results based on strategy."""
        weights = {"semantic": 0.3, "keyword": 0.1, "balanced": 0.25, "hybrid": 0.25}
        return weights.get(strategy, 0.25)
    
    def _get_keyword_weight(self, strategy: str) -> float:
        """Get weight for keyword results based on strategy."""
        weights = {"semantic": 0.1, "keyword": 0.5, "balanced": 0.25, "hybrid": 0.25}
        return weights.get(strategy, 0.25)
    
    def _get_graph_weight(self, strategy: str) -> float:
        """Get weight for graph results based on strategy."""
        weights = {"semantic": 0.1, "keyword": 0.2, "balanced": 0.15, "hybrid": 0.2}
        return weights.get(strategy, 0.15)
    
    def _merge_vector_results(self, vector_results: List[Dict], strategy: str, all_results: Dict):
        """Merge vector search results into combined results."""
        weight = self._get_vector_weight(strategy)
        for result in vector_results:
            result_id = result.get("id", "")
            all_results[result_id] = {
                **result,
                "combined_score": result.get("score", 0) * weight,
                "sources": ["vector"]
            }
    
    def _merge_hyde_results(self, hyde_results: List[Dict], strategy: str, all_results: Dict):
        """Merge HyDE results into combined results."""
        weight = self._get_hyde_weight(strategy)
        for result in hyde_results:
            result_id = result.get("id", "")
            if result_id in all_results:
                all_results[result_id]["combined_score"] += result.get("score", 0) * weight
                all_results[result_id]["sources"].append("hyde")
            else:
                all_results[result_id] = {
                    **result,
                    "combined_score": result.get("score", 0) * weight,
                    "sources": ["hyde"]
                }
    
    def _merge_keyword_results(self, keyword_results: List[Dict], strategy: str, all_results: Dict):
        """Merge keyword search results into combined results."""
        weight = self._get_keyword_weight(strategy)
        for result in keyword_results:
            result_id = result.get("id", "")
            normalized_score = min(result.get("score", 0) / 10.0, 1.0)
            if result_id in all_results:
                all_results[result_id]["combined_score"] += normalized_score * weight
                all_results[result_id]["sources"].append("keyword")
            else:
                all_results[result_id] = {
                    **result,
                    "combined_score": normalized_score * weight,
                    "sources": ["keyword"]
                }
    
    def _merge_graph_results(self, graph_results: List[Dict], strategy: str, all_results: Dict):
        """Merge graph traversal results into combined results."""
        weight = self._get_graph_weight(strategy)
        for result in graph_results:
            result_id = result.get("id", "")
            if result_id in all_results:
                all_results[result_id]["combined_score"] += result.get("score", 0) * weight
                all_results[result_id]["sources"].append("graph")
            else:
                all_results[result_id] = {
                    **result,
                    "combined_score": result.get("score", 0) * weight,
                    "sources": ["graph"]
                }

    async def _advanced_merge_and_rerank(self, vector_results: List[Dict], hyde_results: List[Dict],
                                       keyword_results: List[Dict], graph_results: List[Dict], 
                                       query: str, processing_strategy: str) -> List[Dict]:
        """Advanced merging and reranking with adaptive strategies."""
        try:
            # Combine all results with weighted scores based on strategy
            all_results = {}
            
            # Merge all result types
            self._merge_vector_results(vector_results, processing_strategy, all_results)
            self._merge_hyde_results(hyde_results, processing_strategy, all_results)
            self._merge_keyword_results(keyword_results, processing_strategy, all_results)
            self._merge_graph_results(graph_results, processing_strategy, all_results)
            
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
        await asyncio.sleep(0)  # Use async feature
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
    
    # ============================================================================
    # Phase 3: ADVANCED GRAPH RAG INTEGRATION
    # ============================================================================
    
    def _extract_document_ids(self, vector_results: List[Dict]) -> set:
        """Extract unique document IDs from vector results."""
        document_ids = set()
        for result in vector_results:
            doc_id = result.get("document_id") or result.get("_id")
            if doc_id:
                document_ids.add(doc_id)
        return document_ids
    
    async def _find_related_entities(self, entities: List[str]) -> List[Dict]:
        """Find entities related to query entities."""
        if not entities:
            return []
        try:
            return await self.mongo_client.database.entities.find(
                {"name": {"$in": entities}}
            ).to_list(5)
        except Exception as e:
            logger.warning(f"Failed to find related entities: {e}")
            return []
    
    def _extract_insights_from_entities(self, entity_docs: List[Dict], document_ids: set) -> tuple:
        """Extract graph insights and additional document IDs from entities."""
        graph_insights = []
        additional_docs = set()
        
        for entity_doc in entity_docs:
            doc_ids_from_entity = entity_doc.get("documents", [])
            for doc_id in doc_ids_from_entity:
                if doc_id not in document_ids:
                    additional_docs.add(doc_id)
                    graph_insights.append({
                        "entity": entity_doc.get("name", ""),
                        "related_document": doc_id,
                        "type": "entity_expansion"
                    })
        
        return graph_insights, additional_docs

    async def retrieve_with_graph(
        self,
        query: str,
        entities: List[str],
        vector_results: List[Dict],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Advanced graph-enhanced retrieval combining vector search with graph traversal.
        
        Args:
            query: User query
            entities: Extracted entities from query
            vector_results: Initial vector search results
            top_k: Number of results to return
            
        Returns:
            Dictionary with combined results and graph insights
        """
        try:
            start_time = time.time()
            
            graph_enrichment = {
                "vector_base": len(vector_results),
                "graph_additions": 0,
                "graph_insights": [],
                "relationships_found": 0,
                "total_with_graph": len(vector_results)
            }
            
            # Step 1: Extract document IDs from vector results
            document_ids = self._extract_document_ids(vector_results)
            
            # Step 2: Find related entities
            related_entity_docs = await self._find_related_entities(entities)
            graph_enrichment["relationships_found"] = len(related_entity_docs)
            
            # Step 3: Extract insights and additional documents
            if related_entity_docs:
                graph_insights, additional_docs = self._extract_insights_from_entities(
                    related_entity_docs, document_ids
                )
                graph_enrichment["graph_insights"] = graph_insights
                graph_enrichment["graph_additions"] = len(additional_docs)
                
                # Optionally retrieve additional documents (placeholder for future enhancement)
                # additional_results = await self._retrieve_additional_docs(additional_docs)
            
            # Step 4: Rerank combined results
            reranked_results = vector_results
            if self.reranker_model and settings.use_reranker:
                reranked_results = await self._rerank_results(vector_results, query)
            
            # Sort by score and get top-k
            reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_results = reranked_results[:top_k]
            
            graph_enrichment["total_with_graph"] = len(final_results)
            graph_enrichment["processing_time_ms"] = int((time.time() - start_time) * 1000)
            graph_enrichment["enhancement_ratio"] = (
                graph_enrichment["graph_additions"] / max(1, graph_enrichment["vector_base"])
            )
            
            logger.info("  📊 Phase 3 Graph RAG Results:")
            logger.info(f"    🔵 Vector base: {graph_enrichment['vector_base']} results")
            logger.info(f"    🟣 Graph additions: {graph_enrichment['graph_additions']} new docs")
            logger.info(f"    🔗 Relationships: {graph_enrichment['relationships_found']} found")
            logger.info(f"    📈 Total final: {graph_enrichment['total_with_graph']} results")
            logger.info(f"    ⏱️  Graph processing: {graph_enrichment['processing_time_ms']}ms")
            
            return {
                "results": final_results,
                "enrichment_metrics": graph_enrichment
            }
            
        except Exception as e:
            logger.warning(f"Graph-enhanced retrieval failed: {e}")
            return {
                "results": vector_results,
                "enrichment_metrics": {"error": str(e)}
            }
    
    def _extract_entities_from_doc(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities from document content."""
        if not self.nlp_model:
            return []
        
        doc = self.nlp_model(content[:5000])  # Limit for performance
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "DATE"]:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        return entities
    
    def _extract_entity_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities based on co-occurrence."""
        relationships = []
        
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                relationships.append({
                    "from_entity": ent1["text"],
                    "from_type": ent1["label"],
                    "to_entity": ent2["text"],
                    "to_type": ent2["label"],
                    "relation_type": "co_occurrence",
                    "confidence": 0.7
                })
        
        return relationships
    
    async def _store_relationships_to_db(self, document_id: str, relationships: List[Dict[str, Any]]) -> int:
        """Store relationships to MongoDB."""
        if not relationships:
            return 0
        
        try:
            for rel in relationships:
                rel["document_id"] = document_id
                rel["created_at"] = datetime.now(timezone.utc)
                await self.mongo_client.database.entity_relationships.insert_one(rel)
            
            logger.info(f"✅ Stored {len(relationships)} relationships for doc {document_id}")
            return len(relationships)
        except Exception as e:
            logger.warning(f"Failed to store relationships: {e}")
            return 0
    
    def _process_chunks_for_entities(self, chunk_texts: List[str]) -> int:
        """Process chunks to extract additional entities."""
        if not self.nlp_model:
            return 0
        
        chunks_processed = 0
        for chunk in chunk_texts[:10]:  # Limit to first 10
            try:
                chunk_doc = self.nlp_model(chunk[:1000])
                for ent in chunk_doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]:
                        pass  # Additional relationship tracking
                chunks_processed += 1
            except Exception:
                pass
        
        return chunks_processed
    
    async def extract_and_store_relationships(
        self,
        document_id: str,
        content: str,
        chunk_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract relationships from document content and store in MongoDB.
        
        Args:
            document_id: Document ID
            content: Document content
            chunk_texts: Optional list of chunk texts for relationship extraction
            
        Returns:
            Extraction and storage results
        """
        try:
            if not self.nlp_model:
                return {"status": "skipped", "reason": "NLP model unavailable"}
            
            # Step 1: Extract entities
            entities = self._extract_entities_from_doc(content)
            
            # Step 2: Extract relationships
            relationships = self._extract_entity_relationships(entities)
            
            # Step 3: Store relationships
            stored_count = await self._store_relationships_to_db(document_id, relationships)
            
            # Step 4: Process chunks if provided
            chunks_processed = 0
            if chunk_texts:
                chunks_processed = self._process_chunks_for_entities(chunk_texts)
            
            return {
                "entities_found": len(entities),
                "relationships_stored": stored_count,
                "chunks_processed": chunks_processed
            }
            
        except Exception as e:
            logger.warning(f"Relationship extraction failed: {e}")
            return {"status": "error", "error": str(e)}

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
            except Exception:
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


# Global chat service instance
_chat_service_instance = None

async def get_chat_service() -> ChatService:
    """Get or create the global chat service instance."""
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
        await _chat_service_instance.initialize()
    return _chat_service_instance