"""
Enhanced LLM Handler Service with multi-provider support and robustness.
This module manages communication with various LLM providers including LMStudio,
Ollama, and API-based models with automatic fallback and error recovery.
"""

import logging
import httpx
import json
import asyncio
import re
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import tiktoken

from ..config import settings

logger = logging.getLogger(__name__)

class LLMHandler:
    """Robust LLM handler with multi-provider support and fallback mechanisms."""
    
    def __init__(self):
        self.providers = []
        self.current_provider_idx = 0
        self.tokenizer = None
        self.http_client = None
        self.max_retries = 3
        self.timeout = 300.0  # 5 minutes for LMStudio responses
        self.provider_health = {}
        
        # Initialize tokenizer immediately
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.debug("Tokenizer initialized in __init__")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
        
        # Initialize HTTP client immediately
        try:
            self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
            logger.debug("HTTP client initialized in __init__ with 5-minute timeout")
        except Exception as e:
            logger.warning(f"Failed to initialize HTTP client: {e}")
            self.http_client = None
        
        # Setup providers immediately  
        self._setup_providers()
        logger.debug(f"Providers initialized in __init__: {[p['name'] for p in self.providers]}")
        
    async def initialize(self):
        """Initialize LLM handler with available providers."""
        try:
            await asyncio.sleep(0)
            logger.info("Initializing LLM handler...")
            
            # Initialize tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Initialize HTTP client with extended timeout for LMStudio
            self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
            logger.info("HTTP client initialized with 5-minute timeout")
            
            # Configure providers
            self._setup_providers()
            
            # SKIP health check for now - it's causing startup hangs
            # Providers will be tested on first actual use
            logger.info("Skipping provider health check (will test on first use)")
            logger.info(f"LLM handler initialized with {len(self.providers)} providers")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM handler: {e}")
            raise
    
    def _setup_providers(self):
        """Setup available LLM providers."""
        # LMStudio (primary)
        self.providers.append({
            "name": "lmstudio",
            "url": settings.lmstudio_api_url or "http://localhost:1234/v1",
            "type": "openai_compatible",
            "model": settings.lmstudio_model_name or "mistral-7b-instruct-v0.3",
            "enabled": True,
            "priority": 1
        })
        
        # Ollama (fallback)
        self.providers.append({
            "name": "ollama",
            "url": "http://localhost:11434/v1",
            "type": "openai_compatible",
            "model": "mistral",
            "enabled": True,
            "priority": 2
        })
        
        # Sort by priority
        self.providers.sort(key=lambda x: x.get("priority", 999))
    
    async def _check_provider_health(self):
        """Check health of all providers with aggressive timeouts."""
        logger.info("Checking provider health...")
        for provider in self.providers:
            try:
                logger.info(f"Checking health of '{provider['name']}' at {provider['url']}/models")
                
                # Use asyncio.wait_for to enforce strict timeout
                async with httpx.AsyncClient(timeout=3.0) as client:
                    try:
                        # Wrap the request in wait_for for double protection
                        response = await asyncio.wait_for(
                            client.get(
                                f"{provider['url']}/models",
                                headers={"Accept": "application/json"}
                            ),
                            timeout=3.0  # 3 second absolute timeout
                        )
                        
                        if response.status_code == 200:
                            self.provider_health[provider["name"]] = {
                                "healthy": True,
                                "last_check": datetime.now().isoformat(),
                                "response_time": response.elapsed.total_seconds()
                            }
                            logger.info(f"✅ Provider '{provider['name']}' is healthy (response_time: {response.elapsed.total_seconds():.3f}s)")
                        else:
                            self.provider_health[provider["name"]] = {"healthy": False}
                            logger.warning(f"⚠️ Provider '{provider['name']}' returned status {response.status_code}")
                    
                    except asyncio.TimeoutError:
                        self.provider_health[provider["name"]] = {"healthy": False}
                        logger.warning(f"⚠️ Provider '{provider['name']}' health check timed out after 3s")
                        
            except Exception as e:
                self.provider_health[provider["name"]] = {"healthy": False}
                logger.warning(f"⚠️ Provider '{provider['name']}' health check failed: {e}")
                logger.debug(f"Health check error details: {e}", exc_info=True)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            if not self.tokenizer:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, estimating...")
            return len(text) // 4  # Rough estimate
    
    def _prepare_messages(self, prompt: str, system_prompt: Optional[str]) -> tuple:
        """Prepare messages and count tokens."""
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            combined_prompt = prompt
        
        messages = [{"role": "user", "content": combined_prompt}]
        prompt_tokens = self.count_tokens(combined_prompt)
        
        return messages, prompt_tokens
    
    def _calculate_max_tokens(self, max_tokens: Optional[int], prompt_tokens: int) -> int:
        """Calculate appropriate max_tokens for generation."""
        if max_tokens:
            return max_tokens
        
        configured_max = settings.max_llm_output_tokens
        available_for_output = 32768 - prompt_tokens - 256  # Leave buffer
        calculated_max = min(configured_max, available_for_output)
        return max(calculated_max, 512)  # Ensure minimum
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response from an available LLM provider.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Prepare messages and count tokens
            messages, prompt_tokens = self._prepare_messages(prompt, system_prompt)
            
            # Calculate max tokens
            max_tokens = self._calculate_max_tokens(max_tokens, prompt_tokens)
            
            # Try providers with retry logic
            logger.debug(f"Provider health status: {self.provider_health}")
            logger.debug(f"Available providers: {[p['name'] for p in self.providers]}")
            
            for attempt in range(self.max_retries):
                provider = self._get_next_provider()
                
                if not provider:
                    logger.error("No available LLM providers found")
                    raise RuntimeError("No available LLM providers")
                
                logger.info(f"Attempting to use provider '{provider['name']}' (attempt {attempt + 1}/{self.max_retries})")
                logger.debug(f"Provider URL: {provider['url']}/chat/completions")
                
                try:
                    if stream:
                        return await self._stream_response(
                            provider, messages, temperature, top_p, max_tokens, prompt_tokens
                        )
                    else:
                        return await self._get_response(
                            provider, messages, temperature, top_p, max_tokens, prompt_tokens
                        )
                        
                except Exception as e:
                    logger.warning(f"Provider '{provider['name']}' failed (attempt {attempt + 1}/{self.max_retries}): {e}", exc_info=True)
                    
                    # Safely update health status
                    if provider["name"] not in self.provider_health:
                        self.provider_health[provider["name"]] = {}
                    self.provider_health[provider["name"]]["healthy"] = False
                    
                    logger.warning(f"Marked '{provider['name']}' as unhealthy")
                    
                    if attempt < self.max_retries - 1:
                        logger.info("Retrying with next provider in 1 second...")
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        logger.error(f"All retry attempts exhausted for provider '{provider['name']}'")
                        raise
                        
        except Exception as e:
            logger.error(f"LLM generation failed after {self.max_retries} attempts: {e}", exc_info=True)
            raise
    
    async def _get_response(
        self,
        provider: Dict[str, Any],
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        prompt_tokens: int
    ) -> Dict[str, Any]:
        """Get non-streaming response from provider. Returns response dict."""
        try:
            url = f"{provider['url']}/chat/completions"
            logger.info(f"🔗 Connecting to LMStudio at {url}")
            logger.debug(f"Making request to {url} with model {provider['model']}")
            
            payload = {
                "model": provider["model"],
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            logger.info(f"📤 Sending request to LMStudio (model={payload['model']}, max_tokens={payload['max_tokens']})")
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)[:500]}")
            
            start_time = datetime.now()
            
            try:
                response = await self.http_client.post(url, json=payload)
            except httpx.ConnectError as e:
                    logger.error(f"❌ Connection failed to LMStudio at {url}")
                    logger.error(f"   Error: {e}")
                    logger.error(f"   Make sure LMStudio is running and accessible at {provider['url']}")
                    raise ConnectionError(f"Cannot connect to LMStudio at {provider['url']}. Is it running?")
            except httpx.TimeoutException as e:
                logger.error(f"⏰ Request to LMStudio timed out after {self.timeout}s")
                logger.error("   The model might be too slow or not responding")
                raise TimeoutError(f"LMStudio request timed out after {self.timeout}s")
            
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            logger.info(f"📥 Received response from {provider['name']} in {latency:.2f}s (status: {response.status_code})")
            
            if response.status_code != 200:
                logger.error(f"Provider returned error status {response.status_code}")
                logger.error(f"Response text: {response.text[:500]}")
                raise RuntimeError(f"Provider returned status {response.status_code}: {response.text}")
            
            data = response.json()
            logger.debug(f"📦 Response data structure: {json.dumps(data, indent=2)[:500]}")
            
            completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
            total_tokens = data.get("usage", {}).get("total_tokens", prompt_tokens + completion_tokens)
            
            # Handle both chat completion and text completion response formats
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                
                # Try message.content first (chat completion format)
                if "message" in choice and "content" in choice["message"]:
                    response_content = choice["message"]["content"]
                    logger.debug("✓ Extracted response from message.content (chat format)")
                # Fall back to text field (text completion format - LMStudio default)
                elif "text" in choice:
                    response_content = choice["text"]
                    logger.debug("✓ Extracted response from text field (completion format)")
                else:
                    logger.error(f"❌ Unknown response format: {json.dumps(choice, indent=2)}")
                    raise ValueError("Response format not recognized")
            else:
                logger.error(f"❌ No choices in response: {json.dumps(data, indent=2)}")
                raise ValueError("No choices in LLM response")
            
            logger.info(f"✅ Successfully generated response ({len(response_content)} chars, {completion_tokens} completion tokens)")
            
            # Log token usage
            logger.debug(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            
            return {
                "response": response_content,
                "tokens_generated": completion_tokens,
                "model": provider["model"]
            }
            
            
        except Exception as e:
            logger.error(f"Error with provider '{provider['name']}': {e}", exc_info=True)
            raise
    
    def _parse_stream_line(self, line: str) -> Optional[str]:
        """Parse a single SSE line and extract content."""
        if not line.startswith("data: "):
            return None
        
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            return None
        
        try:
            data = json.loads(data_str)
            if "choices" not in data or len(data["choices"]) == 0:
                return None
            
            choice = data["choices"][0]
            # Try delta.content first (chat completion streaming)
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            
            # Fall back to text field (text completion streaming)
            if not content:
                content = choice.get("text", "")
            
            return content if content else None
        except json.JSONDecodeError:
            return None
    
    async def _stream_response(
        self,
        provider: Dict[str, Any],
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        _prompt_tokens: int
    ) -> AsyncGenerator[str, None]:
        """Stream response from provider."""
        try:
            url = f"{provider['url']}/chat/completions"
            
            payload = {
                "model": provider["model"],
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True
            }
            
            async with self.http_client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Provider returned status {response.status_code}")
                
                async for line in response.aiter_lines():
                    content = self._parse_stream_line(line)
                    if content:
                        yield content
                            
        except Exception as e:
            logger.error(f"Stream error with provider '{provider['name']}': {e}")
            raise
    
    def _get_next_provider(self) -> Optional[Dict[str, Any]]:
        """Get next healthy provider."""
        # Try current provider first if healthy
        current = self.providers[self.current_provider_idx % len(self.providers)]
        if self.provider_health.get(current["name"], {}).get("healthy", True):
            self.current_provider_idx += 1
            return current
        
        # Try to find healthy provider
        for i, provider in enumerate(self.providers):
            if self.provider_health.get(provider["name"], {}).get("healthy", True):
                self.current_provider_idx = i + 1
                return provider
        
        return None
    
    async def validate_response(self, response: str, max_length: int = 10000) -> bool:
        """
        Validate response quality and safety.
        
        Args:
            response: Response text to validate
            max_length: Maximum allowed response length
            
        Returns:
            bool: Whether response is valid
        """
        try:
            await asyncio.sleep(0)
            # Check length
            if len(response) > max_length:
                logger.warning(f"Response exceeds max length: {len(response)} > {max_length}")
                return False
            
            # Check for minimum content
            if len(response.strip()) < 10:
                logger.warning("Response too short")
                return False
            
            # Check for common error patterns
            error_patterns = [
                "error:",
                "sorry, i can't",
                "i don't know",
                "[error]",
                "<!doctype",  # HTML responses
                "<?xml"  # XML responses
            ]
            
            response_lower = response.lower()
            for pattern in error_patterns:
                if pattern in response_lower:
                    logger.warning(f"Response contains error pattern: {pattern}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Response validation error: {e}")
            return False
    
    # ============================================================================
    # Phase 4: ADAPTIVE LLM INFERENCE
    # ============================================================================
    
    def _check_factual_patterns(self, query_lower: str) -> Optional[Dict[str, Any]]:
        """Check for factual query patterns."""
        factual_patterns = [
            r"\b(what|when|where|who)\b.*\?",
            r"\b(define|explain|describe|meaning)\b",
            r"\b(list|enumerate|how many)\b",
            r"\b(fact|truth|definition)\b"
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, query_lower):
                return {
                    "query_type": "factual",
                    "temperature": 0.3,
                    "top_p": 0.85,
                    "max_tokens": min(settings.max_llm_output_tokens, 1024),
                    "reasoning_level": "direct",
                    "parameter_reasoning": "Low temperature for factual accuracy"
                }
        return None
    
    def _check_analytical_patterns(self, query_lower: str) -> Optional[Dict[str, Any]]:
        """Check for analytical query patterns."""
        analytical_patterns = [
            r"\b(compare|difference|contrast|versus)\b",
            r"\b(analyze|evaluate|assess)\b",
            r"\b(pros|cons|advantages|disadvantages)\b",
            r"\b(better|best|vs|versus)\b"
        ]
        
        for pattern in analytical_patterns:
            if re.search(pattern, query_lower):
                return {
                    "query_type": "analytical",
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "max_tokens": min(settings.max_llm_output_tokens, 2048),
                    "reasoning_level": "analytical",
                    "parameter_reasoning": "Medium temperature for balanced analysis"
                }
        return None
    
    def _check_creative_patterns(self, query_lower: str) -> Optional[Dict[str, Any]]:
        """Check for creative query patterns."""
        creative_patterns = [
            r"\b(create|generate|write|compose|imagine)\b",
            r"\b(example|scenario|use case)\b",
            r"\b(brainstorm|idea)\b",
            r"\b(suggest|recommend)\b"
        ]
        
        for pattern in creative_patterns:
            if re.search(pattern, query_lower):
                return {
                    "query_type": "creative",
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "max_tokens": min(settings.max_llm_output_tokens, 2048),
                    "reasoning_level": "creative",
                    "parameter_reasoning": "Higher temperature for creative outputs"
                }
        return None
    
    def _check_technical_patterns(self, query_lower: str) -> Optional[Dict[str, Any]]:
        """Check for technical query patterns."""
        technical_patterns = [
            r"\b(debug|troubleshoot|error|fix)\b",
            r"\b(code|script|program|implement)\b",
            r"\b(syntax|error message)\b",
            r"\b(how to solve|solution)\b"
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, query_lower):
                return {
                    "query_type": "technical",
                    "temperature": 0.4,
                    "top_p": 0.85,
                    "max_tokens": min(settings.max_llm_output_tokens, 2048),
                    "reasoning_level": "step_by_step",
                    "parameter_reasoning": "Low temperature for technical precision, step-by-step reasoning"
                }
        return None
    
    def _check_procedural_patterns(self, query_lower: str) -> Optional[Dict[str, Any]]:
        """Check for procedural query patterns."""
        procedural_patterns = [
            r"\b(how to|steps|process|procedure)\b",
            r"\b(guide|tutorial|manual)\b",
            r"\b(step by step)\b"
        ]
        
        for pattern in procedural_patterns:
            if re.search(pattern, query_lower):
                return {
                    "query_type": "procedural",
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "max_tokens": min(settings.max_llm_output_tokens, 2048),
                    "reasoning_level": "structured",
                    "parameter_reasoning": "Medium temperature with structured output for procedures"
                }
        return None
    
    def _check_opinion_patterns(self, query_lower: str) -> Optional[Dict[str, Any]]:
        """Check for opinion/discussion query patterns."""
        opinion_patterns = [
            r"\b(opinion|think|believe|discuss)\b",
            r"\b(what do you|your thoughts)\b",
            r"\b(why|possible reasons)\b"
        ]
        
        for pattern in opinion_patterns:
            if re.search(pattern, query_lower):
                return {
                    "query_type": "discussion",
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": min(settings.max_llm_output_tokens, 2048),
                    "reasoning_level": "exploratory",
                    "parameter_reasoning": "Balanced temperature for discussion"
                }
        return None
    
    def get_adaptive_inference_params(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect query type and determine optimal LLM inference parameters.
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            Dictionary with optimized temperature, top_p, and other parameters
        """
        try:
            query_lower = query.lower()
            
            # Default params
            adaptive_params = {
                "query_type": "general",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": settings.max_llm_output_tokens,
                "reasoning_level": "standard",
                "parameter_reasoning": ""
            }
            
            # Check each pattern type
            pattern_checkers = [
                self._check_factual_patterns,
                self._check_analytical_patterns,
                self._check_creative_patterns,
                self._check_technical_patterns,
                self._check_procedural_patterns,
                self._check_opinion_patterns
            ]
            
            for checker in pattern_checkers:
                result = checker(query_lower)
                if result:
                    adaptive_params.update(result)
                    break
            
            # Adjust based on query length
            query_tokens = self.count_tokens(query)
            if query_tokens > 200:
                adaptive_params["temperature"] = max(0.3, adaptive_params["temperature"] - 0.1)
                adaptive_params["max_tokens"] = min(adaptive_params["max_tokens"], 1024)
            
            logger.info(f"Adaptive params - Type: {adaptive_params['query_type']}, "
                       f"Temp: {adaptive_params['temperature']}, Top-p: {adaptive_params['top_p']}")
            
            return adaptive_params
            
        except Exception as e:
            logger.warning(f"Adaptive parameter detection failed: {e}")
            return {
                "query_type": "general",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": settings.max_llm_output_tokens,
                "reasoning_level": "standard",
                "parameter_reasoning": "Default parameters (error in detection)"
            }
    
    async def generate_response_adaptive(
        self,
        prompt: str,
        query: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate response with adaptive parameters based on query type.
        
        Args:
            prompt: The LLM prompt
            query: Original user query (for type detection)
            system_prompt: Optional system context
            stream: Whether to stream response
            
        Returns:
            Response dict with adaptive parameters applied
        """
        try:
            # Step 1: Detect query type and get adaptive params
            adaptive_params = self.get_adaptive_inference_params(query, system_prompt)
            
            logger.info("🎯 Using adaptive parameters:")
            logger.info(f"  Type: {adaptive_params['query_type']}")
            logger.info(f"  Temperature: {adaptive_params['temperature']}")
            logger.info(f"  Top-P: {adaptive_params['top_p']}")
            logger.info(f"  Reasoning: {adaptive_params['reasoning_level']}")
            
            # Step 2: Generate response with adaptive params
            response = await self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=adaptive_params["temperature"],
                top_p=adaptive_params["top_p"],
                max_tokens=adaptive_params["max_tokens"],
                stream=stream
            )
            
            # Step 3: Add adaptive metadata to response
            if isinstance(response, dict):
                response["adaptive_metadata"] = {
                    "query_type": adaptive_params["query_type"],
                    "temperature_applied": adaptive_params["temperature"],
                    "reasoning_level": adaptive_params["reasoning_level"],
                    "parameter_reasoning": adaptive_params["parameter_reasoning"]
                }
            
            return response
            
        except Exception as e:
            logger.warning(f"Adaptive response generation failed, falling back to default: {e}")
            return await self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                top_p=0.9
            )
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()


# Global instance
llm_handler = LLMHandler()
