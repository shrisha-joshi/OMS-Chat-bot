"""
Enhanced LLM Handler Service with multi-provider support and robustness.
This module manages communication with various LLM providers including LMStudio,
Ollama, and API-based models with automatic fallback and error recovery.
"""

import logging
import httpx
import json
import asyncio
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
        self.timeout = 120.0
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
            self.http_client = httpx.AsyncClient(timeout=self.timeout)
            logger.debug("HTTP client initialized in __init__")
        except Exception as e:
            logger.warning(f"Failed to initialize HTTP client: {e}")
            self.http_client = None
        
        # Setup providers immediately  
        self._setup_providers()
        logger.debug(f"Providers initialized in __init__: {[p['name'] for p in self.providers]}")
        
    async def initialize(self):
        """Initialize LLM handler with available providers."""
        try:
            logger.info("Initializing LLM handler...")
            
            # Initialize tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=self.timeout)
            
            # Configure providers
            self._setup_providers()
            
            # Test provider health
            await self._check_provider_health()
            
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
        """Check health of all providers."""
        logger.info("Checking provider health...")
        for provider in self.providers:
            try:
                logger.info(f"Checking health of '{provider['name']}' at {provider['url']}/models")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{provider['url']}/models",
                        headers={"Accept": "application/json"}
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
            # Prepare messages
            # Note: Some models don't support 'system' role, so we prepend system prompt to user message
            messages = []
            
            if system_prompt:
                # Combine system prompt with user prompt
                combined_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                combined_prompt = prompt
            
            messages.append({"role": "user", "content": combined_prompt})
            
            # Count tokens
            prompt_tokens = self.count_tokens(combined_prompt)
            system_tokens = 0  # Already included in combined_prompt
            
            # Set max tokens if not provided
            if not max_tokens:
                # Use configured max output tokens, but ensure it fits in context window
                configured_max = settings.max_llm_output_tokens
                available_for_output = 32768 - prompt_tokens - system_tokens - 256  # Leave buffer
                max_tokens = min(configured_max, available_for_output)
                max_tokens = max(max_tokens, 512)  # Ensure minimum
            
            # Try providers with retry logic
            logger.debug(f"Provider health status: {self.provider_health}")
            logger.debug(f"Available providers: {[p['name'] for p in self.providers]}")
            
            for attempt in range(self.max_retries):
                provider = self._get_next_provider()
                
                if not provider:
                    logger.error("No available LLM providers found")
                    raise Exception("No available LLM providers")
                
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
                    self.provider_health[provider["name"]]["healthy"] = False
                    logger.warning(f"Marked '{provider['name']}' as unhealthy")
                    
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying with next provider in 1 second...")
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
    ) -> str:
        """Get non-streaming response from provider. Returns response text directly."""
        try:
            url = f"{provider['url']}/chat/completions"
            logger.debug(f"Making request to {url} with model {provider['model']}")
            
            payload = {
                "model": provider["model"],
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            logger.debug(f"Request payload: model={payload['model']}, max_tokens={payload['max_tokens']}, temperature={payload['temperature']}")
            
            start_time = datetime.now()
            response = await self.http_client.post(url, json=payload)
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            logger.info(f"Received response from {provider['name']} in {latency:.2f}s (status: {response.status_code})")
            
            if response.status_code != 200:
                logger.error(f"Provider returned error status {response.status_code}")
                logger.error(f"Response text: {response.text[:500]}")
                raise Exception(f"Provider returned status {response.status_code}: {response.text}")
            
            data = response.json()
            
            completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
            total_tokens = data.get("usage", {}).get("total_tokens", prompt_tokens + completion_tokens)
            
            response_content = data["choices"][0]["message"]["content"]
            logger.info(f"✅ Successfully generated response ({len(response_content)} chars, {completion_tokens} completion tokens)")
            
            # Log token usage
            logger.debug(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            
            return response_content  # Return just the string, not dict
            
            
        except Exception as e:
            logger.error(f"Error with provider '{provider['name']}': {e}", exc_info=True)
            raise
    
    async def _stream_response(
        self,
        provider: Dict[str, Any],
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        prompt_tokens: int
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
                    raise Exception(f"Provider returned status {response.status_code}")
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
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
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()


# Global instance
llm_handler = LLMHandler()
