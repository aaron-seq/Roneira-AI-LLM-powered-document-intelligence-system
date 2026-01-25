"""
Ollama LLM Provider Implementation.

Provides integration with Ollama for local LLM inference,
supporting streaming, batch processing, and model management.
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from .base_provider import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    LLMProviderError,
    ModelNotFoundError,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local inference.
    
    Connects to an Ollama server for running local LLMs like
    Llama 3, Mistral, Qwen, and other supported models.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize Ollama provider.
        
        Args:
            config: Provider configuration with ollama_base_url.
        """
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._available_models: list = []
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "ollama"
    
    @property
    def base_url(self) -> str:
        """Get the Ollama API base URL."""
        return self.config.ollama_base_url.rstrip("/")
    
    async def initialize(self) -> None:
        """Initialize connection to Ollama server."""
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.config.timeout, connect=10.0),
            )
            
            # Check server availability
            response = await self._client.get("/api/tags")
            if response.status_code != 200:
                raise LLMProviderError(
                    f"Ollama server returned status {response.status_code}",
                    self.provider_name
                )
            
            # Cache available models
            data = response.json()
            self._available_models = [m["name"] for m in data.get("models", [])]
            
            # Verify requested model is available
            if not self._is_model_available(self.config.model_name):
                logger.warning(
                    f"Model {self.config.model_name} not found locally. "
                    "Will attempt to pull on first request."
                )
            
            self.is_initialized = True
            logger.info(
                f"Ollama provider initialized with model {self.config.model_name}"
            )
            
        except httpx.ConnectError as e:
            raise LLMProviderError(
                f"Failed to connect to Ollama at {self.base_url}: {e}",
                self.provider_name,
                retryable=True,
                original_error=e
            )
        except Exception as e:
            raise LLMProviderError(
                f"Failed to initialize Ollama provider: {e}",
                self.provider_name,
                original_error=e
            )
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if model is available locally."""
        # Handle both 'llama3' and 'llama3:latest' formats
        base_name = model_name.split(":")[0]
        return any(
            m.startswith(base_name) for m in self._available_models
        )
    
    async def _ensure_model(self) -> None:
        """Ensure model is available, pulling if necessary."""
        if self._is_model_available(self.config.model_name):
            return
        
        logger.info(f"Pulling model {self.config.model_name}...")
        try:
            response = await self._client.post(
                "/api/pull",
                json={"name": self.config.model_name},
                timeout=600.0  # Model pulls can take a long time
            )
            if response.status_code != 200:
                raise ModelNotFoundError(self.provider_name, self.config.model_name)
            
            # Refresh model list
            tags_response = await self._client.get("/api/tags")
            if tags_response.status_code == 200:
                data = tags_response.json()
                self._available_models = [m["name"] for m in data.get("models", [])]
                
        except httpx.TimeoutException:
            raise LLMProviderError(
                f"Timeout while pulling model {self.config.model_name}",
                self.provider_name,
                retryable=True
            )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate response using Ollama.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system context.
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse with generated content.
        """
        if not self.is_initialized:
            await self.initialize()
        
        await self._ensure_model()
        
        start_time = time.perf_counter()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build request payload
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        # Add JSON format if requested
        if kwargs.get("json_format", False):
            payload["format"] = "json"
        
        try:
            response = await self._client.post(
                "/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                error_msg = response.text
                raise LLMProviderError(
                    f"Ollama request failed: {error_msg}",
                    self.provider_name
                )
            
            data = response.json()
            content = data.get("message", {}).get("content", "")
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract token usage
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            }
            
            self._request_count += 1
            self._total_tokens += usage["total_tokens"]
            
            return LLMResponse(
                content=content,
                model=self.config.model_name,
                provider=self.provider_name,
                usage=usage,
                latency_ms=latency_ms,
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "eval_duration": data.get("eval_duration"),
                }
            )
            
        except httpx.TimeoutException as e:
            raise LLMProviderError(
                f"Ollama request timed out after {self.config.timeout}s",
                self.provider_name,
                retryable=True,
                original_error=e
            )
        except httpx.RequestError as e:
            raise LLMProviderError(
                f"Ollama request failed: {e}",
                self.provider_name,
                retryable=True,
                original_error=e
            )
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response from Ollama.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system context.
            **kwargs: Additional parameters.
            
        Yields:
            StreamChunk with incremental content.
        """
        if not self.is_initialized:
            await self.initialize()
        
        await self._ensure_model()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json=payload,
                timeout=self.config.timeout
            ) as response:
                if response.status_code != 200:
                    raise LLMProviderError(
                        f"Ollama stream failed with status {response.status_code}",
                        self.provider_name
                    )
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    import json
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    content = data.get("message", {}).get("content", "")
                    is_done = data.get("done", False)
                    
                    if content:
                        yield StreamChunk(
                            content=content,
                            is_final=is_done,
                            metadata={"model": self.config.model_name}
                        )
                    
                    if is_done:
                        self._request_count += 1
                        break
                        
        except httpx.TimeoutException as e:
            raise LLMProviderError(
                f"Ollama stream timed out",
                self.provider_name,
                retryable=True,
                original_error=e
            )
    
    async def health_check(self) -> bool:
        """Check if Ollama server is healthy."""
        if not self._client:
            return False
        
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        info = super().get_model_info()
        info["available_models"] = self._available_models
        info["base_url"] = self.base_url
        return info
    
    async def cleanup(self) -> None:
        """Clean up Ollama client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().cleanup()
