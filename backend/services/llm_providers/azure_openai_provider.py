"""
Azure OpenAI LLM Provider Implementation.

Provides integration with Azure OpenAI Service for enterprise-grade
LLM inference with support for GPT-4, GPT-4 Turbo, and GPT-3.5.
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from .base_provider import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    LLMProviderError,
    RateLimitError,
    StreamChunk,
)

logger = logging.getLogger(__name__)

# Azure OpenAI SDK import with graceful fallback
try:
    from openai import AsyncAzureOpenAI, APIError, RateLimitError as OpenAIRateLimitError
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    AsyncAzureOpenAI = None
    APIError = Exception
    OpenAIRateLimitError = Exception
    logger.warning(
        "openai package not installed. Install with: pip install openai"
    )


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider for enterprise LLM inference.
    
    Supports GPT-4, GPT-4 Turbo, GPT-3.5 Turbo with features like:
    - Automatic retry with exponential backoff
    - Rate limit handling
    - Streaming responses
    - Token usage tracking
    """
    
    # Model context windows for truncation handling
    MODEL_CONTEXT_LIMITS = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4o": 128000,
        "gpt-35-turbo": 16385,
        "gpt-3.5-turbo": 16385,
    }
    
    def __init__(self, config: LLMConfig):
        """Initialize Azure OpenAI provider.
        
        Args:
            config: Provider configuration with Azure credentials.
        """
        super().__init__(config)
        self._client: Optional[AsyncAzureOpenAI] = None
        self._rate_limit_reset: float = 0
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "azure_openai"
    
    @property
    def deployment_name(self) -> str:
        """Get the Azure deployment name."""
        return self.config.azure_deployment_name or self.config.model_name
    
    async def initialize(self) -> None:
        """Initialize Azure OpenAI client."""
        if not AZURE_OPENAI_AVAILABLE:
            raise LLMProviderError(
                "openai package not installed",
                self.provider_name
            )
        
        if not self.config.azure_api_key or not self.config.azure_endpoint:
            raise LLMProviderError(
                "Azure OpenAI API key and endpoint are required",
                self.provider_name
            )
        
        try:
            self._client = AsyncAzureOpenAI(
                api_key=self.config.azure_api_key,
                api_version=self.config.azure_api_version,
                azure_endpoint=self.config.azure_endpoint,
                timeout=self.config.timeout,
                max_retries=0  # We handle retries ourselves
            )
            
            # Verify connectivity with a minimal request
            # Note: We skip this to avoid unnecessary API calls on init
            self.is_initialized = True
            logger.info(
                f"Azure OpenAI provider initialized with deployment {self.deployment_name}"
            )
            
        except Exception as e:
            raise LLMProviderError(
                f"Failed to initialize Azure OpenAI: {e}",
                self.provider_name,
                original_error=e
            )
    
    async def _execute_with_retry(
        self,
        operation: str,
        func,
        *args,
        **kwargs
    ) -> Any:
        """Execute a function with retry logic.
        
        Args:
            operation: Name of the operation for logging.
            func: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
            
        Returns:
            Function result.
        """
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                # Check rate limit cooldown
                if time.time() < self._rate_limit_reset:
                    wait_time = self._rate_limit_reset - time.time()
                    logger.warning(f"Rate limit active, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                
                return await func(*args, **kwargs)
                
            except OpenAIRateLimitError as e:
                # Extract retry-after if available
                retry_after = getattr(e, "retry_after", 60)
                self._rate_limit_reset = time.time() + retry_after
                
                logger.warning(
                    f"Rate limit hit on {operation}, retry after {retry_after}s "
                    f"(attempt {attempt + 1}/{self.config.retry_attempts})"
                )
                
                if attempt == self.config.retry_attempts - 1:
                    raise RateLimitError(self.provider_name, retry_after)
                
                await asyncio.sleep(min(retry_after, 30))
                last_error = e
                
            except APIError as e:
                # Retry on 5xx errors
                if hasattr(e, "status_code") and 500 <= e.status_code < 600:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Azure API error on {operation}: {e}, "
                        f"retrying in {delay}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                else:
                    raise LLMProviderError(
                        f"Azure OpenAI API error: {e}",
                        self.provider_name,
                        original_error=e
                    )
        
        raise LLMProviderError(
            f"Failed after {self.config.retry_attempts} attempts: {last_error}",
            self.provider_name,
            retryable=True,
            original_error=last_error
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate response using Azure OpenAI.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system context.
            **kwargs: Additional parameters.
            
        Returns:
            LLMResponse with generated content.
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async def _call():
            return await self._client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                stream=False,
            )
        
        response = await self._execute_with_retry("generate", _call)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        
        self._request_count += 1
        self._total_tokens += usage["total_tokens"]
        
        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            usage=usage,
            latency_ms=latency_ms,
            metadata={
                "deployment": self.deployment_name,
                "finish_reason": response.choices[0].finish_reason,
            }
        )
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response from Azure OpenAI.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system context.
            **kwargs: Additional parameters.
            
        Yields:
            StreamChunk with incremental content.
        """
        if not self.is_initialized:
            await self.initialize()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = await self._client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    is_final = chunk.choices[0].finish_reason is not None
                    
                    yield StreamChunk(
                        content=content,
                        is_final=is_final,
                        metadata={"model": self.deployment_name}
                    )
                    
                    if is_final:
                        self._request_count += 1
                        break
                        
        except OpenAIRateLimitError as e:
            raise RateLimitError(self.provider_name, getattr(e, "retry_after", 60))
        except APIError as e:
            raise LLMProviderError(
                f"Azure OpenAI stream error: {e}",
                self.provider_name,
                original_error=e
            )
    
    async def health_check(self) -> bool:
        """Check Azure OpenAI availability."""
        if not self._client:
            return False
        
        try:
            # Use a minimal request to check health
            response = await self._client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            return response is not None
        except Exception as e:
            logger.warning(f"Azure OpenAI health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Azure OpenAI model information."""
        info = super().get_model_info()
        info["deployment"] = self.deployment_name
        info["context_limit"] = self.MODEL_CONTEXT_LIMITS.get(
            self.config.model_name.lower(), 8192
        )
        info["rate_limit_reset"] = max(0, self._rate_limit_reset - time.time())
        return info
    
    async def cleanup(self) -> None:
        """Clean up Azure OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
        await super().cleanup()
