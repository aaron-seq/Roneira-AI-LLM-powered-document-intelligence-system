"""
Base LLM Provider Interface.

Defines the abstract interface that all LLM providers must implement,
enabling seamless switching between Azure OpenAI, Ollama, and other backends.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    VLLM = "vllm"


@dataclass
class LLMConfig:
    """Configuration for LLM provider.
    
    Attributes:
        provider_type: The type of LLM provider to use.
        model_name: Name of the model to use.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.
        retry_attempts: Number of retry attempts on failure.
        retry_delay: Delay between retries in seconds.
    """
    provider_type: LLMProviderType = LLMProviderType.OLLAMA
    model_name: str = "llama3"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: float = 120.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Provider-specific configs
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_version: str = "2024-02-01"
    azure_deployment_name: Optional[str] = None
    
    ollama_base_url: str = "http://localhost:11434"
    
    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 150000


@dataclass
class LLMResponse:
    """Response from LLM generation.
    
    Attributes:
        content: Generated text content.
        model: Model that generated the response.
        provider: Provider that handled the request.
        usage: Token usage statistics.
        latency_ms: Response latency in milliseconds.
        metadata: Additional provider-specific metadata.
    """
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class StreamChunk:
    """A chunk from a streaming LLM response."""
    content: str
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM providers must implement this interface to ensure
    consistent behavior across different backends.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize provider with configuration.
        
        Args:
            config: Provider configuration.
        """
        self.config = config
        self.is_initialized = False
        self._request_count = 0
        self._total_tokens = 0
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider connection.
        
        Should establish connections, validate credentials,
        and prepare the provider for requests.
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a response for the given prompt.
        
        Args:
            prompt: User prompt to generate response for.
            system_prompt: Optional system prompt for context.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            LLMResponse with generated content and metadata.
            
        Raises:
            LLMProviderError: If generation fails.
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response for the given prompt.
        
        Args:
            prompt: User prompt to generate response for.
            system_prompt: Optional system prompt for context.
            **kwargs: Additional provider-specific parameters.
            
        Yields:
            StreamChunk objects with incremental content.
            
        Raises:
            LLMProviderError: If streaming fails.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and operational.
        
        Returns:
            True if provider is healthy, False otherwise.
        """
        pass
    
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts.
        
        Default implementation processes sequentially.
        Providers can override for optimized batch processing.
        
        Args:
            prompts: List of prompts to process.
            system_prompt: Optional shared system prompt.
            **kwargs: Additional parameters.
            
        Returns:
            List of LLMResponse objects.
        """
        responses = []
        for prompt in prompts:
            response = await self.generate(prompt, system_prompt, **kwargs)
            responses.append(response)
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "provider": self.provider_name,
            "model": self.config.model_name,
            "is_initialized": self.is_initialized,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
        }
    
    async def cleanup(self) -> None:
        """Clean up provider resources.
        
        Override to implement provider-specific cleanup.
        """
        self.is_initialized = False
        logger.info(f"{self.provider_name} provider cleaned up")


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    
    def __init__(
        self,
        message: str,
        provider: str,
        retryable: bool = False,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.original_error = original_error


class RateLimitError(LLMProviderError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, provider: str, retry_after: Optional[float] = None):
        super().__init__(
            f"Rate limit exceeded for {provider}",
            provider,
            retryable=True
        )
        self.retry_after = retry_after


class ModelNotFoundError(LLMProviderError):
    """Raised when requested model is not available."""
    
    def __init__(self, provider: str, model_name: str):
        super().__init__(
            f"Model '{model_name}' not found in {provider}",
            provider,
            retryable=False
        )
        self.model_name = model_name
