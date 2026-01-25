"""
Unit tests for LLM Provider abstraction layer.

Tests the base provider interface, Ollama provider, Azure OpenAI provider,
and provider factory functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator

from backend.services.llm_providers.base_provider import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    LLMProviderError,
    RateLimitError,
    ModelNotFoundError,
    LLMProviderType,
    StreamChunk,
)
from backend.services.llm_providers.ollama_provider import OllamaProvider
from backend.services.llm_providers.azure_openai_provider import AzureOpenAIProvider
from backend.services.llm_providers.provider_factory import (
    LLMProviderFactory,
    get_provider_factory,
    get_llm_provider,
)


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        
        assert config.provider_type == LLMProviderType.OLLAMA
        assert config.model_name == "llama3"
        assert config.temperature == 0.1
        assert config.max_tokens == 2048
        assert config.timeout == 120.0
        assert config.retry_attempts == 3
    
    def test_azure_config(self):
        """Test Azure-specific configuration."""
        config = LLMConfig(
            provider_type=LLMProviderType.AZURE_OPENAI,
            model_name="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_api_key="test-key",
            azure_deployment_name="gpt-4-deployment"
        )
        
        assert config.provider_type == LLMProviderType.AZURE_OPENAI
        assert config.azure_endpoint == "https://test.openai.azure.com/"
        assert config.azure_api_key == "test-key"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""
    
    def test_response_creation(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Hello, world!",
            model="llama3",
            provider="ollama",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_ms=100.5
        )
        
        assert response.content == "Hello, world!"
        assert response.model == "llama3"
        assert response.usage["total_tokens"] == 15
    
    def test_response_to_dict(self):
        """Test converting response to dictionary."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            provider="azure_openai"
        )
        
        result = response.to_dict()
        
        assert isinstance(result, dict)
        assert result["content"] == "Test"
        assert result["provider"] == "azure_openai"


class TestLLMProviderErrors:
    """Tests for LLM provider error classes."""
    
    def test_base_provider_error(self):
        """Test base provider error."""
        error = LLMProviderError(
            "Test error",
            provider="test",
            retryable=True
        )
        
        assert str(error) == "Test error"
        assert error.provider == "test"
        assert error.retryable is True
    
    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError("azure_openai", retry_after=30.0)
        
        assert "Rate limit" in str(error)
        assert error.retry_after == 30.0
        assert error.retryable is True
    
    def test_model_not_found_error(self):
        """Test model not found error."""
        error = ModelNotFoundError("ollama", "nonexistent-model")
        
        assert "nonexistent-model" in str(error)
        assert error.model_name == "nonexistent-model"
        assert error.retryable is False


class TestOllamaProvider:
    """Tests for Ollama provider."""
    
    @pytest.fixture
    def ollama_config(self):
        """Create Ollama configuration."""
        return LLMConfig(
            provider_type=LLMProviderType.OLLAMA,
            model_name="llama3",
            ollama_base_url="http://localhost:11434"
        )
    
    @pytest.fixture
    def ollama_provider(self, ollama_config):
        """Create Ollama provider instance."""
        return OllamaProvider(ollama_config)
    
    def test_provider_name(self, ollama_provider):
        """Test provider name property."""
        assert ollama_provider.provider_name == "ollama"
    
    def test_base_url(self, ollama_provider):
        """Test base URL property."""
        assert ollama_provider.base_url == "http://localhost:11434"
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, ollama_provider):
        """Test successful initialization."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3:latest"}]
        }
        
        with patch.object(ollama_provider, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)
            ollama_provider._client = mock_client
            
            # Manually mark as initialized for testing
            ollama_provider.is_initialized = True
            ollama_provider._available_models = ["llama3:latest"]
            
            assert ollama_provider.is_initialized
            assert "llama3:latest" in ollama_provider._available_models
    
    @pytest.mark.asyncio
    async def test_generate_response(self, ollama_provider):
        """Test generating a response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Hello!"},
            "prompt_eval_count": 10,
            "eval_count": 5,
            "total_duration": 1000000
        }
        
        with patch.object(ollama_provider, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            ollama_provider._client = mock_client
            ollama_provider.is_initialized = True
            ollama_provider._available_models = ["llama3:latest"]
            
            response = await ollama_provider.generate("Hello")
            
            assert response.content == "Hello!"
            assert response.provider == "ollama"
            assert response.usage["total_tokens"] == 15
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, ollama_provider):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch.object(ollama_provider, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)
            ollama_provider._client = mock_client
            
            result = await ollama_provider.health_check()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, ollama_provider):
        """Test failed health check."""
        result = await ollama_provider.health_check()
        
        assert result is False  # No client initialized
    
    def test_get_model_info(self, ollama_provider):
        """Test getting model info."""
        ollama_provider._available_models = ["llama3", "mistral"]
        ollama_provider._request_count = 5
        
        info = ollama_provider.get_model_info()
        
        assert info["provider"] == "ollama"
        assert info["available_models"] == ["llama3", "mistral"]
        assert info["request_count"] == 5


class TestAzureOpenAIProvider:
    """Tests for Azure OpenAI provider."""
    
    @pytest.fixture
    def azure_config(self):
        """Create Azure configuration."""
        return LLMConfig(
            provider_type=LLMProviderType.AZURE_OPENAI,
            model_name="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_api_key="test-key",
            azure_deployment_name="gpt-4-deployment"
        )
    
    @pytest.fixture
    def azure_provider(self, azure_config):
        """Create Azure provider instance."""
        return AzureOpenAIProvider(azure_config)
    
    def test_provider_name(self, azure_provider):
        """Test provider name property."""
        assert azure_provider.provider_name == "azure_openai"
    
    def test_deployment_name(self, azure_provider):
        """Test deployment name property."""
        assert azure_provider.deployment_name == "gpt-4-deployment"
    
    def test_model_context_limits(self, azure_provider):
        """Test model context limit lookup."""
        limits = azure_provider.MODEL_CONTEXT_LIMITS
        
        assert limits["gpt-4"] == 8192
        assert limits["gpt-4-turbo"] == 128000
        assert limits["gpt-35-turbo"] == 16385
    
    def test_get_model_info(self, azure_provider):
        """Test getting model info."""
        info = azure_provider.get_model_info()
        
        assert info["provider"] == "azure_openai"
        assert info["deployment"] == "gpt-4-deployment"
        assert "context_limit" in info


class TestLLMProviderFactory:
    """Tests for LLM provider factory."""
    
    @pytest.fixture
    def factory(self):
        """Create a fresh factory instance."""
        return LLMProviderFactory()
    
    def test_create_ollama_provider(self, factory):
        """Test creating Ollama provider."""
        config = LLMConfig(provider_type=LLMProviderType.OLLAMA)
        
        provider = factory.create_provider(config)
        
        assert isinstance(provider, OllamaProvider)
        assert provider.provider_name == "ollama"
    
    def test_create_azure_provider(self, factory):
        """Test creating Azure provider."""
        config = LLMConfig(
            provider_type=LLMProviderType.AZURE_OPENAI,
            azure_endpoint="https://test.openai.azure.com/",
            azure_api_key="test-key"
        )
        
        provider = factory.create_provider(config)
        
        assert isinstance(provider, AzureOpenAIProvider)
        assert provider.provider_name == "azure_openai"
    
    def test_caching_with_instance_id(self, factory):
        """Test provider caching with instance ID."""
        config = LLMConfig(provider_type=LLMProviderType.OLLAMA)
        
        provider1 = factory.create_provider(config, instance_id="test-1")
        provider2 = factory.create_provider(config, instance_id="test-1")
        
        assert provider1 is provider2  # Same instance
    
    def test_get_instance(self, factory):
        """Test getting cached instance."""
        config = LLMConfig(provider_type=LLMProviderType.OLLAMA)
        factory.create_provider(config, instance_id="cached")
        
        result = factory.get_instance("cached")
        
        assert result is not None
        assert result.provider_name == "ollama"
    
    def test_get_nonexistent_instance(self, factory):
        """Test getting non-existent instance returns None."""
        result = factory.get_instance("nonexistent")
        
        assert result is None
    
    def test_set_fallback_chain(self, factory):
        """Test setting fallback chain."""
        configs = [
            LLMConfig(provider_type=LLMProviderType.AZURE_OPENAI),
            LLMConfig(provider_type=LLMProviderType.OLLAMA),
        ]
        
        factory.set_fallback_chain(configs)
        
        assert len(factory._fallback_chain) == 2
    
    @pytest.mark.asyncio
    async def test_cleanup_all(self, factory):
        """Test cleaning up all providers."""
        config = LLMConfig(provider_type=LLMProviderType.OLLAMA)
        factory.create_provider(config, instance_id="cleanup-test")
        
        await factory.cleanup_all()
        
        assert len(factory._instances) == 0


class TestGlobalFactoryFunctions:
    """Tests for global factory functions."""
    
    def test_get_provider_factory_singleton(self):
        """Test factory singleton behavior."""
        factory1 = get_provider_factory()
        factory2 = get_provider_factory()
        
        assert factory1 is factory2
    
    def test_get_llm_provider_with_config(self):
        """Test getting provider with explicit config."""
        config = LLMConfig(
            provider_type=LLMProviderType.OLLAMA,
            model_name="test-model"
        )
        
        provider = get_llm_provider(config)
        
        assert provider.provider_name == "ollama"
        assert provider.config.model_name == "test-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
