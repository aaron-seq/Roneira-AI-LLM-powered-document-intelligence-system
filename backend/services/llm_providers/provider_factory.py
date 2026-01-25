"""
LLM Provider Factory.

Factory pattern implementation for creating and managing LLM providers
with support for fallback chains and dynamic provider switching.
"""

import logging
from typing import Dict, Optional, Type

from .base_provider import (
    BaseLLMProvider,
    LLMConfig,
    LLMProviderError,
    LLMProviderType,
)
from .ollama_provider import OllamaProvider
from .azure_openai_provider import AzureOpenAIProvider

logger = logging.getLogger(__name__)


# Registry of available providers
PROVIDER_REGISTRY: Dict[LLMProviderType, Type[BaseLLMProvider]] = {
    LLMProviderType.OLLAMA: OllamaProvider,
    LLMProviderType.AZURE_OPENAI: AzureOpenAIProvider,
}


class LLMProviderFactory:
    """Factory for creating and managing LLM providers.
    
    Supports:
    - Dynamic provider creation based on configuration
    - Fallback chain for provider failures
    - Caching of provider instances
    """
    
    def __init__(self):
        """Initialize the factory."""
        self._instances: Dict[str, BaseLLMProvider] = {}
        self._fallback_chain: list = []
    
    def create_provider(
        self,
        config: LLMConfig,
        instance_id: Optional[str] = None
    ) -> BaseLLMProvider:
        """Create a provider instance from configuration.
        
        Args:
            config: Provider configuration.
            instance_id: Optional ID for caching the instance.
            
        Returns:
            Configured provider instance.
            
        Raises:
            LLMProviderError: If provider type is not supported.
        """
        provider_type = config.provider_type
        
        if provider_type not in PROVIDER_REGISTRY:
            raise LLMProviderError(
                f"Unsupported provider type: {provider_type}",
                provider="factory"
            )
        
        # Check cache
        cache_key = instance_id or f"{provider_type}:{config.model_name}"
        if cache_key in self._instances:
            return self._instances[cache_key]
        
        # Create new instance
        provider_class = PROVIDER_REGISTRY[provider_type]
        provider = provider_class(config)
        
        # Cache if ID provided
        if instance_id:
            self._instances[cache_key] = provider
        
        logger.info(f"Created {provider_type} provider with model {config.model_name}")
        return provider
    
    def set_fallback_chain(self, configs: list) -> None:
        """Set the fallback chain for provider failures.
        
        Args:
            configs: List of LLMConfig objects in priority order.
        """
        self._fallback_chain = configs
        logger.info(f"Set fallback chain with {len(configs)} providers")
    
    async def get_available_provider(self) -> Optional[BaseLLMProvider]:
        """Get the first available provider from the fallback chain.
        
        Returns:
            First healthy provider, or None if all fail.
        """
        for config in self._fallback_chain:
            try:
                provider = self.create_provider(config)
                await provider.initialize()
                
                if await provider.health_check():
                    logger.info(f"Using provider: {provider.provider_name}")
                    return provider
            except Exception as e:
                logger.warning(
                    f"Provider {config.provider_type} unavailable: {e}"
                )
                continue
        
        logger.error("No providers available in fallback chain")
        return None
    
    def get_instance(self, instance_id: str) -> Optional[BaseLLMProvider]:
        """Get a cached provider instance.
        
        Args:
            instance_id: Instance identifier.
            
        Returns:
            Cached provider or None.
        """
        return self._instances.get(instance_id)
    
    async def cleanup_all(self) -> None:
        """Clean up all cached provider instances."""
        for instance_id, provider in self._instances.items():
            try:
                await provider.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up {instance_id}: {e}")
        
        self._instances.clear()
        logger.info("All provider instances cleaned up")


# Global factory instance
_factory: Optional[LLMProviderFactory] = None


def get_provider_factory() -> LLMProviderFactory:
    """Get the global provider factory instance.
    
    Returns:
        LLMProviderFactory singleton.
    """
    global _factory
    if _factory is None:
        _factory = LLMProviderFactory()
    return _factory


def get_llm_provider(config: Optional[LLMConfig] = None) -> BaseLLMProvider:
    """Get an LLM provider instance.
    
    Convenience function to get a provider with default configuration
    from environment if no config is provided.
    
    Args:
        config: Optional provider configuration.
        
    Returns:
        Configured LLM provider.
    """
    if config is None:
        config = _load_default_config()
    
    factory = get_provider_factory()
    return factory.create_provider(config)


def _load_default_config() -> LLMConfig:
    """Load default configuration from environment/settings.
    
    Returns:
        LLMConfig with default values.
    """
    try:
        from backend.core.config import get_settings
        settings = get_settings()
        
        # Determine provider type
        provider_type = LLMProviderType.OLLAMA
        
        # Check if Azure is configured
        if settings.azure_openai_api_key and settings.azure_openai_endpoint:
            provider_type = LLMProviderType.AZURE_OPENAI
        
        return LLMConfig(
            provider_type=provider_type,
            model_name=settings.ollama_model if provider_type == LLMProviderType.OLLAMA 
                      else settings.azure_openai_deployment_name,
            ollama_base_url=settings.ollama_base_url,
            azure_api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_api_version=settings.azure_openai_api_version,
            azure_deployment_name=settings.azure_openai_deployment_name,
        )
    except Exception as e:
        logger.warning(f"Failed to load settings, using defaults: {e}")
        return LLMConfig()


async def create_configured_provider() -> BaseLLMProvider:
    """Create and initialize a provider from application settings.
    
    Convenience function for creating a ready-to-use provider.
    
    Returns:
        Initialized LLM provider.
    """
    config = _load_default_config()
    provider = get_llm_provider(config)
    await provider.initialize()
    return provider
