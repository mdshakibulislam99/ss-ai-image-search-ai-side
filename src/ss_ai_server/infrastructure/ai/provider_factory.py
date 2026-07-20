"""
AI Provider Factory

Factory pattern implementation for creating AI providers
"""

from typing import Any, Dict, List, Type, Optional

from ..domain.interfaces.ai_provider import AIProvider  # type: ignore
from .base_provider import BaseAIProvider


class AIProviderFactory:
    """
    Factory for creating AI provider instances
    
    Supports multiple AI providers and allows dynamic registration
    """
    
    _providers: Dict[str, Type[BaseAIProvider]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseAIProvider]) -> None:
        """
        Register a new AI provider
        
        Args:
            name: Provider name (e.g., 'openclip', 'siglip')
            provider_class: Provider implementation class
        """
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, provider_name: str, **kwargs) -> AIProvider:
        """
        Create an AI provider instance
        
        Args:
            provider_name: Name of the provider to create
            **kwargs: Additional arguments for provider initialization
            
        Returns:
            AI provider instance
            
        Raises:
            ValueError: If provider is not registered
        """
        if provider_name not in cls._providers:
            raise ValueError(
                f"AI provider '{provider_name}' is not registered. "
                f"Available providers: {list(cls._providers.keys())}"
            )
        
        provider_class = cls._providers[provider_name]
        return provider_class(**kwargs)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of available AI providers
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def is_provider_available(cls, provider_name: str) -> bool:
        """
        Check if a provider is available
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if provider is available, False otherwise
        """
        return provider_name in cls._providers
    
    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """
        Get list of available AI models across all providers
        
        Returns:
            List of model information dictionaries
        """
        models = []
        for provider_name in cls._providers:
            try:
                provider_class = cls._providers[provider_name]
                if hasattr(provider_class, 'get_supported_models'):
                    provider_models = provider_class.get_supported_models()
                    for model in provider_models:
                        model_info = {
                            "provider": provider_name,
                            "model": model,
                        }
                        models.append(model_info)
            except Exception:
                pass
        return models


# Register built-in providers
def _register_builtin_providers():
    """Register built-in AI providers"""
    from .openclip_provider import OpenCLIPProvider
    AIProviderFactory.register_provider('openclip', OpenCLIPProvider)


# Initialize built-in providers
_register_builtin_providers()