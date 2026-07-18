"""
AI Provider interface - Abstract base class for AI providers
"""

from abc import ABC, abstractmethod
from typing import Dict,  List, Optional, Union

from ..value_objects.embedding_vector import EmbeddingVector  # type: ignore


class ModelInfo:
    """Model information"""
    
    def __init__(self, name: str, provider: str, dimensions: int, version: Optional[str] = None) -> None:
        self.name = name
        self.provider = provider
        self.dimensions = dimensions
        self.version = version
    
    def to_dict(self) -> Dict[str, Union[str, int, None]]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "provider": self.provider,
            "dimensions": self.dimensions,
            "version": self.version,
        }


class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name (e.g., 'openclip', 'siglip')"""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[ModelInfo]:
        """Return list of supported models"""
        pass
    
    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load specific model into memory"""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload model from memory"""
        pass
    
    @abstractmethod
    def generate_embedding(self, image_data: bytes) -> EmbeddingVector:
        """
        Generate embedding for single image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def generate_embeddings_batch(self, images_data: List[bytes]) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple images
        
        Args:
            images_data: List of raw image bytes
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model information"""
        pass
    
    @abstractmethod
    def warmup(self) -> None:
        """Warm up model for faster inference"""
        pass
    
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        pass