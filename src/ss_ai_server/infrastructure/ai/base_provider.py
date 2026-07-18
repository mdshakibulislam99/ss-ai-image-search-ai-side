"""
Base AI Provider - Base implementation for AI providers
"""

from typing import List, Optional

from ...domain.interfaces.ai_provider import AIProvider, ModelInfo  # type: ignore
from ...domain.value_objects.embedding_vector import EmbeddingVector  # type: ignore


class BaseAIProvider(AIProvider):
    """Base implementation for AI providers"""
    
    def __init__(self) -> None:
        """Initialize base provider"""
        self._model = None
        self._model_name = None
        self._loaded = False
    
    def get_provider_name(self) -> str:
        """Return provider name"""
        raise NotImplementedError("Subclasses must implement get_provider_name")
    
    def get_supported_models(self) -> List[ModelInfo]:
        """Return list of supported models"""
        raise NotImplementedError("Subclasses must implement get_supported_models")
    
    def load_model(self, model_name: str) -> None:
        """Load specific model into memory"""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def unload_model(self) -> None:
        """Unload model from memory"""
        self._model = None
        self._model_name = None
        self._loaded = False
    
    def generate_embedding(self, image_data: bytes) -> EmbeddingVector:
        """
        Generate embedding for single image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Embedding vector
        """
        raise NotImplementedError("Subclasses must implement generate_embedding")
    
    def generate_embeddings_batch(self, images_data: List[bytes]) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple images
        
        Args:
            images_data: List of raw image bytes
            
        Returns:
            List of embedding vectors
        """
        raise NotImplementedError("Subclasses must implement generate_embeddings_batch")
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model information"""
        if not self._model_name:
            return None
        
        models = self.get_supported_models()
        for model in models:
            if model.name == self._model_name:
                return model
        return None
    
    def warmup(self) -> None:
        """Warm up model for faster inference"""
        pass
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded