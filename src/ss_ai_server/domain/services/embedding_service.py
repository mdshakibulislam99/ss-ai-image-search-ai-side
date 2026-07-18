"""
EmbeddingService - Domain service for embedding operations
"""

from typing import Optional

from ..interfaces.ai_provider import AIProvider  # type: ignore
from ..interfaces.cache import Cache  # type: ignore
from ..value_objects.embedding_vector import EmbeddingVector  # type: ignore


class EmbeddingService:
    """Domain service for embedding operations"""
    
    def __init__(self, ai_provider: AIProvider, cache: Optional[Cache] = None) -> None:
        """
        Initialize embedding service
        
        Args:
            ai_provider: AI provider for generating embeddings
            cache: Optional cache for storing embeddings
        """
        self.ai_provider = ai_provider
        self.cache = cache
    
    async def generate_embedding(self, image_data: bytes, cache_key: Optional[str] = None) -> EmbeddingVector:
        """
        Generate embedding for image
        
        Args:
            image_data: Raw image bytes
            cache_key: Optional cache key for caching result
            
        Returns:
            Embedding vector
        """
        # Check cache first
        if self.cache and cache_key:
            cached = await self.cache.get(cache_key)
            if cached:
                return cached
        
        # Generate embedding
        embedding = self.ai_provider.generate_embedding(image_data)
        
        # Cache result
        if self.cache and cache_key:
            await self.cache.set(cache_key, embedding)
        
        return embedding
    
    async def generate_embeddings_batch(self, images_data: List[bytes], 
                                        cache_keys: Optional[List[str]] = None) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple images
        
        Args:
            images_data: List of raw image bytes
            cache_keys: Optional list of cache keys
            
        Returns:
            List of embedding vectors
        """
        # Check cache for each image
        embeddings = []
        uncached_images = []
        uncached_indices = []
        
        if self.cache and cache_keys:
            for i, (image_data, cache_key) in enumerate(zip(images_data, cache_keys)):
                cached = await self.cache.get(cache_key)
                if cached:
                    embeddings.append(cached)
                else:
                    uncached_images.append(image_data)
                    uncached_indices.append(i)
        else:
            uncached_images = images_data
            uncached_indices = list(range(len(images_data)))
        
        # Generate embeddings for uncached images
        if uncached_images:
            new_embeddings = self.ai_provider.generate_embeddings_batch(uncached_images)
            
            # Cache new embeddings
            if self.cache and cache_keys:
                for idx, embedding, cache_key in zip(uncached_indices, new_embeddings, 
                                                      [cache_keys[i] for i in uncached_indices]):
                    await self.cache.set(cache_key, embedding)
                    embeddings.insert(idx, embedding)
            else:
                embeddings = new_embeddings
        
        return embeddings
    
    def get_model_info(self) -> Optional[dict]:
        """Get current model information"""
        model_info = self.ai_provider.get_model_info()
        if model_info:
            return model_info.to_dict()
        return None