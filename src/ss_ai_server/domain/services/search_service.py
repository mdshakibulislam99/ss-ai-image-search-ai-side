"""
SearchService - Domain service for search operations
"""

from typing import Optional

from ..interfaces.vector_store import VectorStore  # type: ignore
from ..interfaces.cache import Cache  # type: ignore
from ..value_objects.embedding_vector import EmbeddingVector  # type: ignore
from ..entities.search_result import SearchResult  # type: ignore


class SearchService:
    """Domain service for search operations"""
    
    def __init__(self, vector_store: VectorStore, cache: Optional[Cache] = None) -> None:
        """
        Initialize search service
        
        Args:
            vector_store: Vector store for searching
            cache: Optional cache for storing search results
        """
        self.vector_store = vector_store
        self.cache = cache
    
    async def search_similar(self, query_embedding: EmbeddingVector, limit: int = 20,
                            threshold: float = 0.7, cache_key: Optional[str] = None) -> List[SearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            cache_key: Optional cache key for results
            
        Returns:
            List of search results
        """
        # Check cache first
        if self.cache and cache_key:
            cached = await self.cache.get(cache_key)
            if cached:
                return cached
        
        # Search vector store
        results = self.vector_store.search(query_embedding, limit=limit)
        
        # Filter by threshold
        filtered_results = [r for r in results if r.similarity_score >= threshold]
        
        # Cache results
        if self.cache and cache_key:
            await self.cache.set(cache_key, filtered_results)
        
        return filtered_results
    
    def get_model_info(self) -> Optional[dict]:
        """Get vector store information"""
        stats = self.vector_store.get_stats()
        return stats.to_dict()