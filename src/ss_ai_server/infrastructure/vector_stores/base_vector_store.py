"""
Base Vector Store - Base implementation for vector stores
"""

from typing import Any,  Dict,  List, Optional, Tuple

from ...domain.interfaces.vector_store import VectorStore, VectorStoreStats  # type: ignore
from ...domain.value_objects.embedding_vector import EmbeddingVector  # type: ignore
from ...domain.entities.search_result import SearchResult  # type: ignore


class BaseVectorStore(VectorStore):
    """Base implementation for vector stores"""
    
    def __init__(self) -> None:
        """Initialize base vector store"""
        self._dimensions = None
        self._metric = None
        self._initialized = False
    
    def initialize(self, dimensions: int, metric: str = "cosine") -> None:
        """
        Initialize vector store
        
        Args:
            dimensions: Vector dimensions
            metric: Distance metric
        """
        self._dimensions = dimensions
        self._metric = metric
        self._initialized = True
    
    def add_vectors(self, vectors: List[Tuple[str, EmbeddingVector, Dict[str, Any]]]) -> None:
        """Add vectors to store"""
        raise NotImplementedError("Subclasses must implement add_vectors")
    
    def search(self, query_vector: EmbeddingVector, limit: int = 10,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors"""
        raise NotImplementedError("Subclasses must implement search")
    
    def delete_vectors(self, product_ids: List[str]) -> int:
        """Delete vectors by product IDs"""
        raise NotImplementedError("Subclasses must implement delete_vectors")
    
    def get_vector(self, product_id: str) -> Optional[Tuple[EmbeddingVector, Dict[str, Any]]]:
        """Get single vector by product ID"""
        raise NotImplementedError("Subclasses must implement get_vector")
    
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics"""
        raise NotImplementedError("Subclasses must implement get_stats")
    
    def save(self, path: str) -> None:
        """Persist vector store to disk"""
        raise NotImplementedError("Subclasses must implement save")
    
    def load(self, path: str) -> None:
        """Load vector store from disk"""
        raise NotImplementedError("Subclasses must implement load")
    
    def clear(self) -> None:
        """Clear all vectors"""
        raise NotImplementedError("Subclasses must implement clear")
    
    def get_all_vectors(self) -> List[Tuple[str, EmbeddingVector]]:
        """Get all vectors"""
        raise NotImplementedError("Subclasses must implement get_all_vectors")