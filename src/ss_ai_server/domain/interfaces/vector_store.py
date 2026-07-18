"""
Vector Store interface - Abstract base class for vector databases
"""

from abc import ABC, abstractmethod
from typing import Any,  Dict,  List, Optional, Tuple

from ..value_objects.embedding_vector import EmbeddingVector  # type: ignore
from ..entities.search_result import SearchResult  # type: ignore


class VectorStoreStats:
    """Vector store statistics"""
    
    def __init__(self, total_vectors: int, dimensions: int, index_size: int) -> None:
        self.total_vectors = total_vectors
        self.dimensions = dimensions
        self.index_size = index_size
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return {
            "total_vectors": self.total_vectors,
            "dimensions": self.dimensions,
            "index_size": self.index_size,
        }


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def initialize(self, dimensions: int, metric: str = "cosine") -> None:
        """
        Initialize vector store with dimensions and metric
        
        Args:
            dimensions: Vector dimensions
            metric: Distance metric (cosine, l2, inner_product)
        """
        pass
    
    @abstractmethod
    def add_vectors(self, vectors: List[Tuple[str, EmbeddingVector, Dict[str, Any]]]) -> None:
        """
        Add vectors to store
        
        Args:
            vectors: List of (product_id, embedding, metadata) tuples
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: EmbeddingVector, limit: int = 10,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding
            limit: Maximum results
            filter_dict: Metadata filters
            
        Returns:
            List of search results sorted by similarity
        """
        pass
    
    @abstractmethod
    def delete_vectors(self, product_ids: List[str]) -> int:
        """
        Delete vectors by product IDs
        
        Args:
            product_ids: List of product IDs
            
        Returns:
            Number of vectors deleted
        """
        pass
    
    @abstractmethod
    def get_vector(self, product_id: str) -> Optional[Tuple[EmbeddingVector, Dict[str, Any]]]:
        """
        Get single vector by product ID
        
        Args:
            product_id: Product ID
            
        Returns:
            Tuple of (embedding, metadata) or None
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Persist vector store to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load vector store from disk"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors"""
        pass
    
    @abstractmethod
    def get_all_vectors(self) -> List[Tuple[str, EmbeddingVector]]:
        """Get all vectors (for backup/migration)"""
        pass