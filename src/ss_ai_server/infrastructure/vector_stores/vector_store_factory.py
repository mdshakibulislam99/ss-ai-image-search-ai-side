"""
Vector Store Factory

Factory pattern implementation for creating vector store instances
"""

from typing import Dict,  Type

from ..domain.interfaces.vector_store import VectorStore  # type: ignore
from .base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """
    Factory for creating vector store instances
    
    Supports multiple vector databases and allows dynamic registration
    """
    
    _stores: Dict[str, Type[BaseVectorStore]] = {}
    
    @classmethod
    def register_store(cls, name: str, store_class: Type[BaseVectorStore]) -> None:
        """
        Register a new vector store
        
        Args:
            name: Store name (e.g., 'faiss', 'hnswlib', 'chromadb')
            store_class: Store implementation class
        """
        cls._stores[name] = store_class
    
    @classmethod
    def create_store(cls, store_name: str, **kwargs) -> VectorStore:
        """
        Create a vector store instance
        
        Args:
            store_name: Name of the vector store to create
            **kwargs: Additional arguments for store initialization
            
        Returns:
            Vector store instance
            
        Raises:
            ValueError: If vector store is not registered
        """
        if store_name not in cls._stores:
            raise ValueError(
                f"Vector store '{store_name}' is not registered. "
                f"Available stores: {list(cls._stores.keys())}"
            )
        
        store_class = cls._stores[store_name]
        return store_class(**kwargs)
    
    @classmethod
    def get_available_stores(cls) -> List[str]:
        """
        Get list of available vector stores
        
        Returns:
            List of store names
        """
        return list(cls._stores.keys())
    
    @classmethod
    def is_store_available(cls, store_name: str) -> bool:
        """
        Check if a vector store is available
        
        Args:
            store_name: Name of the vector store
            
        Returns:
            True if store is available, False otherwise
        """
        return store_name in cls._stores


# Register built-in vector stores
def _register_builtin_stores():
    """Register built-in vector stores"""
    from .faiss_vector_store import FAISSVectorStore
    VectorStoreFactory.register_store('faiss', FAISSVectorStore)


# Initialize built-in vector stores
_register_builtin_stores()