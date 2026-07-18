"""
Infrastructure-level exceptions
"""

from typing import Optional

from .base_exceptions import InfrastructureException


class ProviderError(InfrastructureException):
    """Raised when an AI provider encounters an error"""
    
    def __init__(self, provider: str, operation: str, reason: str, details: Optional[dict] = None) -> None:
        """
        Initialize provider error
        
        Args:
            provider: Provider name (e.g., 'openclip', 'siglip')
            operation: Operation that failed (e.g., 'load_model', 'generate_embedding')
            reason: Reason for failure
            details: Additional details
        """
        message = f"AI Provider '{provider}' failed during '{operation}': {reason}"
        error_details = details or {}
        error_details.update({
            "provider": provider,
            "operation": operation,
            "reason": reason,
        })
        
        super().__init__(
            message=message,
            code="PROVIDER_ERROR",
            details=error_details
        )


class VectorStoreError(InfrastructureException):
    """Raised when a vector store encounters an error"""
    
    def __init__(self, store_type: str, operation: str, reason: str, details: Optional[dict] = None) -> None:
        """
        Initialize vector store error
        
        Args:
            store_type: Type of vector store (e.g., 'faiss', 'hnswlib')
            operation: Operation that failed (e.g., 'search', 'add_vectors')
            reason: Reason for failure
            details: Additional details
        """
        message = f"Vector store '{store_type}' failed during '{operation}': {reason}"
        error_details = details or {}
        error_details.update({
            "store_type": store_type,
            "operation": operation,
            "reason": reason,
        })
        
        super().__init__(
            message=message,
            code="VECTOR_STORE_ERROR",
            details=error_details
        )


class DatabaseError(InfrastructureException):
    """Raised when a database operation fails"""
    
    def __init__(self, operation: str, reason: str, details: Optional[dict] = None) -> None:
        """
        Initialize database error
        
        Args:
            operation: Database operation that failed (e.g., 'query', 'insert')
            reason: Reason for failure
            details: Additional details
        """
        message = f"Database operation '{operation}' failed: {reason}"
        error_details = details or {}
        error_details.update({
            "operation": operation,
            "reason": reason,
        })
        
        super().__init__(
            message=message,
            code="DATABASE_ERROR",
            details=error_details
        )


class CacheError(InfrastructureException):
    """Raised when a cache operation fails"""
    
    def __init__(self, operation: str, reason: str, details: Optional[dict] = None) -> None:
        """
        Initialize cache error
        
        Args:
            operation: Cache operation that failed (e.g., 'get', 'set')
            reason: Reason for failure
            details: Additional details
        """
        message = f"Cache operation '{operation}' failed: {reason}"
        error_details = details or {}
        error_details.update({
            "operation": operation,
            "reason": reason,
        })
        
        super().__init__(
            message=message,
            code="CACHE_ERROR",
            details=error_details
        )


class QueueError(InfrastructureException):
    """Raised when a queue operation fails"""
    
    def __init__(self, operation: str, reason: str, details: Optional[dict] = None) -> None:
        """
        Initialize queue error
        
        Args:
            operation: Queue operation that failed (e.g., 'enqueue', 'dequeue')
            reason: Reason for failure
            details: Additional details
        """
        message = f"Queue operation '{operation}' failed: {reason}"
        error_details = details or {}
        error_details.update({
            "operation": operation,
            "reason": reason,
        })
        
        super().__init__(
            message=message,
            code="QUEUE_ERROR",
            details=error_details
        )


class StorageError(InfrastructureException):
    """Raised when a storage operation fails"""
    
    def __init__(self, operation: str, path: str, reason: str, details: Optional[dict] = None) -> None:
        """
        Initialize storage error
        
        Args:
            operation: Storage operation that failed (e.g., 'save', 'load')
            path: File path involved
            reason: Reason for failure
            details: Additional details
        """
        message = f"Storage operation '{operation}' failed on '{path}': {reason}"
        error_details = details or {}
        error_details.update({
            "operation": operation,
            "path": path,
            "reason": reason,
        })
        
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            details=error_details
        )


class ModelLoadException(InfrastructureException):
    """Raised when model loading fails"""
    
    def __init__(self, reason: str, model_name: Optional[str] = None, details: Optional[dict] = None) -> None:
        """
        Initialize model load error
        
        Args:
            reason: Reason for failure
            model_name: Name of the model that failed to load
            details: Additional details
        """
        message = f"Failed to load model"
        if model_name:
            message += f" '{model_name}'"
        message += f": {reason}"
        
        error_details = details or {}
        if model_name:
            error_details["model_name"] = model_name
        
        super().__init__(
            message=message,
            code="MODEL_LOAD_ERROR",
            details=error_details
        )


class InferenceException(InfrastructureException):
    """Raised when inference fails"""
    
    def __init__(self, reason: str, provider: str, details: Optional[dict] = None) -> None:
        """
        Initialize inference error
        
        Args:
            reason: Reason for failure
            provider: Provider name (e.g., 'openclip', 'siglip')
            details: Additional details
        """
        message = f"Inference failed on provider '{provider}': {reason}"
        error_details = details or {}
        error_details["provider"] = provider
        
        super().__init__(
            message=message,
            code="INFERENCE_ERROR",
            details=error_details
        )


class ConfigurationException(InfrastructureException):
    """Raised when configuration is invalid"""
    
    def __init__(self, reason: str, details: Optional[dict] = None) -> None:
        """
        Initialize configuration error
        
        Args:
            reason: Reason for failure
            details: Additional details
        """
        message = f"Configuration error: {reason}"
        error_details = details or {}
        
        super().__init__(
            message=message,
            code="CONFIGURATION_ERROR",
            details=error_details
        )
