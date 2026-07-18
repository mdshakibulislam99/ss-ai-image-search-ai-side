"""
Domain-level exceptions
"""

from typing import Optional

from .base_exceptions import DomainException


class EntityNotFoundError(DomainException):
    """Raised when a domain entity is not found"""
    
    def __init__(self, entity_type: str, entity_id: str, details: Optional[dict] = None):
        """
        Initialize entity not found error
        
        Args:
            entity_type: Type of entity (e.g., 'Product', 'Embedding')
            entity_id: ID of the entity
            details: Additional details
        """
        message = f"{entity_type} with id '{entity_id}' not found"
        error_details = details or {}
        error_details.update({
            "entity_type": entity_type,
            "entity_id": entity_id,
        })
        
        super().__init__(
            message=message,
            code="ENTITY_NOT_FOUND",
            details=error_details
        )


class BusinessRuleViolationError(DomainException):
    """Raised when a business rule is violated"""
    
    def __init__(self, rule: str, message: str, details: Optional[dict] = None):
        """
        Initialize business rule violation error
        
        Args:
            rule: Business rule that was violated
            message: Description of the violation
            details: Additional details
        """
        error_details = details or {}
        error_details["rule"] = rule
        
        super().__init__(
            message=message,
            code="BUSINESS_RULE_VIOLATION",
            details=error_details
        )


class InvalidEmbeddingError(DomainException):
    """Raised when an embedding is invalid"""
    
    def __init__(self, reason: str, details: Optional[dict] = None):
        """
        Initialize invalid embedding error
        
        Args:
            reason: Reason why embedding is invalid
            details: Additional details
        """
        message = f"Invalid embedding: {reason}"
        error_details = details or {}
        error_details["reason"] = reason
        
        super().__init__(
            message=message,
            code="INVALID_EMBEDDING",
            details=error_details
        )


class ModelNotLoadedError(DomainException):
    """Raised when an AI model is not loaded"""
    
    def __init__(self, model_name: str, provider: str, details: Optional[dict] = None):
        """
        Initialize model not loaded error
        
        Args:
            model_name: Name of the model
            provider: Provider name
            details: Additional details
        """
        message = f"Model '{model_name}' from provider '{provider}' is not loaded"
        error_details = details or {}
        error_details.update({
            "model_name": model_name,
            "provider": provider,
        })
        
        super().__init__(
            message=message,
            code="MODEL_NOT_LOADED",
            details=error_details
        )