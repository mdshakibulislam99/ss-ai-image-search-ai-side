"""
SS AI Server Exception Hierarchy

All custom exceptions for the application
"""

from .base_exceptions import (
    SSException,
    ValidationError,
    NotFoundError,
    AuthorizationError,
)
from .domain_exceptions import (
    DomainException,
    EntityNotFoundError,
    BusinessRuleViolationError,
    InvalidEmbeddingError,
    ModelNotLoadedError,
)
from .application_exceptions import (
    ApplicationException,
    UseCaseExecutionError,
    InvalidRequestError,
    OperationNotAllowedError,
)
from .infrastructure_exceptions import (
    InfrastructureException,
    ProviderError,
    VectorStoreError,
    DatabaseError,
    CacheError,
    QueueError,
    StorageError,
)

__all__ = [
    # Base
    "SSException",
    "ValidationError",
    "NotFoundError",
    "AuthorizationError",
    # Domain
    "DomainException",
    "EntityNotFoundError",
    "BusinessRuleViolationError",
    "InvalidEmbeddingError",
    "ModelNotLoadedError",
    # Application
    "ApplicationException",
    "UseCaseExecutionError",
    "InvalidRequestError",
    "OperationNotAllowedError",
    # Infrastructure
    "InfrastructureException",
    "ProviderError",
    "VectorStoreError",
    "DatabaseError",
    "CacheError",
    "QueueError",
    "StorageError",
]