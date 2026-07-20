"""
Base exceptions for SS AI Server
"""

from typing import Optional


class SSException(Exception):
    """Base exception for all SS AI Server exceptions"""
    
    def __init__(self, message: str, code: str, details: Optional[dict] = None) -> None:
        """
        Initialize exception
        
        Args:
            message: Human-readable error message
            code: Machine-readable error code
            details: Additional error details
        """
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class DomainException(SSException):
    """Base exception for domain-level errors"""
    
    def __init__(self, message: str, code: str, details: Optional[dict] = None) -> None:
        """
        Initialize domain exception
        
        Args:
            message: Error message
            code: Error code
            details: Additional details
        """
        super().__init__(message, code, details)


class ValidationError(SSException):
    """Raised when validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[dict] = None) -> None:
        """
        Initialize validation error
        
        Args:
            message: Error message
            field: Field that failed validation
            details: Additional details
        """
        error_details = details or {}
        if field:
            error_details["field"] = field
        
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details=error_details
        )


class NotFoundError(SSException):
    """Raised when a resource is not found"""
    
    def __init__(self, resource_type: str, resource_id: str, details: Optional[dict] = None) -> None:
        """
        Initialize not found error
        
        Args:
            resource_type: Type of resource (e.g., 'Product', 'Embedding')
            resource_id: ID of the resource
            details: Additional details
        """
        message = f"{resource_type} with id '{resource_id}' not found"
        error_details = details or {}
        error_details.update({
            "resource_type": resource_type,
            "resource_id": resource_id,
        })
        
        super().__init__(
            message=message,
            code="NOT_FOUND_ERROR",
            details=error_details
        )


class AuthorizationError(SSException):
    """Raised when authorization fails"""
    
    def __init__(self, message: str, required_permission: Optional[str] = None, 
                 details: Optional[dict] = None) -> None:
        """
        Initialize authorization error
        
        Args:
            message: Error message
            required_permission: Permission that was missing
            details: Additional details
        """
        error_details = details or {}
        if required_permission:
            error_details["required_permission"] = required_permission
        
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            details=error_details
        )


class ApplicationException(SSException):
    """Base exception for application-level errors"""
    
    def __init__(self, message: str, code: str, details: Optional[dict] = None) -> None:
        """
        Initialize application exception
        
        Args:
            message: Error message
            code: Error code
            details: Additional details
        """
        super().__init__(message, code, details)


class SSAIServerException(SSException):
    """Base exception for all SS AI Server exceptions (alias for compatibility)"""
    
    def __init__(self, message: str, status_code: int = 500, details: Optional[dict] = None) -> None:
        """
        Initialize SS AI Server exception
        
        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional details
        """
        self.status_code = status_code
        error_details = details or {}
        error_details["status_code"] = status_code
        super().__init__(message, "SSAI_SERVER_ERROR", error_details)


class InfrastructureException(SSException):
    """Base exception for infrastructure-level errors"""
    
    def __init__(self, message: str, code: str, details: Optional[dict] = None) -> None:
        """
        Initialize infrastructure exception
        
        Args:
            message: Error message
            code: Error code
            details: Additional details
        """
        super().__init__(message, code, details)
