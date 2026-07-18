"""
Application-level exceptions
"""

from typing import Optional

from .base_exceptions import ApplicationException


class UseCaseExecutionError(ApplicationException):
    """Raised when a use case fails to execute"""
    
    def __init__(self, use_case: str, reason: str, details: Optional[dict] = None):
        """
        Initialize use case execution error
        
        Args:
            use_case: Name of the use case that failed
            reason: Reason for failure
            details: Additional details
        """
        message = f"Use case '{use_case}' failed: {reason}"
        error_details = details or {}
        error_details.update({
            "use_case": use_case,
            "reason": reason,
        })
        
        super().__init__(
            message=message,
            code="USE_CASE_EXECUTION_ERROR",
            details=error_details
        )


class InvalidRequestError(ApplicationException):
    """Raised when a request is invalid"""
    
    def __init__(self, message: str, request_data: Optional[dict] = None, 
                 details: Optional[dict] = None):
        """
        Initialize invalid request error
        
        Args:
            message: Error message
            request_data: The invalid request data
            details: Additional details
        """
        error_details = details or {}
        if request_data:
            error_details["request_data"] = request_data
        
        super().__init__(
            message=message,
            code="INVALID_REQUEST",
            details=error_details
        )


class OperationNotAllowedError(ApplicationException):
    """Raised when an operation is not allowed"""
    
    def __init__(self, operation: str, reason: str, details: Optional[dict] = None):
        """
        Initialize operation not allowed error
        
        Args:
            operation: Name of the operation
            reason: Reason why operation is not allowed
            details: Additional details
        """
        message = f"Operation '{operation}' not allowed: {reason}"
        error_details = details or {}
        error_details.update({
            "operation": operation,
            "reason": reason,
        })
        
        super().__init__(
            message=message,
            code="OPERATION_NOT_ALLOWED",
            details=error_details
        )