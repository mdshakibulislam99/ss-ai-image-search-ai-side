"""
Rate Limiting Middleware

Implements rate limiting to prevent abuse
"""

from typing import Callable, Optional, Dict, Tuple
import time
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from ...config.settings import Settings, settings
from ...domain.interfaces.cache import Cache


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm
    
    Limits requests based on client IP or API key
    """
    
    def __init__(
        self,
        app,
        settings: Optional[Settings] = None,
        cache: Optional[Cache] = None,
        default_limit: int = 60,
        window_seconds: int = 60,
        exclude_paths: Optional[list] = None
    ):
        """
        Initialize rate limit middleware
        
        Args:
            app: ASGI application
            settings: Application settings
            cache: Cache service for rate limit tracking
            default_limit: Default requests per window
            window_seconds: Time window in seconds
            exclude_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        self.settings = settings or Settings()
        self.cache = cache
        self.default_limit = default_limit or self.settings.rate_limit_default
        self.window_seconds = window_seconds or self.settings.rate_limit_window
        self.exclude_paths = exclude_paths or [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ):
        """
        Process request with rate limiting
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Skip if rate limiting is disabled
        if not self.settings.rate_limit_enabled:
            return await call_next(request)
        
        # Get client identifier (API key or IP)
        client_id = self._get_client_id(request)
        
        # Check rate limit
        is_allowed, remaining, reset_time = self._check_rate_limit(client_id)
        
        # Add rate limit headers
        response = None
        if is_allowed:
            response = await call_next(request)
        
        # Add rate limit headers to response
        if response:
            response.headers["X-RateLimit-Limit"] = str(self.default_limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(reset_time))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting
        
        Args:
            request: HTTP request
            
        Returns:
            Client identifier string
        """
        # Use client IP address for rate limiting
        client_host = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_host = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_host}"
    
    def _check_rate_limit(self, client_id: str) -> Tuple[bool, int, float]:
        """
        Check if request is within rate limit
        
        Args:
            client_id: Client identifier
            
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Use cache for rate limit tracking
        if self.cache:
            return self._check_rate_limit_with_cache(client_id, current_time, window_start)
        else:
            return self._check_rate_limit_in_memory(client_id, current_time, window_start)
    
    def _check_rate_limit_with_cache(
        self,
        client_id: str,
        current_time: float,
        window_start: float
    ) -> Tuple[bool, int, float]:
        """
        Check rate limit using cache
        
        Args:
            client_id: Client identifier
            current_time: Current timestamp
            window_start: Window start timestamp
            
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        cache_key = f"rate_limit:{client_id}"
        
        # Get existing requests from cache
        requests = self.cache.get(cache_key, [])
        
        # Filter out old requests outside the window
        requests = [req_time for req_time in requests if req_time > window_start]
        
        # Check if limit exceeded
        request_count = len(requests)
        is_allowed = request_count < self.default_limit
        
        if is_allowed:
            # Add current request
            requests.append(current_time)
            # Store back in cache
            self.cache.set(cache_key, requests, ttl=self.window_seconds + 1)
        
        remaining = max(0, self.default_limit - request_count - 1)
        reset_time = current_time + self.window_seconds
        
        return is_allowed, remaining, reset_time
    
    def _check_rate_limit_in_memory(
        self,
        client_id: str,
        current_time: float,
        window_start: float
    ) -> Tuple[bool, int, float]:
        """
        Check rate limit using in-memory storage (fallback)
        
        Args:
            client_id: Client identifier
            current_time: Current timestamp
            window_start: Window start timestamp
            
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        # In-memory storage (not suitable for multi-worker deployments)
        if not hasattr(self, '_request_history'):
            self._request_history: Dict[str, list] = {}
        
        requests = self._request_history.get(client_id, [])
        requests = [req_time for req_time in requests if req_time > window_start]
        
        request_count = len(requests)
        is_allowed = request_count < self.default_limit
        
        if is_allowed:
            requests.append(current_time)
            self._request_history[client_id] = requests
        
        remaining = max(0, self.default_limit - request_count - 1)
        reset_time = current_time + self.window_seconds
        
        return is_allowed, remaining, reset_time