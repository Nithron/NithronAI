"""
nAI Core Middleware
Request logging, rate limiting, and request ID tracking
"""

import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings
from .utils.logging import get_logger, log_request, request_id_var

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and timing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request_id_var.set(request_id)
        
        # Record start time
        start_time = time.perf_counter()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Log request
        log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.settings = get_settings()
        self._requests: dict[str, list[float]] = {}
    
    def _get_client_key(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use X-Forwarded-For if behind proxy, otherwise client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _clean_old_requests(self, client_key: str, current_time: float) -> None:
        """Remove requests outside the time window."""
        if client_key not in self._requests:
            self._requests[client_key] = []
            return
        
        window_start = current_time - self.settings.rate_limit_window_seconds
        self._requests[client_key] = [
            t for t in self._requests[client_key]
            if t > window_start
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.settings.rate_limit_enabled:
            return await call_next(request)
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/detailed"]:
            return await call_next(request)
        
        client_key = self._get_client_key(request)
        current_time = time.time()
        
        # Clean old requests
        self._clean_old_requests(client_key, current_time)
        
        # Check rate limit
        if len(self._requests[client_key]) >= self.settings.rate_limit_requests:
            logger.warning(f"Rate limit exceeded for {client_key}")
            return Response(
                content='{"detail": "Too many requests. Please slow down."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(self.settings.rate_limit_window_seconds),
                    "X-RateLimit-Limit": str(self.settings.rate_limit_requests),
                    "X-RateLimit-Remaining": "0",
                }
            )
        
        # Record this request
        self._requests[client_key].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.settings.rate_limit_requests - len(self._requests[client_key])
        response.headers["X-RateLimit-Limit"] = str(self.settings.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        
        return response


def setup_middleware(app: FastAPI) -> None:
    """Configure all middleware for the application."""
    # Order matters: first added = outermost
    
    # Rate limiting (should be before logging)
    app.add_middleware(RateLimitMiddleware)
    
    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

