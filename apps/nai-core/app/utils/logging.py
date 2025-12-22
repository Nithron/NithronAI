"""
nAI Core Logging Utilities
Structured logging with request tracking
"""

import sys
import logging
import time
from datetime import datetime
from typing import Optional, Any, Dict
from functools import wraps
from contextvars import ContextVar

# Context variable for request ID
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Add request ID if available
        record.request_id = request_id_var.get() or "-"
        
        # Format message
        if hasattr(record, 'extra_data') and record.extra_data:
            extra = " | " + " ".join(f"{k}={v}" for k, v in record.extra_data.items())
        else:
            extra = ""
        
        return f"[{record.timestamp}] [{record.levelname}] [{record.request_id}] {record.name}: {record.getMessage()}{extra}"


class RequestLogger(logging.LoggerAdapter):
    """Logger adapter that includes request context."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        extra = kwargs.get('extra', {})
        extra['request_id'] = request_id_var.get() or "-"
        kwargs['extra'] = extra
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    json_format: bool = False
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format for logs (for production)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    
    if json_format:
        # JSON format for production
        import json
        
        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "request_id": request_id_var.get(),
                }
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                if hasattr(record, 'extra_data'):
                    log_data.update(record.extra_data)
                return json.dumps(log_data)
        
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(StructuredFormatter())
    
    root_logger.addHandler(handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an HTTP request with structured data.
    
    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration_ms: Request duration in milliseconds
        request_id: Request ID for tracing
        extra: Additional data to log
    """
    logger = get_logger("nai.request")
    
    log_data = {
        "method": method,
        "path": path,
        "status": status_code,
        "duration_ms": round(duration_ms, 2),
    }
    
    if extra:
        log_data.update(extra)
    
    # Choose log level based on status
    if status_code >= 500:
        level = logging.ERROR
    elif status_code >= 400:
        level = logging.WARNING
    else:
        level = logging.INFO
    
    logger.log(
        level,
        f"{method} {path} -> {status_code} ({duration_ms:.1f}ms)",
        extra={"extra_data": log_data}
    )


def log_duration(logger_name: str = "nai"):
    """
    Decorator to log function execution duration.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                logger.debug(f"{func.__name__} completed in {duration:.2f}ms")
                return result
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                logger.error(f"{func.__name__} failed after {duration:.2f}ms: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                logger.debug(f"{func.__name__} completed in {duration:.2f}ms")
                return result
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                logger.error(f"{func.__name__} failed after {duration:.2f}ms: {e}")
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

