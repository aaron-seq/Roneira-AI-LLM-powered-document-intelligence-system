"""
Structured Logging with Correlation IDs.

Provides JSON-structured logging with request correlation,
context propagation, and performance timing.
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

# Context variables for correlation
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
request_path_var: ContextVar[str] = ContextVar("request_path", default="")


@dataclass
class LogContext:
    """Context for structured log entries.
    
    Attributes:
        correlation_id: Request correlation ID.
        user_id: User identifier.
        service: Service name.
        operation: Current operation.
        extra: Additional context fields.
    """
    correlation_id: str = ""
    user_id: str = ""
    service: str = "roneira"
    operation: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding empty values."""
        result = {
            "correlation_id": self.correlation_id,
            "service": self.service,
        }
        if self.user_id:
            result["user_id"] = self.user_id
        if self.operation:
            result["operation"] = self.operation
        if self.extra:
            result.update(self.extra)
        return result


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging.
    
    Outputs logs in JSON format with correlation IDs,
    timestamps, and context information.
    """
    
    def __init__(
        self,
        service_name: str = "roneira",
        include_caller: bool = True,
        include_exception: bool = True
    ):
        """Initialize structured formatter.
        
        Args:
            service_name: Name of the service.
            include_caller: Include file/line info.
            include_exception: Include exception details.
        """
        super().__init__()
        self.service_name = service_name
        self.include_caller = include_caller
        self.include_exception = include_exception
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format.
            
        Returns:
            JSON-formatted log string.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }
        
        # Add correlation ID from context
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        # Add user ID if available
        user_id = user_id_var.get()
        if user_id:
            log_entry["user_id"] = user_id
        
        # Add request path if available
        request_path = request_path_var.get()
        if request_path:
            log_entry["request_path"] = request_path
        
        # Add caller information
        if self.include_caller:
            log_entry["caller"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }
        
        # Add exception info
        if self.include_exception and record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add any extra fields from the record
        if hasattr(record, "extra_fields"):
            log_entry["extra"] = record.extra_fields
        
        return json.dumps(log_entry, default=str)


class ContextLogger:
    """Logger wrapper with context support.
    
    Provides structured logging with automatic context
    propagation and timing support.
    """
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """Initialize context logger.
        
        Args:
            name: Logger name.
            logger: Optional existing logger.
        """
        self._logger = logger or logging.getLogger(name)
        self.name = name
    
    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """Internal log method with extra fields.
        
        Args:
            level: Log level.
            message: Log message.
            extra: Optional extra fields.
            exc_info: Include exception info.
        """
        if extra:
            record_extra = {"extra_fields": extra}
        else:
            record_extra = {}
        
        self._logger.log(level, message, exc_info=exc_info, extra=record_extra)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, kwargs if kwargs else None)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, kwargs if kwargs else None)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, kwargs if kwargs else None)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, kwargs if kwargs else None, exc_info)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._log(logging.ERROR, message, kwargs if kwargs else None, exc_info=True)
    
    def timed(self, operation: str) -> "TimedContext":
        """Create a timed context for logging operation duration.
        
        Args:
            operation: Operation name.
            
        Returns:
            TimedContext context manager.
        """
        return TimedContext(self, operation)


class TimedContext:
    """Context manager for timing operations."""
    
    def __init__(self, logger: ContextLogger, operation: str):
        """Initialize timed context.
        
        Args:
            logger: Logger to use.
            operation: Operation name.
        """
        self._logger = logger
        self._operation = operation
        self._start_time = 0.0
    
    def __enter__(self) -> "TimedContext":
        """Start timing."""
        self._start_time = time.perf_counter()
        self._logger.debug(f"Starting {self._operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End timing and log duration."""
        duration_ms = (time.perf_counter() - self._start_time) * 1000
        
        if exc_type is not None:
            self._logger.error(
                f"Failed {self._operation}",
                duration_ms=round(duration_ms, 2),
                error=str(exc_val)
            )
        else:
            self._logger.info(
                f"Completed {self._operation}",
                duration_ms=round(duration_ms, 2)
            )


def setup_structured_logging(
    service_name: str = "roneira",
    log_level: str = "INFO",
    json_output: bool = True
) -> None:
    """Configure structured logging for the application.
    
    Args:
        service_name: Name of the service.
        log_level: Minimum log level.
        json_output: Whether to use JSON format.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))
    
    if json_output:
        handler.setFormatter(StructuredFormatter(service_name))
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(correlation_id)s] %(message)s"
        ))
    
    root_logger.addHandler(handler)


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one.
    
    Returns:
        Correlation ID string.
    """
    correlation_id = correlation_id_var.get()
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context.
    
    Args:
        correlation_id: Correlation ID to set.
    """
    correlation_id_var.set(correlation_id)


def set_user_context(user_id: str) -> None:
    """Set user ID for current context.
    
    Args:
        user_id: User ID to set.
    """
    user_id_var.set(user_id)


def set_request_context(path: str) -> None:
    """Set request path for current context.
    
    Args:
        path: Request path to set.
    """
    request_path_var.set(path)


def with_correlation_id(func: Callable) -> Callable:
    """Decorator to ensure correlation ID exists for function.
    
    Args:
        func: Function to wrap.
        
    Returns:
        Wrapped function.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        get_correlation_id()  # Ensure ID exists
        return await func(*args, **kwargs)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        get_correlation_id()  # Ensure ID exists
        return func(*args, **kwargs)
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def get_logger(name: str) -> ContextLogger:
    """Get a context-aware logger.
    
    Args:
        name: Logger name.
        
    Returns:
        ContextLogger instance.
    """
    return ContextLogger(name)


# FastAPI middleware for request context
class RequestContextMiddleware:
    """FastAPI middleware to set up request context."""
    
    def __init__(self, app):
        """Initialize middleware.
        
        Args:
            app: FastAPI application.
        """
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process request and set up context."""
        if scope["type"] == "http":
            # Extract or generate correlation ID
            headers = dict(scope.get("headers", []))
            correlation_id = headers.get(
                b"x-correlation-id",
                headers.get(b"x-request-id", b"")
            ).decode() or str(uuid.uuid4())
            
            # Set context
            correlation_id_var.set(correlation_id)
            request_path_var.set(scope.get("path", ""))
        
        await self.app(scope, receive, send)
