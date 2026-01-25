"""
OpenTelemetry Distributed Tracing.

Provides distributed tracing for request tracking across services,
enabling performance analysis and debugging of LLM pipelines.
"""

import logging
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Dict, Optional
import uuid

logger = logging.getLogger(__name__)

# Trace context for correlation
_trace_context: ContextVar[Dict[str, str]] = ContextVar(
    "trace_context", default={}
)

# OpenTelemetry imports with fallback
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    logger.warning(
        "opentelemetry packages not installed. Tracing will be disabled. "
        "Install with: pip install opentelemetry-api opentelemetry-sdk "
        "opentelemetry-instrumentation-fastapi"
    )


class TracingService:
    """Service for distributed tracing with OpenTelemetry.
    
    Provides:
    - Automatic span creation for LLM operations
    - Request correlation IDs
    - Performance timing
    - Error tracking
    """
    
    def __init__(
        self,
        service_name: str = "roneira-document-intelligence",
        enable_console_export: bool = False
    ):
        """Initialize tracing service.
        
        Args:
            service_name: Name of this service for traces.
            enable_console_export: Whether to export traces to console.
        """
        self.service_name = service_name
        self.enabled = OTEL_AVAILABLE
        self._tracer = None
        self._provider = None
        
        if self.enabled:
            self._init_tracer(enable_console_export)
    
    def _init_tracer(self, enable_console: bool) -> None:
        """Initialize OpenTelemetry tracer."""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "2.0.0"
        })
        
        self._provider = TracerProvider(resource=resource)
        
        if enable_console:
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            self._provider.add_span_processor(processor)
        
        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer(__name__)
        
        logger.info(f"Tracing initialized for {self.service_name}")
    
    def get_tracer(self):
        """Get the OpenTelemetry tracer."""
        return self._tracer
    
    def create_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Create a new trace span.
        
        Args:
            name: Span name.
            attributes: Optional span attributes.
            
        Returns:
            Context manager for the span.
        """
        if not self.enabled or not self._tracer:
            return _NoOpSpan()
        
        return self._tracer.start_as_current_span(
            name,
            attributes=attributes or {}
        )
    
    def instrument_fastapi(self, app) -> None:
        """Instrument a FastAPI application.
        
        Args:
            app: FastAPI application instance.
        """
        if not self.enabled:
            logger.warning("Tracing disabled, skipping FastAPI instrumentation")
            return
        
        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI instrumented for tracing")
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")
    
    def get_correlation_id(self) -> str:
        """Get or create a correlation ID for the current request.
        
        Returns:
            Correlation ID string.
        """
        context = _trace_context.get()
        if "correlation_id" not in context:
            context = {"correlation_id": str(uuid.uuid4())}
            _trace_context.set(context)
        return context["correlation_id"]
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set the correlation ID for the current request.
        
        Args:
            correlation_id: Correlation ID to set.
        """
        context = _trace_context.get().copy()
        context["correlation_id"] = correlation_id
        _trace_context.set(context)
    
    def shutdown(self) -> None:
        """Shutdown the tracer provider."""
        if self._provider:
            self._provider.shutdown()


class _NoOpSpan:
    """No-op span for when tracing is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def set_attribute(self, key, value):
        pass
    
    def set_status(self, status):
        pass
    
    def add_event(self, name, attributes=None):
        pass


def traced(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> Callable:
    """Decorator for tracing async functions.
    
    Args:
        span_name: Optional span name (defaults to function name).
        attributes: Optional span attributes.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            tracer = get_tracing_service()
            name = span_name or func.__name__
            
            with tracer.create_span(name, attributes):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def trace_llm_operation(
    provider: str,
    model: str,
    operation: str = "generate"
) -> Callable:
    """Decorator specialized for LLM operations.
    
    Args:
        provider: LLM provider name.
        model: Model name.
        operation: Operation type.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            tracer = get_tracing_service()
            span_name = f"llm.{operation}"
            
            attrs = {
                "llm.provider": provider,
                "llm.model": model,
                "llm.operation": operation,
            }
            
            with tracer.create_span(span_name, attrs) as span:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add result metadata if available
                    if hasattr(result, "usage") and span:
                        span.set_attribute("llm.tokens.total", 
                                          result.usage.get("total_tokens", 0))
                    
                    return result
                except Exception as e:
                    if span and OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator


# Global tracing service instance
_tracing_service: Optional[TracingService] = None


def get_tracing_service() -> TracingService:
    """Get the global tracing service instance.
    
    Returns:
        TracingService singleton.
    """
    global _tracing_service
    if _tracing_service is None:
        _tracing_service = TracingService()
    return _tracing_service


def init_tracing(
    service_name: str = "roneira-document-intelligence",
    enable_console: bool = False
) -> TracingService:
    """Initialize the global tracing service.
    
    Args:
        service_name: Service name for traces.
        enable_console: Enable console trace export.
        
    Returns:
        Initialized TracingService.
    """
    global _tracing_service
    _tracing_service = TracingService(
        service_name=service_name,
        enable_console_export=enable_console
    )
    return _tracing_service
