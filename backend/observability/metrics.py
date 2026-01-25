"""
Prometheus Metrics Service.

Provides application metrics collection and exposure for monitoring
LLM latency, request counts, cache performance, and system health.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Prometheus client imports with fallback
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus-client not installed. Metrics will be disabled. "
        "Install with: pip install prometheus-client"
    )


class MetricsService:
    """Service for collecting and exposing application metrics.
    
    Provides Prometheus-compatible metrics for:
    - LLM inference latency and throughput
    - Request counts and error rates
    - Cache hit/miss ratios
    - Document processing statistics
    """
    
    def __init__(self, namespace: str = "roneira"):
        """Initialize metrics service.
        
        Args:
            namespace: Prefix for all metric names.
        """
        self.namespace = namespace
        self.enabled = PROMETHEUS_AVAILABLE
        self._registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        
        if self.enabled:
            self._init_metrics()
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # LLM metrics
        self.llm_request_latency = Histogram(
            f"{self.namespace}_llm_request_latency_seconds",
            "LLM request latency in seconds",
            labelnames=["provider", "model", "operation"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self._registry
        )
        
        self.llm_request_total = Counter(
            f"{self.namespace}_llm_requests_total",
            "Total LLM requests",
            labelnames=["provider", "model", "status"],
            registry=self._registry
        )
        
        self.llm_tokens_total = Counter(
            f"{self.namespace}_llm_tokens_total",
            "Total tokens processed",
            labelnames=["provider", "model", "type"],
            registry=self._registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            f"{self.namespace}_cache_hits_total",
            "Cache hit count",
            labelnames=["cache_type"],
            registry=self._registry
        )
        
        self.cache_misses = Counter(
            f"{self.namespace}_cache_misses_total",
            "Cache miss count",
            labelnames=["cache_type"],
            registry=self._registry
        )
        
        # Document processing metrics
        self.documents_processed = Counter(
            f"{self.namespace}_documents_processed_total",
            "Total documents processed",
            labelnames=["status", "document_type"],
            registry=self._registry
        )
        
        self.document_processing_latency = Histogram(
            f"{self.namespace}_document_processing_latency_seconds",
            "Document processing latency in seconds",
            labelnames=["document_type"],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
            registry=self._registry
        )
        
        # Active connections
        self.active_connections = Gauge(
            f"{self.namespace}_active_connections",
            "Number of active connections",
            labelnames=["type"],
            registry=self._registry
        )
        
        # Provider health
        self.provider_health = Gauge(
            f"{self.namespace}_provider_health",
            "Provider health status (1=healthy, 0=unhealthy)",
            labelnames=["provider"],
            registry=self._registry
        )
        
        # Application info
        self.app_info = Info(
            f"{self.namespace}_app",
            "Application information",
            registry=self._registry
        )
    
    def record_llm_request(
        self,
        provider: str,
        model: str,
        latency_seconds: float,
        status: str = "success",
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> None:
        """Record an LLM request.
        
        Args:
            provider: LLM provider name.
            model: Model name.
            latency_seconds: Request latency in seconds.
            status: Request status ('success', 'error', 'timeout').
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
        """
        if not self.enabled:
            return
        
        self.llm_request_latency.labels(
            provider=provider, model=model, operation="generate"
        ).observe(latency_seconds)
        
        self.llm_request_total.labels(
            provider=provider, model=model, status=status
        ).inc()
        
        if prompt_tokens > 0:
            self.llm_tokens_total.labels(
                provider=provider, model=model, type="prompt"
            ).inc(prompt_tokens)
        
        if completion_tokens > 0:
            self.llm_tokens_total.labels(
                provider=provider, model=model, type="completion"
            ).inc(completion_tokens)
    
    def record_cache_access(self, cache_type: str, hit: bool) -> None:
        """Record a cache access.
        
        Args:
            cache_type: Type of cache ('embedding', 'response', 'document').
            hit: Whether the cache access was a hit.
        """
        if not self.enabled:
            return
        
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_document_processed(
        self,
        status: str,
        document_type: str,
        latency_seconds: Optional[float] = None
    ) -> None:
        """Record document processing.
        
        Args:
            status: Processing status ('success', 'error').
            document_type: Type of document (e.g., 'pdf', 'docx').
            latency_seconds: Processing latency if available.
        """
        if not self.enabled:
            return
        
        self.documents_processed.labels(
            status=status, document_type=document_type
        ).inc()
        
        if latency_seconds is not None:
            self.document_processing_latency.labels(
                document_type=document_type
            ).observe(latency_seconds)
    
    def set_active_connections(self, connection_type: str, count: int) -> None:
        """Set active connection count.
        
        Args:
            connection_type: Type of connection ('websocket', 'sse').
            count: Number of active connections.
        """
        if not self.enabled:
            return
        
        self.active_connections.labels(type=connection_type).set(count)
    
    def set_provider_health(self, provider: str, healthy: bool) -> None:
        """Set provider health status.
        
        Args:
            provider: Provider name.
            healthy: Whether provider is healthy.
        """
        if not self.enabled:
            return
        
        self.provider_health.labels(provider=provider).set(1 if healthy else 0)
    
    def set_app_info(self, version: str, environment: str) -> None:
        """Set application info.
        
        Args:
            version: Application version.
            environment: Environment name.
        """
        if not self.enabled:
            return
        
        self.app_info.info({
            "version": version,
            "environment": environment
        })
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format.
        
        Returns:
            Prometheus metrics as bytes.
        """
        if not self.enabled:
            return b"# Metrics disabled - prometheus-client not installed\n"
        
        return generate_latest(self._registry)
    
    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        if not self.enabled:
            return "text/plain"
        return CONTENT_TYPE_LATEST


def timed_operation(
    metrics: "MetricsService",
    provider: str,
    model: str
) -> Callable:
    """Decorator for timing LLM operations.
    
    Args:
        metrics: MetricsService instance.
        provider: Provider name.
        model: Model name.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                latency = time.perf_counter() - start
                metrics.record_llm_request(
                    provider=provider,
                    model=model,
                    latency_seconds=latency,
                    status=status
                )
        return wrapper
    return decorator


# Global metrics instance
_metrics: Optional[MetricsService] = None


def get_metrics() -> MetricsService:
    """Get the global metrics service instance.
    
    Returns:
        MetricsService singleton.
    """
    global _metrics
    if _metrics is None:
        _metrics = MetricsService()
    return _metrics
