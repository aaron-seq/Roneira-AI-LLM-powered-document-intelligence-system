"""HTTP middleware: request correlation, security headers, and metrics.

The frontend already sent ``X-Correlation-ID`` on every request and the global
exception handler already read ``request.state.correlation_id`` — but nothing
populated it, so every error log said ``correlation_id: unknown``. This closes
that loop and adds the response hardening a browser-facing API needs.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from backend.observability.structured_logging import (
    correlation_id_var,
    request_path_var,
    user_id_var,
)

logger = logging.getLogger("backend.api.access")

CORRELATION_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"

#: Headers applied to every response. CSP is deliberately strict: this API
#: serves JSON and user-supplied files, never its own HTML.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-site",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), interest-cohort=()",
    "Content-Security-Policy": (
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
        "img-src 'self' data:; style-src 'unsafe-inline'"
    ),
}


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to every request, log, and response.

    An inbound ``X-Correlation-ID`` (or ``X-Request-ID``) is honoured so a
    trace can span the browser and the API; otherwise one is generated.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = (
            request.headers.get(CORRELATION_HEADER)
            or request.headers.get(REQUEST_ID_HEADER)
            or str(uuid.uuid4())
        )
        # Bound so a hostile client cannot write unbounded data into our logs.
        correlation_id = correlation_id[:128]

        request.state.correlation_id = correlation_id
        correlation_token = correlation_id_var.set(correlation_id)
        path_token = request_path_var.set(request.url.path)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            correlation_id_var.reset(correlation_token)
            request_path_var.reset(path_token)

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[CORRELATION_HEADER] = correlation_id
        response.headers["X-Response-Time-ms"] = f"{duration_ms:.1f}"

        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={
                "correlation_id": correlation_id,
                "http_method": request.method,
                "http_path": request.url.path,
                "http_status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply standard hardening headers to every response."""

    def __init__(self, app: ASGIApp, hsts: bool = False):
        super().__init__(app)
        # HSTS is only meaningful over TLS and actively harmful on a plain-HTTP
        # local run, so it is opt-in and enabled for production only.
        self.hsts = hsts

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if self.hsts:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request counts and latency into the Prometheus registry.

    Uses the route *template* (``/api/documents/{document_id}/status``) rather
    than the concrete path, so document IDs do not explode metric cardinality.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        from backend.observability.metrics import get_metrics

        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - started
            route = request.scope.get("route")
            endpoint = getattr(route, "path", None) or "unmatched"
            try:
                get_metrics().record_request(
                    method=request.method,
                    endpoint=endpoint,
                    status_code=status_code,
                    duration_seconds=duration,
                )
            except Exception as exc:  # pragma: no cover - metrics must never 500
                logger.debug("Failed to record request metrics: %s", exc)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-client rate limiting.

    Document processing and LLM inference are the most expensive things this
    service does and both were previously reachable without any budget: one
    client could saturate the worker pool indefinitely.

    Scope: counters live in this process, so with N replicas the effective
    limit is N x the configured value. That is the same guarantee slowapi's
    default in-memory backend gives. For a hard global limit, enforce it at
    the ingress/gateway or move the counters to Redis.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int,
        expensive_per_minute: int,
        expensive_paths: tuple = (),
        exempt_paths: tuple = (),
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.expensive_per_minute = expensive_per_minute
        self.expensive_paths = expensive_paths
        self.exempt_paths = exempt_paths
        # bucket key -> (window_start_epoch_minute, count)
        self._buckets: dict = {}

    def _client_key(self, request: Request) -> str:
        """Identify the caller, preferring the authenticated subject.

        Falls back to the peer address. ``X-Forwarded-For`` is deliberately
        NOT trusted: unless the proxy is known to overwrite it, any client can
        set it and reset its own budget.
        """
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            # Hash so tokens never become dictionary keys or appear in a dump.
            return "t:" + str(hash(auth[7:]))
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    def _limit_for(self, path: str) -> int:
        if any(path.startswith(prefix) for prefix in self.expensive_paths):
            return self.expensive_per_minute
        return self.requests_per_minute

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if path in self.exempt_paths:
            return await call_next(request)

        limit = self._limit_for(path)
        window = int(time.time() // 60)
        bucket = "expensive" if limit == self.expensive_per_minute else "default"
        key = f"{self._client_key(request)}|{bucket}"

        stored_window, count = self._buckets.get(key, (window, 0))
        if stored_window != window:
            stored_window, count = window, 0

        count += 1
        self._buckets[key] = (stored_window, count)

        # Opportunistic sweep so the dictionary cannot grow without bound.
        if len(self._buckets) > 10_000:
            self._buckets = {k: v for k, v in self._buckets.items() if v[0] >= window - 1}

        if count > limit:
            retry_after = 60 - int(time.time() % 60)
            logger.warning(
                "Rate limit exceeded for %s on %s (%s/%s)", key, path, count, limit
            )
            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "message": (
                        f"Rate limit of {limit} requests/minute exceeded. "
                        f"Retry in {retry_after}s."
                    ),
                    "details": {"limit_per_minute": limit},
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - count))
        return response


def bind_user_to_logs(user_id: str) -> None:
    """Attach ``user_id`` to the logging context for the current request."""
    user_id_var.set(user_id)
