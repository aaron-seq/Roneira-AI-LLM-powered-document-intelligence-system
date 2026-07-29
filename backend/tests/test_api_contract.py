"""API contract, middleware behaviour and error handling.

Covers the cross-cutting guarantees clients depend on: honest health
reporting, correlation IDs, security headers, and error envelopes that do not
leak internals.
"""

from __future__ import annotations

import pytest


class TestHealth:
    def test_health_reports_each_component(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert set(body["components"]) >= {
            "database",
            "embeddings",
            "vector_store",
            "llm",
        }

    def test_health_distinguishes_degraded_from_healthy(self, client):
        """Tests run without an LLM, so health must say 'degraded'."""
        body = client.get("/api/health").json()
        assert body["status"] in ("healthy", "degraded", "unhealthy")
        if body["components"]["llm"]["status"] != "ok":
            assert body["status"] == "degraded"

    def test_degraded_components_explain_themselves(self, client):
        components = client.get("/api/health").json()["components"]
        for name, component in components.items():
            if component["status"] != "ok":
                assert component["detail"], (
                    f"{name} is {component['status']} with no detail"
                )

    def test_liveness_does_not_depend_on_the_database(self, client):
        assert client.get("/api/health/live").json() == {"status": "alive"}

    def test_readiness_reports_ready(self, client):
        assert client.get("/api/health/ready").json()["status"] == "ready"


class TestCorrelationIds:
    def test_every_response_carries_a_correlation_id(self, client):
        response = client.get("/api/health")
        assert response.headers.get("X-Correlation-ID")

    def test_an_inbound_correlation_id_is_preserved(self, client):
        """A trace must span the browser and the API."""
        response = client.get(
            "/api/health", headers={"X-Correlation-ID": "trace-abc-123"}
        )
        assert response.headers["X-Correlation-ID"] == "trace-abc-123"

    def test_x_request_id_is_accepted_as_a_fallback(self, client):
        response = client.get("/api/health", headers={"X-Request-ID": "req-999"})
        assert response.headers["X-Correlation-ID"] == "req-999"

    def test_an_overlong_correlation_id_is_truncated(self, client):
        """Otherwise a client can write unbounded data into our logs."""
        response = client.get("/api/health", headers={"X-Correlation-ID": "z" * 5000})
        assert len(response.headers["X-Correlation-ID"]) <= 128

    def test_responses_report_their_handling_time(self, client):
        response = client.get("/api/health")
        assert float(response.headers["X-Response-Time-ms"]) >= 0


class TestSecurityHeaders:
    @pytest.mark.parametrize(
        "header,expected",
        [
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
            ("Referrer-Policy", "strict-origin-when-cross-origin"),
        ],
    )
    def test_hardening_headers_are_present(self, client, header, expected):
        assert client.get("/api/health").headers.get(header) == expected

    def test_a_content_security_policy_is_set(self, client):
        csp = client.get("/api/health").headers.get("Content-Security-Policy")
        assert csp and "frame-ancestors 'none'" in csp

    def test_hsts_is_absent_outside_production(self, client):
        """HSTS over plain HTTP would pin developers into a broken state."""
        assert "Strict-Transport-Security" not in client.get("/api/health").headers

    def test_headers_are_applied_to_error_responses_too(self, client):
        response = client.get("/api/documents/does-not-exist/status")
        assert response.status_code == 401
        assert response.headers.get("X-Content-Type-Options") == "nosniff"


class TestErrorEnvelope:
    def test_unknown_paths_return_404(self, client):
        assert client.get("/api/no-such-endpoint").status_code == 404

    def test_validation_errors_return_422(self, client, auth_headers):
        response = client.post("/api/chat", headers=auth_headers, json={})
        assert response.status_code == 422

    def test_empty_chat_message_is_rejected(self, client, auth_headers):
        response = client.post("/api/chat", headers=auth_headers, json={"message": ""})
        assert response.status_code == 422

    def test_rag_top_k_is_bounded(self, client, auth_headers):
        """An unbounded top_k is a cheap way to force a huge retrieval."""
        response = client.post(
            "/api/chat",
            headers=auth_headers,
            json={"message": "hello", "rag_top_k": 10_000},
        )
        assert response.status_code == 422

    def test_error_responses_do_not_leak_stack_traces(self, client, auth_headers):
        response = client.get("/api/documents/../../etc/passwd", headers=auth_headers)
        assert "Traceback" not in response.text
        assert "/home/" not in response.text


class TestMetricsEndpoint:
    def test_metrics_are_exposed_in_prometheus_format(self, client):
        response = client.get("/api/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_http_request_metrics_are_recorded(self, client):
        client.get("/api/health")
        body = client.get("/api/metrics").text
        assert "roneira_http_requests_total" in body

    def test_metrics_use_route_templates_not_concrete_paths(self, client, auth_headers):
        """Document IDs as label values would explode metric cardinality."""
        client.get("/api/documents/abc-123-def/status", headers=auth_headers)
        body = client.get("/api/metrics").text
        assert "abc-123-def" not in body

    def test_embedding_backend_is_published_as_a_gauge(self, client):
        assert "roneira_embedding_backend_real" in client.get("/api/metrics").text


class TestOpenApiSchema:
    def test_schema_is_served(self, client):
        response = client.get("/api/openapi.json")
        assert response.status_code == 200
        assert response.json()["info"]["title"]

    def test_documented_endpoints_exist(self, client):
        paths = client.get("/api/openapi.json").json()["paths"]
        for path in [
            "/api/auth/token",
            "/api/auth/me",
            "/api/documents/upload",
            "/api/documents/{document_id}",
            "/api/chat",
            "/api/search",
            "/api/health",
        ]:
            assert path in paths, f"{path} is missing from the OpenAPI schema"

    def test_protected_endpoints_declare_their_security_scheme(self, client):
        """Generated clients rely on this to know a token is required."""
        schema = client.get("/api/openapi.json").json()
        upload = schema["paths"]["/api/documents/upload"]["post"]
        assert "security" in upload


class TestRateLimiting:
    def test_responses_advertise_the_remaining_budget(self, client, auth_headers):
        response = client.get("/api/documents", headers=auth_headers)
        assert int(response.headers["X-RateLimit-Limit"]) > 0
        assert "X-RateLimit-Remaining" in response.headers

    def test_health_checks_are_exempt(self, client):
        """Infrastructure polls these on a fixed schedule."""
        assert "X-RateLimit-Limit" not in client.get("/api/health").headers

    def test_exceeding_the_budget_returns_429(self):
        """Built with a deliberately tiny budget rather than by hammering."""
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from backend.api.middleware import RateLimitMiddleware

        app = Starlette(
            routes=[Route("/ping", lambda request: PlainTextResponse("pong"))]
        )
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=3, expensive_per_minute=1
        )

        with TestClient(app) as limited:
            statuses = [limited.get("/ping").status_code for _ in range(5)]

        assert statuses[:3] == [200, 200, 200]
        assert statuses[3:] == [429, 429]

    def test_a_429_tells_the_client_when_to_retry(self):
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from backend.api.middleware import RateLimitMiddleware

        app = Starlette(
            routes=[Route("/ping", lambda request: PlainTextResponse("pong"))]
        )
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=1, expensive_per_minute=1
        )

        with TestClient(app) as limited:
            limited.get("/ping")
            blocked = limited.get("/ping")

        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) > 0
        assert blocked.json()["error"] == "RateLimitExceeded"
