"""Authentication and authorization.

These tests exist because every one of these endpoints was previously
reachable with no credentials at all.
"""

from __future__ import annotations

import pytest

PROTECTED_ENDPOINTS = [
    ("get", "/api/documents"),
    ("get", "/api/documents/some-id/status"),
    ("get", "/api/documents/some-id"),
    ("delete", "/api/documents/some-id"),
    ("get", "/api/dashboard/metrics"),
    ("get", "/api/rag/stats"),
    ("get", "/api/auth/me"),
]


class TestTokenIssuance:
    def test_valid_credentials_return_a_token(self, client):
        response = client.post(
            "/api/auth/token", data={"username": "demo", "password": "demo"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["access_token"]
        assert body["user_id"]
        assert "user" in body["roles"]

    def test_wrong_password_is_rejected(self, client):
        response = client.post(
            "/api/auth/token", data={"username": "demo", "password": "wrong"}
        )
        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"

    def test_unknown_user_is_rejected(self, client):
        response = client.post(
            "/api/auth/token", data={"username": "nobody", "password": "whatever"}
        )
        assert response.status_code == 401

    def test_failure_message_does_not_reveal_whether_the_user_exists(self, client):
        """A different message for 'no such user' is a username oracle."""
        unknown = client.post(
            "/api/auth/token", data={"username": "nobody", "password": "x"}
        ).json()["detail"]
        wrong_password = client.post(
            "/api/auth/token", data={"username": "demo", "password": "x"}
        ).json()["detail"]
        assert unknown == wrong_password

    def test_login_alias_still_works(self, client):
        """The original /login path is kept so clients do not break."""
        response = client.post(
            "/api/auth/login", data={"username": "demo", "password": "demo"}
        )
        assert response.status_code == 200
        assert response.json()["access_token"]


class TestEndpointProtection:
    @pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
    def test_requires_a_token(self, client, method, path):
        response = getattr(client, method)(path)
        assert response.status_code == 401, (
            f"{method.upper()} {path} answered {response.status_code} without "
            "credentials; it must require authentication"
        )

    @pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
    def test_rejects_a_garbage_token(self, client, method, path):
        response = getattr(client, method)(
            path, headers={"Authorization": "Bearer not-a-real-jwt"}
        )
        assert response.status_code == 401

    def test_upload_requires_a_token(self, client, sample_text_bytes):
        response = client.post(
            "/api/documents/upload",
            files={"file": ("report.txt", sample_text_bytes, "text/plain")},
        )
        assert response.status_code == 401

    def test_chat_requires_a_token(self, client):
        response = client.post("/api/chat", json={"message": "hello"})
        assert response.status_code == 401

    def test_search_requires_a_token(self, client):
        response = client.post("/api/search", json={"query": "revenue"})
        assert response.status_code == 401


class TestIdentity:
    def test_me_describes_the_caller(self, client, auth_headers):
        response = client.get("/api/auth/me", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "demo"
        assert body["is_anonymous"] is False

    def test_admin_carries_the_admin_role(self, client, admin_headers):
        response = client.get("/api/auth/me", headers=admin_headers)
        assert response.status_code == 200
        assert "admin" in response.json()["roles"]

    def test_tampered_token_signature_is_rejected(self, client, auth_token):
        """Flipping the payload must invalidate the signature."""
        header, payload, signature = auth_token.split(".")
        tampered = f"{header}.{payload}.{signature[:-4]}AAAA"
        response = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {tampered}"}
        )
        assert response.status_code == 401


class TestTokenIntegrity:
    """A token is valid exactly as issued, or not at all.

    PyJWT 2.13.0 accepted compact signature segments containing characters
    outside the Base64URL alphabet (PYSEC-2026-4147), so appending `!!!!` to a
    valid token produced a *different string* that still verified and returned
    the same claims. The claims were not forgeable, but any code treating the
    token string as an identifier — a revocation list, deduplication, a
    per-token counter — could be bypassed by mutating it. Fixed in 2.14.0;
    this pins the behaviour so a downgrade of the dependency fails here rather
    than silently reopening it.
    """

    # No whitespace suffix here: HTTP header parsing strips trailing spaces
    # before the value reaches the application, so " " is not a mutated token,
    # it is the same token.
    @pytest.mark.parametrize("suffix", ["!!!!", "====", "....", "~~", "??"])
    def test_a_token_with_trailing_junk_is_rejected(self, client, auth_token, suffix):
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {auth_token}{suffix}"},
        )
        assert response.status_code == 401

    def test_the_unmodified_token_still_works(self, client, auth_token):
        """The guard above must not have been achieved by rejecting everything."""
        response = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    # Non-ASCII is absent on purpose: an HTTP header cannot carry it, so httpx
    # refuses to send it and the server is never reached. The service-level
    # equivalent is covered by test_non_ascii_is_rejected_at_the_service below.
    @pytest.mark.parametrize(
        "malformed",
        [
            "not-a-token",
            "only.two",
            "a.b.c.d",
            "...",
            "..",
            "eyJhbGciOiJIUzI1NiJ9..",
        ],
    )
    def test_malformed_tokens_are_401_not_500(self, client, malformed):
        """A bad token is the client's error, never a server crash.

        PYSEC-2026-4142: `PyJWS._load()` translated only `ValueError`, so some
        malformed inputs raised something that was not a `PyJWTError` — which
        the handler does not catch, turning a 401 into an unhandled 500.
        """
        response = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {malformed}"}
        )
        assert response.status_code == 401, response.text

    @pytest.mark.parametrize("token", ["ÿÿÿ.ÿÿÿ.ÿÿÿ", "ä.ö.ü", "\x00.\x00.\x00"])
    def test_non_ascii_is_rejected_at_the_service(self, app, token):
        """Verified below HTTP, where such a token can actually arrive.

        An HTTP header cannot carry these bytes, so the API cannot be the test
        surface — but a token reaching ``verify_token`` from a websocket query
        parameter or another caller can.
        """
        auth = app.state.container.auth_service

        assert auth.verify_token(token) is None


class TestPublicEndpoints:
    """Endpoints infrastructure must reach without credentials."""

    @pytest.mark.parametrize(
        "path", ["/api/", "/api/health", "/api/health/live", "/api/health/ready"]
    )
    def test_remain_public(self, client, path):
        assert client.get(path).status_code == 200
