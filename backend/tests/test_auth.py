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


class TestPublicEndpoints:
    """Endpoints infrastructure must reach without credentials."""

    @pytest.mark.parametrize(
        "path", ["/api/", "/api/health", "/api/health/live", "/api/health/ready"]
    )
    def test_remain_public(self, client, path):
        assert client.get(path).status_code == 200
