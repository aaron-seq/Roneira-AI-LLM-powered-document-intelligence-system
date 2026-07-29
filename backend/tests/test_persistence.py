"""Durability of document state across application restarts.

Document status used to live in ``DocumentProcessorService.processing_status``,
a plain dict on a singleton. Restarting the process — or simply running a
second worker — lost every upload, and ``GET /api/documents`` returned only
whatever that worker happened to hold.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


def _wait(client, headers, document_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/documents/{document_id}/status", headers=headers).json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    pytest.fail("document did not finish processing")


@pytest.fixture
def restartable_env(tmp_path, monkeypatch):
    """Point the app at a fresh database and vector store under tmp_path.

    These tests deliberately rebuild the settings singleton, so the cache is
    restored afterwards. Without that, every later test in the session would
    keep resolving this fixture's temporary paths and throwaway secret key.
    """
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("SECRET_KEY", "restart-test-secret-key-abcdefghijklmnop")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'restart.db'}")
    monkeypatch.setenv("UPLOAD_DIRECTORY", str(tmp_path / "uploads"))
    monkeypatch.setenv("PROCESSED_FILES_DIRECTORY", str(tmp_path / "processed"))
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "vectors"))
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")
    monkeypatch.setenv("RATE_LIMIT_EXPENSIVE_PER_MINUTE", "10000")

    yield tmp_path

    # monkeypatch has now reverted the environment; rebuild the cached
    # settings from it so the session-scoped app keeps working.
    from backend.core.config import reload_settings

    reload_settings()


def _boot():
    """Build and start a fresh application instance."""
    from backend.core.config import reload_settings
    from backend.main import create_application

    reload_settings()
    return create_application()


class TestDocumentsSurviveRestart:
    def test_a_document_is_still_listed_after_a_restart(self, restartable_env):
        payload = b"Board minutes: the Zephyr acquisition was approved."

        with TestClient(_boot()) as client:
            token = client.post(
                "/api/auth/token", data={"username": "demo", "password": "demo"}
            ).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            upload = client.post(
                "/api/documents/upload",
                headers=headers,
                files={"file": ("minutes.txt", payload, "text/plain")},
            )
            document_id = upload.json()["document_id"]
            _wait(client, headers, document_id)

        # Second boot: new process-equivalent, same storage.
        with TestClient(_boot()) as client:
            token = client.post(
                "/api/auth/token", data={"username": "demo", "password": "demo"}
            ).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            status = client.get(f"/api/documents/{document_id}/status", headers=headers)
            assert status.status_code == 200, "document was lost across restart"
            assert status.json()["status"] == "completed"

            listing = client.get("/api/documents", headers=headers).json()
            assert document_id in {d["document_id"] for d in listing["documents"]}

    def test_extracted_text_survives_a_restart(self, restartable_env):
        payload = b"Contract clause 7.2 governs termination for convenience."

        with TestClient(_boot()) as client:
            token = client.post(
                "/api/auth/token", data={"username": "demo", "password": "demo"}
            ).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            document_id = client.post(
                "/api/documents/upload",
                headers=headers,
                files={"file": ("contract.txt", payload, "text/plain")},
            ).json()["document_id"]
            _wait(client, headers, document_id)

        with TestClient(_boot()) as client:
            token = client.post(
                "/api/auth/token", data={"username": "demo", "password": "demo"}
            ).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            detail = client.get(f"/api/documents/{document_id}", headers=headers)
            assert "clause 7.2" in detail.json()["result"]["original_text"]

    def test_ownership_is_still_enforced_after_a_restart(self, restartable_env):
        payload = b"Private note about the Contoso renewal."

        with TestClient(_boot()) as client:
            token = client.post(
                "/api/auth/token", data={"username": "demo", "password": "demo"}
            ).json()["access_token"]
            document_id = client.post(
                "/api/documents/upload",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("note.txt", payload, "text/plain")},
            ).json()["document_id"]

        with TestClient(_boot()) as client:
            admin = client.post(
                "/api/auth/token", data={"username": "admin", "password": "admin123"}
            ).json()["access_token"]
            probe = client.get(
                f"/api/documents/{document_id}/status",
                headers={"Authorization": f"Bearer {admin}"},
            )
            assert probe.status_code == 404


class TestSourceFileRetention:
    def test_the_source_file_is_kept_on_disk_by_default(self, restartable_env):
        """Deleting sources made every citation impossible to verify."""
        with TestClient(_boot()) as client:
            token = client.post(
                "/api/auth/token", data={"username": "demo", "password": "demo"}
            ).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            document_id = client.post(
                "/api/documents/upload",
                headers=headers,
                files={"file": ("kept.txt", b"Retained content.", "text/plain")},
            ).json()["document_id"]
            _wait(client, headers, document_id)

        stored = list((restartable_env / "uploads").rglob("*"))
        assert any(path.is_file() for path in stored), "source file was deleted"

    def test_retention_can_be_disabled(self, restartable_env, monkeypatch):
        """Deployments that must not retain documents at rest need this."""
        monkeypatch.setenv("RETAIN_SOURCE_FILES", "false")

        with TestClient(_boot()) as client:
            token = client.post(
                "/api/auth/token", data={"username": "demo", "password": "demo"}
            ).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            document_id = client.post(
                "/api/documents/upload",
                headers=headers,
                files={"file": ("transient.txt", b"Ephemeral content.", "text/plain")},
            ).json()["document_id"]
            _wait(client, headers, document_id)

            # The document record remains; only the file is gone.
            assert (
                client.get(
                    f"/api/documents/{document_id}/status", headers=headers
                ).status_code
                == 200
            )
            assert (
                client.get(
                    f"/api/documents/{document_id}/source", headers=headers
                ).status_code
                == 410
            )

        files = [p for p in (restartable_env / "uploads").rglob("*") if p.is_file()]
        assert not files, f"source files remained despite retention being off: {files}"
