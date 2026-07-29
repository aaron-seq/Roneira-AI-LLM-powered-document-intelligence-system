"""Shared pytest fixtures.

Every test runs against a real application instance with a throwaway SQLite
database and a throwaway vector store, so tests exercise the same wiring
production uses rather than a parallel mock of it.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

# Configure the environment before anything imports the settings singleton.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-used-in-production-abcdef123456")
os.environ.setdefault("REQUIRE_REAL_EMBEDDINGS", "false")
# Rate limits high enough that ordinary test traffic never trips them; the
# limiter has its own dedicated test that sets them low.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "10000")
os.environ.setdefault("RATE_LIMIT_EXPENSIVE_PER_MINUTE", "10000")


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Session-scoped loop so the app starts once for all tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def workspace() -> Iterator[Path]:
    """An isolated directory for uploads, the database and the vector store."""
    with tempfile.TemporaryDirectory(prefix="roneira-tests-") as directory:
        root = Path(directory)
        (root / "uploads").mkdir()
        (root / "processed").mkdir()

        os.environ["UPLOAD_DIRECTORY"] = str(root / "uploads")
        os.environ["PROCESSED_FILES_DIRECTORY"] = str(root / "processed")
        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{root / 'test.db'}"
        os.environ["VECTOR_STORE_PATH"] = str(root / "vectors")

        from backend.core.config import reload_settings

        reload_settings()
        yield root


@pytest.fixture(scope="session")
def app(workspace: Path):
    """The FastAPI application, built after the environment is configured."""
    from backend.main import create_application

    return create_application()


@pytest.fixture(scope="session")
def client(app) -> Iterator:
    """A TestClient that runs the real startup and shutdown lifespan."""
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def auth_token(client) -> str:
    """A valid bearer token for the built-in demo user."""
    response = client.post(
        "/api/auth/token", data={"username": "demo", "password": "demo"}
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest.fixture(scope="session")
def auth_headers(auth_token: str) -> dict:
    """Authorization header for the demo user."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture(scope="session")
def admin_headers(client) -> dict:
    """Authorization header for the built-in admin user."""
    response = client.post(
        "/api/auth/token", data={"username": "admin", "password": "admin123"}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def sample_text_bytes() -> bytes:
    """A small text document with distinctive, searchable content."""
    return (
        "ACME Corporation Quarterly Report\n\n"
        "Revenue for the quarter was 4.2 million dollars, up 18 percent "
        "year over year. The engineering team shipped the Zephyr platform "
        "migration ahead of schedule.\n\n"
        "Action items: renew the Contoso support contract before March 31, "
        "and hire two additional platform engineers.\n"
    ).encode("utf-8")


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    """A structurally valid single-page PDF containing extractable text."""
    content = b"BT /F1 24 Tf 72 700 Td (Zephyr platform migration report) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_position = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(pdf)
