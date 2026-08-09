#!/usr/bin/env python3
"""Load the sample corpus into a running instance.

Turns a first run into something you can immediately ask questions of, rather
than an empty index and a login page.

    python scripts/load_samples.py
    python scripts/load_samples.py --url http://localhost:8000 --pattern 'INV-*.pdf'

Requires the API to be running. Uses only the standard library plus httpx,
which is already a dependency.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover - dependency is in requirements.txt
    sys.exit("httpx is required: pip install httpx")

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "samples"

# Exit codes, so this is usable from a script or a Makefile.
EXIT_OK = 0
EXIT_UNREACHABLE = 2
EXIT_AUTH_FAILED = 3
EXIT_SOME_FAILED = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--username", default="demo")
    parser.add_argument("--password", default="demo")
    parser.add_argument(
        "--pattern", default="*", help="Glob to select a subset, e.g. 'INV-*.pdf'"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for each document to finish processing",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Queue the uploads and exit without waiting",
    )
    return parser.parse_args()


def authenticate(client: httpx.Client, username: str, password: str) -> str:
    """Exchange credentials for a bearer token."""
    response = client.post(
        "/api/auth/token", data={"username": username, "password": password}
    )
    if response.status_code != 200:
        print(f"  Authentication failed ({response.status_code}): {response.text}")
        sys.exit(EXIT_AUTH_FAILED)
    return response.json()["access_token"]


def check_capabilities(client: httpx.Client) -> None:
    """Report degraded capabilities before uploading anything.

    Better to know now that search will be keyword-only than to conclude the
    system is bad at answering questions.
    """
    try:
        health = client.get("/api/health").json()
    except Exception as exc:
        print(f"  Could not reach {client.base_url} — is the API running?  ({exc})")
        sys.exit(EXIT_UNREACHABLE)

    components = health.get("components", {})
    if health.get("status") != "healthy":
        print(f"  Service reports: {health.get('status')}")
        for name, component in components.items():
            if component.get("status") != "ok":
                print(f"    - {name}: {component.get('detail')}")
        print()


def upload(
    client: httpx.Client, path: Path, headers: dict, max_retries: int = 3
) -> str | None:
    """Upload one file, returning its document id or None on rejection.

    Uploads are rate limited (they cost a document parse), and a corpus larger
    than the per-minute budget will hit it. Honouring Retry-After is the
    correct client behaviour and keeps the demo working out of the box.
    """
    for attempt in range(max_retries + 1):
        with path.open("rb") as handle:
            response = client.post(
                "/api/documents/upload",
                headers=headers,
                files={"file": (path.name, handle)},
            )

        if response.status_code == 202:
            return response.json()["document_id"]

        if response.status_code == 429 and attempt < max_retries:
            wait = int(response.headers.get("Retry-After", "30")) + 1
            print(f"  rate limited — waiting {wait}s before retrying {path.name}")
            time.sleep(wait)
            continue

        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        print(f"  {path.name}: rejected ({response.status_code}) — {detail}")
        return None

    return None


def wait_for(
    client: httpx.Client, document_id: str, headers: dict, timeout: float
) -> str:
    """Poll until the document reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/documents/{document_id}/status", headers=headers).json()
        if body["status"] in ("completed", "failed"):
            return body["status"]
        time.sleep(0.5)
    return "timeout"


def main() -> int:
    args = parse_args()

    if not SAMPLES_DIR.is_dir():
        sys.exit(f"Sample directory not found: {SAMPLES_DIR}")

    files = sorted(
        path
        for path in SAMPLES_DIR.glob(args.pattern)
        if path.is_file() and path.suffix.lower() != ".md"
    )
    if not files:
        sys.exit(f"No files matched {args.pattern!r} in {SAMPLES_DIR}")

    with httpx.Client(base_url=args.url, timeout=60.0) as client:
        check_capabilities(client)

        token = authenticate(client, args.username, args.password)
        headers = {"Authorization": f"Bearer {token}"}

        print(f"Uploading {len(files)} document(s) to {args.url}\n")

        queued: list[tuple[str, Path]] = []
        for path in files:
            document_id = upload(client, path, headers)
            if document_id:
                queued.append((document_id, path))
                print(f"  {path.name}: queued")

        if args.no_wait:
            print(f"\n{len(queued)} document(s) queued.")
            return EXIT_OK

        print("\nProcessing…\n")
        completed = failed = 0
        for document_id, path in queued:
            status = wait_for(client, document_id, headers, args.timeout)
            if status == "completed":
                completed += 1
                print(f"  {path.name}: indexed")
            else:
                failed += 1
                detail = client.get(
                    f"/api/documents/{document_id}/status", headers=headers
                ).json()
                print(f"  {path.name}: {status} — {detail.get('error', '')}")

        print(
            f"\n{completed} indexed, {failed} failed, {len(files) - len(queued)} rejected"
        )

        if completed:
            stats = client.get("/api/rag/stats", headers=headers).json()
            chunks = stats.get("vector_store", {}).get("document_count", 0)
            print(f"{chunks} searchable chunks in the index.")

            if not stats.get("embeddings_are_real"):
                print(
                    "\nNote: search is keyword-only. Install sentence-transformers "
                    "for semantic matching."
                )

            print("\nTry:")
            print(f"  curl -X POST {args.url}/api/search \\")
            print(f'    -H "Authorization: Bearer {token[:16]}…" \\')
            print('    -H "Content-Type: application/json" \\')
            print("""    -d '{"query":"total amount due","top_k":3}'""")

        return EXIT_SOME_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
