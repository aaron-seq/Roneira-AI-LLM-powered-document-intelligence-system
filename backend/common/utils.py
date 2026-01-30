"""
Shared utilities for Roneira AI.
Generic functions that are project-agnostic.
"""

import uuid
import hashlib
from datetime import datetime

def generate_uuid() -> str:
    """Generate a standard UUID string."""
    return str(uuid.uuid4())

def get_current_utc_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat()

def hash_string(input_string: str) -> str:
    """Return SHA-256 hash of a string."""
    return hashlib.sha256(input_string.encode()).hexdigest()
