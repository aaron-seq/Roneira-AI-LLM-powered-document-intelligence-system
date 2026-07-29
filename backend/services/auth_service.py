"""JWT authentication.

The built-in users are a development convenience so the system is usable the
moment it starts. They are *not* a user management system: see
``docs/SECURITY.md`` for what to replace before running this with real data.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

# pbkdf2_sha256 is in passlib's core, needs no compiled backend, and is a
# genuine password KDF. The previous sha256_crypt choice worked but the bcrypt
# comment above it was misleading about what was actually running.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

#: Generic message for every failure mode. Distinguishing "no such user" from
#: "wrong password" turns the login endpoint into a username oracle.
_AUTH_FAILURE_MESSAGE = "Invalid username or password"


class AuthService:
    """Issues and verifies JWT access tokens."""

    def __init__(self, users: Optional[Dict[str, Dict[str, Any]]] = None):
        self.settings = get_settings()
        self._users = users if users is not None else self._build_default_users()
        # Pre-computed hash used to equalise timing when a username does not
        # exist, so an attacker cannot enumerate accounts by response latency.
        self._dummy_hash = pwd_context.hash("timing-equalisation-placeholder")

    def _build_default_users(self) -> Dict[str, Dict[str, Any]]:
        """Create the development users, honouring password overrides.

        ``DEMO_USER_PASSWORD`` / ``ADMIN_USER_PASSWORD`` let a deployment change
        these without a code edit. They are still development accounts.
        """
        demo_password = os.getenv("DEMO_USER_PASSWORD", "demo")
        admin_password = os.getenv("ADMIN_USER_PASSWORD", "admin123")

        if self.settings.is_production:
            logger.warning(
                "Built-in demo accounts are enabled in production. Replace "
                "AuthService with a real user store before serving real data."
            )

        return {
            "demo": {
                "username": "demo",
                "password_hash": pwd_context.hash(demo_password),
                "user_id": "demo_user_001",
                "roles": ["user"],
            },
            "admin": {
                "username": "admin",
                "password_hash": pwd_context.hash(admin_password),
                "user_id": "admin_user_001",
                "roles": ["user", "admin"],
            },
        }

    # ------------------------------------------------------------- passwords
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Check a password against its hash."""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as exc:
            logger.warning("Password verification error: %s", exc)
            return False

    def hash_password(self, password: str) -> str:
        """Hash a password for storage."""
        return pwd_context.hash(password)

    # ---------------------------------------------------------------- tokens
    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Mint a signed JWT.

        Args:
            data: Claims to embed. Must include ``sub``.
            expires_delta: Lifetime override.

        Returns:
            The encoded token.
        """
        now = datetime.now(timezone.utc)
        expires = now + (
            expires_delta or timedelta(minutes=self.settings.access_token_expire_minutes)
        )

        payload = {**data, "exp": expires, "iat": now, "nbf": now, "type": "access"}
        return jwt.encode(
            payload, self.settings.secret_key, algorithm=self.settings.algorithm
        )

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate a token.

        Returns:
            The claims, or ``None`` if the token is invalid, expired, or is not
            an access token.
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                # Pinning the algorithm is what stops an attacker presenting a
                # token signed with "none" or with a different scheme.
                algorithms=[self.settings.algorithm],
            )
        except JWTError as exc:
            logger.debug("Token verification failed: %s", exc)
            return None

        if payload.get("type") != "access":
            logger.debug("Rejected token with type=%r", payload.get("type"))
            return None

        return payload

    # ------------------------------------------------------------ user login
    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Verify credentials and return token material.

        Raises:
            ValueError: on any authentication failure, always with the same
                message regardless of cause.
        """
        user = self._users.get(username)

        if user is None:
            # Still perform a hash comparison so the timing matches the
            # valid-username path.
            self.verify_password(password, self._dummy_hash)
            logger.info("Authentication failed for unknown user %r", username)
            raise ValueError(_AUTH_FAILURE_MESSAGE)

        if not self.verify_password(password, user["password_hash"]):
            logger.info("Authentication failed for user %r", username)
            raise ValueError(_AUTH_FAILURE_MESSAGE)

        access_token = self.create_access_token(
            {
                "sub": user["user_id"],
                "username": user["username"],
                "roles": user["roles"],
            }
        )

        logger.info("User %r authenticated", username)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.settings.access_token_expire_minutes * 60,
            "user_id": user["user_id"],
            "username": user["username"],
            "roles": list(user["roles"]),
        }

    async def get_current_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Resolve a token to a user record, or ``None`` when invalid."""
        payload = self.verify_token(token)
        if not payload:
            return None

        return {
            "user_id": payload.get("sub"),
            "username": payload.get("username"),
            "roles": payload.get("roles", []),
        }

    @property
    def usernames(self) -> List[str]:
        """Names of the configured users (for diagnostics, never for auth)."""
        return list(self._users)
