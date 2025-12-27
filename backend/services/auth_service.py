"""Authentication service for the Document Intelligence System.

Provides JWT-based authentication with secure password handling
and token management.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from jose import jwt, JWTError
from passlib.context import CryptContext

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

# Password hashing context - using sha256_crypt for development compatibility
# In production with proper bcrypt setup, use: schemes=["bcrypt"]
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


class AuthService:
    """Handles user authentication and token management.

    Provides methods for password verification, JWT token creation,
    and token validation.
    """

    def __init__(self):
        self.settings = get_settings()
        # Demo users for development - in production use database
        self._demo_users = {
            "demo": {
                "username": "demo",
                "password_hash": pwd_context.hash("demo"),
                "user_id": "demo_user_001",
                "roles": ["user"],
            },
            "admin": {
                "username": "admin",
                "password_hash": pwd_context.hash("admin123"),
                "user_id": "admin_user_001",
                "roles": ["user", "admin"],
            },
        }

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.

        Args:
            plain_password: Plain text password to verify.
            hashed_password: Hashed password to compare against.

        Returns:
            True if password matches, False otherwise.
        """
        return pwd_context.verify(plain_password, hashed_password)

    def hash_password(self, password: str) -> str:
        """Hash a password for storage.

        Args:
            password: Plain text password to hash.

        Returns:
            Hashed password string.
        """
        return pwd_context.hash(password)

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token.

        Args:
            data: Data to encode in the token.
            expires_delta: Optional custom expiration time.

        Returns:
            Encoded JWT token string.
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.settings.access_token_expire_minutes
            )

        to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "access"})

        encoded_jwt = jwt.encode(
            to_encode, self.settings.secret_key, algorithm=self.settings.algorithm
        )

        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token.

        Args:
            token: JWT token string to verify.

        Returns:
            Decoded token payload if valid, None otherwise.
        """
        try:
            payload = jwt.decode(
                token, self.settings.secret_key, algorithms=[self.settings.algorithm]
            )
            return payload
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None

    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate a user and return token data.

        Args:
            username: User's username.
            password: User's password.

        Returns:
            Dictionary containing access token and metadata.

        Raises:
            ValueError: If authentication fails.
        """
        # Look up user (demo implementation)
        user = self._demo_users.get(username)

        if not user:
            logger.warning(f"Authentication failed: user '{username}' not found")
            raise ValueError("Invalid username or password")

        if not self.verify_password(password, user["password_hash"]):
            logger.warning(f"Authentication failed: invalid password for '{username}'")
            raise ValueError("Invalid username or password")

        # Create access token
        token_data = {
            "sub": user["user_id"],
            "username": username,
            "roles": user["roles"],
        }

        access_token = self.create_access_token(token_data)

        logger.info(f"User '{username}' authenticated successfully")

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.settings.access_token_expire_minutes * 60,
            "user_id": user["user_id"],
        }

    async def get_current_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Get the current user from a token.

        Args:
            token: JWT access token.

        Returns:
            User data dictionary if token is valid, None otherwise.
        """
        payload = self.verify_token(token)

        if not payload:
            return None

        return {
            "user_id": payload.get("sub"),
            "username": payload.get("username"),
            "roles": payload.get("roles", []),
        }
