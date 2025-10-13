"""Authentication and authorization module.

Provides JWT token management, user authentication, and authorization
functionalities with secure best practices.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from config import application_settings
from core.exceptions import AuthenticationError, AuthorizationError

logger = logging.getLogger(__name__)

# Security configurations
security_scheme = HTTPBearer()
password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token with expiration."""
    
    token_data = data.copy()
    
    if expires_delta:
        expire_time = datetime.utcnow() + expires_delta
    else:
        expire_time = datetime.utcnow() + timedelta(
            minutes=application_settings.token_expiry_minutes
        )
    
    token_data.update({
        "exp": expire_time,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    try:
        encoded_token = jwt.encode(
            token_data,
            application_settings.secret_key,
            algorithm=application_settings.jwt_algorithm
        )
        
        logger.info(f"Access token created for user: {data.get('sub')}")
        return encoded_token
        
    except Exception as e:
        logger.error(f"Failed to create access token: {e}")
        raise AuthenticationError("Token creation failed")


def verify_access_token(token: str) -> Dict[str, Any]:
    """Verify and decode a JWT access token."""
    
    try:
        payload = jwt.decode(
            token,
            application_settings.secret_key,
            algorithms=[application_settings.jwt_algorithm]
        )
        
        # Verify token type
        if payload.get("type") != "access":
            raise AuthenticationError("Invalid token type")
        
        # Verify expiration
        expiration = payload.get("exp")
        if not expiration or datetime.fromtimestamp(expiration) < datetime.utcnow():
            raise AuthenticationError("Token has expired")
        
        return payload
        
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise AuthenticationError("Invalid token")
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise AuthenticationError("Token verification failed")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return password_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return password_context.verify(plain_password, hashed_password)


async def get_authenticated_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme)
) -> str:
    """Extract and validate user from JWT token."""
    
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        payload = verify_access_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise AuthenticationError("Invalid token payload")
        
        return user_id
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[str]:
    """Extract user from JWT token if provided, otherwise return None."""
    
    if not credentials or not credentials.credentials:
        return None
    
    try:
        return await get_authenticated_user(credentials)
    except HTTPException:
        return None


def create_user_session_token(
    user_id: str,
    session_data: Optional[Dict[str, Any]] = None
) -> str:
    """Create a session token for user tracking."""
    
    data = {
        "sub": user_id,
        "session": True
    }
    
    if session_data:
        data.update(session_data)
    
    return create_access_token(data)


class RequireRole:
    """Dependency class for role-based access control."""
    
    def __init__(self, required_roles: list[str]):
        self.required_roles = required_roles
    
    async def __call__(self, user_id: str = Depends(get_authenticated_user)) -> str:
        """Check if user has required roles."""
        
        # In a real application, you would check user roles from database
        # For now, we'll implement basic admin check
        user_roles = await self._get_user_roles(user_id)
        
        if not any(role in user_roles for role in self.required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return user_id
    
    async def _get_user_roles(self, user_id: str) -> list[str]:
        """Get user roles from database or cache."""
        # Implement actual role lookup here
        # For demo purposes, return default roles
        return ["user", "admin"] if user_id == "admin" else ["user"]


# Common role dependencies
require_admin = RequireRole(["admin"])
require_user = RequireRole(["user", "admin"])
