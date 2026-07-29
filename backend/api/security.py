"""Request authentication and authorization dependencies.

``AuthService`` could mint and verify tokens before this module existed, but
nothing ever called it from a route: every document and chat endpoint was
reachable anonymously and uploads were hard-coded to a ``demo_user`` owner.
These dependencies are what actually attach an identity to a request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.api.dependencies import get_auth_service
from backend.core.config import get_settings
from backend.services.auth_service import AuthService

logger = logging.getLogger(__name__)

#: ``auto_error=False`` so we can emit our own WWW-Authenticate response and
#: so optional-auth routes can see "no credentials" without a 403.
bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")

#: Identity used when authentication is explicitly disabled for local runs.
ANONYMOUS_USER_ID = "local-dev-user"


@dataclass(frozen=True)
class CurrentUser:
    """The authenticated principal for a request."""

    user_id: str
    username: str
    roles: List[str]
    is_anonymous: bool = False

    def has_role(self, role: str) -> bool:
        return role in self.roles


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> CurrentUser:
    """Resolve the caller's identity, or reject the request.

    When ``REQUIRE_AUTHENTICATION=false`` (development convenience, refused in
    production by the settings validator) a stable anonymous principal is
    returned so ownership scoping still behaves consistently.
    """
    settings = get_settings()

    if not settings.require_authentication:
        return CurrentUser(
            user_id=ANONYMOUS_USER_ID,
            username="local",
            roles=["user"],
            is_anonymous=True,
        )

    if credentials is None or not credentials.credentials:
        raise _unauthorized("Not authenticated. Supply an 'Authorization: Bearer' token.")

    payload = auth_service.verify_token(credentials.credentials)
    if payload is None:
        raise _unauthorized("Invalid or expired token.")

    user_id = payload.get("sub")
    if not user_id:
        raise _unauthorized("Token is missing a subject claim.")

    return CurrentUser(
        user_id=user_id,
        username=payload.get("username", user_id),
        roles=list(payload.get("roles", [])),
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[CurrentUser]:
    """Resolve the caller if a valid token is present, else ``None``.

    For endpoints that are readable anonymously but richer when signed in.
    Never raises on a bad token.
    """
    settings = get_settings()
    if not settings.require_authentication:
        return CurrentUser(
            user_id=ANONYMOUS_USER_ID,
            username="local",
            roles=["user"],
            is_anonymous=True,
        )
    if credentials is None or not credentials.credentials:
        return None

    payload = auth_service.verify_token(credentials.credentials)
    if payload is None or not payload.get("sub"):
        return None

    return CurrentUser(
        user_id=payload["sub"],
        username=payload.get("username", payload["sub"]),
        roles=list(payload.get("roles", [])),
    )


def require_roles(*roles: str):
    """Build a dependency that requires the caller to hold one of ``roles``."""

    async def _dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not any(user.has_role(role) for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of the following roles: {', '.join(roles)}",
            )
        return user

    return _dependency


#: Convenience dependency for administrative endpoints.
require_admin = require_roles("admin")
