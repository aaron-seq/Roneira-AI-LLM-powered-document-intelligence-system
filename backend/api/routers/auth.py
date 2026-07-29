"""Authentication endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from backend.api.dependencies import get_auth_service
from backend.api.security import CurrentUser, get_current_user
from backend.models.responses import AuthTokenResponse, CurrentUserResponse
from backend.services.auth_service import AuthService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/token",
    response_model=AuthTokenResponse,
    summary="Exchange credentials for an access token",
)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthTokenResponse:
    """Authenticate and return a bearer token.

    Uses the standard OAuth2 password form so the Swagger "Authorize" button
    and generated clients work without custom glue.
    """
    try:
        token_data = await auth_service.authenticate_user(
            form_data.username, form_data.password
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except Exception as exc:
        logger.exception("Authentication failed unexpectedly")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed",
        ) from exc

    return AuthTokenResponse(**token_data)


# The original endpoint was /login. Kept as an alias so existing clients and
# the frontend do not break on upgrade.
@router.post(
    "/login",
    response_model=AuthTokenResponse,
    summary="Alias of /token",
    include_in_schema=False,
)
async def login_alias(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthTokenResponse:
    return await login_for_access_token(form_data, auth_service)


@router.get(
    "/me",
    response_model=CurrentUserResponse,
    summary="Describe the caller's identity",
)
async def read_current_user(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUserResponse:
    """Return the identity attached to the presented token.

    Lets a client verify a stored token is still valid without performing a
    side-effecting request.
    """
    return CurrentUserResponse(
        user_id=user.user_id,
        username=user.username,
        roles=user.roles,
        is_anonymous=user.is_anonymous,
    )
