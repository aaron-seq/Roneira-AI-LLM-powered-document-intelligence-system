from fastapi import APIRouter, HTTPException, Form, status, Depends
from backend.services.auth_service import AuthService
from backend.api.dependencies import get_auth_service
from backend.models.responses import AuthTokenResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/login", response_model=AuthTokenResponse)
async def authenticate_user(
    username: str = Form(...), 
    password: str = Form(...),
    auth_service: AuthService = Depends(get_auth_service)
) -> AuthTokenResponse:
    """User authentication endpoint"""
    try:
        token_data = await auth_service.authenticate_user(username, password)
        return AuthTokenResponse(**token_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed",
        )
