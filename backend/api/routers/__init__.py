from fastapi import APIRouter

from .auth import router as auth_router
from .documents import router as documents_router
from .chat import router as chat_router
from .system import router as system_router

# Main API Router that aggregates all feature routers
api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(documents_router, prefix="/documents", tags=["Documents"])
api_router.include_router(chat_router, tags=["Chat"]) # Prefix handled in sub-routes for chat/search distinction
api_router.include_router(system_router, tags=["System"])
