from fastapi import APIRouter

from .auth import router as auth_router
from .chat import router as chat_router
from .documents import router as documents_router
from .system import router as system_router

# Main API Router that aggregates all feature routers
api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(documents_router, prefix="/documents", tags=["Documents"])

# No router-level tag for these two. FastAPI *appends* an include-level tag to
# whatever a route declares, so a route tagged "RAG" inside a router included
# as "Chat" carried both — and /api/docs listed the same eight endpoints twice,
# under two headings. Their routes tag themselves; the prefix likewise lives on
# the individual routes so /chat and /search can sit side by side.
api_router.include_router(chat_router)
api_router.include_router(system_router)
