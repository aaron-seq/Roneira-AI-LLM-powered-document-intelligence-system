import uvicorn
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from backend.core.config import get_settings
from backend.utils.telemetry import setup_telemetry
from backend.utils.exceptions import AppError, global_exception_handler
from backend.api.routers import api_router
from backend.api.dependencies import initialize_services, cleanup_services

# Configure Telemetry (Structured Logging)
setup_telemetry()
logger = logging.getLogger("backend.api")

@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """Enhanced application lifecycle management"""
    settings = get_settings()
    logger.info("🚀 Starting Document Intelligence System...")

    try:
        # Initialize all services via dependencies module
        await initialize_services()

        # Create upload directories
        os.makedirs(settings.upload_directory, exist_ok=True)
        os.makedirs(settings.processed_files_directory, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        logger.info("✅ System initialization complete")
        yield

    except Exception as e:
        logger.error(f" Startup failed: {e}")
        raise
    finally:
        logger.info(" Shutting down gracefully...")
        await cleanup_services()


def create_application() -> FastAPI:
    """Factory function to create FastAPI application"""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        description="Enterprise-grade document processing with AI/ML capabilities",
        version=settings.version,
        docs_url="/api/docs" if settings.debug else None,
        redoc_url="/api/redoc" if settings.debug else None,
        lifespan=application_lifespan,
    )

    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # API Routes
    # Mount the aggregated router at /api
    application.include_router(api_router, prefix="/api")

    # Serve static files (for uploaded documents preview)
    if os.path.exists("uploads"):
        application.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

    return application


app = create_application()


# Exception Handlers
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return await global_exception_handler(request, exc)

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return await global_exception_handler(request, exc)


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
