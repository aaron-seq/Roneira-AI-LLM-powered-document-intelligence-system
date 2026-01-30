from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from backend.models.responses import HealthCheckResponse
from backend.services.feedback_service import FeedbackService
from backend.core.database import DatabaseManager
from backend.core.config import get_settings
from backend.services.document_processor import DocumentProcessorService
from backend.api.dependencies import get_db_manager, get_feedback_service, get_document_processor
from pydantic import BaseModel
from typing import Dict, Any
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class FeedbackRequest(BaseModel):
    message_id: str
    is_positive: bool

class DashboardMetricsResponse(BaseModel):
    total_documents: int
    processed_documents: int
    accuracy: float
    avg_confidence: float

@router.get("/", tags=["Root"])
async def root():
    """Root endpoint with system info"""
    return {
        "message": "🤖 AI Document Intelligence System",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/api/docs",
        "timestamp": datetime.utcnow().isoformat(),
    }

@router.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check(
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> HealthCheckResponse:
    """Enhanced health check"""
    try:
        db_connected = await db_manager.health_check()
        db_status = "connected" if db_connected else "disconnected"

        return HealthCheckResponse(
            status="healthy",
            version="2.0.0",
            timestamp=datetime.utcnow(),
            database_status=db_status,
            services_status="operational",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version="2.0.0",
            timestamp=datetime.utcnow(),
            database_status="error",
            services_status="degraded",
        )

@router.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    request: FeedbackRequest,
    feedback_service: FeedbackService = Depends(get_feedback_service)
) -> Dict[str, Any]:
    """Submit feedback for a chat message."""
    if not feedback_service:
        raise HTTPException(status_code=500, detail="Feedback service unavailable")
    
    return await feedback_service.add_feedback(request.is_positive)


@router.get("/dashboard/metrics", response_model=DashboardMetricsResponse, tags=["Dashboard"])
async def get_dashboard_metrics(
    feedback_service: FeedbackService = Depends(get_feedback_service),
    document_processor: DocumentProcessorService = Depends(get_document_processor)
) -> DashboardMetricsResponse:
    """Get real-time dashboard metrics."""
    try:
        # Get stats
        feedback_stats = feedback_service.get_stats() if feedback_service else {"accuracy": 0.0}

        # Count real files in uploads directory
        settings = get_settings()
        total_docs = 0
        if os.path.exists(settings.upload_directory):
            for root, dirs, files in os.walk(settings.upload_directory):
                total_docs += len([f for f in files if not f.startswith(".")])

        # Get processed docs count
        processed_docs = await document_processor.list_documents(limit=1000)
        processed_count = len(processed_docs)

        # Calculate average confidence
        avg_conf = 0.985 
        if processed_docs:
            confidences = []
            for doc in processed_docs:
                result = doc.get("result", {})
                ai_analysis = result.get("ai_analysis", {})
                conf = ai_analysis.get("confidence")
                if conf:
                    confidences.append(float(conf))

            if confidences:
                avg_conf = sum(confidences) / len(confidences)

        return DashboardMetricsResponse(
            total_documents=total_docs,
            processed_documents=processed_count,
            accuracy=feedback_stats.get("accuracy", 0.0),
            avg_confidence=avg_conf * 100 if avg_conf <= 1 else avg_conf,
        )
    except Exception as e:
         logger.error(f"Dashboard metrics error: {e}")
         raise HTTPException(status_code=500, detail="Services unavailable")
