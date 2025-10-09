# ==============================================================================
# Free Version - Main FastAPI Application
# Using only open source and free tools for document intelligence
# ==============================================================================

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer

# Pydantic models
from pydantic import BaseModel, Field

# Database and caching
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import redis.asyncio as redis

# Free services
from backend.services.free_llm_service import get_llm_service
from backend.services.free_ocr_service import get_ocr_service

# Utilities
import aiofiles
import uuid
import json
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./roneira.db")
engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Redis setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# App configuration
APP_CONFIG = {
    "title": "Roneira AI - Free Document Intelligence",
    "description": "Open source document intelligence system using free AI tools",
    "version": "2.0.0-free",
    "max_file_size": int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024,
    "upload_path": Path(os.getenv("UPLOAD_PATH", "./uploads")),
    "allowed_extensions": os.getenv("ALLOWED_FILE_EXTENSIONS", ".pdf,.docx,.jpg,.jpeg,.png,.txt").split(",")
}

# Create upload directory
APP_CONFIG["upload_path"].mkdir(exist_ok=True)

# Database Models
class DocumentModel(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(String, default="uploaded")  # uploaded, processing, completed, failed
    extracted_text = Column(Text)
    analysis_result = Column(Text)  # JSON string
    ocr_confidence = Column(Float)
    processing_time = Column(Float)
    error_message = Column(String)
    is_processed = Column(Boolean, default=False)

# Pydantic Models
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_size: int
    status: str
    message: str

class DocumentAnalysis(BaseModel):
    document_id: str
    text: str
    analysis: Dict[str, Any]
    confidence: float
    processing_time: float
    status: str

class ProcessingStatus(BaseModel):
    document_id: str
    status: str
    progress: int
    message: str
    error: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    services: Dict[str, Any]
    timestamp: datetime

# Global services
llm_service = None
ocr_service = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global llm_service, ocr_service, redis_client
    
    # Startup
    logger.info("Starting Roneira AI Free Document Intelligence System")
    
    try:
        # Initialize services
        llm_service = get_llm_service()
        ocr_service = get_ocr_service()
        
        # Initialize Redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection established")
        
        # Create database tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized")
        
        # Health check services
        llm_health = await llm_service.health_check()
        ocr_health = await ocr_service.health_check()
        
        logger.info(f"LLM Services: {llm_health['status']} - {llm_health['services_available']}")
        logger.info(f"OCR Services: {ocr_health['status']} - {ocr_health['services_available']}")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down services")
    if redis_client:
        await redis_client.close()
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=APP_CONFIG["title"],
    description=APP_CONFIG["description"],
    version=APP_CONFIG["version"],
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
if Path("./static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Security
security = HTTPBearer(auto_error=False)

# Database dependency
async def get_db() -> AsyncSession:
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Utility functions
def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""
    # Check file size
    if file.size and file.size > APP_CONFIG["max_file_size"]:
        return False
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in APP_CONFIG["allowed_extensions"]:
        return False
    
    return True

async def save_uploaded_file(file: UploadFile) -> tuple[str, str]:
    """Save uploaded file and return document ID and file path"""
    document_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    filename = f"{document_id}{file_ext}"
    file_path = APP_CONFIG["upload_path"] / filename
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return document_id, str(file_path)

async def update_processing_status(document_id: str, status: str, progress: int = 0, message: str = ""):
    """Update processing status in Redis"""
    if redis_client:
        status_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        await redis_client.setex(f"status:{document_id}", 3600, json.dumps(status_data))

# Background task for document processing
async def process_document_task(document_id: str, file_path: str, db: AsyncSession):
    """Background task to process document"""
    try:
        await update_processing_status(document_id, "processing", 10, "Starting OCR extraction")
        
        # OCR extraction
        start_time = datetime.utcnow()
        ocr_result = await ocr_service.extract_text_from_document(file_path)
        
        if not ocr_result["success"]:
            raise Exception(f"OCR failed: {ocr_result.get('error', 'Unknown error')}")
        
        await update_processing_status(document_id, "processing", 50, "Analyzing document with AI")
        
        # AI Analysis
        extracted_text = ocr_result.get("text", "")
        if extracted_text:
            analysis_result = await llm_service.analyze_document(extracted_text, "general")
            insights = await llm_service.generate_insights(analysis_result)
        else:
            analysis_result = {"error": "No text extracted"}
            insights = {"error": "No insights generated"}
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        await update_processing_status(document_id, "processing", 90, "Saving results")
        
        # Update database
        from sqlalchemy import select, update
        stmt = (
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(
                processing_status="completed",
                extracted_text=extracted_text,
                analysis_result=json.dumps({
                    "analysis": analysis_result,
                    "insights": insights,
                    "ocr_details": ocr_result
                }),
                ocr_confidence=ocr_result.get("confidence", 0),
                processing_time=processing_time,
                is_processed=True
            )
        )
        await db.execute(stmt)
        await db.commit()
        
        await update_processing_status(document_id, "completed", 100, "Processing completed successfully")
        logger.info(f"Document {document_id} processed successfully in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {e}")
        
        # Update database with error
        from sqlalchemy import update
        stmt = (
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(
                processing_status="failed",
                error_message=str(e),
                is_processed=False
            )
        )
        await db.execute(stmt)
        await db.commit()
        
        await update_processing_status(document_id, "failed", 0, f"Processing failed: {str(e)}")

# API Endpoints
@app.get("/", response_class=FileResponse)
async def root():
    """Serve the main application"""
    static_path = Path("./static/index.html")
    if static_path.exists():
        return FileResponse(static_path)
    return {"message": "Roneira AI Document Intelligence System - API is running"}

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        # Check services
        llm_health = await llm_service.health_check() if llm_service else {"status": "unavailable"}
        ocr_health = await ocr_service.health_check() if ocr_service else {"status": "unavailable"}
        
        # Check Redis
        redis_health = {"status": "unavailable"}
        if redis_client:
            try:
                await redis_client.ping()
                redis_health = {"status": "healthy"}
            except Exception as e:
                redis_health = {"status": "unhealthy", "error": str(e)}
        
        # Check database
        db_health = {"status": "unavailable"}
        try:
            async with engine.begin() as conn:
                await conn.execute("SELECT 1")
            db_health = {"status": "healthy"}
        except Exception as e:
            db_health = {"status": "unhealthy", "error": str(e)}
        
        overall_status = "healthy"
        if any(service.get("status") != "healthy" for service in [llm_health, ocr_health, redis_health, db_health]):
            overall_status = "degraded"
        
        return HealthCheck(
            status=overall_status,
            services={
                "llm": llm_health,
                "ocr": ocr_health,
                "redis": redis_health,
                "database": db_health
            },
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            services={"error": str(e)},
            timestamp=datetime.utcnow()
        )

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process a document"""
    try:
        # Validate file
        if not validate_file(file):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file. Check file size and format."
            )
        
        # Save file
        document_id, file_path = await save_uploaded_file(file)
        
        # Create database record
        document = DocumentModel(
            id=document_id,
            filename=Path(file_path).name,
            original_filename=file.filename,
            file_size=file.size or 0,
            file_type=Path(file.filename).suffix.lower(),
            processing_status="uploaded"
        )
        
        db.add(document)
        await db.commit()
        
        # Start background processing
        background_tasks.add_task(process_document_task, document_id, file_path, db)
        
        await update_processing_status(document_id, "queued", 0, "Document queued for processing")
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_size=file.size or 0,
            status="uploaded",
            message="Document uploaded successfully and queued for processing"
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/status", response_model=ProcessingStatus)
async def get_processing_status(document_id: str):
    """Get processing status for a document"""
    try:
        if redis_client:
            status_data = await redis_client.get(f"status:{document_id}")
            if status_data:
                data = json.loads(status_data)
                return ProcessingStatus(
                    document_id=document_id,
                    status=data["status"],
                    progress=data["progress"],
                    message=data["message"]
                )
        
        # Fallback to database
        from sqlalchemy import select
        stmt = select(DocumentModel).where(DocumentModel.id == document_id)
        async with SessionLocal() as db:
            result = await db.execute(stmt)
            document = result.scalar_one_or_none()
            
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            return ProcessingStatus(
                document_id=document_id,
                status=document.processing_status,
                progress=100 if document.is_processed else 0,
                message="Status from database",
                error=document.error_message
            )
            
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}", response_model=DocumentAnalysis)
async def get_document_analysis(document_id: str, db: AsyncSession = Depends(get_db)):
    """Get document analysis results"""
    try:
        from sqlalchemy import select
        stmt = select(DocumentModel).where(DocumentModel.id == document_id)
        result = await db.execute(stmt)
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not document.is_processed:
            raise HTTPException(status_code=202, detail="Document is still processing")
        
        analysis_data = json.loads(document.analysis_result) if document.analysis_result else {}
        
        return DocumentAnalysis(
            document_id=document_id,
            text=document.extracted_text or "",
            analysis=analysis_data,
            confidence=document.ocr_confidence or 0,
            processing_time=document.processing_time or 0,
            status=document.processing_status
        )
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)):
    """List all documents"""
    try:
        from sqlalchemy import select
        stmt = (
            select(DocumentModel)
            .offset(skip)
            .limit(limit)
            .order_by(DocumentModel.upload_time.desc())
        )
        result = await db.execute(stmt)
        documents = result.scalars().all()
        
        return [
            {
                "id": doc.id,
                "filename": doc.original_filename,
                "upload_time": doc.upload_time,
                "status": doc.processing_status,
                "file_size": doc.file_size,
                "is_processed": doc.is_processed
            }
            for doc in documents
        ]
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Roneira AI Free on {host}:{port}")
    
    uvicorn.run(
        "main_free:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development"
    )