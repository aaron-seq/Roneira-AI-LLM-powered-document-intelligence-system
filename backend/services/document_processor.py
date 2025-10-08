"""
Enhanced document processing service with AI integration
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Callable, Optional
import aiofiles
from fastapi import UploadFile
import logging

from backend.core.config import get_application_settings
from backend.services.ai.azure_intelligence import AzureDocumentIntelligenceService
from backend.services.ai.llm_enhancer import LLMEnhancementService
from backend.core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


class DocumentProcessorService:
    """Enhanced document processing with AI capabilities"""
    
    def __init__(self):
        self.settings = get_application_settings()
        self.azure_service = AzureDocumentIntelligenceService()
        self.llm_service = LLMEnhancementService()
        self.processing_status: Dict[str, Dict[str, Any]] = {}
    
    async def save_uploaded_file(self, file: UploadFile, document_id: str, user_id: str) -> str:
        """Securely save uploaded file with proper naming"""
        
        # Create user-specific subdirectory
        user_upload_dir = os.path.join(self.settings.upload_directory, user_id)
        os.makedirs(user_upload_dir, exist_ok=True)
        
        # Generate secure filename
        file_extension = os.path.splitext(file.filename)[1]
        secure_filename = f"{document_id}_{uuid.uuid4().hex[:8]}{file_extension}"
        file_path = os.path.join(user_upload_dir, secure_filename)
        
        try:
            async with aiofiles.open(file_path, 'wb') as buffer:
                content = await file.read()
                await buffer.write(content)
                
            logger.info(f"File saved successfully: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"File save error: {e}")
            raise DocumentProcessingError(
                message="Failed to save uploaded file",
                error_code="FILE_SAVE_ERROR"
            )
    
    async def process_document_with_ai(
        self, 
        document_id: str, 
        file_path: str, 
        options: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ):
        """Process document with comprehensive AI analysis"""
        
        try:
            # Initialize processing status
            self.processing_status[document_id] = {
                "status": "processing",
                "progress": 0,
                "current_stage": "initializing",
                "started_at": datetime.utcnow(),
                "stages": []
            }
            
            if progress_callback:
                await progress_callback(self.processing_status[document_id])
            
            # Stage 1: Document analysis with Azure
            await self._update_progress(
                document_id, 20, "Analyzing document structure", progress_callback
            )
            
            azure_result = await self.azure_service.analyze_document(
                file_path, options.get("extract_tables", True)
            )
            
            # Stage 2: AI enhancement
            await self._update_progress(
                document_id, 50, "Enhancing with AI insights", progress_callback
            )
            
            enhanced_result = await self.llm_service.enhance_document_data(
                azure_result, options
            )
            
            # Stage 3: Structured data extraction
            await self._update_progress(
                document_id, 80, "Extracting structured data", progress_callback
            )
            
            structured_data = await self._extract_structured_insights(enhanced_result)
            
            # Stage 4: Finalization
            await self._update_progress(
                document_id, 100, "Processing complete", progress_callback
            )
            
            # Store final result
            final_result = {
                "status": "completed",
                "document_id": document_id,
                "processed_at": datetime.utcnow().isoformat(),
                "azure_analysis": azure_result,
                "ai_insights": enhanced_result,
                "structured_data": structured_data,
                "processing_options": options
            }
            
            self.processing_status[document_id] = final_result
            
        except Exception as e:
            logger.error(f"Document processing failed for {document_id}: {e}")
            
            self.processing_status[document_id] = {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }
            
            if progress_callback:
                await progress_callback(self.processing_status[document_id])
            
        finally:
            # Cleanup temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    async def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document processing status"""
        return self.processing_status.get(document_id)
    
    async def _update_progress(
        self, 
        document_id: str, 
        progress: int, 
        stage: str, 
        callback: Optional[Callable] = None
    ):
        """Update processing progress with detailed stage information"""
        
        if document_id in self.processing_status:
            self.processing_status[document_id].update({
                "progress": progress,
                "current_stage": stage,
                "last_updated": datetime.utcnow().isoformat()
            })
            
            if callback:
                await callback(self.processing_status[document_id])
    
    async def _extract_structured_insights(self, enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure key insights from processed document"""
        
        insights = {
            "key_entities": enhanced_data.get("entities", []),
            "document_summary": enhanced_data.get("summary", ""),
            "confidence_scores": enhanced_data.get("confidence", {}),
            "extracted_tables": enhanced_data.get("tables", []),
            "metadata": {
                "page_count": enhanced_data.get("page_count", 0),
                "word_count": enhanced_data.get("word_count", 0),
                "language": enhanced_data.get("language", "unknown")
            }
        }
        
        return insights
