"""Document Intelligence Service for processing and analyzing documents.

Provides comprehensive document processing capabilities including OCR,
text extraction, AI-powered analysis, and data structuring.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio

from config import application_settings
from app.core.exceptions import DocumentProcessingError, ExternalServiceError
from app.services.azure_document_service import AzureDocumentAnalyzer
from app.services.language_model_service import LanguageModelProcessor
from app.core.database_manager import get_database_session, DocumentRecord
from sqlalchemy import select, update

logger = logging.getLogger(__name__)


class DocumentIntelligenceService:
    """Main service for document processing and intelligence extraction."""

    def __init__(self):
        self.azure_analyzer: Optional[AzureDocumentAnalyzer] = None
        self.language_processor: Optional[LanguageModelProcessor] = None
        self.is_initialized = False
        self.processing_stats = {
            "documents_processed": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
            "success_rate": 0,
        }

    async def initialize(self) -> None:
        """Initialize the document intelligence service."""
        try:
            self.azure_analyzer = AzureDocumentAnalyzer()
            self.language_processor = LanguageModelProcessor()

            await self.azure_analyzer.initialize()
            await self.language_processor.initialize()

            self.is_initialized = True
            logger.info("Document Intelligence Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Document Intelligence Service: {e}")
            raise DocumentProcessingError(
                f"Service initialization failed: {e}", error_code="INITIALIZATION_ERROR"
            )

    async def health_check(self) -> bool:
        """Check the health of the document intelligence service."""
        if not self.is_initialized:
            return False

        try:
            azure_healthy = await self.azure_analyzer.health_check()
            llm_healthy = await self.language_processor.health_check()

            return azure_healthy and llm_healthy

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def process_document(
        self, document_id: str, file_path: str, user_id: str, original_filename: str
    ) -> Dict[str, Any]:
        """Process a document through the complete intelligence pipeline."""

        if not self.is_initialized:
            raise DocumentProcessingError(
                "Service not initialized",
                document_id=document_id,
                error_code="SERVICE_NOT_INITIALIZED",
            )

        processing_start_time = datetime.utcnow()

        try:
            # Save initial document record
            await self._save_document_record(
                document_id=document_id,
                user_id=user_id,
                original_filename=original_filename,
                file_path=file_path,
                status="processing",
                processing_stage="initialization",
            )

            # Step 1: Extract content using Azure Document Intelligence
            logger.info(f"Starting Azure analysis for document {document_id}")
            await self._update_processing_stage(document_id, "azure_analysis")

            azure_result = await self.azure_analyzer.analyze_document_from_file(
                file_path=file_path, analysis_type="comprehensive"
            )

            if not azure_result.get("success"):
                raise DocumentProcessingError(
                    f"Azure analysis failed: {azure_result.get('error')}",
                    document_id=document_id,
                    stage="azure_analysis",
                )

            # Step 2: Enhance with AI language model
            logger.info(f"Starting LLM enhancement for document {document_id}")
            await self._update_processing_stage(document_id, "llm_enhancement")

            enhanced_result = await self.language_processor.enhance_document_data(
                extracted_data=azure_result.get("data", {}),
                document_context={
                    "filename": original_filename,
                    "document_id": document_id,
                    "user_id": user_id,
                },
            )

            if not enhanced_result.get("success"):
                logger.warning(
                    f"LLM enhancement failed for {document_id}, using raw data"
                )
                final_data = azure_result.get("data", {})
            else:
                final_data = enhanced_result.get("data", {})

            # Step 3: Structure and validate final results
            await self._update_processing_stage(document_id, "finalization")

            structured_result = await self._structure_final_result(
                document_id=document_id,
                azure_data=azure_result.get("data", {}),
                enhanced_data=final_data,
                metadata={
                    "original_filename": original_filename,
                    "file_size": Path(file_path).stat().st_size,
                    "processing_time": (
                        datetime.utcnow() - processing_start_time
                    ).total_seconds(),
                },
            )

            # Step 4: Save final results
            await self._save_processing_results(
                document_id=document_id,
                results=structured_result,
                processing_time=(
                    datetime.utcnow() - processing_start_time
                ).total_seconds(),
            )

            # Update statistics
            await self._update_processing_statistics(success=True)

            logger.info(f"Successfully processed document {document_id}")

            return {
                "success": True,
                "document_id": document_id,
                "data": structured_result,
                "processing_time": (
                    datetime.utcnow() - processing_start_time
                ).total_seconds(),
            }

        except DocumentProcessingError as e:
            # Re-raise document processing errors
            await self._save_error_result(document_id, str(e))
            await self._update_processing_statistics(success=False)
            raise

        except Exception as e:
            logger.error(f"Unexpected error processing document {document_id}: {e}")
            error_msg = f"Unexpected processing error: {e}"

            await self._save_error_result(document_id, error_msg)
            await self._update_processing_statistics(success=False)

            return {"success": False, "document_id": document_id, "error": error_msg}

    async def get_user_documents(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
        status_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get documents for a specific user with pagination."""

        try:
            async with get_database_session() as session:
                query = select(DocumentRecord).where(DocumentRecord.user_id == user_id)

                if status_filter:
                    query = query.where(DocumentRecord.status == status_filter)

                query = query.order_by(DocumentRecord.created_at.desc())
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                documents = result.scalars().all()

                return [
                    {
                        "document_id": doc.id,
                        "filename": doc.original_filename,
                        "status": doc.status,
                        "created_at": doc.created_at,
                        "processed_at": doc.processed_at,
                        "file_size_bytes": doc.file_size_bytes,
                        "processing_time_seconds": doc.processing_time_seconds,
                        "error_message": doc.error_message,
                    }
                    for doc in documents
                ]

        except Exception as e:
            logger.error(f"Failed to get user documents: {e}")
            return []

    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics for monitoring."""
        return self.processing_stats.copy()

    async def cleanup(self) -> None:
        """Clean up service resources."""
        if self.azure_analyzer:
            await self.azure_analyzer.cleanup()

        if self.language_processor:
            await self.language_processor.cleanup()

        logger.info("Document Intelligence Service cleaned up")

    # Private helper methods
    async def _save_document_record(
        self,
        document_id: str,
        user_id: str,
        original_filename: str,
        file_path: str,
        status: str,
        processing_stage: Optional[str] = None,
    ) -> None:
        """Save document record to database."""

        try:
            file_size = Path(file_path).stat().st_size

            async with get_database_session() as session:
                document = DocumentRecord(
                    id=document_id,
                    user_id=user_id,
                    original_filename=original_filename,
                    file_path=file_path,
                    status=status,
                    processing_stage=processing_stage,
                    file_size_bytes=file_size,
                )

                session.add(document)
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to save document record: {e}")
            raise DocumentProcessingError(f"Database save failed: {e}")

    async def _update_processing_stage(self, document_id: str, stage: str) -> None:
        """Update document processing stage."""

        try:
            async with get_database_session() as session:
                query = (
                    update(DocumentRecord)
                    .where(DocumentRecord.id == document_id)
                    .values(processing_stage=stage, updated_at=datetime.utcnow())
                )

                await session.execute(query)
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to update processing stage: {e}")

    async def _structure_final_result(
        self,
        document_id: str,
        azure_data: Dict[str, Any],
        enhanced_data: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Structure the final processing result."""

        return {
            "document_id": document_id,
            "metadata": metadata,
            "extraction": {
                "raw_text": azure_data.get("content", ""),
                "structured_data": azure_data.get("structured_fields", {}),
                "tables": azure_data.get("tables", []),
                "key_value_pairs": azure_data.get("key_value_pairs", {}),
            },
            "analysis": {
                "summary": enhanced_data.get("summary", ""),
                "key_insights": enhanced_data.get("insights", []),
                "entities": enhanced_data.get("entities", []),
                "sentiment": enhanced_data.get("sentiment", {}),
                "classification": enhanced_data.get("classification", {}),
            },
            "confidence_scores": {
                "extraction_confidence": azure_data.get("confidence", 0.0),
                "analysis_confidence": enhanced_data.get("confidence", 0.0),
            },
        }

    async def _save_processing_results(
        self, document_id: str, results: Dict[str, Any], processing_time: float
    ) -> None:
        """Save final processing results to database."""

        try:
            async with get_database_session() as session:
                query = (
                    update(DocumentRecord)
                    .where(DocumentRecord.id == document_id)
                    .values(
                        status="completed",
                        extracted_data=json.dumps(results),
                        processed_at=datetime.utcnow(),
                        processing_time_seconds=int(processing_time),
                        updated_at=datetime.utcnow(),
                    )
                )

                await session.execute(query)
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to save processing results: {e}")

    async def _save_error_result(self, document_id: str, error_message: str) -> None:
        """Save error result to database."""

        try:
            async with get_database_session() as session:
                query = (
                    update(DocumentRecord)
                    .where(DocumentRecord.id == document_id)
                    .values(
                        status="failed",
                        error_message=error_message,
                        processed_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                )

                await session.execute(query)
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to save error result: {e}")

    async def _update_processing_statistics(self, success: bool) -> None:
        """Update processing statistics."""

        self.processing_stats["documents_processed"] += 1

        if success:
            # Update success rate calculation
            total_docs = self.processing_stats["documents_processed"]
            current_successes = (self.processing_stats["success_rate"] / 100.0) * (
                total_docs - 1
            ) + 1
            self.processing_stats["success_rate"] = (
                current_successes / total_docs
            ) * 100
        else:
            # Recalculate success rate
            total_docs = self.processing_stats["documents_processed"]
            current_successes = (self.processing_stats["success_rate"] / 100.0) * (
                total_docs - 1
            )
            self.processing_stats["success_rate"] = (
                (current_successes / total_docs) * 100 if total_docs > 0 else 0
            )
