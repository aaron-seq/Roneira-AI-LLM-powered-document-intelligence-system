"""Azure Document Service Adapter.

Provides the AzureDocumentAnalyzer class expected by
document_intelligence_service.py by wrapping the root-level
AzureDocumentIntelligenceService implementation.
"""

import logging
import sys
import os
from typing import Any, Dict

# Add parent directory to path to import root-level modules
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from azure_document_intelligence import AzureDocumentIntelligenceService

logger = logging.getLogger(__name__)


class AzureDocumentAnalyzer:
    """Adapter class that wraps AzureDocumentIntelligenceService.

    Provides the interface expected by DocumentIntelligenceService while
    delegating to the actual Azure Document Intelligence implementation.
    """

    def __init__(self):
        self._service = AzureDocumentIntelligenceService()
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the Azure Document Analyzer service."""
        try:
            await self._service.initialize()
            self.is_initialized = self._service.is_initialized
            logger.info("AzureDocumentAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AzureDocumentAnalyzer: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if the service is healthy and operational."""
        if not self.is_initialized:
            return False
        return await self._service.check_health()

    async def analyze_document_from_file(
        self, file_path: str, analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze a document from a file path.

        Args:
            file_path: Path to the document file.
            analysis_type: Type of analysis to perform.

        Returns:
            Dictionary containing analysis results with success flag.
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # Map analysis_type to model_id for Azure API
            model_mapping = {
                "comprehensive": "prebuilt-document",
                "layout": "prebuilt-layout",
                "invoice": "prebuilt-invoice",
                "receipt": "prebuilt-receipt",
            }
            model_id = model_mapping.get(analysis_type, "prebuilt-document")

            result = await self._service.analyze_document_from_file(
                file_path=file_path, model_id=model_id
            )

            # Transform result to expected format
            return {
                "success": True,
                "data": {
                    "content": result.get("content", {}).get("full_text", ""),
                    "structured_fields": result.get("content", {}),
                    "tables": result.get("content", {}).get("tables", []),
                    "key_value_pairs": result.get("content", {}).get(
                        "key_value_pairs", {}
                    ),
                    "confidence": 0.95,
                },
                "processing_time": result.get("processing_time_seconds", 0),
            }

        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {"success": False, "error": str(e), "data": {}}

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.is_initialized = False
        logger.info("AzureDocumentAnalyzer cleaned up")
