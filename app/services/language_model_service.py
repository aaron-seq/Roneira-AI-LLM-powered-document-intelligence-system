"""Language Model Service Adapter.

Provides the LanguageModelProcessor class expected by
document_intelligence_service.py by wrapping the root-level
LargeLanguageModelService implementation.
"""

import logging
import sys
import os
from typing import Any, Dict, Optional

# Add parent directory to path to import root-level modules
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from llm_service import LargeLanguageModelService

logger = logging.getLogger(__name__)


class LanguageModelProcessor:
    """Adapter class that wraps LargeLanguageModelService.

    Provides the interface expected by DocumentIntelligenceService while
    delegating to the actual LLM service implementation.
    """

    def __init__(self):
        self._service = LargeLanguageModelService()
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the Language Model Processor service."""
        try:
            await self._service.initialize()
            self.is_initialized = self._service.is_initialized
            logger.info("LanguageModelProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LanguageModelProcessor: {e}")
            # Don't raise - LLM enhancement is optional
            self.is_initialized = False

    async def health_check(self) -> bool:
        """Check if the service is healthy and operational."""
        if not self.is_initialized:
            return False
        return await self._service.check_health()

    async def enhance_document_data(
        self,
        extracted_data: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enhance extracted document data using the language model.

        Args:
            extracted_data: Data extracted from Azure Document Intelligence.
            document_context: Optional context about the document.

        Returns:
            Dictionary containing enhanced data with success flag.
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "LLM service not initialized",
                "data": extracted_data,
            }

        try:
            # Use the underlying service's enhancement method
            result = await self._service.enhance_extracted_data(
                extracted_data=extracted_data, extraction_type="hybrid"
            )

            # Check for errors in the result
            if "error" in result:
                return {
                    "success": False,
                    "error": result.get("error"),
                    "data": extracted_data,
                }

            # Transform to expected format
            enhanced_data = result.get("data", {})

            return {
                "success": True,
                "data": {
                    "summary": enhanced_data.get("summary", ""),
                    "insights": enhanced_data.get("key_insights", []),
                    "entities": enhanced_data.get("entities", []),
                    "sentiment": enhanced_data.get("sentiment", {}),
                    "classification": enhanced_data.get("classification", {}),
                    "confidence": 0.85,
                },
                "metrics": result.get("metrics", {}),
            }

        except Exception as e:
            logger.error(f"Error enhancing document data: {e}")
            return {"success": False, "error": str(e), "data": extracted_data}

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.is_initialized = False
        logger.info("LanguageModelProcessor cleaned up")
