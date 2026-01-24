"""Language Model Service Adapter.

Provides the LanguageModelProcessor class expected by
document_intelligence_service.py by implementing LLM logic directly
using LangChain and Ollama.
"""

import logging
import json
from typing import Any, Dict, Optional

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Import global settings securely
try:
    from config import settings
except ImportError:
    # Fallback for when running inside deep package structure without proper path
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import settings

logger = logging.getLogger(__name__)


# Define expected output structure for robust parsing
class EnhancedDocumentData(BaseModel):
    summary: str = Field(description="A concise summary of the document")
    key_insights: list[str] = Field(description="List of key insights key_insights")
    entities: list[str] = Field(description="List of important entities named entities")
    sentiment: Dict[str, float] = Field(description="Sentiment analysis scores")
    classification: Dict[str, float] = Field(description="Document classification confidence")


class LanguageModelProcessor:
    """Service that enhances document data using Ollama (via LangChain)."""

    def __init__(self):
        self._llm = None
        self.is_initialized = False
        self._parser = JsonOutputParser(pydantic_object=EnhancedDocumentData)

    async def initialize(self) -> None:
        """Initialize the Language Model connection (Ollama)."""
        try:
            # Initialize Ollama chat model
            self._llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.1,  # Low temperature for factual extraction
                format="json",  # Force JSON mode
                timeout=120.0   # Increase timeout for large docs
            )
            
            # Simple connection check
            logger.info(f"Initializing Ollama connection to {settings.ollama_base_url} with model {settings.ollama_model}...")
            # We don't await invoke here to keep startup fast, but we mark initialized.
            # Real check happens in health_check.
            self.is_initialized = True
            logger.info("LanguageModelProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LanguageModelProcessor: {e}")
            self.is_initialized = False

    async def health_check(self) -> bool:
        """Check if the service is healthy and operational."""
        if not self.is_initialized or not self._llm:
            return False
        
        try:
            # Send a minimal prompt to verify connectivity
            response = await self._llm.ainvoke("hi")
            return True if response else False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

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
        if not self.is_initialized or not self._llm:
            return {
                "success": False,
                "error": "LLM service not initialized (Ollama)",
                "data": extracted_data,
            }

        try:
            # Prepare context from extracted content
            content = extracted_data.get("content", "")
            if not content:
                # Fallback to key-value pairs if raw content is missing
                content = json.dumps(extracted_data.get("key_value_pairs", {}))

            # Truncate content to avoid context window issues (approx 6000 chars)
            truncated_content = content[:12000]

            system_prompt = (
                "You are an expert document analyst. Analyze the following document text "
                "and extract structured insights in JSON format. "
                "Return code ONLY JSON. \n"
                f"{self._parser.get_format_instructions()}"
            )

            user_prompt = f"Document content:\n{truncated_content}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            # Invoke LLM
            response = await self._llm.ainvoke(messages)
            
            # Parse output
            try:
                # ChatOllama usually returns .content string
                parsed_data = self._parser.parse(response.content)
            except Exception as parse_error:
                logger.warning(f"Failed to parse JSON directly, attempting relaxed parsing: {parse_error}")
                # Fallback: Extract JSON substring if needed
                text = response.content
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end != -1:
                    parsed_data = json.loads(text[start:end])
                else:
                    raise parse_error

            return {
                "success": True,
                "data": {
                    "summary": parsed_data.get("summary", "No summary available"),
                    "insights": parsed_data.get("key_insights", []),
                    "entities": parsed_data.get("entities", []),
                    "sentiment": parsed_data.get("sentiment", {}),
                    "classification": parsed_data.get("classification", {}),
                    "confidence": 0.85, # Estimated
                },
                "metrics": {
                    "model": settings.ollama_model,
                    "source": "ollama"
                },
            }

        except Exception as e:
            logger.error(f"Error enhancing document data with Ollama: {e}")
            return {"success": False, "error": str(e), "data": extracted_data}

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.is_initialized = False
        self._llm = None
        logger.info("LanguageModelProcessor cleaned up")
