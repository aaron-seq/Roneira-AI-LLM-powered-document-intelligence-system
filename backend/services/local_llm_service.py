"""
Local LLM service using Ollama for document intelligence
"""

import asyncio
# Remove standard logging to enforce structured logging
# import logging 
import httpx
from typing import Dict, Any, Optional, List
import json

from backend.core.config import get_settings
from backend.observability.structured_logging import get_logger, with_correlation_id

# Initialize structured logger for telemetry
logger = get_logger(__name__)


class LocalLLMService:
    """Local LLM service using Ollama"""

    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.is_initialized = False

    @with_correlation_id
    async def initialize(self):
        """Initialize the local LLM service with connection checks."""
        # WHY: We wrap this in a try/except to prevent the entire backend from crashing
        # if the optional LLM service (Ollama) is down. This ensures "Graceful Degradation".
        try:
            self.client = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=self.settings.ollama_timeout,
            )

            # Test connection and model availability
            with logger.timed("llm_initialization_check"):
                await self._check_model_availability()
                
            self.is_initialized = True
            logger.info(
                f"✅ Local LLM service initialized",
                model=self.settings.ollama_model
            )

        except Exception as e:
            # Critical: Log full stack trace for debugging
            logger.error(
                "❌ Failed to initialize local LLM service",
                exc_info=True,
                error_type=type(e).__name__
            )
            logger.warning(
                "⚠️ Application will continue in DEGRADED mode (No AI features)"
            )
            self.is_initialized = False
            # as per requirement: Do NOT raise exception to allow app to start

    async def _check_model_availability(self):
        """Check if the specified model is available"""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]

                if self.settings.ollama_model not in model_names:
                    logger.warning(
                        f"Model {self.settings.ollama_model} not found. Available models: {model_names}"
                    )
                    logger.info(f"Pulling model {self.settings.ollama_model}...")
                    await self._pull_model()
                else:
                    logger.info(f"Model {self.settings.ollama_model} is available")
            else:
                raise Exception(f"Failed to check models: {response.status_code}")

        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            raise

    async def _pull_model(self):
        """Pull the model if not available"""
        try:
            response = await self.client.post(
                "/api/pull", json={"name": self.settings.ollama_model}
            )

            if response.status_code == 200:
                logger.info(f"Successfully pulled model {self.settings.ollama_model}")
            else:
                raise Exception(f"Failed to pull model: {response.status_code}")

        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            raise

    @with_correlation_id
    async def enhance_document_data(
        self, extracted_text: str, document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance extracted document data with AI insights using structured prompts."""
        if not self.is_initialized:
            logger.warning("LLM service not initialized, returning basic processing")
            return {
                "enhanced_text": extracted_text,
                "summary": "Document processing completed without AI enhancement",
                "key_points": [],
                "entities": [],
                "confidence": 0.5,
            }

        try:
            # Create a comprehensive prompt for document analysis
            prompt = self._create_document_analysis_prompt(
                extracted_text, document_metadata
            )

            # Generate AI response with timing telemetry
            # WHY: We time this specific call because LLM inference is the most expensive operation
            # and we need to monitor latency trends.
            with logger.timed("llm_inference_enhance_data"):
                ai_response = await self._generate_response(prompt)

            # Parse and structure the response
            enhanced_data = await self._parse_ai_response(ai_response, extracted_text)

            return enhanced_data

        except Exception as e:
            logger.error(f"Error enhancing document data: {e}")
            return {
                "enhanced_text": extracted_text,
                "summary": f"Document processed with error: {str(e)}",
                "key_points": [],
                "entities": [],
                "confidence": 0.3,
            }

    def _create_document_analysis_prompt(
        self, text: str, metadata: Dict[str, Any]
    ) -> str:
        """Create a comprehensive prompt for document analysis"""

        prompt = f"""
You are an expert document analyst. Analyze the following document and provide structured insights.

Document Text:
{text[:4000]}  # Limit text to avoid token limits

Document Metadata:
- Filename: {metadata.get("filename", "Unknown")}
- File Type: {metadata.get("file_type", "Unknown")}
- Pages: {metadata.get("pages", "Unknown")}

Please provide your analysis in the following JSON format:
{{
    "summary": "A concise 2-3 sentence summary of the document's main content",
    "key_points": ["Important point 1", "Important point 2", "Important point 3"],
    "document_type": "Type of document (e.g., contract, report, invoice, letter)",
    "entities": [
        {{"type": "person", "value": "John Doe"}},
        {{"type": "organization", "value": "Company Name"}},
        {{"type": "date", "value": "2025-01-01"}},
        {{"type": "amount", "value": "$1,000"}}
    ],
    "sentiment": "positive/neutral/negative",
    "urgency": "low/medium/high",
    "action_items": ["Action 1", "Action 2"],
    "confidence": 0.85
}}

Respond ONLY with valid JSON, no additional text.
"""
        return prompt

    async def _generate_response(self, prompt: str) -> str:
        """Generate response from local LLM with retry logic and telemetry."""
        try:
            payload = {
                "model": self.settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for more consistent results
                    "top_p": 0.9,
                    "num_predict": 1000,  # Limit response length
                },
            }

            response = await self.client.post("/api/generate", json=payload)

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(
                    f"LLM generation failed",
                    status_code=response.status_code,
                    response_text=response.text
                )
                raise Exception(
                    f"LLM generation failed: {response.status_code} - {response.text}"
                )

        except Exception as e:
            logger.error("Error generating LLM response", exc_info=True)
            raise

    async def _parse_ai_response(
        self, ai_response: str, original_text: str
    ) -> Dict[str, Any]:
        """Parse and validate AI response"""
        try:
            # Try to parse as JSON
            parsed_response = json.loads(ai_response.strip())

            # Validate required fields and add defaults
            enhanced_data = {
                "enhanced_text": original_text,
                "summary": parsed_response.get(
                    "summary", "Document processed successfully"
                ),
                "key_points": parsed_response.get("key_points", []),
                "document_type": parsed_response.get("document_type", "Unknown"),
                "entities": parsed_response.get("entities", []),
                "sentiment": parsed_response.get("sentiment", "neutral"),
                "urgency": parsed_response.get("urgency", "medium"),
                "action_items": parsed_response.get("action_items", []),
                "confidence": float(parsed_response.get("confidence", 0.7)),
                "word_count": len(original_text.split()),
                "processing_notes": "Enhanced with local LLM analysis",
            }

            return enhanced_data

        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response as JSON, using fallback")
            return {
                "enhanced_text": original_text,
                "summary": ai_response[:200] + "..."
                if len(ai_response) > 200
                else ai_response,
                "key_points": [],
                "document_type": "Unknown",
                "entities": [],
                "sentiment": "neutral",
                "urgency": "medium",
                "action_items": [],
                "confidence": 0.6,
                "word_count": len(original_text.split()),
                "processing_notes": "Basic processing - AI response could not be parsed",
            }

    async def generate_chat_response(self, prompt: str) -> str:
        """Generate a direct text response for chat conversations.

        Unlike enhance_document_data which returns structured JSON,
        this method returns raw text suitable for conversational AI.
        """
        if not self.is_initialized:
            logger.warning("LLM service not initialized")
            return "I apologize, but the AI service is currently unavailable. Please ensure Ollama is running and try again."

        try:
            return await self._generate_response(prompt)
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return f"I encountered an error generating a response: {str(e)}"

    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the given text"""
        if not self.is_initialized:
            return text[:max_length] + "..." if len(text) > max_length else text

        try:
            prompt = f"""
Summarize the following text in {max_length} characters or less. Focus on the most important information:

{text[:2000]}

Summary:
"""
            response = await self._generate_response(prompt)
            return response.strip()[:max_length]

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text

    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
            logger.info("Local LLM service closed")
