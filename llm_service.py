"""
LLM Service for Document Intelligence

This module integrates with Azure OpenAI via the LangChain library to provide
advanced language model capabilities for document processing. It enhances the
data extracted by Azure Document Intelligence, providing deeper insights.

Key Features:
- Asynchronous interaction with the language model.
- Methods for enhancing structured, unstructured, and hybrid data.
- Document summarization and classification.
- Custom callback handler for monitoring LLM performance.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_core.callbacks import BaseCallbackHandler

from config import settings

logger = logging.getLogger(__name__)


class LLMPerformanceTracker(BaseCallbackHandler):
    """
    A custom LangChain callback handler to track the performance of LLM calls.
    It records duration, token usage, and any errors.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.total_tokens = 0
        self.errors = []

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        self.start_time = datetime.utcnow()

    def on_llm_end(self, response, **kwargs) -> None:
        self.end_time = datetime.utcnow()
        if hasattr(response, "llm_output") and response.llm_output:
            self.total_tokens = response.llm_output.get("token_usage", {}).get(
                "total_tokens", 0
            )

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.end_time = datetime.utcnow()
        self.errors.append(str(error))

    def get_metrics(self) -> Dict[str, Any]:
        """Returns a dictionary of the collected performance metrics."""
        duration = (
            (self.end_time - self.start_time).total_seconds()
            if self.start_time and self.end_time
            else 0
        )
        return {
            "duration_seconds": duration,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
        }


class LargeLanguageModelService:
    """
    A service class for interacting with the Azure OpenAI language model.
    """

    def __init__(self):
        self.llm: Optional[AzureChatOpenAI] = None
        self.is_initialized = False
        self._initialization_lock = asyncio.Lock()

    async def initialize(self):
        """Initializes the Azure OpenAI client."""
        async with self._initialization_lock:
            if self.is_initialized:
                return

            try:
                azure_config = settings.get_azure_openai_configuration()
                self.llm = AzureChatOpenAI(
                    openai_api_version=azure_config["api_version"],
                    azure_deployment=azure_config["deployment_name"],
                    azure_endpoint=azure_config["api_base"],
                    openai_api_key=azure_config["api_key"],
                )
                self.is_initialized = True
                logger.info("LLM service initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize LLM service: {e}")
                raise

    async def check_health(self) -> bool:
        """Performs a health check on the LLM service."""
        if not self.is_initialized:
            return False

        try:
            await self.llm.agenerate([[HumanMessage(content="Health check")]])
            return True
        except Exception:
            return False

    async def enhance_extracted_data(
        self, extracted_data: Dict[str, Any], extraction_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Enhances the data extracted by Azure Document Intelligence using the LLM.

        Args:
            extracted_data: The data from Azure Document Intelligence.
            extraction_type: The type of extraction ('structured', 'unstructured', 'hybrid').

        Returns:
            A dictionary containing the LLM-enhanced data.
        """
        performance_tracker = LLMPerformanceTracker()

        try:
            prompt_template = self._get_enhancement_prompt(extraction_type)
            chain = LLMChain(
                llm=self.llm, prompt=prompt_template, callbacks=[performance_tracker]
            )

            result_str = await chain.arun(
                azure_data=json.dumps(extracted_data, indent=2)
            )
            enhanced_data = json.loads(result_str)

            return {
                "data": enhanced_data,
                "metrics": performance_tracker.get_metrics(),
            }

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON.")
            return {"enhanced_text": result_str, "raw_azure_data": extracted_data}
        except Exception as e:
            logger.error(f"Error enhancing extraction: {e}")
            return {"error": str(e), "raw_azure_data": extracted_data}

    async def answer_question(
        self, document_context: str, question: str
    ) -> Dict[str, Any]:
        """
        Answers a user's question based on the document's content.

        Args:
            document_context: The text content of the document.
            question: The user's question.

        Returns:
            A dictionary containing the answer and performance metrics.
        """
        performance_tracker = LLMPerformanceTracker()

        try:
            messages = [
                SystemMessage(
                    content="You are a helpful assistant that answers questions based on the provided document context. "
                    "Do not make up information that is not in the context."
                ),
                HumanMessage(
                    content=f"Document Context:\n---\n{document_context}\n---\n\nQuestion: {question}"
                ),
            ]

            response = await self.llm.agenerate(
                [messages], callbacks=[performance_tracker]
            )
            answer = response.generations[0][0].text

            return {
                "answer": answer,
                "metrics": performance_tracker.get_metrics(),
            }
        except Exception as e:
            logger.error(f"Error answering question: {e}", exc_info=True)
            return {"error": str(e)}

    def _get_enhancement_prompt(self, extraction_type: str) -> PromptTemplate:
        """Returns the appropriate prompt template for the given extraction type."""

        if extraction_type == "structured":
            template = """
            Analyze the following structured data extracted from a document.
            Validate fields, standardize formats, and infer missing values.
            Return your analysis as a JSON object.
            
            Extracted Data:
            {azure_data}
            
            JSON Response:
            """
        elif extraction_type == "unstructured":
            template = """
            Analyze the following unstructured text. Extract key entities,
            summarize the content, and identify the main topics.
            Return your analysis as a JSON object.
            
            Extracted Text:
            {azure_data}
            
            JSON Response:
            """
        else:  # hybrid
            template = """
            Perform a comprehensive analysis of the following document data,
            which contains both structured and unstructured elements.
            Cross-reference the data and provide a unified summary.
            Return your analysis as a JSON object.
            
            Document Data:
            {azure_data}
            
            JSON Response:
            """

        return PromptTemplate(template=template, input_variables=["azure_data"])


# --- Singleton Instance ---
llm_service = LargeLanguageModelService()
