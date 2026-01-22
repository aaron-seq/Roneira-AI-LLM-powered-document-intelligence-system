"""
Chat Service for RAG-Powered Conversations

Provides chat completion functionality with RAG integration,
streaming responses, and conversation management.
"""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import asyncio
import json

from backend.services.retrieval_service import RetrievalService
from backend.services.memory_service import MemoryService
from backend.services.prompt_service import PromptService
from backend.services.local_llm_service import LocalLLMService

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Response from a chat completion."""

    message: str
    session_id: str
    sources: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "session_id": self.session_id,
            "sources": self.sources,
            "model": self.model,
            "usage": self.usage,
        }


@dataclass
class ChatRequest:
    """Request for a chat completion."""

    message: str
    session_id: Optional[str] = None
    use_rag: bool = True
    rag_top_k: int = 5
    document_filter: Optional[str] = None
    stream: bool = False


class ChatService:
    """
    Main service for RAG-powered chat completions.

    Integrates retrieval, memory, prompts, and LLM for
    context-aware conversations.
    """

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        memory_service: Optional[MemoryService] = None,
        prompt_service: Optional[PromptService] = None,
        llm_service: Optional[LocalLLMService] = None,
    ):
        self.retrieval = retrieval_service or RetrievalService()
        self.memory = memory_service or MemoryService()
        self.prompts = prompt_service or PromptService()
        self.llm = llm_service or LocalLLMService()

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize all component services."""
        await self.retrieval.initialize()
        await self.llm.initialize()
        self.is_initialized = True
        logger.info("ChatService initialized successfully")

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        use_rag: bool = True,
        rag_top_k: int = 5,
        document_filter: Optional[str] = None,
    ) -> ChatResponse:
        """
        Process a chat message with optional RAG.

        Args:
            message: User message
            session_id: Conversation session ID
            use_rag: Whether to use retrieval augmentation
            rag_top_k: Number of chunks to retrieve
            document_filter: Filter to specific document

        Returns:
            ChatResponse with generated message
        """
        if not self.is_initialized:
            await self.initialize()

        # Generate session ID if not provided
        import uuid

        session_id = session_id or str(uuid.uuid4())

        # Store user message in memory
        await self.memory.add_user_message(session_id, message)

        # Get conversation history
        history = await self.memory.get_context_messages(session_id)

        # Retrieve relevant context if RAG enabled
        context = ""
        sources = []

        if use_rag:
            filter_dict = None
            if document_filter:
                filter_dict = {"document_id": document_filter}

            retrieval_result = await self.retrieval.retrieve(
                query=message, top_k=rag_top_k, filter_dict=filter_dict
            )

            context = retrieval_result.combined_context
            sources = [
                {
                    "document_id": r.document_id,
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "content_preview": r.content[:200] + "..."
                    if len(r.content) > 200
                    else r.content,
                }
                for r in retrieval_result.results
            ]

        # Build prompt with context
        chat_messages = self.prompts.build_chat_messages(
            user_message=message,
            context=context if context else None,
            history=history[:-1] if len(history) > 1 else None,  # Exclude current
        )

        # Generate response
        response_text = await self._generate_response(chat_messages)

        # Store assistant response in memory
        await self.memory.add_assistant_message(session_id, response_text)

        return ChatResponse(
            message=response_text,
            session_id=session_id,
            sources=sources,
            model=self.llm.settings.ollama_model,
            usage={"context_chunks": len(sources), "history_messages": len(history)},
        )

    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from LLM."""
        # Combine messages into a single prompt
        prompt_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)

        # Use LLM service's chat-specific method
        response = await self.llm.generate_chat_response(full_prompt)

        return (
            response
            if response
            else "I couldn't generate a response. Please try again."
        )

    async def stream_chat(
        self, message: str, session_id: Optional[str] = None, use_rag: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response.

        Yields chunks of the response as they're generated.
        """
        # For now, generate full response and simulate streaming
        response = await self.chat(
            message=message, session_id=session_id, use_rag=use_rag
        )

        # Simulate streaming by yielding words
        words = response.message.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)  # Small delay for streaming effect

    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        return await self.memory.get_conversation_history(session_id)

    async def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        return await self.memory.clear_session(session_id)

    async def index_document(
        self, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Index a document for RAG retrieval."""
        result = await self.retrieval.index_document(
            document_id=document_id, content=content, metadata=metadata
        )
        return result.to_dict()

    async def search_documents(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search indexed documents."""
        result = await self.retrieval.retrieve(query=query, top_k=top_k)
        return [r.to_dict() for r in result.results]

    async def get_stats(self) -> Dict[str, Any]:
        """Get chat service statistics."""
        retrieval_stats = await self.retrieval.get_stats()
        memory_stats = self.memory.get_stats()
        prompt_stats = self.prompts.get_stats()

        return {
            "initialized": self.is_initialized,
            "retrieval": retrieval_stats,
            "memory": memory_stats,
            "prompts": prompt_stats,
        }

    async def cleanup(self) -> None:
        """Clean up all resources."""
        await self.retrieval.cleanup()
        await self.memory.cleanup()
        await self.llm.close()
        self.is_initialized = False
        logger.info("ChatService cleaned up")
