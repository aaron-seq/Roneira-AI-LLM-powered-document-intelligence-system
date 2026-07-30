"""
Chat Service for RAG-Powered Conversations

Provides chat completion functionality with RAG integration,
streaming responses, and conversation management.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.services.local_llm_service import LocalLLMService
from backend.services.memory_service import MemoryService
from backend.services.prompt_service import PromptService
from backend.services.retrieval_service import RetrievalService
from backend.utils.exceptions import LLMUnavailableError

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
        # The retrieval service is injected, not constructed here. When chat
        # built its own instance it also built its own vector store, so
        # documents indexed by the upload pipeline were invisible to chat —
        # every RAG answer came back with zero sources.
        self.retrieval = retrieval_service or RetrievalService()
        self.memory = memory_service or MemoryService()
        self.prompts = prompt_service or PromptService()
        self.llm = llm_service or LocalLLMService()

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize all component services."""
        if self.is_initialized:
            return
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
        owner_id: Optional[str] = None,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> ChatResponse:
        """
        Process a chat message with optional RAG.

        Args:
            message: User message
            session_id: Conversation session ID
            use_rag: Whether to use retrieval augmentation
            rag_top_k: Number of chunks to retrieve
            document_filter: Restrict retrieval to a single document
            owner_id: Restrict retrieval to this user's documents
            allowed_document_ids: Explicit allow-list of retrievable documents

        Returns:
            ChatResponse with generated message and grounded citations.
        """
        if not self.is_initialized:
            await self.initialize()

        session_id = session_id or str(uuid.uuid4())

        await self.memory.add_user_message(session_id, message)
        history = await self.memory.get_context_messages(session_id)

        context = ""
        sources: List[Dict[str, Any]] = []
        grounded = False

        if use_rag:
            document_ids = allowed_document_ids
            if document_filter:
                # Honour the narrower of the two: an explicit document filter
                # can only ever restrict within what the caller may read.
                if document_ids is not None and document_filter not in document_ids:
                    document_ids = []
                else:
                    document_ids = [document_filter]

            retrieval_result = await self.retrieval.retrieve(
                query=message,
                top_k=rag_top_k,
                owner_id=owner_id,
                document_ids=document_ids,
            )

            context = retrieval_result.combined_context
            grounded = bool(retrieval_result.results)
            sources = [
                {
                    "document_id": r.document_id,
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "page_number": r.metadata.get("page_number"),
                    "filename": r.metadata.get("filename"),
                    "content_preview": (
                        r.content[:200] + "..." if len(r.content) > 200 else r.content
                    ),
                }
                for r in retrieval_result.results
            ]

        chat_messages = self.prompts.build_chat_messages(
            user_message=message,
            context=context if context else None,
            history=history[:-1] if len(history) > 1 else None,  # Exclude current
        )

        retrieved_anything = grounded
        llm_failed = False
        try:
            response_text = await self._generate_response(chat_messages)
        except LLMUnavailableError as exc:
            llm_failed = True
            response_text = exc.message

        # `grounded` promises the answer was *built from* the retrieved
        # passages. Retrieval succeeding is not enough: if generation failed
        # there is no answer to ground, and reporting True would put a
        # "grounded" badge and a citation list beside a failure notice.
        grounded = retrieved_anything and not llm_failed

        if use_rag and not retrieved_anything:
            # Say so rather than letting the model answer from parametric
            # memory while the UI shows an empty citation list.
            preamble = (
                "I could not find anything in your indexed documents that answers this. "
            )
            if not llm_failed:
                preamble += (
                    "Answering from general knowledge instead, so please "
                    "verify independently.\n\n"
                )
            response_text = preamble + response_text
        elif llm_failed:
            response_text += (
                " The sources below are the closest matches in your documents; "
                "read them directly to verify."
            )

        await self.memory.add_assistant_message(session_id, response_text)

        return ChatResponse(
            message=response_text,
            session_id=session_id,
            sources=sources,
            model=self.llm.settings.ollama_model,
            usage={
                "context_chunks": len(sources),
                "history_messages": len(history),
                "grounded": grounded,
                "embeddings_are_real": self.retrieval.embeddings_are_real,
            },
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
            response if response else "I couldn't generate a response. Please try again."
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
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        owner_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index a document for RAG retrieval."""
        result = await self.retrieval.index_document(
            document_id=document_id,
            content=content,
            metadata=metadata,
            owner_id=owner_id,
        )
        return result.to_dict()

    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        owner_id: Optional[str] = None,
        allowed_document_ids: Optional[List[str]] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search indexed documents, scoped to what the caller may read."""
        document_ids = allowed_document_ids
        if document_id:
            if document_ids is not None and document_id not in document_ids:
                return []
            document_ids = [document_id]

        result = await self.retrieval.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            owner_id=owner_id,
            document_ids=document_ids,
        )
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
