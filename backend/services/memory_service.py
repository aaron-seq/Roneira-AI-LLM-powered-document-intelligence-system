"""
Memory Service for Conversation History

Manages conversation memory and context for multi-turn
chat interactions in RAG applications.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import asyncio
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in a conversation."""

    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(
                data.get("timestamp", datetime.utcnow().isoformat())
            ),
            message_id=data.get("message_id", str(uuid.uuid4())),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationSession:
    """Represents a conversation session with message history."""

    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict] = None
    ) -> Message:
        """Add a message to the conversation."""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_messages_for_context(
        self, max_messages: int = 10, max_tokens: Optional[int] = None
    ) -> List[Message]:
        """Get recent messages for context."""
        if not self.messages:
            return []

        recent = self.messages[-max_messages:]

        if max_tokens:
            # Rough token estimation (4 chars per token)
            filtered = []
            total_chars = 0
            max_chars = max_tokens * 4

            for msg in reversed(recent):
                if total_chars + len(msg.content) <= max_chars:
                    filtered.insert(0, msg)
                    total_chars += len(msg.content)
                else:
                    break

            return filtered

        return recent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.updated_at = datetime.utcnow()


class MemoryStore:
    """
    In-memory storage for conversation sessions.
    Uses LRU eviction for memory management.
    """

    def __init__(self, max_sessions: int = 1000):
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, ConversationSession] = OrderedDict()

    def get(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session, updating access order."""
        if session_id in self._sessions:
            # Move to end (most recently accessed)
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        return None

    def create(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationSession:
        """Create a new session."""
        session_id = session_id or str(uuid.uuid4())

        # Evict oldest if at capacity
        while len(self._sessions) >= self.max_sessions:
            self._sessions.popitem(last=False)

        session = ConversationSession(session_id=session_id, metadata=metadata or {})
        self._sessions[session_id] = session
        return session

    def get_or_create(
        self, session_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationSession:
        """Get existing session or create new one."""
        existing = self.get(session_id)
        if existing:
            return existing
        return self.create(session_id, metadata)

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self, limit: int = 50) -> List[str]:
        """List session IDs (most recent first)."""
        return list(reversed(list(self._sessions.keys())))[:limit]

    @property
    def session_count(self) -> int:
        return len(self._sessions)


class MemoryService:
    """
    Main service for conversation memory management.

    Provides session-based memory with context window management
    for multi-turn conversations.
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        default_context_messages: int = 10,
        default_context_tokens: int = 4000,
    ):
        self.store = MemoryStore(max_sessions=max_sessions)
        self.default_context_messages = default_context_messages
        self.default_context_tokens = default_context_tokens
        self.is_initialized = True

        logger.info("MemoryService initialized")

    async def get_session(
        self, session_id: str, create_if_missing: bool = True
    ) -> Optional[ConversationSession]:
        """
        Get a conversation session.

        Args:
            session_id: Session identifier
            create_if_missing: Create new session if not found

        Returns:
            ConversationSession or None
        """
        if create_if_missing:
            return self.store.get_or_create(session_id)
        return self.store.get(session_id)

    async def add_user_message(
        self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add a user message to session."""
        session = self.store.get_or_create(session_id)
        return session.add_message("user", content, metadata)

    async def add_assistant_message(
        self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add an assistant message to session."""
        session = self.store.get_or_create(session_id)
        return session.add_message("assistant", content, metadata)

    async def add_system_message(self, session_id: str, content: str) -> Message:
        """Add a system message to session."""
        session = self.store.get_or_create(session_id)
        return session.add_message("system", content)

    async def get_context_messages(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM context.

        Returns messages in the format expected by chat models:
        [{"role": "user", "content": "..."}, ...]
        """
        session = self.store.get(session_id)
        if not session:
            return []

        messages = session.get_messages_for_context(
            max_messages=max_messages or self.default_context_messages,
            max_tokens=max_tokens or self.default_context_tokens,
        )

        return [{"role": m.role, "content": m.content} for m in messages]

    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get full conversation history."""
        session = self.store.get(session_id)
        if not session:
            return []

        return [m.to_dict() for m in session.messages]

    async def clear_session(self, session_id: str) -> bool:
        """Clear all messages in a session."""
        session = self.store.get(session_id)
        if session:
            session.clear()
            return True
        return False

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session entirely."""
        return self.store.delete(session_id)

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a session."""
        session = self.store.get(session_id)
        if not session:
            return {"exists": False}

        return {
            "exists": True,
            "session_id": session_id,
            "message_count": len(session.messages),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
        }

    async def list_sessions(self, limit: int = 50) -> List[str]:
        """List active session IDs."""
        return self.store.list_sessions(limit)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory service statistics."""
        return {
            "active_sessions": self.store.session_count,
            "max_sessions": self.store.max_sessions,
            "default_context_messages": self.default_context_messages,
            "default_context_tokens": self.default_context_tokens,
        }

    async def cleanup(self) -> None:
        """Clean up all sessions."""
        self.store._sessions.clear()
        logger.info("MemoryService cleaned up")
