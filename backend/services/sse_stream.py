"""
Server-Sent Events (SSE) Streaming Service.

Provides SSE support for streaming LLM responses to clients,
enabling real-time updates and reduced time-to-first-token.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


@dataclass
class SSEEvent:
    """Server-Sent Event data structure.
    
    Attributes:
        data: Event data (will be JSON serialized).
        event: Optional event type name.
        id: Optional event ID for client tracking.
        retry: Optional retry interval in milliseconds.
    """
    data: Any
    event: Optional[str] = None
    id: Optional[str] = None
    retry: Optional[int] = None
    
    def encode(self) -> str:
        """Encode event to SSE format.
        
        Returns:
            SSE-formatted string.
        """
        lines = []
        
        if self.event:
            lines.append(f"event: {self.event}")
        
        if self.id:
            lines.append(f"id: {self.id}")
        
        if self.retry:
            lines.append(f"retry: {self.retry}")
        
        # Serialize data to JSON if not a string
        if isinstance(self.data, str):
            data_str = self.data
        else:
            data_str = json.dumps(self.data, ensure_ascii=False)
        
        # Split data into multiple lines if needed
        for line in data_str.split("\n"):
            lines.append(f"data: {line}")
        
        return "\n".join(lines) + "\n\n"


class SSEStreamManager:
    """Manager for SSE streaming connections.
    
    Handles streaming LLM responses with proper SSE formatting,
    heartbeat keep-alive, and connection management.
    """
    
    def __init__(
        self,
        heartbeat_interval: float = 15.0,
        max_reconnect_time: int = 30000
    ):
        """Initialize SSE stream manager.
        
        Args:
            heartbeat_interval: Seconds between heartbeat events.
            max_reconnect_time: Client reconnection timeout in ms.
        """
        self.heartbeat_interval = heartbeat_interval
        self.max_reconnect_time = max_reconnect_time
        self._active_streams: Dict[str, bool] = {}
    
    async def stream_llm_response(
        self,
        response_generator: AsyncGenerator[str, None],
        stream_id: str,
        include_metadata: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response chunks as SSE events.
        
        Args:
            response_generator: Async generator yielding response chunks.
            stream_id: Unique identifier for this stream.
            include_metadata: Whether to include timing metadata.
            
        Yields:
            SSE-formatted strings.
        """
        self._active_streams[stream_id] = True
        start_time = time.perf_counter()
        chunk_count = 0
        total_content = ""
        
        try:
            # Send initial connection event
            yield SSEEvent(
                data={"status": "connected", "stream_id": stream_id},
                event="stream_start",
                retry=self.max_reconnect_time
            ).encode()
            
            # Stream response chunks
            async for chunk in response_generator:
                if not self._active_streams.get(stream_id, False):
                    break
                
                chunk_count += 1
                total_content += chunk
                
                event_data = {"content": chunk, "chunk_index": chunk_count}
                
                if include_metadata:
                    event_data["elapsed_ms"] = int(
                        (time.perf_counter() - start_time) * 1000
                    )
                
                yield SSEEvent(data=event_data, event="chunk").encode()
            
            # Send completion event
            completion_data = {
                "status": "completed",
                "total_chunks": chunk_count,
                "total_length": len(total_content),
            }
            
            if include_metadata:
                completion_data["total_time_ms"] = int(
                    (time.perf_counter() - start_time) * 1000
                )
            
            yield SSEEvent(data=completion_data, event="stream_end").encode()
            
        except Exception as e:
            logger.error(f"SSE stream error for {stream_id}: {e}")
            yield SSEEvent(
                data={"error": str(e), "status": "error"},
                event="error"
            ).encode()
        finally:
            self._active_streams.pop(stream_id, None)
    
    async def create_streaming_response(
        self,
        response_generator: AsyncGenerator[str, None],
        stream_id: str,
        request: Optional[Request] = None
    ) -> StreamingResponse:
        """Create a FastAPI StreamingResponse for SSE.
        
        Args:
            response_generator: Async generator yielding response chunks.
            stream_id: Unique identifier for this stream.
            request: Optional FastAPI request for disconnect detection.
            
        Returns:
            Configured StreamingResponse.
        """
        async def event_generator():
            async for event in self.stream_llm_response(
                response_generator, stream_id
            ):
                # Check if client disconnected
                if request and await request.is_disconnected():
                    logger.info(f"Client disconnected: {stream_id}")
                    self._active_streams[stream_id] = False
                    break
                
                yield event
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    
    def cancel_stream(self, stream_id: str) -> bool:
        """Cancel an active stream.
        
        Args:
            stream_id: Stream identifier to cancel.
            
        Returns:
            True if stream was cancelled, False if not found.
        """
        if stream_id in self._active_streams:
            self._active_streams[stream_id] = False
            return True
        return False
    
    @property
    def active_stream_count(self) -> int:
        """Get count of active streams."""
        return sum(1 for active in self._active_streams.values() if active)


# Global SSE manager instance
_sse_manager: Optional[SSEStreamManager] = None


def get_sse_manager() -> SSEStreamManager:
    """Get the global SSE manager instance.
    
    Returns:
        SSEStreamManager singleton.
    """
    global _sse_manager
    if _sse_manager is None:
        _sse_manager = SSEStreamManager()
    return _sse_manager


async def stream_chat_response(
    prompt: str,
    session_id: str,
    system_prompt: Optional[str] = None,
    request: Optional[Request] = None
) -> StreamingResponse:
    """Stream a chat response using SSE.
    
    Convenience function for streaming LLM responses in chat context.
    
    Args:
        prompt: User prompt.
        session_id: Chat session identifier.
        system_prompt: Optional system context.
        request: FastAPI request for disconnect detection.
        
    Returns:
        StreamingResponse with SSE events.
    """
    from backend.services.llm_providers import get_llm_provider
    
    provider = get_llm_provider()
    await provider.initialize()
    
    async def chunk_generator():
        async for chunk in provider.generate_stream(prompt, system_prompt):
            yield chunk.content
    
    manager = get_sse_manager()
    return await manager.create_streaming_response(
        chunk_generator(),
        stream_id=session_id,
        request=request
    )
