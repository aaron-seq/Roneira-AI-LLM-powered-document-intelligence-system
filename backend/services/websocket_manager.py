"""WebSocket connection manager for real-time updates.

Provides WebSocket connection management for broadcasting
document processing progress and status updates to clients.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time client updates.

    Tracks active connections and provides methods for broadcasting
    messages to individual clients or all connected clients.
    """

    def __init__(self):
        # Map of client_id to WebSocket connection
        self._active_connections: Dict[str, WebSocket] = {}
        # Map of document_id to list of subscribed client_ids
        self._document_subscriptions: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to register.
            client_id: Unique identifier for the client.
        """
        await websocket.accept()
        self._active_connections[client_id] = websocket
        logger.info(f"WebSocket client connected: {client_id}")

        # Send connection confirmation
        await self._send_message(
            websocket,
            {
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection.

        Args:
            client_id: Client identifier to disconnect.
        """
        if client_id in self._active_connections:
            del self._active_connections[client_id]

            # Remove from all document subscriptions
            for doc_id in list(self._document_subscriptions.keys()):
                if client_id in self._document_subscriptions[doc_id]:
                    self._document_subscriptions[doc_id].remove(client_id)
                    if not self._document_subscriptions[doc_id]:
                        del self._document_subscriptions[doc_id]

            logger.info(f"WebSocket client disconnected: {client_id}")

    async def subscribe_to_document(self, client_id: str, document_id: str) -> None:
        """Subscribe a client to document updates.

        Args:
            client_id: Client to subscribe.
            document_id: Document to subscribe to.
        """
        if document_id not in self._document_subscriptions:
            self._document_subscriptions[document_id] = []

        if client_id not in self._document_subscriptions[document_id]:
            self._document_subscriptions[document_id].append(client_id)
            logger.debug(f"Client {client_id} subscribed to document {document_id}")

    async def broadcast_progress(
        self, document_id: str, progress: int, stage: Optional[str] = None
    ) -> None:
        """Broadcast processing progress to subscribed clients.

        Args:
            document_id: Document being processed.
            progress: Progress percentage (0-100).
            stage: Optional processing stage description.
        """
        message = {
            "type": "progress",
            "document_id": document_id,
            "progress": progress,
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self._broadcast_to_document_subscribers(document_id, message)

    async def broadcast_status(
        self, document_id: str, status: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast status update to subscribed clients.

        Args:
            document_id: Document identifier.
            status: New status value.
            data: Optional additional data to include.
        """
        message = {
            "type": "status",
            "document_id": document_id,
            "status": status,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self._broadcast_to_document_subscribers(document_id, message)

    async def broadcast_error(
        self, document_id: str, error: str, error_code: Optional[str] = None
    ) -> None:
        """Broadcast error message to subscribed clients.

        Args:
            document_id: Document that encountered an error.
            error: Error message.
            error_code: Optional error code.
        """
        message = {
            "type": "error",
            "document_id": document_id,
            "error": error,
            "error_code": error_code,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self._broadcast_to_document_subscribers(document_id, message)
        logger.error(f"Document {document_id} error broadcast: {error}")

    async def handle_client_message(self, client_id: str, message: str) -> None:
        """Handle incoming message from a client.

        Args:
            client_id: Client that sent the message.
            message: Raw message string.
        """
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")

            if message_type == "subscribe":
                document_id = data.get("document_id")
                if document_id:
                    await self.subscribe_to_document(client_id, document_id)
                    await self._send_to_client(
                        client_id, {"type": "subscribed", "document_id": document_id}
                    )

            elif message_type == "ping":
                await self._send_to_client(
                    client_id,
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                )

            else:
                logger.debug(f"Unknown message type from {client_id}: {message_type}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client {client_id}: {message}")

    async def _broadcast_to_document_subscribers(
        self, document_id: str, message: Dict[str, Any]
    ) -> None:
        """Send message to all clients subscribed to a document.

        Args:
            document_id: Document identifier.
            message: Message to broadcast.
        """
        subscribers = self._document_subscriptions.get(document_id, [])

        for client_id in subscribers:
            await self._send_to_client(client_id, message)

    async def _send_to_client(self, client_id: str, message: Dict[str, Any]) -> None:
        """Send message to a specific client.

        Args:
            client_id: Target client identifier.
            message: Message to send.
        """
        websocket = self._active_connections.get(client_id)
        if websocket:
            await self._send_message(websocket, message)

    async def _send_message(
        self, websocket: WebSocket, message: Dict[str, Any]
    ) -> None:
        """Send a message through a WebSocket connection.

        Args:
            websocket: WebSocket connection.
            message: Message dictionary to send as JSON.
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")

    @property
    def active_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self._active_connections)

    def get_document_subscriber_count(self, document_id: str) -> int:
        """Get the number of subscribers for a document.

        Args:
            document_id: Document to check.

        Returns:
            Number of subscribed clients.
        """
        return len(self._document_subscriptions.get(document_id, []))
