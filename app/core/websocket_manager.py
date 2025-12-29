"""WebSocket connection manager for real-time document processing updates.

Provides centralized WebSocket connection management with document-specific
broadcasting capabilities for real-time status updates.
"""

import asyncio
import json
import logging
from typing import Dict, Set, Any, Optional
from datetime import datetime

from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from app.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """Manages WebSocket connections for real-time document updates.

    Provides document-specific connection grouping and broadcasting
    capabilities with proper connection lifecycle management.
    """

    def __init__(self):
        """Initialize the WebSocket connection manager."""
        # Document ID -> Set of WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        # WebSocket -> Document ID mapping for cleanup
        self._websocket_to_document: Dict[WebSocket, str] = {}
        # Connection metadata for monitoring
        self._connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info("WebSocket connection manager initialized")

    async def connect(self, websocket: WebSocket, document_id: str) -> None:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to register
            document_id: The document ID to associate with this connection

        Raises:
            Exception: If connection acceptance fails
        """
        try:
            # Accept the WebSocket connection
            await websocket.accept()

            async with self._lock:
                # Initialize document group if needed
                if document_id not in self._connections:
                    self._connections[document_id] = set()

                # Add connection to document group
                self._connections[document_id].add(websocket)
                self._websocket_to_document[websocket] = document_id

                # Store connection metadata
                self._connection_metadata[websocket] = {
                    "document_id": document_id,
                    "connected_at": datetime.utcnow(),
                    "messages_sent": 0,
                    "client_info": {
                        "headers": dict(websocket.headers),
                        "query_params": dict(websocket.query_params),
                    },
                }

            logger.info(
                f"WebSocket connected for document {document_id}. "
                f"Total connections: {self.get_connection_count()}"
            )

            # Send initial connection confirmation
            await self._send_to_websocket(
                websocket,
                {
                    "type": "connection_established",
                    "document_id": document_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Real-time updates enabled",
                },
            )

        except Exception as e:
            logger.error(f"Failed to connect WebSocket for document {document_id}: {e}")
            await self._cleanup_websocket(websocket)
            raise

    async def disconnect(self, websocket: WebSocket, document_id: str) -> None:
        """Disconnect and clean up a WebSocket connection.

        Args:
            websocket: The WebSocket connection to disconnect
            document_id: The document ID associated with the connection
        """
        await self._cleanup_websocket(websocket)
        logger.info(
            f"WebSocket disconnected for document {document_id}. "
            f"Remaining connections: {self.get_connection_count()}"
        )

    async def broadcast_to_document(
        self, document_id: str, message: Dict[str, Any]
    ) -> int:
        """Broadcast a message to all connections for a specific document.

        Args:
            document_id: The document ID to broadcast to
            message: The message data to send

        Returns:
            int: Number of connections the message was sent to

        Raises:
            ServiceUnavailableError: If no connections exist for the document
        """
        if document_id not in self._connections:
            logger.warning(f"No connections found for document {document_id}")
            return 0

        connections = self._connections[document_id].copy()
        if not connections:
            logger.warning(f"No active connections for document {document_id}")
            return 0

        # Add metadata to message
        enhanced_message = {
            **message,
            "timestamp": datetime.utcnow().isoformat(),
            "broadcast_id": f"{document_id}_{datetime.utcnow().timestamp()}",
        }

        successful_sends = 0
        failed_connections = []

        # Send to all connections for this document
        for websocket in connections:
            try:
                await self._send_to_websocket(websocket, enhanced_message)
                successful_sends += 1

                # Update metadata
                if websocket in self._connection_metadata:
                    self._connection_metadata[websocket]["messages_sent"] += 1

            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                failed_connections.append(websocket)

        # Clean up failed connections
        for websocket in failed_connections:
            await self._cleanup_websocket(websocket)

        logger.info(
            f"Broadcast to document {document_id}: {successful_sends} successful, "
            f"{len(failed_connections)} failed"
        )

        return successful_sends

    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast a message to all connected WebSocket clients.

        Args:
            message: The message data to send

        Returns:
            int: Total number of connections the message was sent to
        """
        total_sent = 0

        for document_id in list(self._connections.keys()):
            sent_count = await self.broadcast_to_document(document_id, message)
            total_sent += sent_count

        logger.info(f"Global broadcast sent to {total_sent} connections")
        return total_sent

    async def get_document_connections(self, document_id: str) -> int:
        """Get the number of active connections for a document.

        Args:
            document_id: The document ID to check

        Returns:
            int: Number of active connections for the document
        """
        if document_id not in self._connections:
            return 0

        # Clean up any dead connections while counting
        active_connections = set()
        for websocket in self._connections[document_id].copy():
            if websocket.client_state == WebSocketState.CONNECTED:
                active_connections.add(websocket)
            else:
                await self._cleanup_websocket(websocket)

        self._connections[document_id] = active_connections
        return len(active_connections)

    def get_connection_count(self) -> int:
        """Get the total number of active WebSocket connections.

        Returns:
            int: Total number of active connections across all documents
        """
        return sum(len(connections) for connections in self._connections.values())

    def get_document_list(self) -> list[str]:
        """Get a list of document IDs with active connections.

        Returns:
            List of document IDs that have at least one active connection
        """
        return [
            doc_id
            for doc_id, connections in self._connections.items()
            if len(connections) > 0
        ]

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics.

        Returns:
            Dictionary containing connection statistics and metadata
        """
        stats = {
            "total_connections": self.get_connection_count(),
            "documents_with_connections": len(self.get_document_list()),
            "connection_details": {},
            "generated_at": datetime.utcnow().isoformat(),
        }

        for document_id in self._connections:
            active_count = await self.get_document_connections(document_id)
            if active_count > 0:
                stats["connection_details"][document_id] = {
                    "active_connections": active_count,
                    "connection_metadata": [],
                }

                # Add metadata for connections to this document
                for websocket in self._connections[document_id]:
                    if websocket in self._connection_metadata:
                        metadata = self._connection_metadata[websocket].copy()
                        # Remove potentially sensitive client info from stats
                        metadata["client_info"] = {
                            "headers_count": len(metadata["client_info"]["headers"]),
                            "has_query_params": len(
                                metadata["client_info"]["query_params"]
                            )
                            > 0,
                        }
                        stats["connection_details"][document_id][
                            "connection_metadata"
                        ].append(metadata)

        return stats

    async def cleanup_all_connections(self) -> None:
        """Clean up all WebSocket connections.

        Useful for application shutdown or testing cleanup.
        """
        logger.info("Cleaning up all WebSocket connections")

        # Get all websockets to avoid modifying dict during iteration
        all_websockets = []
        for connections in self._connections.values():
            all_websockets.extend(list(connections))

        # Close all connections
        for websocket in all_websockets:
            await self._cleanup_websocket(websocket)

        # Clear all data structures
        async with self._lock:
            self._connections.clear()
            self._websocket_to_document.clear()
            self._connection_metadata.clear()

        logger.info("All WebSocket connections cleaned up")

    async def _send_to_websocket(
        self, websocket: WebSocket, message: Dict[str, Any]
    ) -> None:
        """Send a message to a specific WebSocket connection.

        Args:
            websocket: The WebSocket connection to send to
            message: The message data to send

        Raises:
            Exception: If sending fails (connection closed, etc.)
        """
        if websocket.client_state != WebSocketState.CONNECTED:
            raise Exception("WebSocket is not connected")

        try:
            message_json = json.dumps(message, default=str)
            await websocket.send_text(message_json)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            raise

    async def _cleanup_websocket(self, websocket: WebSocket) -> None:
        """Clean up a WebSocket connection and its metadata.

        Args:
            websocket: The WebSocket connection to clean up
        """
        async with self._lock:
            # Remove from document group
            if websocket in self._websocket_to_document:
                document_id = self._websocket_to_document[websocket]
                if document_id in self._connections:
                    self._connections[document_id].discard(websocket)

                    # Remove empty document groups
                    if len(self._connections[document_id]) == 0:
                        del self._connections[document_id]

                del self._websocket_to_document[websocket]

            # Remove metadata
            if websocket in self._connection_metadata:
                del self._connection_metadata[websocket]

        # Close connection if still open
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except Exception as e:
            logger.debug(f"Error closing WebSocket (may already be closed): {e}")
