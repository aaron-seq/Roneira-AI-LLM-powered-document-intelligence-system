"""Comprehensive tests for WebSocket connection manager.

Provides unit tests for WebSocket connection management,
broadcasting, and error handling scenarios.
"""

import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from app.core.websocket_manager import WebSocketConnectionManager
from app.core.exceptions import ServiceUnavailableError


class TestWebSocketConnectionManager:
    """Test WebSocket connection manager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh WebSocketConnectionManager for each test."""
        return WebSocketConnectionManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket connection."""
        websocket = MagicMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.close = AsyncMock()
        websocket.client_state = WebSocketState.CONNECTED
        websocket.headers = {"user-agent": "test-client", "host": "localhost:8000"}
        websocket.query_params = {"token": "test-token"}
        return websocket
    
    @pytest.fixture
    def multiple_websockets(self):
        """Create multiple mock WebSocket connections."""
        websockets = []
        for i in range(3):
            ws = MagicMock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            ws.close = AsyncMock()
            ws.client_state = WebSocketState.CONNECTED
            ws.headers = {"user-agent": f"test-client-{i}"}
            ws.query_params = {"client_id": f"client-{i}"}
            websockets.append(ws)
        return websockets


class TestConnection(TestWebSocketConnectionManager):
    """Test WebSocket connection establishment."""
    
    @pytest.mark.asyncio
    async def test_connect_single_websocket(self, manager, mock_websocket):
        """Test connecting a single WebSocket."""
        document_id = "test-doc-123"
        
        await manager.connect(mock_websocket, document_id)
        
        # Verify WebSocket was accepted
        mock_websocket.accept.assert_called_once()
        
        # Verify connection was registered
        assert manager.get_connection_count() == 1
        assert await manager.get_document_connections(document_id) == 1
        assert document_id in manager.get_document_list()
        
        # Verify initial message was sent
        mock_websocket.send_text.assert_called_once()
        sent_message = mock_websocket.send_text.call_args[0][0]
        message_data = json.loads(sent_message)
        assert message_data["type"] == "connection_established"
        assert message_data["document_id"] == document_id
    
    @pytest.mark.asyncio
    async def test_connect_multiple_websockets_same_document(self, manager, multiple_websockets):
        """Test connecting multiple WebSockets to the same document."""
        document_id = "test-doc-456"
        
        for websocket in multiple_websockets:
            await manager.connect(websocket, document_id)
        
        # Verify all connections were registered
        assert manager.get_connection_count() == 3
        assert await manager.get_document_connections(document_id) == 3
        
        # Verify all WebSockets were accepted
        for websocket in multiple_websockets:
            websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_multiple_documents(self, manager, multiple_websockets):
        """Test connecting WebSockets to different documents."""
        document_ids = ["doc-1", "doc-2", "doc-3"]
        
        for websocket, doc_id in zip(multiple_websockets, document_ids):
            await manager.connect(websocket, doc_id)
        
        # Verify connections were distributed correctly
        assert manager.get_connection_count() == 3
        for doc_id in document_ids:
            assert await manager.get_document_connections(doc_id) == 1
        
        assert set(manager.get_document_list()) == set(document_ids)
    
    @pytest.mark.asyncio
    async def test_connect_failure_cleanup(self, manager, mock_websocket):
        """Test cleanup when WebSocket connection fails."""
        document_id = "test-doc-789"
        
        # Mock WebSocket accept to raise an exception
        mock_websocket.accept.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            await manager.connect(mock_websocket, document_id)
        
        # Verify no connections were registered
        assert manager.get_connection_count() == 0
        assert await manager.get_document_connections(document_id) == 0


class TestDisconnection(TestWebSocketConnectionManager):
    """Test WebSocket disconnection and cleanup."""
    
    @pytest.mark.asyncio
    async def test_disconnect_single_websocket(self, manager, mock_websocket):
        """Test disconnecting a single WebSocket."""
        document_id = "test-doc-disconnect"
        
        # Connect first
        await manager.connect(mock_websocket, document_id)
        assert manager.get_connection_count() == 1
        
        # Disconnect
        await manager.disconnect(mock_websocket, document_id)
        
        # Verify cleanup
        assert manager.get_connection_count() == 0
        assert await manager.get_document_connections(document_id) == 0
        assert document_id not in manager.get_document_list()
    
    @pytest.mark.asyncio
    async def test_disconnect_one_of_multiple(self, manager, multiple_websockets):
        """Test disconnecting one WebSocket when multiple are connected."""
        document_id = "test-doc-multi"
        
        # Connect all WebSockets
        for websocket in multiple_websockets:
            await manager.connect(websocket, document_id)
        
        assert manager.get_connection_count() == 3
        
        # Disconnect one
        await manager.disconnect(multiple_websockets[0], document_id)
        
        # Verify partial cleanup
        assert manager.get_connection_count() == 2
        assert await manager.get_document_connections(document_id) == 2
        assert document_id in manager.get_document_list()
    
    @pytest.mark.asyncio
    async def test_cleanup_all_connections(self, manager, multiple_websockets):
        """Test cleaning up all connections."""
        document_ids = ["doc-1", "doc-2", "doc-3"]
        
        # Connect WebSockets to different documents
        for websocket, doc_id in zip(multiple_websockets, document_ids):
            await manager.connect(websocket, doc_id)
        
        assert manager.get_connection_count() == 3
        
        # Cleanup all
        await manager.cleanup_all_connections()
        
        # Verify complete cleanup
        assert manager.get_connection_count() == 0
        assert len(manager.get_document_list()) == 0
        
        # Verify all WebSockets were closed
        for websocket in multiple_websockets:
            websocket.close.assert_called_once()


class TestBroadcasting(TestWebSocketConnectionManager):
    """Test WebSocket message broadcasting functionality."""
    
    @pytest.mark.asyncio
    async def test_broadcast_to_document_single_connection(self, manager, mock_websocket):
        """Test broadcasting to a document with one connection."""
        document_id = "test-doc-broadcast"
        message = {"status": "processing", "progress": 50}
        
        await manager.connect(mock_websocket, document_id)
        
        # Clear the connection establishment message
        mock_websocket.send_text.reset_mock()
        
        # Broadcast message
        sent_count = await manager.broadcast_to_document(document_id, message)
        
        assert sent_count == 1
        mock_websocket.send_text.assert_called_once()
        
        # Verify message content
        sent_message = mock_websocket.send_text.call_args[0][0]
        message_data = json.loads(sent_message)
        assert message_data["status"] == "processing"
        assert message_data["progress"] == 50
        assert "timestamp" in message_data
        assert "broadcast_id" in message_data
    
    @pytest.mark.asyncio
    async def test_broadcast_to_document_multiple_connections(self, manager, multiple_websockets):
        """Test broadcasting to a document with multiple connections."""
        document_id = "test-doc-multi-broadcast"
        message = {"status": "completed", "result": "success"}
        
        # Connect all WebSockets to the same document
        for websocket in multiple_websockets:
            await manager.connect(websocket, document_id)
            websocket.send_text.reset_mock()  # Clear connection messages
        
        # Broadcast message
        sent_count = await manager.broadcast_to_document(document_id, message)
        
        assert sent_count == 3
        
        # Verify all WebSockets received the message
        for websocket in multiple_websockets:
            websocket.send_text.assert_called_once()
            sent_message = websocket.send_text.call_args[0][0]
            message_data = json.loads(sent_message)
            assert message_data["status"] == "completed"
            assert message_data["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_broadcast_to_nonexistent_document(self, manager):
        """Test broadcasting to a document with no connections."""
        document_id = "nonexistent-doc"
        message = {"status": "test"}
        
        sent_count = await manager.broadcast_to_document(document_id, message)
        
        assert sent_count == 0
    
    @pytest.mark.asyncio
    async def test_broadcast_to_all_connections(self, manager, multiple_websockets):
        """Test broadcasting to all connections across multiple documents."""
        document_ids = ["doc-1", "doc-2", "doc-3"]
        message = {"type": "system_announcement", "message": "Maintenance window"}
        
        # Connect WebSockets to different documents
        for websocket, doc_id in zip(multiple_websockets, document_ids):
            await manager.connect(websocket, doc_id)
            websocket.send_text.reset_mock()  # Clear connection messages
        
        # Broadcast to all
        sent_count = await manager.broadcast_to_all(message)
        
        assert sent_count == 3
        
        # Verify all WebSockets received the message
        for websocket in multiple_websockets:
            websocket.send_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broadcast_with_failed_connections(self, manager, multiple_websockets):
        """Test broadcasting with some failed connections."""
        document_id = "test-doc-failed"
        message = {"status": "test"}
        
        # Connect all WebSockets
        for websocket in multiple_websockets:
            await manager.connect(websocket, document_id)
            websocket.send_text.reset_mock()
        
        # Make one WebSocket fail
        multiple_websockets[1].send_text.side_effect = Exception("Connection lost")
        
        # Broadcast message
        sent_count = await manager.broadcast_to_document(document_id, message)
        
        # Should succeed for 2 out of 3 connections
        assert sent_count == 2
        
        # Failed connection should be cleaned up
        assert await manager.get_document_connections(document_id) == 2


class TestConnectionMetadata(TestWebSocketConnectionManager):
    """Test connection metadata and statistics."""
    
    @pytest.mark.asyncio
    async def test_connection_statistics(self, manager, multiple_websockets):
        """Test getting connection statistics."""
        document_ids = ["doc-stats-1", "doc-stats-2"]
        
        # Connect WebSockets
        await manager.connect(multiple_websockets[0], document_ids[0])
        await manager.connect(multiple_websockets[1], document_ids[0])
        await manager.connect(multiple_websockets[2], document_ids[1])
        
        # Get statistics
        stats = await manager.get_connection_stats()
        
        assert stats["total_connections"] == 3
        assert stats["documents_with_connections"] == 2
        assert "generated_at" in stats
        assert "connection_details" in stats
        
        # Verify document-specific stats
        doc1_stats = stats["connection_details"][document_ids[0]]
        assert doc1_stats["active_connections"] == 2
        
        doc2_stats = stats["connection_details"][document_ids[1]]
        assert doc2_stats["active_connections"] == 1
    
    @pytest.mark.asyncio
    async def test_connection_metadata_tracking(self, manager, mock_websocket):
        """Test that connection metadata is properly tracked."""
        document_id = "test-doc-metadata"
        
        await manager.connect(mock_websocket, document_id)
        
        # Send a few messages to update metadata
        mock_websocket.send_text.reset_mock()
        await manager.broadcast_to_document(document_id, {"test": "message1"})
        await manager.broadcast_to_document(document_id, {"test": "message2"})
        
        stats = await manager.get_connection_stats()
        connection_metadata = stats["connection_details"][document_id]["connection_metadata"][0]
        
        assert connection_metadata["document_id"] == document_id
        assert connection_metadata["messages_sent"] == 2
        assert "connected_at" in connection_metadata
        assert "client_info" in connection_metadata


class TestErrorHandling(TestWebSocketConnectionManager):
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_dead_connection_cleanup(self, manager, mock_websocket):
        """Test cleanup of dead connections."""
        document_id = "test-doc-dead"
        
        await manager.connect(mock_websocket, document_id)
        assert manager.get_connection_count() == 1
        
        # Simulate dead connection
        mock_websocket.client_state = WebSocketState.DISCONNECTED
        
        # Getting document connections should clean up dead connections
        active_count = await manager.get_document_connections(document_id)
        
        assert active_count == 0
        assert manager.get_connection_count() == 0
    
    @pytest.mark.asyncio
    async def test_websocket_close_error_handling(self, manager, mock_websocket):
        """Test handling of WebSocket close errors."""
        document_id = "test-doc-close-error"
        
        await manager.connect(mock_websocket, document_id)
        
        # Make close raise an exception
        mock_websocket.close.side_effect = Exception("Already closed")
        
        # Should not raise exception
        await manager.disconnect(mock_websocket, document_id)
        
        # Connection should still be cleaned up
        assert manager.get_connection_count() == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, manager, multiple_websockets):
        """Test concurrent connection operations."""
        document_id = "test-doc-concurrent"
        
        # Run concurrent connections
        tasks = []
        for websocket in multiple_websockets:
            task = asyncio.create_task(manager.connect(websocket, document_id))
            tasks.append(task)
        
        # Wait for all connections to complete
        await asyncio.gather(*tasks)
        
        # All connections should be registered
        assert manager.get_connection_count() == 3
        assert await manager.get_document_connections(document_id) == 3


if __name__ == "__main__":
    pytest.main([__file__])
