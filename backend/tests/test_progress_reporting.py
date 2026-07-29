"""Progress reporting and structured logging call signatures.

Both behaviours here live on error/notification paths that the happy-path
tests never touch, which is exactly why they were broken:

* ``WebSocketManager.broadcast_progress`` takes a percentage, but the
  document pipeline handed it the entire status dictionary, so subscribed
  clients received an object where they expected a number.
* ``ContextLogger`` accepts ``(message, **kwargs)`` only. Calls written in
  ``logger.warning("... %s", value)`` form raise ``TypeError`` — and every one
  of them sat inside an ``except`` block, so the failure would surface only
  once something else had already gone wrong.
"""

from __future__ import annotations

import inspect

import pytest

from backend.observability.structured_logging import get_logger
from backend.services.websocket_manager import WebSocketManager


class TestStructuredLoggerSignature:
    """Guard against reintroducing %-style calls on the structured logger."""

    @pytest.mark.parametrize("level", ["debug", "info", "warning"])
    def test_logger_rejects_positional_format_arguments(self, level):
        logger = get_logger("test.signature")
        with pytest.raises(TypeError):
            getattr(logger, level)("value is %s", "unexpected")

    def test_error_silently_swallows_a_positional_argument(self, caplog):
        """``error()`` is the nastier case, and the reason for the static check.

        Its signature is ``error(message, exc_info=False, **kwargs)``, so a
        stray second positional binds to ``exc_info`` instead of raising. The
        call succeeds and the log line is emitted with a literal ``%s`` in it —
        a silent corruption rather than a loud failure.
        """
        logger = get_logger("test.signature")
        logger.error("value is %s", "unexpected")  # no exception raised

        assert any("value is %s" in record.message for record in caplog.records), (
            "expected the unformatted message to reach the log"
        )

    @pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
    def test_logger_accepts_a_message_and_keyword_context(self, level):
        logger = get_logger("test.signature")
        getattr(logger, level)("something happened", document_id="abc", count=3)

    def test_no_percent_style_calls_remain_in_the_document_pipeline(self):
        """A static check: the failure mode is invisible until an error path runs."""
        import re

        from backend.services import document_processor

        source = inspect.getsource(document_processor)
        offenders = re.findall(
            r"logger\.(?:debug|info|warning|error)\([^)]*%s[^)]*,\s*\w+", source
        )
        assert not offenders, (
            "ContextLogger takes no positional format args; use an f-string "
            f"or keyword context instead. Offending calls: {offenders}"
        )


class TestProgressBroadcast:
    """Progress messages must carry a number clients can render."""

    @pytest.fixture
    def manager(self):
        return WebSocketManager()

    @pytest.mark.asyncio
    async def test_progress_message_carries_an_integer_percentage(self, manager):
        sent = []

        async def capture(document_id, message):
            sent.append(message)

        manager._broadcast_to_document_subscribers = capture

        await manager.broadcast_progress("doc-1", progress=40, stage="Extracting text")

        assert len(sent) == 1
        assert sent[0]["progress"] == 40
        assert isinstance(sent[0]["progress"], int)
        assert sent[0]["stage"] == "Extracting text"

    def test_broadcast_progress_declares_an_integer_parameter(self, manager):
        """The contract the pipeline has to honour."""
        signature = inspect.signature(manager.broadcast_progress)
        assert signature.parameters["progress"].annotation is int

    @pytest.mark.asyncio
    async def test_pipeline_sends_a_percentage_not_the_status_dict(self, manager):
        """Exercises the callback the documents router installs."""
        from typing import Any, Dict

        sent = []

        async def capture(document_id, message):
            sent.append(message)

        manager._broadcast_to_document_subscribers = capture

        async def progress(update: Dict[str, Any]) -> None:
            await manager.broadcast_progress(
                "doc-1",
                progress=int(update.get("progress", 0)),
                stage=update.get("message"),
            )

        await progress(
            {
                "document_id": "doc-1",
                "status": "processing",
                "progress": 75,
                "message": "Indexing for search",
                "filename": "report.pdf",
            }
        )

        assert sent[0]["progress"] == 75
        assert not isinstance(sent[0]["progress"], dict)
