"""Logging configuration.

One logging setup for the whole process. Previously ``utils/telemetry`` and
``observability/structured_logging`` each configured logging independently and
whichever ran last won, so the correlation-ID-aware formatter was installed but
never used.

Handlers are attached to the root logger, and the structured formatter from
``observability`` supplies correlation IDs from the request context.
"""

from __future__ import annotations

import logging
import sys

from backend.observability.structured_logging import StructuredFormatter

#: Libraries that log at INFO on every request and drown out our own output.
NOISY_LOGGERS = {
    "uvicorn.access": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "chromadb": logging.WARNING,
    "urllib3": logging.WARNING,
}


def setup_telemetry(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure process-wide logging.

    Args:
        level: Root log level name.
        json_logs: Emit one JSON object per line. Enabled in production so a
            log aggregator can index correlation IDs and status codes;
            human-readable text is used locally.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Replace existing handlers so repeated calls (tests, reload) do not
    # produce duplicated log lines.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    if json_logs:
        handler.setFormatter(StructuredFormatter(service_name="roneira"))
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.addHandler(handler)

    for name, noisy_level in NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(noisy_level)

    logging.getLogger("backend").info(
        "Logging configured (level=%s, format=%s)",
        level.upper(),
        "json" if json_logs else "text",
    )
