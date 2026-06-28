"""Structured (JSON) logging primitives shared across the app.

A request ID is stored in a :class:`contextvars.ContextVar` so it is available to
every log record emitted while handling a request — including deep inside the
predictor — without threading it through call signatures.
"""

import datetime as dt
import json
import logging
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"request_id", "message", "asctime", "taskName"}


class RequestIDFilter(logging.Filter):
    """Attach the current request ID to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JSONFormatter(logging.Formatter):
    """Render a log record as a single JSON line.

    Any non-reserved keyword passed via ``logger.info(..., extra={...})`` is
    promoted to a top-level field, so structured context (latency, status,
    category) stays queryable.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)
