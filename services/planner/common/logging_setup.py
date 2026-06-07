"""
services/planner/common/logging_setup.py

Shared logging configuration for the planner service.
Extracted from server.py (SCRUM-341 PR review) so that:
  - server.py and test_planner_hardening.py import the same module
  - Tests exercise the actual formatter, not an inline copy
  - _JsonFormatter can be updated in one place

Usage:
    from common.logging_setup import build_logger, _POLICY_CONFIG_PATH

Log format (one JSON object per line):
    {"time": "...", "level": "INFO", "service": "planner", "msg": "...", ...extra_fields}

Structured fields are passed via the logging extra= parameter and merged
into the top-level JSON payload by _JsonFormatter. This avoids double-encoding:

    # CORRECT
    log.info("covariance fetched", extra={"event": "covariance_fetched", "pair": "226/35929"})
    # emits: {"time":..., "level":"INFO", "service":"planner", "msg":"covariance fetched",
    #          "event":"covariance_fetched", "pair":"226/35929"}

    # WRONG (double-encoded -- do not use)
    log.info(json.dumps({"event": "covariance_fetched", "pair": "226/35929"}))
    # emits: {"time":..., "level":"INFO", "service":"planner",
    #          "msg": "{\"event\": \"covariance_fetched\", ...}"}
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SERVICE_NAME = "planner"

# Default config path -- baked into image via COPY config/ /app/config/
# Can be overridden via env var for future operator policy hot-swap support.
_POLICY_CONFIG_PATH = Path(
    os.environ.get("OPERATOR_POLICY_PATH", "/app/config/operator_policy_leo.yaml")
)


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line matching the documented service format.

    Base fields (always present):
        time    -- UTC ISO-8601 with Z suffix
        level   -- log level name (INFO, WARNING, ERROR, ...)
        service -- SERVICE_NAME constant ("planner")
        msg     -- the log message string

    Structured fields passed via extra= are merged into the top-level payload:
        log.info("cdm fetched", extra={"event": "covariance_fetched", "pair": "226/35929"})
        → {"time":..., "level":"INFO", "service":"planner", "msg":"cdm fetched",
            "event":"covariance_fetched", "pair":"226/35929"}

    Exception info (when exc_info=True) is added under the "exc" key.
    """

    # Reserved LogRecord attributes that should not be forwarded as extra fields.
    _RESERVED = frozenset({
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "time":    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level":   record.levelname,
            "service": SERVICE_NAME,
            "msg":     record.getMessage(),
        }

        # Merge extra fields into top-level payload.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def build_logger(name: str = SERVICE_NAME) -> logging.Logger:
    """Build and return a logger using _JsonFormatter.

    Idempotent -- safe to call multiple times with the same name.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
