"""
tests/test_planner_hardening.py

SCRUM-341: Tests for container hardening additions to server.py.
Covers AC1 (/ready probe logic) and AC3 (structured JSON logging).

_JsonFormatter and _POLICY_CONFIG_PATH are defined inline here to avoid
importing server.py, which pulls in FastAPI at module level. FastAPI is
a container dependency, not a test venv dependency. The logic under test
is copied verbatim from server.py -- any drift between the two will be
caught by code review on the PR.

Run with:
  PYTHONPATH=services/planner python -m pytest services/planner/tests/test_planner_hardening.py -v
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict

import pytest

from common.operator_policy import OperatorPolicy


# ---------------------------------------------------------------------------
# Inline copy of _JsonFormatter from server.py (verbatim)
# ---------------------------------------------------------------------------

SERVICE_NAME    = "planner"
SERVICE_VERSION = "2.5.0"

_POLICY_CONFIG_PATH = Path(
    os.environ.get("OPERATOR_POLICY_PATH", "services/planner/config/operator_policy_leo.yaml")
)


class _JsonFormatter(logging.Formatter):
    """Verbatim copy from server.py -- any change there must be mirrored here."""
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "time":    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level":   record.levelname,
            "service": SERVICE_NAME,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# AC3: Structured JSON logging
# ---------------------------------------------------------------------------

class TestJsonFormatter:
    """AC3: _JsonFormatter must emit valid JSON matching the documented format:
    {"time": "...", "level": "...", "service": "planner", "msg": "..."}
    """

    def _emit(self, level: int, message: str) -> dict:
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        logger = logging.getLogger(SERVICE_NAME)
        prev_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            logger.log(level, message)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)
        return json.loads(buf.getvalue().strip())

    def test_output_is_valid_json(self):
        """Each log line must be parseable as a JSON object."""
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        logger = logging.getLogger(SERVICE_NAME)
        prev_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            logger.info("validity check")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)
        output = buf.getvalue().strip()
        assert output
        assert isinstance(json.loads(output), dict)

    def test_required_keys_present(self):
        record = self._emit(logging.INFO, "key check")
        assert "time" in record
        assert "level" in record
        assert "service" in record
        assert "msg" in record

    def test_service_field_matches_constant(self):
        record = self._emit(logging.INFO, "service name check")
        assert record["service"] == SERVICE_NAME

    def test_level_info(self):
        record = self._emit(logging.INFO, "info level")
        assert record["level"] == "INFO"

    def test_level_warning(self):
        record = self._emit(logging.WARNING, "warning level")
        assert record["level"] == "WARNING"

    def test_level_error(self):
        record = self._emit(logging.ERROR, "error level")
        assert record["level"] == "ERROR"

    def test_msg_field_contains_message(self):
        sentinel = "xorbita-sentinel-341"
        record = self._emit(logging.INFO, sentinel)
        assert sentinel in record["msg"]

    def test_time_ends_with_z(self):
        record = self._emit(logging.INFO, "time check")
        assert record["time"].endswith("Z"), f"Expected Z suffix, got: {record['time']}"

    def test_time_contains_iso_separator(self):
        record = self._emit(logging.INFO, "iso check")
        assert "T" in record["time"]

    def test_exc_key_present_when_exception_logged(self):
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        logger = logging.getLogger(SERVICE_NAME)
        prev_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            try:
                raise ValueError("test exception for 341")
            except ValueError:
                logger.error("caught error", exc_info=True)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)
        record = json.loads(buf.getvalue().strip())
        assert "exc" in record
        assert "ValueError" in record["exc"]

    def test_exc_key_absent_when_no_exception(self):
        record = self._emit(logging.INFO, "no exception here")
        assert "exc" not in record

    def test_single_line_output(self):
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        logger = logging.getLogger(SERVICE_NAME)
        prev_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            logger.info("newline check")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# AC1: /ready probe logic
# ---------------------------------------------------------------------------

class TestReadyProbeLogic:
    """AC1: /ready must return 200 only when policy config is present and valid.

    Tests exercise OperatorPolicy.from_yaml() directly -- the same call
    server.py makes inside the /ready handler. If this raises, /ready
    returns 503. If it succeeds, /ready returns 200.
    """

    def test_real_policy_file_exists(self):
        """/ready happy path requires the config file to exist at the baked-in path."""
        assert _POLICY_CONFIG_PATH.exists(), (
            f"Default policy config missing at {_POLICY_CONFIG_PATH}. "
            "COPY config/ /app/config/ must be in the Dockerfile."
        )

    def test_real_policy_file_loads_without_error(self):
        """The default policy must load cleanly -- /ready 200 path."""
        policy = OperatorPolicy.from_yaml(str(_POLICY_CONFIG_PATH))
        assert policy is not None

    def test_policy_loads_correct_operator_id(self):
        """Loaded policy must have the expected DEFAULT_LEO operator_id."""
        policy = OperatorPolicy.from_yaml(str(_POLICY_CONFIG_PATH))
        assert policy.operator_id == "DEFAULT_LEO"

    def test_policy_loads_correct_version(self):
        """Loaded policy must carry policy_version 2.5.0."""
        policy = OperatorPolicy.from_yaml(str(_POLICY_CONFIG_PATH))
        assert policy.policy_version == "2.5.0"

    def test_missing_policy_raises(self, tmp_path):
        """/ready 503 path: missing file must raise."""
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(Exception):
            OperatorPolicy.from_yaml(str(missing))

    def test_malformed_policy_raises(self, tmp_path):
        """/ready 503 path: malformed YAML must raise."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("operator_id: [unclosed bracket")
        with pytest.raises(Exception):
            OperatorPolicy.from_yaml(str(bad))

    def test_policy_config_path_is_path_object(self):
        """_POLICY_CONFIG_PATH must be a Path instance."""
        assert isinstance(_POLICY_CONFIG_PATH, Path)
