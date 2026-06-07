"""
tests/test_planner_hardening.py

SCRUM-341: Tests for container hardening additions to server.py.
Covers AC1 (/ready probe logic) and AC3 (structured JSON logging).

Imports _JsonFormatter and _POLICY_CONFIG_PATH from common.logging_setup
(the module both server.py and this file share) so tests exercise the
actual production code path rather than an inline copy.

Run with (from repo root -- conftest.py handles PYTHONPATH):
    python -m pytest services/planner/tests/test_planner_hardening.py -v
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

import pytest

from common.logging_setup import _JsonFormatter, _POLICY_CONFIG_PATH, SERVICE_NAME
from common.operator_policy import OperatorPolicy


# ---------------------------------------------------------------------------
# AC3: Structured JSON logging
# ---------------------------------------------------------------------------

class TestJsonFormatter:
    """AC3: _JsonFormatter must emit valid JSON matching the documented format:
    {"time": "...", "level": "...", "service": "planner", "msg": "...", ...extras}
    """

    def _emit(self, level: int, message: str, extra: dict | None = None) -> dict:
        """Emit one log record through _JsonFormatter, return parsed dict."""
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        logger = logging.getLogger(SERVICE_NAME)
        prev_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            logger.log(level, message, extra=extra or {})
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
        """Record must contain time, level, service, msg."""
        record = self._emit(logging.INFO, "key check")
        assert "time" in record
        assert "level" in record
        assert "service" in record
        assert "msg" in record

    def test_service_field_matches_constant(self):
        """service field must equal SERVICE_NAME."""
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

    def test_extra_fields_merged_into_top_level(self):
        """Extra fields must appear at top level, not nested inside msg."""
        record = self._emit(logging.INFO, "structured event", extra={
            "event": "test_event",
            "pair": "226/35929",
        })
        assert record["event"] == "test_event"
        assert record["pair"] == "226/35929"
        # msg must be the plain string, not a JSON blob
        assert record["msg"] == "structured event"

    def test_msg_is_not_double_encoded(self):
        """msg field must be a plain string, never a JSON-encoded string."""
        record = self._emit(logging.INFO, "plain message", extra={"key": "value"})
        assert isinstance(record["msg"], str)
        # If double-encoded, msg would start with '{'
        assert not record["msg"].startswith("{")

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
    """AC1: /ready must return 200 only when policy config is present and valid."""

    def test_real_policy_file_exists(self):
        """/ready happy path requires the config file to exist."""
        # When running locally (not in container), resolve relative to repo root.
        local_path = Path("services/planner/config/operator_policy_leo.yaml")
        container_path = _POLICY_CONFIG_PATH
        assert local_path.exists() or container_path.exists(), (
            f"Default policy config not found at {local_path} or {container_path}. "
            "COPY config/ /app/config/ must be in the Dockerfile."
        )

    def test_real_policy_file_loads_without_error(self):
        """The default policy must load cleanly -- /ready 200 path."""
        local_path = Path("services/planner/config/operator_policy_leo.yaml")
        path = local_path if local_path.exists() else _POLICY_CONFIG_PATH
        policy = OperatorPolicy.from_yaml(str(path))
        assert policy is not None

    def test_policy_loads_correct_operator_id(self):
        local_path = Path("services/planner/config/operator_policy_leo.yaml")
        path = local_path if local_path.exists() else _POLICY_CONFIG_PATH
        policy = OperatorPolicy.from_yaml(str(path))
        assert policy.operator_id == "DEFAULT_LEO"

    def test_policy_loads_correct_version(self):
        local_path = Path("services/planner/config/operator_policy_leo.yaml")
        path = local_path if local_path.exists() else _POLICY_CONFIG_PATH
        policy = OperatorPolicy.from_yaml(str(path))
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
        assert isinstance(_POLICY_CONFIG_PATH, Path)
