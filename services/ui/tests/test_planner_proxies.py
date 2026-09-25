"""tests/test_planner_proxies.py

SCRUM-457: the dashboard's proxy to the SCRUM-456 asynchronous secondary screen.

The front end paints the Secondary conflict row CLEAR or NOT CLEAR from what this
route returns, so the property worth testing is not that it forwards -- it is that
nothing it can return on a bad day looks like a clear screen. A planner that is
down, slow, returning an error envelope, returning a 404 for an evicted job, or
returning something that is not JSON at all must all read NOT CLEAR.

These are the first tests in the ui service; there was no suite here before, so
conftest.py alongside mirrors the planner's.

Run from repo root:
    python -m pytest services/ui/tests/test_planner_proxies.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from app.main import app

_JOB = "9596f2a685a84f6cae69aa107ecec383"
_URL = f"/api/planner/secondary-screen/{_JOB}"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class _Resp:
    """A stand-in for requests' response, with the two things the route reads."""

    def __init__(self, payload, status_code=200, invalid_json=False):
        self._payload = payload
        self.status_code = status_code
        self._invalid = invalid_json

    def json(self):
        if self._invalid:
            raise ValueError("not json")
        return self._payload


def _pending():
    return {"job_id": _JOB, "status": "pending", "clear": False, "pending": True,
            "screening_id": None, "conjunctions": [], "verdict": None,
            "error": None}


def _clear():
    return {"job_id": _JOB, "status": "clear", "clear": True, "pending": False,
            "screening_id": "603390", "conjunctions": [], "error": None,
            "verdict": {"clear": True, "evaluated": 0, "breaches": []}}


def _not_clear():
    return {"job_id": _JOB, "status": "not_clear", "clear": False,
            "pending": False, "screening_id": "603390", "error": None,
            "conjunctions": [{"object_id": "STARLINK-1661",
                              "miss_distance_km": 4.397}],
            "verdict": {"clear": False, "evaluated": 1210,
                        "breaches": [{"object_id": "STARLINK-1661",
                                      "limbs": ["mahalanobis"]}]}}


# ---------------------------------------------------------------------------
# Forwarding
# ---------------------------------------------------------------------------

class TestForwarding:
    def test_it_calls_the_planner_poll_endpoint_for_that_job(self, client):
        with patch("app.main.requests.get", return_value=_Resp(_pending())) as get:
            client.get(_URL)
        assert get.call_args[0][0].endswith(f"/v1/secondary-screen/{_JOB}")

    def test_the_timeout_is_short_because_the_endpoint_is_a_store_read(self, client):
        """Not the 60 s of the evaluate proxy: the screen is not on this path."""
        with patch("app.main.requests.get", return_value=_Resp(_pending())) as get:
            client.get(_URL)
        assert get.call_args.kwargs["timeout"] <= 15

    def test_a_pending_payload_is_relayed_untouched(self, client):
        with patch("app.main.requests.get", return_value=_Resp(_pending())):
            body = client.get(_URL).json()
        assert body["status"] == "pending"
        assert body["clear"] is False
        assert body["pending"] is True

    def test_a_clear_payload_is_relayed_untouched(self, client):
        """The one case that may report clear, and it must survive intact."""
        with patch("app.main.requests.get", return_value=_Resp(_clear())):
            resp = client.get(_URL)
        assert resp.status_code == 200
        assert resp.json()["status"] == "clear"
        assert resp.json()["clear"] is True

    def test_a_not_clear_payload_keeps_its_breaches(self, client):
        with patch("app.main.requests.get", return_value=_Resp(_not_clear())):
            body = client.get(_URL).json()
        assert body["status"] == "not_clear"
        assert body["clear"] is False
        assert body["verdict"]["breaches"]
        assert body["conjunctions"][0]["object_id"] == "STARLINK-1661"


# ---------------------------------------------------------------------------
# Failing closed
# ---------------------------------------------------------------------------

class TestFailsClosed:
    def _assert_not_clear(self, body):
        """The shape the front end reads as NOT CLEAR, on every failure path."""
        assert body["status"] == "error"
        assert body["clear"] is False
        assert body["pending"] is False
        assert body["error"]
        assert "NOT CLEAR" in body["operator_note"]
        # and nothing that could be mistaken for a result
        assert body["conjunctions"] == []
        assert body["verdict"] is None

    def test_a_planner_connection_error_reads_not_clear(self, client):
        with patch("app.main.requests.get",
                   side_effect=requests.exceptions.ConnectionError("refused")):
            resp = client.get(_URL)
        assert resp.status_code == 503
        self._assert_not_clear(resp.json())

    def test_a_planner_timeout_reads_not_clear(self, client):
        with patch("app.main.requests.get",
                   side_effect=requests.exceptions.Timeout("slow")):
            resp = client.get(_URL)
        assert resp.status_code == 504
        self._assert_not_clear(resp.json())

    def test_an_unexpected_exception_reads_not_clear(self, client):
        with patch("app.main.requests.get", side_effect=RuntimeError("boom")):
            resp = client.get(_URL)
        assert resp.status_code == 500
        self._assert_not_clear(resp.json())

    def test_invalid_json_reads_not_clear(self, client):
        with patch("app.main.requests.get",
                   return_value=_Resp(None, invalid_json=True)):
            self._assert_not_clear(client.get(_URL).json())

    def test_the_planner_404_for_an_evicted_job_reads_not_clear(self, client):
        """An unknown or evicted job id is not a clear screen.

        The planner returns an error envelope with no status field; handing that
        to the front end unexamined would leave it guessing.
        """
        envelope = {"error": {"message": "no secondary screen with id ..."}}
        with patch("app.main.requests.get", return_value=_Resp(envelope, 404)):
            resp = client.get(_URL)
        assert resp.status_code == 404
        self._assert_not_clear(resp.json())

    def test_an_error_envelope_with_a_200_still_reads_not_clear(self, client):
        """Defensive: a body without status/clear can never be judged clear."""
        with patch("app.main.requests.get",
                   return_value=_Resp({"detail": "something odd"}, 200)):
            resp = client.get(_URL)
        self._assert_not_clear(resp.json())

    def test_a_payload_missing_the_clear_flag_reads_not_clear(self, client):
        """status alone is not enough: the front end judges on clear."""
        with patch("app.main.requests.get",
                   return_value=_Resp({"status": "clear"}, 200)):
            self._assert_not_clear(client.get(_URL).json())

    def test_a_payload_missing_the_status_reads_not_clear(self, client):
        with patch("app.main.requests.get",
                   return_value=_Resp({"clear": True}, 200)):
            self._assert_not_clear(client.get(_URL).json())

    def test_no_failure_path_can_report_clear_true(self, client):
        """The single property this route exists to guarantee."""
        failures = [
            requests.exceptions.ConnectionError("refused"),
            requests.exceptions.Timeout("slow"),
            RuntimeError("boom"),
        ]
        for failure in failures:
            with patch("app.main.requests.get", side_effect=failure):
                assert client.get(_URL).json()["clear"] is False
        bodies = [
            _Resp(None, invalid_json=True),
            _Resp({"error": "gone"}, 404),
            _Resp({"clear": True}, 200),          # no status
            _Resp({"status": "clear"}, 200),      # no clear flag
        ]
        for resp in bodies:
            with patch("app.main.requests.get", return_value=resp):
                assert client.get(_URL).json()["clear"] is False
