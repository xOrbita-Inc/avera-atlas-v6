"""
Tests for SCRUM-373 AC2: feeding observations through the live tracker
/v1/observations interface, rather than emitting a file.

Mocking convention matches services/detector/tests/test_tracker_link.py's
TestPushToTracker (patch the httpx client, MagicMock the response), adapted
for the sync httpx.Client used here rather than AsyncClient.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.sandbox.observation_bundle import ObservationBundle
from services.sandbox.observations import generate_angular_observation
from services.sandbox.schema_emission import (
    TRACKER_OBSERVATIONS_URL,
    TrackerIngestError,
    TrackerIngestResult,
    push_observations_to_tracker,
)
from services.sandbox.sensor_knowledge import SensorKnowledgeStore
from services.sandbox.sensor_model import SensorConfig
from services.sandbox.tests.test_observations import make_debris, make_host
from services.sandbox.tests.test_tracker_contract_emission import (
    FORBIDDEN_GOD_VIEW_FIELDS,
    make_detected_bundle,
)


def _mock_client(status_code: int, json_body: dict) -> MagicMock:
    """Build a MagicMock standing in for httpx.Client's context manager."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_body
    mock_response.text = str(json_body)

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


class TestPushObservationsToTracker:
    def test_posts_to_correct_url(self):
        bundle = make_detected_bundle()
        mock_client = _mock_client(
            200, {"accepted": True, "observation_count": 1, "observation_ids": ["obs-1"]}
        )

        with patch(
            "services.sandbox.schema_emission.httpx.Client",
            return_value=mock_client,
        ):
            push_observations_to_tracker(bundle)

        called_url = mock_client.post.call_args.args[0]
        assert called_url == TRACKER_OBSERVATIONS_URL

    def test_posts_the_same_body_bundle_to_tracker_ingest_request_builds(self):
        from services.sandbox.schema_emission import bundle_to_tracker_ingest_request

        bundle = make_detected_bundle()
        expected_body = bundle_to_tracker_ingest_request(bundle)
        mock_client = _mock_client(
            200, {"accepted": True, "observation_count": 1, "observation_ids": ["obs-1"]}
        )

        with patch(
            "services.sandbox.schema_emission.httpx.Client",
            return_value=mock_client,
        ):
            push_observations_to_tracker(bundle)

        sent_body = mock_client.post.call_args.kwargs["json"]
        assert sent_body == expected_body

    def test_returns_result_on_success(self):
        bundle = make_detected_bundle()
        mock_client = _mock_client(
            200,
            {"accepted": True, "observation_count": 1, "observation_ids": ["obs-test-1"]},
        )

        with patch(
            "services.sandbox.schema_emission.httpx.Client",
            return_value=mock_client,
        ):
            result = push_observations_to_tracker(bundle)

        assert isinstance(result, TrackerIngestResult)
        assert result.accepted is True
        assert result.observation_count == 1
        assert result.observation_ids == ("obs-test-1",)
        assert result.status_code == 200

    def test_raises_on_non_200_response(self):
        bundle = make_detected_bundle()
        mock_client = _mock_client(400, {"error": "Invalid observation ingest request."})

        with patch(
            "services.sandbox.schema_emission.httpx.Client",
            return_value=mock_client,
        ):
            with pytest.raises(TrackerIngestError):
                push_observations_to_tracker(bundle)

    def test_raises_on_connection_failure(self):
        import httpx

        bundle = make_detected_bundle()
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "services.sandbox.schema_emission.httpx.Client",
            return_value=mock_client,
        ):
            with pytest.raises(TrackerIngestError):
                push_observations_to_tracker(bundle)

    def test_rejects_empty_bundle_before_any_network_call(self):
        """Same zero-detections guard as bundle_to_tracker_ingest_request,
        which push_observations_to_tracker calls first -- no network call
        should be attempted for an empty bundle."""
        bundle = ObservationBundle(observations=[])
        mock_client = _mock_client(200, {})

        with patch(
            "services.sandbox.schema_emission.httpx.Client",
            return_value=mock_client,
        ):
            with pytest.raises(ValueError, match="zero detections"):
                push_observations_to_tracker(bundle)

        mock_client.post.assert_not_called()

    def test_does_not_leak_god_view_fields_in_the_request_actually_sent(self):
        """Mirrors test_tracker_contract_excludes_god_view_fields, but
        checks the body that would actually go over the wire, not just
        the dict bundle_to_tracker_ingest_request returns."""
        import json

        bundle = make_detected_bundle()
        mock_client = _mock_client(
            200, {"accepted": True, "observation_count": 1, "observation_ids": ["obs-1"]}
        )

        with patch(
            "services.sandbox.schema_emission.httpx.Client",
            return_value=mock_client,
        ):
            push_observations_to_tracker(bundle)

        sent_body = mock_client.post.call_args.kwargs["json"]
        encoded = json.dumps(sent_body)
        for forbidden_field in FORBIDDEN_GOD_VIEW_FIELDS:
            assert forbidden_field not in encoded


class TestSensorKnowledgeStorePushToTracker:
    """SensorKnowledgeStore.push_to_tracker: same isolation guarantees as
    write_schema_artifact, since it goes through the identical sensor-facing
    conversion and the store still has no god-view access at all."""

    def test_delegates_to_push_observations_to_tracker(self):
        bundle = make_detected_bundle()
        store = SensorKnowledgeStore.from_observation_bundle(bundle)
        mock_client = _mock_client(
            200, {"accepted": True, "observation_count": 1, "observation_ids": ["obs-1"]}
        )

        with patch(
            "services.sandbox.schema_emission.httpx.Client",
            return_value=mock_client,
        ):
            result = store.push_to_tracker(bundle)

        assert result.accepted is True
        mock_client.post.assert_called_once()

    def test_store_still_has_no_god_view_access(self):
        """push_to_tracker must not open a new god-view surface: the
        existing GodViewAccessError properties still raise."""
        from services.sandbox.sensor_knowledge import GodViewAccessError

        bundle = make_detected_bundle()
        store = SensorKnowledgeStore.from_observation_bundle(bundle)

        with pytest.raises(GodViewAccessError):
            _ = store.truth_states
        with pytest.raises(GodViewAccessError):
            _ = store.objects
        with pytest.raises(GodViewAccessError):
            _ = store.snapshots
