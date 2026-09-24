"""tests/test_leolabs_screening.py

SCRUM-441: the on-demand screening client and result parser.

Offline, with a mocked LeoLabsClient. No live LeoLabs in CI.

The property most worth protecting here is that a screen which could not run
never looks like a screen that found nothing. Several tests exist only to hold
that line: timeout, transport failure, rate limiting and no-access each raise,
and none of them returns an empty conjunction set.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_screening.py -v
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from common import leolabs_runtime, leolabs_screening
from common.leolabs_asset_map import AssetRegistry
from common.leolabs_client import LeoLabsAuthError, LeoLabsError, LeoLabsHTTPError
from common.leolabs_screening import (
    LeoLabsScreeningError,
    LeoLabsScreeningTimeout,
    LeoLabsScreeningUnavailable,
    ScreeningThresholds,
    build_screening_request,
    run_screening,
    thresholds_from_policy,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_OBJECTS = [{"catalogNumber": "L2669", "noradCatalogNumber": 36508, "name": "CRYOSAT 2"}]
_THRESHOLDS = ScreeningThresholds(
    min_probability_of_collision=1.0e-4,
    max_miss_distance_km=1.0,
    max_mahalanobis=4.0,
    combined_hbr_m=15.0,
)
_EPHEMERIS = {
    "frame": "EME2000",
    "covarianceFrame": "EME2000",
    "states": [{
        "timestamp": "2026-09-24T12:00:00Z",
        "position": [6792000.0, 0.0, 0.0],
        "velocity": [0.0, 5000.0, 5800.0],
        "covariance": [[40000.0] + [0.0] * 5] + [[0.0] * 6] * 5,
    }],
}


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


@pytest.fixture
def registry() -> AssetRegistry:
    return AssetRegistry.from_objects(_OBJECTS)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    """The module is gated behind LEOLABS_ENABLED, matching leolabs_runtime."""
    monkeypatch.setattr(leolabs_screening, "LEOLABS_ENABLED", True)
    leolabs_runtime.reset_caches()
    yield
    leolabs_runtime.reset_caches()


def _client(cdms, *, created=None, status=None):
    client = MagicMock()
    client.create_screening.return_value = created or {"id": "scr-001"}
    client.wait_for_screening.return_value = status or {"status": "completed"}
    client.get_screening_cdms.return_value = cdms
    return client


def _run(client, registry, **kw):
    return run_screening(
        _EPHEMERIS, _THRESHOLDS, "L2669",
        client=client, registry=registry, **kw
    )


# ---------------------------------------------------------------------------
# The request body
# ---------------------------------------------------------------------------

class TestRequestBody:
    """SCRUM-441 fix: multipart form-data with a file upload, not a JSON body."""

    def _req(self, **kw):
        return build_screening_request(_EPHEMERIS, _THRESHOLDS, "L2669", **kw)

    def test_the_ephemeris_is_an_uploaded_file_part(self):
        req = self._req()
        assert set(req.files) == {"file"}
        filename, payload, content_type = req.files["file"]
        assert filename == "ephemeris.json"
        assert content_type == "application/json"
        # Bytes, not a file object: _request retries, and a file object would be
        # consumed by the first attempt and upload empty on the second.
        assert isinstance(payload, bytes)
        assert json.loads(payload.decode("utf-8")) == _EPHEMERIS

    def test_the_form_fields_use_the_confirmed_names(self):
        assert set(self._req().data) == {
            "primaryCatalogNumber",
            "missDistance",
            "probabilityOfCollision",
            "mahalanobisDistance",
            "primaryHardBodyRadius",
        }

    def test_the_primary_catalog_number_is_the_primary_object(self):
        assert self._req().data["primaryCatalogNumber"] == "L2669"

    def test_miss_distance_is_kilometres_with_no_conversion(self):
        """The one km field in an otherwise metres API. 1.0 must stay 1.0."""
        data = self._req().data
        assert data["missDistance"] == 1.0
        assert data["missDistance"] != 1000.0      # not metres
        assert data["missDistance"] <= 100.0       # the API's stated maximum

    def test_probability_of_collision_is_a_string(self):
        """Typed as a string by the reference, so it is sent as one."""
        value = self._req().data["probabilityOfCollision"]
        assert isinstance(value, str)
        assert float(value) == 1.0e-4

    def test_the_primary_hard_body_radius_is_the_15_m_combined_value(self):
        data = self._req().data
        assert data["primaryHardBodyRadius"] == 15.0

    def test_the_secondary_hard_body_radius_is_omitted(self):
        """Interim decision: each catalog secondary keeps its own radius."""
        assert "secondaryHardBodyRadius" not in self._req().data

    def test_mahalanobis_distance_carries_our_threshold(self):
        assert self._req().data["mahalanobisDistance"] == 4.0

    def test_screening_the_whole_catalog_is_expressed_by_omission(self):
        """There is no screenAgainstAllObjects flag; absence selects it.

        Which means an accidental addition would silently narrow the screen to
        one secondary rather than failing, so absence is asserted.
        """
        data = self._req().data
        assert "secondaryCatalogNumber" not in data
        assert "secondaryFile" not in data
        assert "secondaryFile" not in self._req().files

    def test_using_the_files_own_uncertainty_is_expressed_by_omission(self):
        """No useFileUncertainty flag either; omitting these selects it.

        Sending any of them would screen against a LeoLabs default instead of the
        covariance SCRUM-440 put in the file, hiding that covariance floor.
        """
        data = self._req().data
        for field_name in ("radialUncertainty", "inTrackUncertainty",
                           "crossTrackUncertainty"):
            assert field_name not in data

    @pytest.mark.parametrize("dead", [
        "hardBodyRadius",          # the deprecated combined value
        "primaryObject",           # our old guess
        "ephemeris",               # the inline body we used to send
        "screenAgainstAllObjects",
        "useFileUncertainty",
        "thresholds",
    ])
    def test_none_of_the_rejected_json_era_fields_is_sent(self, dead):
        """Each of these was in the body LeoLabs rejected. None exists in the API."""
        assert dead not in self._req().data

    def test_extra_merges_into_the_form_fields_last(self):
        req = self._req(extra={"mahalanobisDistance": 9.0, "newField": 7})
        assert req.data["mahalanobisDistance"] == 9.0
        assert req.data["newField"] == 7

    def test_extra_can_drop_a_field_a_live_submit_rejects(self):
        """The plan's contingency: if 422 names mahalanobisDistance, drop it."""
        req = self._req()
        req.data.pop("mahalanobisDistance")
        assert "mahalanobisDistance" not in req.data

    @pytest.mark.parametrize("bad", [None, {}, {"states": []}, {"frame": "EME2000"}])
    def test_an_ephemeris_without_states_is_refused(self, bad):
        with pytest.raises(LeoLabsScreeningError):
            build_screening_request(bad, _THRESHOLDS, "L2669")

    def test_a_missing_primary_object_is_refused(self):
        with pytest.raises(LeoLabsScreeningError):
            build_screening_request(_EPHEMERIS, _THRESHOLDS, "")


class TestThresholdsFromPolicy:
    def test_the_values_come_from_the_policy_not_from_here(self):
        """No screening threshold is invented in this module."""
        from common.operator_policy import OperatorPolicy

        policy = OperatorPolicy(operator_id="test", policy_version="v1")
        thresholds = thresholds_from_policy(policy, primary_radius_m=1.0)

        assert thresholds.min_probability_of_collision == policy.pc_maneuver_threshold
        assert thresholds.max_miss_distance_km == policy.min_miss_distance_km
        assert thresholds.max_mahalanobis == policy.mahalanobis_screen_threshold
        # Small satellite: the 15 m screening convention dominates its physical size.
        assert thresholds.combined_hbr_m == 15.0
        assert thresholds.hbr_source == "screening_convention"

    def test_a_large_primary_uses_its_physical_radius(self):
        from common.operator_policy import OperatorPolicy

        policy = OperatorPolicy(operator_id="test", policy_version="v1")
        thresholds = thresholds_from_policy(policy, primary_radius_m=40.0)
        assert thresholds.combined_hbr_m > 15.0


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestRunScreening:
    def test_it_submits_polls_retrieves_and_parses(self, cdm, registry):
        client = _client([cdm])
        result = _run(client, registry)

        client.create_screening.assert_called_once()
        client.wait_for_screening.assert_called_once()
        client.get_screening_cdms.assert_called_once_with("scr-001")

        assert result.screening_id == "scr-001"
        assert result.cdm_count == 1
        assert len(result.conjunctions) == 1
        assert result.is_clear is False

    def test_the_parsed_conjunctions_carry_real_covariance(self, cdm, registry):
        """The point of the on-demand screen: covariance-bearing results."""
        import numpy as np

        parsed = _run(_client([cdm]), registry).conjunctions[0]
        cov = np.asarray(parsed.secondary.cov_eci_pos_m2, dtype=float)
        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)
        assert float(np.linalg.eigvalsh(cov).min()) >= -1e-6
        assert parsed.t_ca_utc

    def test_it_posts_the_built_multipart_request(self, cdm, registry):
        client = _client([cdm])
        _run(client, registry)
        kwargs = client.create_screening.call_args.kwargs

        # Multipart, not JSON: no positional body is passed at all.
        assert client.create_screening.call_args.args == ()
        assert set(kwargs) == {"files", "data"}
        assert kwargs["data"]["primaryCatalogNumber"] == "L2669"
        assert kwargs["data"]["primaryHardBodyRadius"] == 15.0
        assert json.loads(kwargs["files"]["file"][1].decode("utf-8")) == _EPHEMERIS

    def test_results_are_deduped_to_one_row_per_event(self, cdm, registry):
        """Reissues of one event collapse, the same way the live listing collapses."""
        a = copy.deepcopy(cdm)
        b = copy.deepcopy(cdm)
        a["COMMENT_ID"], b["COMMENT_ID"] = 1, 2
        a["COMMENT_EVENT_ID"] = b["COMMENT_EVENT_ID"] = 900
        result = _run(_client([a, b]), registry)
        assert result.cdm_count == 2
        assert len(result.conjunctions) == 1

    def test_an_unparseable_result_is_counted_not_silently_dropped(
        self, cdm, registry
    ):
        good = copy.deepcopy(cdm)
        broken = {"COMMENT_ID": 99, "COMMENT_EVENT_ID": 42}   # nothing parseable
        result = _run(_client([good, broken]), registry)

        assert len(result.conjunctions) == 1
        assert len(result.skipped) == 1
        assert result.skipped[0]["cdm_id"] == 99
        # A caller can see the set it is judging is incomplete.
        assert result.cdm_count == 2


class TestEmptyResult:
    def test_an_empty_result_is_a_clear_sky_not_an_error(self, registry):
        result = _run(_client([]), registry)
        assert result.conjunctions == []
        assert result.is_clear is True
        assert result.cdm_count == 0

    def test_a_none_result_is_also_treated_as_empty(self, registry):
        result = _run(_client(None), registry)
        assert result.is_clear is True


# ---------------------------------------------------------------------------
# Failing closed: none of these may return empty
# ---------------------------------------------------------------------------

class TestFailsClosed:
    def test_a_poll_timeout_raises(self, cdm, registry):
        client = _client([cdm])
        client.wait_for_screening.side_effect = LeoLabsError(
            "screening scr-001 did not complete within 300s")
        with pytest.raises(LeoLabsScreeningTimeout):
            _run(client, registry)

    def test_a_transport_failure_on_create_raises(self, registry):
        client = _client([])
        client.create_screening.side_effect = LeoLabsHTTPError(500, "upstream")
        with pytest.raises(LeoLabsScreeningError):
            _run(client, registry)

    def test_a_rate_limit_refusal_raises_unavailable(self, registry):
        """429 means the screening was not run, not that the sky is clear."""
        client = _client([])
        client.create_screening.side_effect = LeoLabsHTTPError(429, "slow down")
        with pytest.raises(LeoLabsScreeningUnavailable) as excinfo:
            _run(client, registry)
        assert "rate-limited" in str(excinfo.value)

    @pytest.mark.parametrize("status", [402, 403])
    def test_no_on_demand_access_raises_unavailable(self, registry, status):
        client = _client([])
        client.create_screening.side_effect = LeoLabsHTTPError(status, "nope")
        with pytest.raises(LeoLabsScreeningUnavailable) as excinfo:
            _run(client, registry)
        assert "on-demand screening access" in str(excinfo.value)

    def test_bad_credentials_raise_unavailable(self, registry):
        client = _client([])
        client.create_screening.side_effect = LeoLabsAuthError("bad key")
        with pytest.raises(LeoLabsScreeningUnavailable):
            _run(client, registry)

    def test_the_real_422_from_the_first_live_submit_fails_closed(self, registry):
        """The actual response LeoLabs gave on 2026-09-24, pinned.

        A rejected body means the screening never ran. It must raise, not come
        back as a clear sky -- which is the whole failure mode this module
        exists to prevent, and the one a wrong request body would trigger.
        """
        client = _client([])
        client.create_screening.side_effect = LeoLabsHTTPError(
            422, 'LeoLabs request failed (HTTP 422): {"error": "Invalid Miss Distance"}'
        )
        with pytest.raises(LeoLabsScreeningError) as excinfo:
            _run(client, registry)
        # Not an Unavailable: the account has access, the body was wrong.
        assert not isinstance(excinfo.value, LeoLabsScreeningUnavailable)
        assert "Invalid Miss Distance" in str(excinfo.value)

    def test_a_retrieval_failure_raises(self, cdm, registry):
        client = _client([cdm])
        client.get_screening_cdms.side_effect = LeoLabsHTTPError(503, "unavailable")
        with pytest.raises(LeoLabsScreeningError):
            _run(client, registry)

    def test_the_feature_flag_off_raises_rather_than_returning_clear(
        self, registry, monkeypatch
    ):
        monkeypatch.setattr(leolabs_screening, "LEOLABS_ENABLED", False)
        with pytest.raises(LeoLabsScreeningUnavailable):
            _run(_client([]), registry)

    def test_a_create_without_a_usable_id_raises(self, registry):
        """No id means nothing to poll; a result then describes no screening."""
        client = _client([], created={"unexpected": "shape"})
        with pytest.raises(LeoLabsScreeningError):
            _run(client, registry)

    def test_no_failure_path_returns_an_empty_result(self, registry):
        """The property this module exists to protect, asserted as a property."""
        failures = [
            LeoLabsHTTPError(429, "rate"),
            LeoLabsHTTPError(403, "no access"),
            LeoLabsHTTPError(500, "boom"),
            LeoLabsAuthError("bad key"),
        ]
        for failure in failures:
            client = _client([])
            client.create_screening.side_effect = failure
            with pytest.raises(LeoLabsScreeningError):
                _run(client, registry)


# ---------------------------------------------------------------------------
# The status enum is not hard-wired
# ---------------------------------------------------------------------------

class TestStatusVocabulary:
    def test_a_custom_is_complete_is_passed_through(self, cdm, registry):
        """The terminal status enum is confirmed on the first live submit.

        Until then it must be injectable, not baked in.
        """
        client = _client([cdm])
        sentinel = lambda status: status.get("state") == "SCREENING_DONE"  # noqa: E731
        _run(client, registry, is_complete=sentinel)
        assert client.wait_for_screening.call_args.kwargs["is_complete"] is sentinel

    def test_a_custom_timeout_is_passed_through(self, cdm, registry):
        client = _client([cdm])
        _run(client, registry, timeout_s=42.0)
        assert client.wait_for_screening.call_args.kwargs["timeout_s"] == 42.0

    def test_defaults_are_left_to_the_client(self, cdm, registry):
        """No timeout or vocabulary is re-specified here; the client owns them."""
        client = _client([cdm])
        _run(client, registry)
        assert client.wait_for_screening.call_args.kwargs == {}


class TestScreeningIdShapes:
    @pytest.mark.parametrize("created,expected", [
        ({"id": "a"}, "a"),
        ({"screeningId": "b"}, "b"),
        ({"screening_id": "c"}, "c"),
        ({"screenings": [{"id": "d"}]}, "d"),
        ({"data": [{"screeningId": "e"}]}, "e"),
        ("f", "f"),
    ])
    def test_the_id_is_found_whatever_leolabs_calls_it(
        self, cdm, registry, created, expected
    ):
        """Another first-live-submit unknown, so several spellings are accepted."""
        client = _client([cdm], created=created)
        assert _run(client, registry).screening_id == expected
        client.get_screening_cdms.assert_called_once_with(expected)
