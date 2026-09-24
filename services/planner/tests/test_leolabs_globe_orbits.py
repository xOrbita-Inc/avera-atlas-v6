"""tests/test_leolabs_globe_orbits.py

SCRUM-447: GET /v1/leolabs/orbits, the live-data source for the 3D globe.

All offline: the LeoLabs client is mocked and the feature flag is patched, so CI
needs no credential. The live confirmation that get_states really returns what
these fakes pretend it returns is recorded in docs/scrum-447/implementation_plan.md
-- these tests pin the shape that was observed, so a change in that shape fails
here rather than silently drawing nothing.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_globe_orbits.py -v
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import server
from common import leolabs_runtime
from common.leolabs_asset_map import AssetRegistry
from common.leolabs_runtime import LeoLabsStateError, latest_state_km

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_OBJECTS = [{"catalogNumber": "L2669", "noradCatalogNumber": 36508, "name": "CRYOSAT 2"}]
_NOW = datetime(2026, 8, 25, tzinfo=timezone.utc)

# The real frames.EME2000 shape and units, as confirmed against the live API on
# 2026-09-23: a list of states, position/velocity in METRES and metres/second.
# SWARM C measured |r| = 6.79e6 m and |v| = 7.66e3 m/s.
_ASSET_R_M = [-6459825.0, -2059812.84, 438910.73]
_ASSET_V_M_S = [-374.71, -487.73, -7636.96]


def _states_response(r_m=None, v_m_s=None, timestamp="2026-08-25T00:00:00Z"):
    return {
        "states": [
            {
                "id": 1,
                "catalogNumber": "L2669",
                "noradCatalogNumber": 36508,
                "timestamp": timestamp,
                "frames": {
                    "EME2000": {
                        "position": list(r_m if r_m is not None else _ASSET_R_M),
                        "velocity": list(v_m_s if v_m_s is not None else _ASSET_V_M_S),
                    }
                },
            }
        ]
    }


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


@pytest.fixture
def registry() -> AssetRegistry:
    return AssetRegistry.from_objects(_OBJECTS)


@pytest.fixture(autouse=True)
def _clean_caches():
    leolabs_runtime.reset_caches()
    yield
    leolabs_runtime.reset_caches()


def _variant(base, *, pc, tca, norad, designator, cdm_id, event_id, name=None):
    out = copy.deepcopy(base)
    out["COLLISION_PROBABILITY"] = pc
    out["TCA_ISO"] = tca
    out["TCA"] = tca.replace("T", " ").replace("Z", "")
    out["SAT2_COMMENT_NORAD_ID"] = norad
    out["SAT2_OBJECT_DESIGNATOR"] = designator
    out["SAT2_OBJECT_NAME"] = name or f"DEBRIS {norad}"
    out["COMMENT_ID"] = cdm_id
    out["COMMENT_EVENT_ID"] = event_id
    out["SAT2_COMMENT_OBJECT_URL"] = f"https://platform.leolabs.space/catalog/{designator}"
    out["SAT2_COMMENT_STATE_URL"] = (
        f"https://platform.leolabs.space/catalog/{designator}/states/1"
    )
    return out


@pytest.fixture
def three_cdms(cdm) -> list:
    """Three distinct secondaries spanning three risk bands."""
    return [
        _variant(cdm, pc=5.0e-4, tca="2026-08-28T00:00:00Z", norad=100002,
                 designator="L100002", cdm_id=1002, event_id=9002, name="RED OBJ"),
        _variant(cdm, pc=2.0e-5, tca="2026-08-26T00:00:00Z", norad=100003,
                 designator="L100003", cdm_id=1003, event_id=9003, name="AMBER OBJ"),
        _variant(cdm, pc=1.0e-6, tca="2026-08-27T00:00:00Z", norad=100001,
                 designator="L100001", cdm_id=1001, event_id=9001, name="GREEN OBJ"),
    ]


def _serve(cdms, registry, monkeypatch, states=None, states_exc=None):
    """Wire the endpoint to a mocked client and return (TestClient, client)."""
    client = MagicMock()
    client.search_conjunction_cdms.return_value = cdms
    if states_exc is not None:
        client.get_states.side_effect = states_exc
    else:
        client.get_states.return_value = (
            states if states is not None else _states_response()
        )
    monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
    monkeypatch.setattr(leolabs_runtime, "get_client", lambda: client)
    monkeypatch.setattr(leolabs_runtime, "get_registry", lambda c=None: registry)
    return TestClient(server.svc), client


def _get(tc, **params):
    params.setdefault("primary_norad", 36508)
    return tc.get("/v1/leolabs/orbits", params=params)


# ---------------------------------------------------------------------------
# State to track
# ---------------------------------------------------------------------------

class TestStateToTrack:
    def test_the_asset_state_becomes_a_closed_track(
        self, three_cdms, registry, monkeypatch
    ):
        tc, client = _serve(three_cdms, registry, monkeypatch)
        data = _get(tc).json()

        assert data["status"] == "ok"
        track = data["asset_track"]
        assert len(track) == server.GLOBE_TRACK_STEPS + 1
        # Every point sits at a sane LEO radius: the metres-to-km conversion
        # happened exactly once. Skipping it would put these at ~6.8e6.
        radii = [math.sqrt(sum(c * c for c in p)) for p in track]
        assert all(6500 < r < 7200 for r in radii), radii[:3]
        # And the track closes on itself, which is what makes it a ring.
        gap = math.dist(track[0], track[-1])
        assert gap < 0.05 * radii[0]

    def test_the_asset_state_is_sourced_from_get_states_by_catalog_number(
        self, three_cdms, registry, monkeypatch
    ):
        """Not from an npz, and keyed on the catalog number rather than the NORAD.

        Passing the NORAD is a 404 from the live API -- that is how the SCRUM-447
        confirm-live step began -- so the call is pinned here.
        """
        tc, client = _serve(three_cdms, registry, monkeypatch)
        data = _get(tc).json()

        client.get_states.assert_called_once()
        assert client.get_states.call_args.args[0] == "L2669"
        assert data["asset"]["catalog_number"] == "L2669"
        assert data["asset"]["state_source"] == "leolabs_get_states"
        assert data["asset"]["epoch_utc"] == "2026-08-25T00:00:00Z"

    def test_secondaries_come_from_the_cdm_not_from_get_states(
        self, three_cdms, registry, monkeypatch
    ):
        """get_states is 403 for non-subscribed objects, so the CDM is the source.

        One call total, for the asset. A regression to per-secondary get_states
        would both 403 in production and show up here as extra calls.
        """
        tc, client = _serve(three_cdms, registry, monkeypatch)
        data = _get(tc).json()

        assert client.get_states.call_count == 1
        assert len(data["object_tracks"]) == 3
        for obj in data["objects"].values():
            assert obj["state_source"] == "leolabs_cdm_sat2"

    def test_object_tracks_are_closed_rings_at_leo_radius(
        self, three_cdms, registry, monkeypatch
    ):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        tracks = _get(tc).json()["object_tracks"]

        assert tracks
        for pts in tracks.values():
            assert len(pts) == server.GLOBE_TRACK_STEPS + 1
            radii = [math.sqrt(sum(c * c for c in p)) for p in pts]
            assert all(6000 < r < 8000 for r in radii)

    def test_steps_controls_the_track_resolution(
        self, three_cdms, registry, monkeypatch
    ):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        assert len(_get(tc, steps=24).json()["asset_track"]) == 25


# ---------------------------------------------------------------------------
# Risk banding matches the 2D table
# ---------------------------------------------------------------------------

class TestRiskBanding:
    def test_bands_match_the_conjunctions_route_for_the_same_objects(
        self, three_cdms, registry, monkeypatch
    ):
        """The globe and the table must not disagree about what is RED."""
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        globe = _get(tc).json()
        table = tc.get(
            "/v1/leolabs/conjunctions", params={"primary_norad": 36508}
        ).json()

        by_norad = {str(r["secondary_norad"]): r["risk_level"]
                    for r in table["conjunctions"]}
        assert globe["risk"] == by_norad
        assert set(globe["risk"].values()) == {"RED", "AMBER", "GREEN"}

    def test_the_worst_object_is_the_highest_pc(
        self, three_cdms, registry, monkeypatch
    ):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        data = _get(tc).json()

        assert data["worst_object"] == "100002"
        assert data["objects"]["100002"]["risk_level"] == "RED"
        assert data["objects"]["100002"]["pc"] == 5.0e-4

    def test_one_ring_per_object_even_with_several_events(
        self, cdm, registry, monkeypatch
    ):
        """A busy secondary has many events; the globe draws it once.

        Observed live: SWARM C's 116 events collapsed to 18 distinct objects, one
        of them appearing 29 times. Drawing per event would stack 29 identical
        rings.
        """
        repeated = [
            _variant(cdm, pc=5.0e-4, tca="2026-08-28T00:00:00Z", norad=100002,
                     designator="L100002", cdm_id=2001, event_id=8001),
            _variant(cdm, pc=1.0e-4, tca="2026-08-29T00:00:00Z", norad=100002,
                     designator="L100002", cdm_id=2002, event_id=8002),
            _variant(cdm, pc=2.0e-5, tca="2026-08-30T00:00:00Z", norad=100003,
                     designator="L100003", cdm_id=2003, event_id=8003),
        ]
        tc, _ = _serve(repeated, registry, monkeypatch)
        data = _get(tc).json()

        assert data["counts"]["events_in_window"] == 3
        assert data["counts"]["drawn"] == 2
        assert set(data["object_tracks"]) == {"100002", "100003"}
        # The kept row is the worst one for that object, not whichever came last.
        assert data["objects"]["100002"]["risk_level"] == "RED"


# ---------------------------------------------------------------------------
# Covariance (SCRUM-448)
# ---------------------------------------------------------------------------

class TestCovariance:
    def test_every_drawn_object_carries_axes_and_sigmas(
        self, three_cdms, registry, monkeypatch
    ):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        data = _get(tc).json()

        assert data["objects"]
        for key, obj in data["objects"].items():
            cov = obj["cov"]
            assert len(cov["axes"]) == 3
            assert len(cov["sigmas_m"]) == 3
            for axis in cov["axes"]:
                assert len(axis) == 3
                assert math.isclose(
                    math.sqrt(sum(c * c for c in axis)), 1.0, rel_tol=1e-9
                )
            assert all(isinstance(v, float) and v >= 0 for v in cov["sigmas_m"])
            # Sorted largest first, so "3 sigma" has an unambiguous meaning.
            assert cov["sigmas_m"] == sorted(cov["sigmas_m"], reverse=True)
            # Labelled, so nobody mistakes it for a synthetic default.
            assert cov["source"] == "leolabs_cdm_covariance"
            assert cov["frame"] == "EME2000"

    def test_the_asset_carries_a_covariance_from_the_worst_rows_sat1_block(
        self, three_cdms, registry, monkeypatch
    ):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        data = _get(tc).json()

        cov = data["asset"]["cov"]
        assert len(cov["axes"]) == 3 and len(cov["sigmas_m"]) == 3
        assert cov["source"] == "leolabs_cdm_covariance"
        # The asset's state is live but its covariance is a CDM's, so the two are
        # at different epochs and the response says which.
        assert data["asset"]["cov_epoch_utc"] == (
            data["objects"][data["worst_object"]]["epoch_utc"]
        )

    def test_the_axes_are_a_right_handed_orthonormal_basis(
        self, three_cdms, registry, monkeypatch
    ):
        """The renderer builds a rotation from these."""
        import numpy as np

        tc, _ = _serve(three_cdms, registry, monkeypatch)
        data = _get(tc).json()
        for obj in data["objects"].values():
            basis = np.array(obj["cov"]["axes"], dtype=float).T
            assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-9)
            assert float(np.linalg.det(basis)) == pytest.approx(1.0, abs=1e-9)

    def test_the_covariance_is_the_cdms_own_not_a_default(
        self, three_cdms, registry, monkeypatch
    ):
        """Reconstructing A from the axes and sigmas must give the parsed matrix.

        This is the test that distinguishes 'real covariance plumbed through'
        from 'something ellipsoid-shaped'.
        """
        import numpy as np
        from common.leolabs_cdm_parser import parse_leolabs_cdm

        tc, _ = _serve(three_cdms, registry, monkeypatch)
        data = _get(tc).json()

        worst_key = data["worst_object"]
        cov = data["objects"][worst_key]["cov"]
        basis = np.array(cov["axes"], dtype=float).T
        lam = np.diag([s ** 2 for s in cov["sigmas_m"]])
        reconstructed = basis @ lam @ basis.T

        # three_cdms is ordered worst-first, so its first entry is the worst row.
        parsed = parse_leolabs_cdm(three_cdms[0], "L2669")
        assert np.allclose(
            reconstructed, np.asarray(parsed.secondary.cov_eci_pos_m2), rtol=1e-6,
            atol=1e-6,
        )

    def test_the_real_anisotropy_survives(self, three_cdms, registry, monkeypatch):
        """A conjunction covariance is in-track dominated by a huge ratio.

        If that ever comes back near-spherical, something has normalised away the
        one thing the ellipsoid is for.
        """
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        data = _get(tc).json()
        sig = data["objects"][data["worst_object"]]["cov"]["sigmas_m"]
        assert sig[0] / max(sig[2], 1e-9) > 50

    def test_an_object_with_no_usable_covariance_is_drawn_without_one(
        self, cdm, registry, monkeypatch
    ):
        """No cov field, no fabricated ellipsoid -- but the object still draws.

        Same honesty rule as a stateless secondary: absent is absent, and the
        object does not disappear from the globe because of it.
        """
        import common.leolabs_runtime as rt

        good = _variant(cdm, pc=5.0e-4, tca="2026-08-28T00:00:00Z", norad=100002,
                        designator="L100002", cdm_id=1002, event_id=9002)
        broken = _variant(cdm, pc=2.0e-5, tca="2026-08-26T00:00:00Z", norad=100003,
                          designator="L100003", cdm_id=1003, event_id=9003)
        tc, _ = _serve([good, broken], registry, monkeypatch)

        # Strip the covariance of the second object only.
        real = rt.object_covariance_block
        calls = {"n": 0}

        def _second_has_none(cov3):
            calls["n"] += 1
            return None if calls["n"] == 2 else real(cov3)

        monkeypatch.setattr(rt, "object_covariance_block", _second_has_none)
        import server as srv
        monkeypatch.setattr(srv, "object_covariance_block", _second_has_none)

        data = _get(tc).json()

        assert set(data["object_tracks"]) == {"100002", "100003"}
        assert "cov" in data["objects"]["100002"]
        assert "cov" not in data["objects"]["100003"]
        # And it is still a fully drawn object, not a skip.
        assert data["counts"]["drawn"] == 2
        assert data["counts"]["skipped"] == 0

    def test_covariance_does_not_change_the_status_contract(
        self, three_cdms, registry, monkeypatch
    ):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        assert _get(tc).status_code == 200
        assert _get(tc, primary_norad=99999).status_code == 404
        assert _get(tc, lookahead_days=60).status_code == 422

    def test_feed_off_is_still_503_with_no_covariance_leaking(self, monkeypatch):
        monkeypatch.setattr(server, "LEOLABS_ENABLED", False)
        resp = TestClient(server.svc).get(
            "/v1/leolabs/orbits", params={"primary_norad": 36508}
        )
        assert resp.status_code == 503
        assert "asset" not in resp.json()


# ---------------------------------------------------------------------------
# Empty set, skipped objects, and the status contract
# ---------------------------------------------------------------------------

class TestEmptyAndSkipped:
    def test_an_empty_in_volume_set_is_a_200_with_the_asset_only(
        self, registry, monkeypatch
    ):
        """A quiet sky is a real state of the world, not an error."""
        tc, _ = _serve([], registry, monkeypatch)
        resp = _get(tc)

        assert resp.status_code == 200
        data = resp.json()
        assert data["object_tracks"] == {}
        assert data["counts"]["drawn"] == 0
        # The asset is still drawn: an operator must see their own spacecraft.
        assert len(data["asset_track"]) == server.GLOBE_TRACK_STEPS + 1

    def test_a_secondary_with_no_usable_state_is_skipped_not_fatal(
        self, cdm, registry, monkeypatch
    ):
        broken = _variant(cdm, pc=5.0e-4, tca="2026-08-28T00:00:00Z", norad=100002,
                          designator="L100002", cdm_id=1002, event_id=9002)
        good = _variant(cdm, pc=2.0e-5, tca="2026-08-26T00:00:00Z", norad=100003,
                        designator="L100003", cdm_id=1003, event_id=9003)
        tc, _ = _serve([broken, good], registry, monkeypatch)

        # Fail exactly one secondary's propagation, the way an unusable state
        # would. Call 1 is the asset, call 2 is the worst secondary (rows arrive
        # worst-first), call 3 is the good one -- so failing call 2 drops the RED
        # object and leaves the AMBER one to draw.
        import server as srv
        original = srv.propagate_two_body
        calls = {"n": 0}

        def _fail_second(r0, v0, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("no usable state")
            return original(r0, v0, **kw)

        monkeypatch.setattr(srv, "propagate_two_body", _fail_second)
        resp = _get(tc)

        assert resp.status_code == 200
        data = resp.json()
        # The good object still drew; the broken one is reported, not silently gone.
        assert "100003" in data["object_tracks"]
        assert "100002" not in data["object_tracks"]
        assert data["counts"]["skipped"] == 1
        assert data["skipped"][0]["secondary_norad"] == 100002

    def test_feed_off_is_503_not_an_empty_globe(self, monkeypatch):
        """'The feed is off' and 'your sky is clear' must not look alike."""
        monkeypatch.setattr(server, "LEOLABS_ENABLED", False)
        resp = TestClient(server.svc).get(
            "/v1/leolabs/orbits", params={"primary_norad": 36508}
        )

        assert resp.status_code == 503
        assert "asset_track" not in resp.json()

    def test_an_unsubscribed_asset_is_404(self, three_cdms, registry, monkeypatch):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        assert _get(tc, primary_norad=99999).status_code == 404

    def test_an_asset_with_no_state_is_503(self, three_cdms, registry, monkeypatch):
        """No asset state means no globe, and that is a failure, not an empty sky."""
        tc, _ = _serve(three_cdms, registry, monkeypatch, states={"states": []})
        resp = _get(tc)
        assert resp.status_code == 503
        assert "asset state unavailable" in resp.json().get("error", "").lower()

    def test_a_state_fetch_failure_is_503(self, three_cdms, registry, monkeypatch):
        tc, _ = _serve(three_cdms, registry, monkeypatch,
                       states_exc=RuntimeError("upstream 403"))
        assert _get(tc).status_code == 503

    def test_a_conjunction_fetch_failure_is_503(
        self, three_cdms, registry, monkeypatch
    ):
        tc, client = _serve(three_cdms, registry, monkeypatch)
        client.search_conjunction_cdms.side_effect = RuntimeError("upstream 500")
        assert _get(tc).status_code == 503

    def test_a_window_wider_than_the_cap_is_422(
        self, three_cdms, registry, monkeypatch
    ):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        assert _get(tc, lookahead_days=60).status_code == 422

    def test_bad_parameters_are_422(self, three_cdms, registry, monkeypatch):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        assert _get(tc, steps=2).status_code == 422
        assert _get(tc, max_objects=-1).status_code == 422
        assert _get(tc, lookahead_days=-1).status_code == 422


# ---------------------------------------------------------------------------
# The state helper, against the shape confirmed live
# ---------------------------------------------------------------------------

class TestLatestStateKm:
    def test_metres_are_converted_to_km_exactly_once(self):
        client = MagicMock()
        client.get_states.return_value = _states_response()
        r_km, v_km_s, epoch = latest_state_km("L2669", client=client)

        assert math.isclose(r_km[0], _ASSET_R_M[0] / 1000.0)
        assert math.isclose(v_km_s[2], _ASSET_V_M_S[2] / 1000.0)
        # Sanity on magnitudes: LEO radius in km and orbital speed in km/s.
        assert 6500 < math.sqrt(sum(c * c for c in r_km)) < 7200
        assert 6.5 < math.sqrt(sum(c * c for c in v_km_s)) < 8.5
        assert epoch == "2026-08-25T00:00:00Z"

    def test_the_latest_state_is_the_first_entry(self):
        client = MagicMock()
        payload = _states_response()
        payload["states"].append(
            _states_response(r_m=[1.0, 2.0, 3.0])["states"][0]
        )
        client.get_states.return_value = payload
        r_km, _, _ = latest_state_km("L2669", client=client)
        assert math.isclose(r_km[0], _ASSET_R_M[0] / 1000.0)

    @pytest.mark.parametrize("payload", [
        {"states": []},
        {},
        {"states": [{"frames": {}}]},
        {"states": [{"frames": {"EME2000": {"position": [1, 2]}}}]},
        {"states": [{"frames": {"EME2000": {"position": [0, 0, 0],
                                            "velocity": [1, 2, 3]}}}]},
    ])
    def test_an_unusable_state_raises_rather_than_returning_half_a_vector(
        self, payload
    ):
        """A track drawn from a partial state is worse than no track."""
        client = MagicMock()
        client.get_states.return_value = payload
        with pytest.raises(LeoLabsStateError):
            latest_state_km("L2669", client=client)
