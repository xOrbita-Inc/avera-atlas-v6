"""tests/test_leolabs_ephemeris.py

SCRUM-440: the post-burn trajectory ephemeris submitted for LeoLabs on-demand
screening.

Offline, no network. What is worth testing here is not the propagator -- that is
kepler_propagate's own suite -- but the three things this module is responsible
for: the exact emitted shape, the unit conversions, and refusing to fabricate an
uncertainty it does not have.

The expected shape is pinned against what a live GET
/catalog/objects/<catalog>/states returned on 2026-09-24: frames.EME2000 carries
`position` (m), `velocity` (m/s) and a 6x6 `covariance` in m^2 / (m/s)^2 with the
position block top-left. Emitting the same object LeoLabs hands back is the point,
so a change in that shape should fail here.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_ephemeris.py -v
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from common.leolabs_ephemeris import (
    DEFAULT_HORIZON_HOURS,
    DEFAULT_STEP_S,
    FRAME,
    LeoLabsEphemerisError,
    build_screening_ephemeris,
    seed_post_burn_covariance_km2,
)
from common.orbit_propagation import (
    MU_EARTH,
    kepler_propagate,
    propagate_j2_series,
)

# A circular-ish LEO post-burn state, km and km/s.
_EPOCH = "2026-09-24T12:00:00Z"
_R_KM = [6792.0, 0.0, 0.0]
_V_KM_S = [0.0, 5.0, 5.8]
# A realistic post-burn position covariance: metres-scale, expressed in km^2.
# 0.04 km^2 = 4e4 m^2 -> 200 m one-sigma.
# A Swarm-C-like orbit, for the divergence guard: the asset the SCRUM-441 screen
# was actually run against.
_SWARM_C_A_KM = 6838.0
_SWARM_C_INC = math.radians(87.35)
_SWARM_C_SPEED = math.sqrt(MU_EARTH / _SWARM_C_A_KM)
_SWARM_C_R_KM = [_SWARM_C_A_KM, 0.0, 0.0]
_SWARM_C_V_KM_S = [0.0,
                   _SWARM_C_SPEED * math.cos(_SWARM_C_INC),
                   _SWARM_C_SPEED * math.sin(_SWARM_C_INC)]

_P_KM2 = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.09, 0.02],
    [0.00, 0.02, 0.16],
])


def _build(**kw):
    params = dict(
        epoch_utc=_EPOCH, r_sat_km=_R_KM, v_sat_km_s=_V_KM_S,
        p_post_eci_km2=_P_KM2, horizon_hours=2.0, step_s=600.0,
    )
    params.update(kw)
    return build_screening_ephemeris(**params)


# ---------------------------------------------------------------------------
# Shape: what LeoLabs receives
# ---------------------------------------------------------------------------

class TestEmittedShape:
    def test_top_level_is_frame_covariance_frame_and_states(self):
        eph = _build()
        assert sorted(eph) == ["covarianceFrame", "frame", "states"]
        assert eph["frame"] == FRAME == "EME2000"
        assert eph["covarianceFrame"] == "EME2000"

    def test_each_state_has_exactly_the_four_expected_keys(self):
        """Matching LeoLabs' own EME2000 state: timestamp, position, velocity, covariance."""
        for state in _build()["states"]:
            assert sorted(state) == ["covariance", "position", "timestamp", "velocity"]
            assert len(state["position"]) == 3
            assert len(state["velocity"]) == 3

    def test_the_covariance_is_a_6x6_nested_list_of_plain_floats(self):
        """Nested lists, not a numpy array: this dict is going through json.dumps."""
        cov = _build()["states"][0]["covariance"]
        assert isinstance(cov, list) and len(cov) == 6
        for row in cov:
            assert isinstance(row, list) and len(row) == 6
            assert all(isinstance(c, float) for c in row)

    def test_the_whole_document_is_json_serialisable(self):
        import json
        text = json.dumps(_build())
        assert json.loads(text)["frame"] == "EME2000"


# ---------------------------------------------------------------------------
# Units: the conversions a bug would hide in
# ---------------------------------------------------------------------------

class TestUnits:
    def test_the_first_position_is_the_km_input_times_1000(self):
        first = _build()["states"][0]
        assert first["position"] == pytest.approx([c * 1000.0 for c in _R_KM])

    def test_the_first_velocity_is_the_km_s_input_times_1000(self):
        first = _build()["states"][0]
        assert first["velocity"] == pytest.approx([c * 1000.0 for c in _V_KM_S])

    def test_the_seed_covariance_position_block_is_p_post_times_1e6(self):
        """The FIRST state is the seed, since Phi(0) is the identity."""
        cov = np.array(_build()["states"][0]["covariance"], dtype=float)
        assert np.allclose(cov[:3, :3], _P_KM2 * 1.0e6, rtol=1e-9, atol=1e-6)
        # Sanity in the units a human reads: 0.04 km^2 is a 200 m one-sigma.
        assert math.isclose(math.sqrt(cov[0][0]), 200.0, rel_tol=1e-6)

    def test_positions_are_metres_scale_not_kilometres(self):
        """The conversion is the likeliest bug, so check the magnitude directly."""
        for state in _build()["states"]:
            radius_m = math.sqrt(sum(c * c for c in state["position"]))
            assert 6.0e6 < radius_m < 8.0e6

    def test_velocities_are_metres_per_second_scale(self):
        for state in _build()["states"]:
            speed = math.sqrt(sum(c * c for c in state["velocity"]))
            assert 6.0e3 < speed < 9.0e3


# ---------------------------------------------------------------------------
# The 3x3 -> 6x6 decision
# ---------------------------------------------------------------------------

class TestCovarianceBlock:
    def test_the_seed_velocity_block_is_zero_without_a_burn(self):
        """No burn, no execution error -- so no velocity uncertainty asserted."""
        cov = np.array(
            seed_post_burn_covariance_km2(_P_KM2, [0.0, 0.0, 0.0], 0.0, 0.0),
            dtype=float)
        assert np.all(cov[3:, 3:] == 0.0)
        assert np.all(cov[:3, 3:] == 0.0)
        assert np.all(cov[3:, :3] == 0.0)

    def test_the_result_is_symmetric_and_psd(self):
        cov = np.array(
            seed_post_burn_covariance_km2(_P_KM2, [0.0, 0.0, 0.0], 0.0, 0.0),
            dtype=float)
        assert np.allclose(cov, cov.T)
        assert float(np.linalg.eigvalsh(cov).min()) >= -1e-9

    def test_the_covariance_grows_across_the_horizon(self):
        """SCRUM-452 replaced the constant floor this once asserted.

        Real uncertainty fans out, most of it along-track. A repeated covariance
        was the thing that made the screen falsely confident, so growth is now
        the property under test rather than sameness.
        """
        states = _build(horizon_hours=24.0, step_s=3600.0,
                        dv_eci_km_s=[0.0, 1.0e-4, 0.0],
                        execution_error_magnitude_fraction=0.02,
                        execution_error_pointing_sigma_rad=0.0174532925)["states"]
        traces = [np.trace(np.array(st["covariance"], dtype=float)[:3, :3])
                  for st in states]
        assert traces[-1] > traces[0]
        # Monotone across the window, not merely bigger at the end.
        assert all(b >= a * 0.999 for a, b in zip(traces, traces[1:]))
        # And materially bigger: this is the fan-out the floor did not have.
        assert traces[-1] > 100.0 * traces[0]

    def test_states_no_longer_share_one_covariance_object(self):
        states = _build()["states"]
        assert states[-1]["covariance"] != states[0]["covariance"]


# ---------------------------------------------------------------------------
# The trajectory
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_the_first_state_is_the_post_burn_state_unpropagated(self):
        first = _build()["states"][0]
        assert first["timestamp"] == "2026-09-24T12:00:00Z"
        assert first["position"] == pytest.approx([c * 1000.0 for c in _R_KM])
        assert first["velocity"] == pytest.approx([c * 1000.0 for c in _V_KM_S])

    def test_states_span_the_horizon_at_the_step(self):
        eph = _build(horizon_hours=2.0, step_s=600.0)
        # 2 h at 600 s, endpoint inclusive.
        assert len(eph["states"]) == 13
        assert eph["states"][-1]["timestamp"] == "2026-09-24T14:00:00Z"

    def test_timestamps_are_iso8601_z_and_strictly_increasing(self):
        stamps = [s["timestamp"] for s in _build()["states"]]
        assert all(t.endswith("Z") for t in stamps)
        parsed = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in stamps]
        assert all(b > a for a, b in zip(parsed, parsed[1:]))
        assert all(p.tzinfo == timezone.utc for p in parsed)

    def test_the_states_match_the_j2_propagator_exactly(self):
        """SCRUM-451: the ephemeris is the J2 trajectory, not the two-body one.

        This replaced an assertion against kepler_propagate. The screening file
        deliberately no longer describes the same trajectory the planner's
        internal two-body screen does -- it describes where the asset will
        actually be, which is the whole point of 451.
        """
        eph = _build(horizon_hours=1.0, step_s=900.0)
        offsets = np.array([i * 900.0 for i in range(len(eph["states"]))])
        positions, velocities = propagate_j2_series(
            np.asarray(_R_KM, dtype=float), np.asarray(_V_KM_S, dtype=float), offsets)

        for i, state in enumerate(eph["states"]):
            r = np.asarray(_R_KM) if i == 0 else positions[i]
            v = np.asarray(_V_KM_S) if i == 0 else velocities[i]
            assert state["position"] == pytest.approx([c * 1000.0 for c in r])
            assert state["velocity"] == pytest.approx([c * 1000.0 for c in v])

    def test_a_late_state_differs_from_two_body_by_the_expected_order(self):
        """The regression guard at the builder level.

        If a future edit reverts this path to kepler_propagate, the difference
        collapses to zero and this fails. The magnitudes are the measured
        two-body-vs-J2 divergences that justified SCRUM-451.
        """
        eph = build_screening_ephemeris(_EPOCH, _SWARM_C_R_KM, _SWARM_C_V_KM_S, _P_KM2)
        offsets = [i * DEFAULT_STEP_S for i in range(len(eph["states"]))]

        for hours, low_km, high_km in ((24, 330.0, 620.0), (72, 1000.0, 1900.0)):
            target = hours * 3600.0
            index = offsets.index(target)
            j2_km = np.array(eph["states"][index]["position"]) / 1000.0
            two_body_km, _ = kepler_propagate(
                np.asarray(_SWARM_C_R_KM, dtype=float),
                np.asarray(_SWARM_C_V_KM_S, dtype=float), target)
            divergence = float(np.linalg.norm(j2_km - two_body_km))
            assert low_km < divergence < high_km, (
                f"{hours} h divergence {divergence:.1f} km outside "
                f"[{low_km}, {high_km}] -- has J2 been dropped?")

    def test_the_first_state_is_still_the_unpropagated_epoch_state(self):
        """Unchanged by 451: the file still starts exactly where the asset is."""
        first = build_screening_ephemeris(
            _EPOCH, _SWARM_C_R_KM, _SWARM_C_V_KM_S, _P_KM2)["states"][0]
        assert first["timestamp"] == "2026-09-24T12:00:00Z"
        assert first["position"] == pytest.approx(
            [c * 1000.0 for c in _SWARM_C_R_KM])
        assert first["velocity"] == pytest.approx(
            [c * 1000.0 for c in _SWARM_C_V_KM_S])

    def test_the_trajectory_actually_moves(self):
        """Guards against a propagation that silently returns the initial state."""
        states = _build()["states"]
        moved = math.dist(states[0]["position"], states[-1]["position"])
        assert moved > 1.0e6      # metres

    def test_a_bound_orbit_keeps_its_radius_within_a_sane_band(self):
        radii = [math.sqrt(sum(c * c for c in s["position"]))
                 for s in _build()["states"]]
        assert max(radii) / min(radii) < 1.2

    def test_the_default_window_is_the_scrum_381_screen_window(self):
        """72 h is OperatorPolicy.max_hours_before_tca, locked by the envelope."""
        assert DEFAULT_HORIZON_HOURS == 72.0
        eph = build_screening_ephemeris(_EPOCH, _R_KM, _V_KM_S, _P_KM2)
        expected = int(DEFAULT_HORIZON_HOURS * 3600.0 / DEFAULT_STEP_S) + 1
        assert len(eph["states"]) == expected
        last = datetime.fromisoformat(
            eph["states"][-1]["timestamp"].replace("Z", "+00:00"))
        first = datetime.fromisoformat(
            eph["states"][0]["timestamp"].replace("Z", "+00:00"))
        assert (last - first).total_seconds() == DEFAULT_HORIZON_HOURS * 3600.0

    def test_the_default_file_is_not_needlessly_dense(self):
        """A coarse cadence was a deliberate choice; hold it to under a thousand."""
        eph = build_screening_ephemeris(_EPOCH, _R_KM, _V_KM_S, _P_KM2)
        assert len(eph["states"]) < 1000


# ---------------------------------------------------------------------------
# Refusing to fabricate
# ---------------------------------------------------------------------------

class TestDegenerateCovariance:
    def test_a_missing_covariance_is_refused_not_zero_filled(self):
        """post_burn_covariance_km2 returns None for unknown, and None is not zero.

        Emitting a zero covariance would submit a screening against a trajectory
        declared exactly known, and it would look like a successful submission.
        """
        with pytest.raises(LeoLabsEphemerisError) as excinfo:
            _build(p_post_eci_km2=None)
        assert "not zero" in str(excinfo.value)

    def test_an_all_zero_covariance_is_refused(self):
        """Zero is PSD, but it asserts perfect knowledge, which is never true."""
        with pytest.raises(LeoLabsEphemerisError):
            _build(p_post_eci_km2=np.zeros((3, 3)))

    @pytest.mark.parametrize("bad", [
        [[1.0, 0.0], [0.0, 1.0]],
        np.full((3, 3), np.nan),
        np.diag([np.inf, 1.0, 1.0]),
        np.array([[1.0, 5.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.diag([-1.0, 1.0, 1.0]),
    ])
    def test_a_degenerate_covariance_is_refused(self, bad):
        with pytest.raises(LeoLabsEphemerisError):
            _build(p_post_eci_km2=bad)

    def test_nothing_is_emitted_when_the_covariance_is_refused(self):
        """It fails before propagating, so there is no half-built file."""
        with pytest.raises(LeoLabsEphemerisError):
            _build(p_post_eci_km2=None, horizon_hours=72.0, step_s=1.0)


class TestBadInputs:
    @pytest.mark.parametrize("kw", [
        {"epoch_utc": ""},
        {"epoch_utc": "not-a-date"},
        {"r_sat_km": [1.0, 2.0]},
        {"v_sat_km_s": [1.0, 2.0, 3.0, 4.0]},
        {"r_sat_km": [float("nan"), 0.0, 0.0]},
        {"horizon_hours": 0.0},
        {"horizon_hours": -1.0},
        {"step_s": 0.0},
        {"step_s": -60.0},
    ])
    def test_unusable_inputs_raise(self, kw):
        with pytest.raises(LeoLabsEphemerisError):
            _build(**kw)

    def test_a_step_longer_than_the_horizon_raises(self):
        """Rather than silently emitting a one-state 'trajectory'."""
        with pytest.raises(LeoLabsEphemerisError):
            _build(horizon_hours=1.0, step_s=7200.0)

    def test_a_naive_epoch_is_read_as_utc(self):
        eph = _build(epoch_utc="2026-09-24T12:00:00")
        assert eph["states"][0]["timestamp"] == "2026-09-24T12:00:00Z"

    def test_an_offset_epoch_is_normalised_to_utc(self):
        eph = _build(epoch_utc="2026-09-24T14:00:00+02:00")
        assert eph["states"][0]["timestamp"] == "2026-09-24T12:00:00Z"
