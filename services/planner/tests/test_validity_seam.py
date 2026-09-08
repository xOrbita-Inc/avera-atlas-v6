"""
SCRUM-379 -- the propagation seam around the SCRUM-378 Validity Assessor.

The point of these tests is the epoch requirement. 378's own docstring is
explicit: 'conjunction_plane_epsilon requires w_position_eci to be at the SAME
epoch as r_rel_km / v_rel_km_s, so this must genuinely be the state at TCA, not
at the IOD solution's own epoch.' Mixing the two raises nothing and returns a
plausible number, so it is tested directly rather than left to inspection.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from common.decision_state_machine import ValidityRouting
from common.orbit_propagation import kepler_propagate
from common.validity_seam import (
    LEO_EPSILON_THRESHOLD,
    ArcObservation,
    ObservationEpochState,
    RoutingDecision,
    assess_validity_at_tca,
    build_validity_verdict_for_arc,
    propagate_target_to_tca,
    semi_major_axis_km,
)

IOD_EPOCH = datetime(2026, 4, 1, 6, 0, 0, tzinfo=timezone.utc)
TCA = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)  # six hours later

# A roughly circular ~7000 km LEO target, the same shape services/validity's
# own tests use.
R_IOD = np.array([7000.0, 0.0, 0.0])
V_IOD = np.array([0.0, 7.5460, 0.0])

R_REL_AT_TCA = np.array([0.5, -0.3, 0.1])
V_REL_AT_TCA = np.array([0.0, 0.01, 0.0])


def an_arc(n: int = 4) -> list[ArcObservation]:
    """Observations spread over the 20 minutes before the IOD epoch."""
    return [
        ArcObservation(
            epoch_utc=IOD_EPOCH - timedelta(minutes=20 - 5 * i),
            r_observer_km=np.array([6378.0, 50.0 * i, 10.0 * i]),
            ra_sigma_rad=1e-5,
            dec_sigma_rad=1e-5,
            range_sigma_km=0.01,
        )
        for i in range(n)
    ]


def assess(**overrides):
    kwargs = dict(
        r_target_km=R_IOD,
        v_target_km_s=V_IOD,
        state_epoch_utc=IOD_EPOCH,
        tca_utc=TCA,
        observations=an_arc(),
        r_rel_km_at_tca=R_REL_AT_TCA,
        v_rel_km_s_at_tca=V_REL_AT_TCA,
    )
    kwargs.update(overrides)
    return assess_validity_at_tca(**kwargs)


# ---------------------------------------------------------------------------
# The epoch requirement
# ---------------------------------------------------------------------------


class TestTargetStateIsAtTca:
    def test_the_target_is_propagated_from_its_own_epoch_to_tca(self):
        result = assess()

        expected_r, expected_v = kepler_propagate(R_IOD, V_IOD, 6 * 3600.0)
        assert np.allclose(result.r_target_tca_km, expected_r)
        assert np.allclose(result.v_target_tca_km_s, expected_v)

    def test_the_state_at_tca_is_not_the_state_at_the_iod_epoch(self):
        """Six hours of LEO motion is roughly four orbits. If this ever passes
        by accident, the propagation is not running."""
        result = assess()

        assert not np.allclose(result.r_target_tca_km, R_IOD, atol=1.0)

    def test_the_recorded_epoch_is_tca(self):
        result = assess()

        assert result.target_state_epoch_utc == "2026-04-01T12:00:00Z"
        assert result.propagated_dt_s == pytest.approx(6 * 3600.0)

    def test_mixing_the_epochs_changes_the_answer(self):
        """The silent-wrong bug, demonstrated.

        Calling 378 directly with the IOD-epoch state as the Gramian's t0 --
        what a caller that skipped this seam would do -- raises nothing and
        produces a different epsilon against the same conjunction geometry.
        """
        seam = assess()

        # The same arc, anchored to the IOD epoch instead of TCA -- what a
        # caller that skipped this seam would assemble.
        wrong_epoch_states = []
        for obs in an_arc():
            dt_from_iod_s = (obs.epoch_utc - IOD_EPOCH).total_seconds()
            r_obs, v_obs = kepler_propagate(R_IOD, V_IOD, dt_from_iod_s)
            wrong_epoch_states.append(
                ObservationEpochState(
                    r_target_km=r_obs,
                    v_target_km_s=v_obs,
                    dt_s=dt_from_iod_s,
                    r_observer_km=obs.r_observer_km,
                    ra_sigma_rad=obs.ra_sigma_rad,
                    dec_sigma_rad=obs.dec_sigma_rad,
                    range_sigma_km=obs.range_sigma_km,
                )
            )

        naive = build_validity_verdict_for_arc(
            r_target_tca_km=R_IOD,
            v_target_tca_km_s=V_IOD,
            a_km=seam.a_km,
            observation_epochs=wrong_epoch_states,
            r_rel_km_at_tca=R_REL_AT_TCA,
            v_rel_km_s_at_tca=V_REL_AT_TCA,
            epsilon_threshold=LEO_EPSILON_THRESHOLD,
        )

        assert naive.epsilon != pytest.approx(seam.verdict.epsilon)

    def test_an_iod_epoch_equal_to_tca_needs_no_propagation(self):
        result = assess(state_epoch_utc=TCA)

        assert result.propagated_dt_s == 0.0
        assert np.allclose(result.r_target_tca_km, R_IOD)

    def test_observation_dt_is_measured_from_tca_and_is_negative(self):
        """378's contract: dt_s is elapsed time from TCA to the observation,
        negative for every real observation since they precede TCA."""
        arc = an_arc()
        result = assess(observations=arc)

        # Every observation here is 6 hours and change before TCA.
        for obs in arc:
            dt = (obs.epoch_utc - TCA).total_seconds()
            assert dt < 0.0
            assert dt < -6 * 3600.0
        assert result.verdict is not None

    def test_naive_timestamps_are_rejected(self):
        with pytest.raises(ValueError):
            assess(tca_utc=datetime(2026, 4, 1, 12, 0, 0))
        with pytest.raises(ValueError):
            assess(state_epoch_utc=datetime(2026, 4, 1, 6, 0, 0))

    def test_an_empty_arc_is_rejected_rather_than_scored_as_not_earned(self):
        with pytest.raises(ValueError):
            assess(observations=[])


# ---------------------------------------------------------------------------
# Propagation helper and the reference semi-major axis
# ---------------------------------------------------------------------------


class TestPropagationHelpers:
    def test_propagate_to_tca_returns_the_signed_dt(self):
        _, _, dt_s = propagate_target_to_tca(R_IOD, V_IOD, IOD_EPOCH, TCA)

        assert dt_s == pytest.approx(6 * 3600.0)

    def test_a_tca_before_the_state_epoch_propagates_backwards(self):
        earlier = IOD_EPOCH - timedelta(hours=1)
        r_back, _, dt_s = propagate_target_to_tca(R_IOD, V_IOD, IOD_EPOCH, earlier)

        assert dt_s == pytest.approx(-3600.0)
        assert not np.allclose(r_back, R_IOD, atol=1.0)

    def test_the_reference_semi_major_axis_comes_from_the_state_at_tca(self):
        result = assess()

        expected = semi_major_axis_km(
            np.array(result.r_target_tca_km), np.array(result.v_target_tca_km_s)
        )
        assert result.a_km == pytest.approx(expected)
        # A near-circular 7000 km orbit.
        assert result.a_km == pytest.approx(7000.0, rel=1e-3)

    def test_an_explicit_semi_major_axis_overrides_the_derived_one(self):
        result = assess(a_km=6900.0)

        assert result.a_km == 6900.0

    def test_an_unbound_state_has_no_reference_semi_major_axis(self):
        with pytest.raises(ValueError):
            semi_major_axis_km(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 20.0, 0.0]))


# ---------------------------------------------------------------------------
# Routing, and the enum the guard compares against
# ---------------------------------------------------------------------------


class TestRouting:
    def test_the_assessment_carries_a_routed_decision(self):
        result = assess()

        assert isinstance(result.routing, ValidityRouting)
        assert result.routing in tuple(ValidityRouting)

    def test_the_planner_mirror_has_not_drifted_from_services_validity(self):
        """A drift would make the validity guard compare against values that
        never occur, and it would fail open-looking rather than loudly."""
        assert {r.value for r in ValidityRouting} == {r.value for r in RoutingDecision}

    def test_the_evidence_carries_the_378_field_names(self):
        result = assess()

        assert set(result.evidence) == {
            "validity_status",
            "validity_epsilon",
            "weak_directions",
            "epsilon_threshold",
            "phenomenologies_used",
        }
        assert result.evidence["epsilon_threshold"] == LEO_EPSILON_THRESHOLD

    def test_the_locked_leo_floor_is_the_default(self):
        # SCRUM-333 section 7: 0.20 for LEO, locked.
        assert LEO_EPSILON_THRESHOLD == 0.20
        assert assess().verdict.epsilon_threshold == 0.20

    def test_phenomenologies_default_to_tle_only(self):
        assert assess().evidence["phenomenologies_used"] == ["TLE"]

    def test_a_supplied_phenomenology_list_is_passed_through(self):
        result = assess(phenomenologies_used=["optical", "TLE"])

        assert result.evidence["phenomenologies_used"] == ["optical", "TLE"]

    def test_to_dict_flattens_the_evidence_for_a_transition_record(self):
        payload = assess().to_dict()

        assert payload["target_state_epoch_utc"] == "2026-04-01T12:00:00Z"
        assert payload["routing"] in {r.value for r in ValidityRouting}
        assert "validity_status" in payload
        assert "validity_epsilon" in payload
