"""Tests for services/validity/wrapper.py, SCRUM-378.

Covers the rewritten (post-John's Option 2) wrapper: no propagation, no
tracker/iod.py import, caller supplies every state pre-propagated.
"""
from __future__ import annotations

import numpy as np
import pytest

from aps_math.observability import ValidityStatus, ValidityVerdict
from wrapper import (
    ObservationEpochState,
    build_validity_verdict_for_arc,
    validity_evidence_values,
)

# A simple, reusable LEO-like geometry. Not meant to be physically
# exact, just consistent enough to exercise the wiring and boundary
# behavior. a_km/r_tca/v_tca describe a roughly circular ~7000 km orbit.
_A_KM = 7000.0
_R_TCA = np.array([7000.0, 0.0, 0.0])
_V_TCA = np.array([0.0, 7.5, 0.0])


def _good_arc(n_obs: int = 3, ra_sigma=1e-5, dec_sigma=1e-5) -> list[ObservationEpochState]:
    """A well-spread arc: observer moves, dt_s varies -- should produce
    good observability (EARNED) under the module's own conventions."""
    obs = []
    for i in range(n_obs):
        dt_s = -600.0 + i * 200.0  # spread over 10 minutes before TCA
        obs.append(
            ObservationEpochState(
                r_target_km=_R_TCA + np.array([-5.0 * (i + 1), -20.0 * (i + 1), 0.0]),
                v_target_km_s=_V_TCA + np.array([0.02 * (i + 1), 0.0, 0.0]),
                dt_s=dt_s,
                r_observer_km=np.array([6378.0, 50.0 * i, 0.0]),
                ra_sigma_rad=ra_sigma,
                dec_sigma_rad=dec_sigma,
                range_sigma_km=0.01,
            )
        )
    return obs


class TestNoPropagationOrTrackerCoupling:
    """The whole point of the rewrite: confirm the module surface itself
    doesn't smuggle in a propagator or a tracker import."""

    def test_wrapper_module_does_not_import_iod(self):
        import wrapper

        assert "iod" not in dir(wrapper)
        assert not hasattr(wrapper, "kepler_propagate")
        assert not hasattr(wrapper, "_kepler_propagate")

    def test_observation_epoch_state_has_no_timestamp_field(self):
        """Confirms the data contract itself: dt_s is a plain number,
        not a datetime the wrapper would need to do arithmetic on."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ObservationEpochState)}
        assert "timestamp" not in fields
        assert "epoch" not in fields
        assert "dt_s" in fields


class TestBasicWiring:
    def test_produces_a_verdict_with_expected_fields(self):
        verdict = build_validity_verdict_for_arc(
            r_target_tca_km=_R_TCA,
            v_target_tca_km_s=_V_TCA,
            a_km=_A_KM,
            observation_epochs=_good_arc(),
            r_rel_km_at_tca=np.array([0.5, -0.3, 0.1]),
            v_rel_km_s_at_tca=np.array([0.0, 0.01, 0.0]),
            epsilon_threshold=0.20,
        )
        assert verdict.status in (ValidityStatus.EARNED, ValidityStatus.NOT_EARNED)
        assert isinstance(verdict.epsilon, float)
        assert verdict.epsilon_threshold == 0.20
        assert isinstance(verdict.weak_directions, list)
        assert verdict.phenomenologies_used == ["TLE"]

    def test_default_phenomenologies_is_tle_only(self):
        verdict = build_validity_verdict_for_arc(
            r_target_tca_km=_R_TCA,
            v_target_tca_km_s=_V_TCA,
            a_km=_A_KM,
            observation_epochs=_good_arc(),
            r_rel_km_at_tca=np.array([0.5, -0.3, 0.1]),
            v_rel_km_s_at_tca=np.array([0.0, 0.01, 0.0]),
            epsilon_threshold=0.20,
            phenomenologies_used=None,
        )
        assert verdict.phenomenologies_used == ["TLE"]

    def test_explicit_phenomenologies_are_passed_through(self):
        verdict = build_validity_verdict_for_arc(
            r_target_tca_km=_R_TCA,
            v_target_tca_km_s=_V_TCA,
            a_km=_A_KM,
            observation_epochs=_good_arc(),
            r_rel_km_at_tca=np.array([0.5, -0.3, 0.1]),
            v_rel_km_s_at_tca=np.array([0.0, 0.01, 0.0]),
            epsilon_threshold=0.20,
            phenomenologies_used=["optical", "RF"],
        )
        assert verdict.phenomenologies_used == ["optical", "RF"]

    def test_to_dict_matches_gnc_interface_shape(self):
        verdict = build_validity_verdict_for_arc(
            r_target_tca_km=_R_TCA,
            v_target_tca_km_s=_V_TCA,
            a_km=_A_KM,
            observation_epochs=_good_arc(),
            r_rel_km_at_tca=np.array([0.5, -0.3, 0.1]),
            v_rel_km_s_at_tca=np.array([0.0, 0.01, 0.0]),
            epsilon_threshold=0.20,
        )
        d = verdict.to_dict()
        assert set(d.keys()) == {
            "status", "epsilon", "epsilon_threshold",
            "weak_directions", "phenomenologies_used",
        }


class TestArcQualityAffectsEpsilon:
    """Sanity-check the wiring end-to-end: the Gramian assembly must
    actually be sensitive to the inputs, not silently ignoring them.

    Deliberately NOT asserting a specific direction (e.g. "more
    observations always means higher epsilon") with synthetic numbers --
    a single observation can legitimately make W_CP singular (a real,
    documented failure mode in conjunction_plane_epsilon, not a wrapper
    bug), and asserting a monotonic epsilon trend from made-up geometry
    would be asserting physics we haven't independently verified for
    these specific toy numbers. What's actually being checked here is
    narrower and safe to assert: the wrapper propagates whatever
    build_validity_verdict/conjunction_plane_epsilon does, without
    swallowing or distorting it.
    """

    def test_well_spread_three_observation_arc_produces_a_defined_epsilon(self):
        """Three observations spread over time with non-repeating
        observer positions should give a well-posed (non-singular)
        information matrix -- this is the ordinary, expected case, not
        an edge case, so it should not raise."""
        verdict = build_validity_verdict_for_arc(
            r_target_tca_km=_R_TCA,
            v_target_tca_km_s=_V_TCA,
            a_km=_A_KM,
            observation_epochs=_good_arc(n_obs=3),
            r_rel_km_at_tca=np.array([0.5, -0.3, 0.1]),
            v_rel_km_s_at_tca=np.array([0.0, 0.01, 0.0]),
            epsilon_threshold=0.20,
        )
        assert verdict.epsilon >= 0.0

    def test_single_degenerate_observation_raises_not_silently_wrong(self):
        """A single observation can leave the conjunction-plane
        information matrix singular. The wrapper must let that raise
        (per conjunction_plane_epsilon's own contract), not catch it and
        return a fabricated epsilon."""
        with pytest.raises(ValueError):
            build_validity_verdict_for_arc(
                r_target_tca_km=_R_TCA,
                v_target_tca_km_s=_V_TCA,
                a_km=_A_KM,
                observation_epochs=_good_arc(n_obs=1),
                r_rel_km_at_tca=np.array([0.5, -0.3, 0.1]),
                v_rel_km_s_at_tca=np.array([0.0, 0.01, 0.0]),
                epsilon_threshold=0.20,
            )


class TestValidityEvidenceValues:
    """The mapping to evidence_record.py's catalogue field names.

    The key thing under test: two of the five catalogue field names
    (validity_status, validity_epsilon) DIFFER from ValidityVerdict's
    own attribute/to_dict() names (status, epsilon). A regression here
    would silently produce a dict that either has the wrong keys (and
    EvidenceRecord.build's `unknown field` check would catch that
    loudly) or, worse, get merged past that check some other way.
    Pinning the exact key set and exact key names directly.
    """

    def _verdict(self, status=ValidityStatus.EARNED, epsilon=0.73):
        return ValidityVerdict(
            status=status,
            epsilon=epsilon,
            epsilon_threshold=0.20,
            weak_directions=["radial"],
            phenomenologies_used=["TLE"],
        )

    def test_returns_exactly_the_five_catalogue_keys(self):
        d = validity_evidence_values(self._verdict())
        assert set(d.keys()) == {
            "validity_status", "validity_epsilon", "weak_directions",
            "epsilon_threshold", "phenomenologies_used",
        }

    def test_status_and_epsilon_use_the_prefixed_catalogue_names(self):
        """The specific renaming this function exists for: NOT "status"
        and "epsilon" (ValidityVerdict.to_dict()'s names), but
        "validity_status" and "validity_epsilon" (the catalogue's
        names)."""
        d = validity_evidence_values(self._verdict())
        assert "status" not in d
        assert "epsilon" not in d
        assert "validity_status" in d
        assert "validity_epsilon" in d

    def test_status_is_the_plain_string_value_not_the_enum_member(self):
        d = validity_evidence_values(self._verdict(status=ValidityStatus.NOT_EARNED))
        assert d["validity_status"] == "NOT_EARNED"
        assert not isinstance(d["validity_status"], ValidityStatus)

    def test_values_match_the_source_verdict(self):
        v = self._verdict(status=ValidityStatus.EARNED, epsilon=0.55)
        d = validity_evidence_values(v)
        assert d["validity_epsilon"] == 0.55
        assert d["epsilon_threshold"] == 0.20
        assert d["weak_directions"] == ["radial"]
        assert d["phenomenologies_used"] == ["TLE"]

    def test_preserves_an_empty_weak_directions_list_as_is(self):
        """This function is a pure pass-through mapper, not an enforcer
        -- build_validity_verdict is what decides whether
        weak_directions is empty on EARNED (already tested elsewhere in
        test_observability.py). Here, just confirm this function
        doesn't fabricate or drop an empty list."""
        v = ValidityVerdict(
            status=ValidityStatus.EARNED,
            epsilon=0.73,
            epsilon_threshold=0.20,
            weak_directions=[],
            phenomenologies_used=["TLE"],
        )
        d = validity_evidence_values(v)
        assert d["weak_directions"] == []


class TestInvalidPhenomenologyPropagates:
    def test_invalid_phenomenology_raises(self):
        with pytest.raises(ValueError):
            build_validity_verdict_for_arc(
                r_target_tca_km=_R_TCA,
                v_target_tca_km_s=_V_TCA,
                a_km=_A_KM,
                observation_epochs=_good_arc(),
                r_rel_km_at_tca=np.array([0.5, -0.3, 0.1]),
                v_rel_km_s_at_tca=np.array([0.0, 0.01, 0.0]),
                epsilon_threshold=0.20,
                phenomenologies_used=["not_a_real_sensor"],
            )
