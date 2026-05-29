"""
APS 2.5 — Step 9.3 Test Suite
==============================
File: services/planner/tests/test_constellation_geometry.py

Coverage:
  1. Physics helpers (SMA, circular velocity, mean motion)
  2. j2_raan_rate_deg_per_day — CTO AC: J2 implemented
  3. _return_burn_cost_m_s — §2 corrected formula
  4. WalkerDeltaGeometry construction, derived fields, validation
  5. Slot addressing — epoch-0 RAAN, mean anomaly, IDs
  6. slot_raan_deg_at_time — CTO AC: time-varying slot target
  7. J2 unit test: slot RAAN changes over 24-hour window at 482km/53deg
  8. WalkerSlotAddress construction, delegation, J2 propagation
  9. Drift tolerance (drift_km, is_within_tolerance, days_until_breach)
 10. plan_slot_recovery — AC: valid recovery plan from post-avoidance drift
 11. SlotRecoveryPlan fields, feasibility flags, repr
 12. Reference architectures (starlink_shell_1, leo_constellation_482km)
 13. Non-constellated satellite path (no recovery required)
 14. Error paths and edge cases

Run:
    pytest services/planner/tests/test_constellation_geometry.py -v
"""

import math
import pytest

from common.constellation_geometry import (
    WalkerDeltaGeometry,
    WalkerSlotAddress,
    SlotRecoveryPlan,
    starlink_shell_1,
    leo_constellation_482km,
    j2_raan_rate_deg_per_day,
    _mean_motion_to_sma_km,
    _circular_velocity_km_s,
    _mean_motion_rad_s,
    _return_burn_cost_m_s,
    _along_track_drift_km,
    _MU_KM3_S2,
    _R_EARTH_KM,
    _J2,
    _TWO_PI,
    _SEC_PER_DAY,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def walker_60_6_2() -> WalkerDeltaGeometry:
    """60/6/2 Walker-Delta at 482 km, 53 deg — primary test fixture."""
    return WalkerDeltaGeometry(
        total_satellites=60,
        num_planes=6,
        phasing_parameter=2,
        inclination_deg=53.0,
        altitude_km=482.0,
        mean_motion_rev_per_day=15.3020,
    )


@pytest.fixture
def walker_24_3_1() -> WalkerDeltaGeometry:
    """24/3/1 Walker-Delta — smaller geometry for boundary tests."""
    return WalkerDeltaGeometry(
        total_satellites=24,
        num_planes=3,
        phasing_parameter=1,
        inclination_deg=45.0,
        altitude_km=482.0,
        mean_motion_rev_per_day=15.3020,
    )


# ---------------------------------------------------------------------------
# 1. Physics helpers
# ---------------------------------------------------------------------------

class TestPhysicsHelpers:
    def test_sma_from_mean_motion_physical(self):
        """n^2 * a^3 = mu (Kepler residual < 1e-9 relative)."""
        n_rev_day = 15.3020
        n_rad_s = n_rev_day * _TWO_PI / _SEC_PER_DAY
        sma_km = _mean_motion_to_sma_km(n_rev_day)
        residual = abs(n_rad_s**2 * sma_km**3 - _MU_KM3_S2) / _MU_KM3_S2
        assert residual < 1e-9

    def test_sma_range_for_leo(self):
        """SMA for 15.3020 rev/day should be ~6853 km."""
        sma = _mean_motion_to_sma_km(15.3020)
        assert 6840 < sma < 6870

    def test_sma_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _mean_motion_to_sma_km(0.0)

    def test_sma_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _mean_motion_to_sma_km(-1.0)

    def test_circular_velocity_physical(self):
        """v_c = sqrt(mu / a); check at 482 km altitude."""
        sma_km = _mean_motion_to_sma_km(15.3020)
        vc = _circular_velocity_km_s(sma_km)
        # ISS-class: ~7.6 km/s expected
        assert 7.5 < vc < 7.7

    def test_mean_motion_rad_s_consistent(self):
        """_mean_motion_rad_s should invert _mean_motion_to_sma_km."""
        n_in = 15.3020 * _TWO_PI / _SEC_PER_DAY
        sma = _mean_motion_to_sma_km(15.3020)
        n_out = _mean_motion_rad_s(sma)
        assert abs(n_out - n_in) / n_in < 1e-9

    def test_along_track_drift_zero_delta_n(self):
        assert _along_track_drift_km(0.0, 6853.0, 10.0) == pytest.approx(0.0)

    def test_along_track_drift_linear_in_time(self):
        d1 = _along_track_drift_km(0.01, 6853.0, 1.0)
        d5 = _along_track_drift_km(0.01, 6853.0, 5.0)
        assert d5 == pytest.approx(5 * d1, rel=1e-9)

    def test_along_track_drift_symmetric(self):
        d_pos = _along_track_drift_km(+0.01, 6853.0, 3.0)
        d_neg = _along_track_drift_km(-0.01, 6853.0, 3.0)
        assert d_pos == pytest.approx(d_neg, rel=1e-9)

    def test_along_track_drift_negative_elapsed_raises(self):
        with pytest.raises(ValueError, match="elapsed_days"):
            _along_track_drift_km(0.01, 6853.0, -1.0)


# ---------------------------------------------------------------------------
# 2. J2 RAAN rate — CTO acceptance criterion
# ---------------------------------------------------------------------------

class TestJ2RaanRate:
    def test_negative_for_prograde(self):
        """Prograde orbit (i < 90 deg) must have negative RAAN drift."""
        rate = j2_raan_rate_deg_per_day(6853.0, 53.0)
        assert rate < 0.0

    def test_positive_for_retrograde(self):
        """Retrograde orbit (i > 90 deg) must have positive RAAN drift."""
        rate = j2_raan_rate_deg_per_day(6853.0, 98.0)
        assert rate > 0.0

    def test_zero_at_critical_inclination(self):
        """At i = 90 deg, cos(i) = 0, RAAN drift = 0."""
        rate = j2_raan_rate_deg_per_day(6853.0, 90.0)
        assert abs(rate) < 1e-10

    def test_physical_magnitude_482km_53deg(self):
        """At 482 km, 53 deg: expect -4.66 deg/day (exact Brouwer formula).
        Research doc §6.1 states 'approx -2.0' as illustration;
        the exact formula yields -4.66 deg/day."""
        rate = j2_raan_rate_deg_per_day(6853.352, 53.0)
        assert abs(rate) == pytest.approx(4.663, abs=0.05)
        assert rate < 0.0

    def test_increases_magnitude_at_lower_altitude(self):
        """Lower orbit -> stronger J2 effect -> larger |drift rate|."""
        rate_low = j2_raan_rate_deg_per_day(6700.0, 53.0)
        rate_high = j2_raan_rate_deg_per_day(7000.0, 53.0)
        assert abs(rate_low) > abs(rate_high)

    def test_geometry_carries_j2_rate(self, walker_60_6_2):
        """WalkerDeltaGeometry derives j2_raan_rate_deg_per_day in __post_init__."""
        rate = walker_60_6_2.j2_raan_rate_deg_per_day
        assert rate < 0.0
        assert abs(rate) == pytest.approx(4.663, abs=0.05)


# ---------------------------------------------------------------------------
# 3. Return burn physics — §2 corrected formula
# ---------------------------------------------------------------------------

class TestReturnBurnCost:
    def test_return_equals_avoid_on_circular_orbit(self):
        """dv_return = dv_avoid exactly (§2 fundamental result)."""
        sma = _mean_motion_to_sma_km(15.3020)
        for dv in [0.1, 0.21, 1.0, 2.0, 5.0]:
            dv_ret = _return_burn_cost_m_s(dv, sma)
            assert dv_ret == pytest.approx(dv, rel=1e-6), (
                f"Return ratio ≠ 1 for dv={dv}: got {dv_ret}"
            )

    def test_return_scales_with_dv(self):
        sma = _mean_motion_to_sma_km(15.3020)
        dv1 = _return_burn_cost_m_s(1.0, sma)
        dv2 = _return_burn_cost_m_s(2.0, sma)
        assert dv2 == pytest.approx(2 * dv1, rel=1e-9)

    def test_zero_avoid_zero_return(self):
        sma = _mean_motion_to_sma_km(15.3020)
        assert _return_burn_cost_m_s(0.0, sma) == pytest.approx(0.0)

    def test_negative_dv_raises(self):
        with pytest.raises(ValueError, match="dv_avoid_m_s"):
            _return_burn_cost_m_s(-0.1, 6853.0)

    def test_total_cost_is_2x_avoid(self):
        sma = _mean_motion_to_sma_km(15.3020)
        dv_avoid = 0.21
        dv_ret = _return_burn_cost_m_s(dv_avoid, sma)
        assert (dv_avoid + dv_ret) == pytest.approx(2 * dv_avoid, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. WalkerDeltaGeometry — construction and derived fields
# ---------------------------------------------------------------------------

class TestWalkerDeltaGeometryConstruction:
    def test_basic_fields(self, walker_60_6_2):
        g = walker_60_6_2
        assert g.total_satellites == 60
        assert g.num_planes == 6
        assert g.phasing_parameter == 2
        assert g.inclination_deg == 53.0
        assert g.altitude_km == 482.0

    def test_sats_per_plane(self, walker_60_6_2):
        assert walker_60_6_2.sats_per_plane == 10

    def test_raan_spacing(self, walker_60_6_2):
        assert walker_60_6_2.raan_spacing_deg == pytest.approx(60.0)

    def test_in_plane_spacing(self, walker_60_6_2):
        assert walker_60_6_2.in_plane_spacing_deg == pytest.approx(36.0)

    def test_sma_km_derived(self, walker_60_6_2):
        expected = _mean_motion_to_sma_km(15.3020)
        assert walker_60_6_2.sma_km == pytest.approx(expected, rel=1e-9)

    def test_j2_rate_derived(self, walker_60_6_2):
        expected = j2_raan_rate_deg_per_day(walker_60_6_2.sma_km, 53.0)
        assert walker_60_6_2.j2_raan_rate_deg_per_day == pytest.approx(expected, rel=1e-9)

    def test_frozen(self, walker_60_6_2):
        with pytest.raises((AttributeError, TypeError)):
            walker_60_6_2.total_satellites = 99  # type: ignore[misc]

    def test_repr_contains_j2(self, walker_60_6_2):
        r = repr(walker_60_6_2)
        assert "Omega_dot_J2" in r
        assert "60/6/2" in r


# ---------------------------------------------------------------------------
# 5. Validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def _kw(self, **overrides):
        base = dict(total_satellites=60, num_planes=6, phasing_parameter=2,
                    inclination_deg=53.0, altitude_km=482.0)
        base.update(overrides)
        return base

    def test_zero_satellites(self):
        with pytest.raises(ValueError, match="total_satellites"):
            WalkerDeltaGeometry(**self._kw(total_satellites=0))

    def test_not_divisible(self):
        with pytest.raises(ValueError, match="divisible"):
            WalkerDeltaGeometry(**self._kw(total_satellites=61))

    def test_phasing_too_large(self):
        with pytest.raises(ValueError, match="phasing_parameter"):
            WalkerDeltaGeometry(**self._kw(phasing_parameter=6))

    def test_phasing_negative(self):
        with pytest.raises(ValueError, match="phasing_parameter"):
            WalkerDeltaGeometry(**self._kw(phasing_parameter=-1))

    def test_inclination_zero(self):
        with pytest.raises(ValueError, match="inclination_deg"):
            WalkerDeltaGeometry(**self._kw(inclination_deg=0.0))

    def test_altitude_negative(self):
        with pytest.raises(ValueError, match="altitude_km"):
            WalkerDeltaGeometry(**self._kw(altitude_km=-10.0))

    def test_plane_oob(self, walker_60_6_2):
        with pytest.raises(IndexError, match="plane_idx"):
            walker_60_6_2.slot_raan_deg(6)

    def test_seat_oob(self, walker_60_6_2):
        with pytest.raises(IndexError, match="seat_idx"):
            walker_60_6_2.slot_mean_anomaly_deg(0, 10)


# ---------------------------------------------------------------------------
# 6. Slot addressing — epoch-0
# ---------------------------------------------------------------------------

class TestSlotAddressing:
    def test_raan_plane_0(self, walker_60_6_2):
        assert walker_60_6_2.slot_raan_deg(0) == pytest.approx(0.0)

    def test_raan_plane_3(self, walker_60_6_2):
        assert walker_60_6_2.slot_raan_deg(3) == pytest.approx(180.0)

    def test_raan_plane_5(self, walker_60_6_2):
        assert walker_60_6_2.slot_raan_deg(5) == pytest.approx(300.0)

    def test_mean_anomaly_plane0_seat0(self, walker_60_6_2):
        assert walker_60_6_2.slot_mean_anomaly_deg(0, 0) == pytest.approx(0.0)

    def test_mean_anomaly_plane0_seat1(self, walker_60_6_2):
        assert walker_60_6_2.slot_mean_anomaly_deg(0, 1) == pytest.approx(36.0)

    def test_mean_anomaly_in_range(self, walker_60_6_2):
        for p in range(6):
            for s in range(10):
                ma = walker_60_6_2.slot_mean_anomaly_deg(p, s)
                assert 0.0 <= ma < 360.0

    def test_slot_id_format(self, walker_60_6_2):
        assert walker_60_6_2.slot_id(0, 0) == "P00-S00"
        assert walker_60_6_2.slot_id(5, 9) == "P05-S09"

    def test_all_slot_ids_unique(self, walker_60_6_2):
        ids = [s.slot_id for s in walker_60_6_2.slots()]
        assert len(set(ids)) == 60

    def test_slots_count(self, walker_60_6_2):
        assert len(list(walker_60_6_2.slots())) == 60

    def test_slot_count(self, walker_60_6_2):
        assert walker_60_6_2.slot_count() == 60


# ---------------------------------------------------------------------------
# 7. J2 time-varying slot target — CTO acceptance criterion
# ---------------------------------------------------------------------------

class TestJ2SlotPropagation:
    def test_raan_at_t0_equals_epoch(self, walker_60_6_2):
        """At t=0, J2-propagated RAAN must equal epoch-0 RAAN."""
        for p in range(6):
            raan_epoch = walker_60_6_2.slot_raan_deg(p)
            raan_t0 = walker_60_6_2.slot_raan_deg_at_time(p, 0.0)
            assert raan_t0 == pytest.approx(raan_epoch, abs=1e-9)

    def test_raan_changes_over_24h(self, walker_60_6_2):
        """CTO AC: slot RAAN changes over 24-hour window at 482 km, 53 deg.

        J2 rate ≈ -4.66 deg/day, so 24h drift ≈ 4.66 deg.
        The test asserts the drift is non-zero and in the expected direction.
        """
        raan_t0 = walker_60_6_2.slot_raan_deg_at_time(0, 0.0)
        raan_t24 = walker_60_6_2.slot_raan_deg_at_time(0, 86400.0)

        # Compute signed difference (accounting for wrap-around)
        delta = (raan_t24 - raan_t0 + 180.0) % 360.0 - 180.0

        # Must be non-zero (the point of the test)
        assert abs(delta) > 1.0, (
            f"Slot RAAN should change by ~4.66 deg in 24h but changed by {delta:.4f} deg"
        )
        # Must be negative (prograde orbit, J2 regresses RAAN westward)
        assert delta < 0.0, (
            f"J2 should cause negative RAAN drift for prograde orbit, got {delta:.4f} deg"
        )
        # Must be in the physically correct range
        assert abs(delta) == pytest.approx(4.663, abs=0.1), (
            f"Expected ~4.663 deg/day at 482km/53deg, got {abs(delta):.4f}"
        )

    def test_raan_drift_proportional_to_time(self, walker_60_6_2):
        """J2 drift is linear in time (first-order secular term)."""
        r1 = walker_60_6_2.slot_raan_deg_at_time(0, 3600.0)
        r2 = walker_60_6_2.slot_raan_deg_at_time(0, 7200.0)
        r0 = walker_60_6_2.slot_raan_deg_at_time(0, 0.0)

        drift_1h = (r1 - r0 + 180) % 360 - 180
        drift_2h = (r2 - r0 + 180) % 360 - 180
        assert drift_2h == pytest.approx(2 * drift_1h, rel=1e-6)

    def test_all_planes_drift_same_rate(self, walker_60_6_2):
        """All planes drift at the same J2 rate (same orbit, same physics)."""
        drifts = []
        for p in range(6):
            r0 = walker_60_6_2.slot_raan_deg_at_time(p, 0.0)
            r24 = walker_60_6_2.slot_raan_deg_at_time(p, 86400.0)
            d = (r24 - r0 + 180) % 360 - 180
            drifts.append(d)
        # All planes drift by the same amount
        for d in drifts:
            assert d == pytest.approx(drifts[0], abs=1e-9)

    def test_raan_output_in_0_360(self, walker_60_6_2):
        """Output always normalised to [0, 360)."""
        for elapsed in [0, 86400, 30 * 86400, 365 * 86400]:
            raan = walker_60_6_2.slot_raan_deg_at_time(0, elapsed)
            assert 0.0 <= raan < 360.0

    def test_walker_slot_address_j2_delegation(self, walker_60_6_2):
        """WalkerSlotAddress.target_raan_at_time delegates to geometry."""
        slot = WalkerSlotAddress(walker_60_6_2, plane_idx=2, seat_idx=3)
        direct = walker_60_6_2.slot_raan_deg_at_time(2, 86400.0)
        via_slot = slot.target_raan_at_time(86400.0)
        assert direct == pytest.approx(via_slot, rel=1e-9)


# ---------------------------------------------------------------------------
# 8. WalkerSlotAddress
# ---------------------------------------------------------------------------

class TestWalkerSlotAddress:
    def test_slot_address_raan(self, walker_60_6_2):
        slot = WalkerSlotAddress(walker_60_6_2, plane_idx=2, seat_idx=0)
        assert slot.target_raan_deg == pytest.approx(120.0)

    def test_slot_address_id(self, walker_60_6_2):
        slot = WalkerSlotAddress(walker_60_6_2, plane_idx=3, seat_idx=7)
        assert slot.slot_id == "P03-S07"

    def test_slot_address_frozen(self, walker_60_6_2):
        slot = WalkerSlotAddress(walker_60_6_2, plane_idx=0, seat_idx=0)
        with pytest.raises((AttributeError, TypeError)):
            slot.plane_idx = 9  # type: ignore[misc]

    def test_slot_address_repr(self, walker_60_6_2):
        slot = WalkerSlotAddress(walker_60_6_2, plane_idx=1, seat_idx=2)
        r = repr(slot)
        assert "P01-S02" in r


# ---------------------------------------------------------------------------
# 9. Drift tolerance
# ---------------------------------------------------------------------------

class TestDriftTolerance:
    def test_zero_drift_on_target(self, walker_60_6_2):
        assert walker_60_6_2.along_track_drift_km(15.3020, 10.0) == pytest.approx(0.0)

    def test_within_tolerance_on_target(self, walker_60_6_2):
        assert walker_60_6_2.is_within_drift_tolerance(15.3020, 10.0) is True

    def test_outside_tolerance_large_delta_n(self, walker_60_6_2):
        assert walker_60_6_2.is_within_drift_tolerance(
            15.3020 + 0.05, elapsed_days=30.0, acceptable_drift_km=4.459
        ) is False

    def test_days_until_breach_none_for_zero_delta_n(self, walker_60_6_2):
        assert walker_60_6_2.days_until_tolerance_breach(15.3020) is None

    def test_days_until_breach_positive(self, walker_60_6_2):
        days = walker_60_6_2.days_until_tolerance_breach(15.3025)
        assert days is not None and days > 0.0

    def test_breach_time_symmetric(self, walker_60_6_2):
        hi = walker_60_6_2.days_until_tolerance_breach(15.3020 + 0.005)
        lo = walker_60_6_2.days_until_tolerance_breach(15.3020 - 0.005)
        assert hi == pytest.approx(lo, rel=1e-9)

    def test_slot_drift_delegates(self, walker_60_6_2):
        slot = WalkerSlotAddress(walker_60_6_2, 0, 0)
        direct = walker_60_6_2.along_track_drift_km(15.3025, 5.0)
        via_slot = slot.along_track_drift_km(15.3025, 5.0)
        assert direct == pytest.approx(via_slot)


# ---------------------------------------------------------------------------
# 10. plan_slot_recovery — AC: valid recovery plan
# ---------------------------------------------------------------------------

class TestSlotRecovery:
    def test_recovery_required_above_threshold(self, walker_60_6_2):
        """Drift > acceptable_drift_km -> recovery_required = True."""
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=0, seat_idx=0,
            dv_avoid_m_s=0.21,
            post_maneuver_drift_km=5.0,    # > 4.459
            acceptable_drift_km=4.459,
        )
        assert plan.recovery_required is True

    def test_no_recovery_below_threshold(self, walker_60_6_2):
        """Drift <= acceptable_drift_km -> recovery_required = False."""
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=0, seat_idx=0,
            dv_avoid_m_s=0.10,
            post_maneuver_drift_km=3.0,    # < 4.459
            acceptable_drift_km=4.459,
        )
        assert plan.recovery_required is False
        assert plan.dv_return_m_s == pytest.approx(0.0)
        assert plan.dv_total_m_s == pytest.approx(0.10)

    def test_dv_total_is_2x_avoid_when_recovery_required(self, walker_60_6_2):
        """Total cost = 2 * dv_avoid when recovery required (§2 result)."""
        dv_avoid = 0.21
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=1, seat_idx=3,
            dv_avoid_m_s=dv_avoid,
            post_maneuver_drift_km=5.0,
            acceptable_drift_km=4.459,
        )
        assert plan.dv_total_m_s == pytest.approx(2 * dv_avoid, rel=1e-5)
        assert plan.dv_return_m_s == pytest.approx(dv_avoid, rel=1e-5)

    def test_target_raan_at_epoch_0(self, walker_60_6_2):
        """At recovery_epoch_offset_s=0, target RAAN equals epoch-0 value."""
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=2, seat_idx=0,
            dv_avoid_m_s=0.21,
            post_maneuver_drift_km=5.0,
            recovery_epoch_offset_s=0.0,
        )
        assert plan.target_raan_deg == pytest.approx(
            walker_60_6_2.slot_raan_deg(2), abs=1e-9
        )

    def test_target_raan_j2_corrected_at_24h(self, walker_60_6_2):
        """After 24 h, recovery target RAAN differs from epoch-0 by ~4.66 deg."""
        plan_t0 = walker_60_6_2.plan_slot_recovery(
            plane_idx=0, seat_idx=0, dv_avoid_m_s=0.21,
            post_maneuver_drift_km=5.0, recovery_epoch_offset_s=0.0,
        )
        plan_t24 = walker_60_6_2.plan_slot_recovery(
            plane_idx=0, seat_idx=0, dv_avoid_m_s=0.21,
            post_maneuver_drift_km=5.0, recovery_epoch_offset_s=86400.0,
        )
        delta = (plan_t24.target_raan_deg - plan_t0.target_raan_deg + 180) % 360 - 180
        assert abs(delta) == pytest.approx(4.663, abs=0.1)
        assert delta < 0.0  # prograde: RAAN regresses

    def test_slot_id_in_plan(self, walker_60_6_2):
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=3, seat_idx=5, dv_avoid_m_s=0.5,
            post_maneuver_drift_km=6.0,
        )
        assert plan.slot_id == "P03-S05"

    def test_budget_feasible_flag(self, walker_60_6_2):
        """budget_feasible = True when dv_return <= return_dv_budget_m_s."""
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=0, seat_idx=0, dv_avoid_m_s=2.0,
            post_maneuver_drift_km=5.0,
            return_dv_budget_m_s=4.4,
        )
        # dv_return ≈ 2.0 m/s < 4.4 m/s budget
        assert plan.budget_feasible is True

    def test_budget_infeasible_flag(self, walker_60_6_2):
        """budget_feasible = False when dv_return > return_dv_budget_m_s."""
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=0, seat_idx=0, dv_avoid_m_s=5.0,
            post_maneuver_drift_km=10.0,
            return_dv_budget_m_s=4.4,
        )
        # dv_return ≈ 5.0 m/s > 4.4 m/s budget
        assert plan.budget_feasible is False

    def test_within_max_recovery_time(self, walker_60_6_2):
        """One orbit ~5,640 s; default max_recovery_time_s=86400 -> True."""
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=0, seat_idx=0, dv_avoid_m_s=0.21,
            post_maneuver_drift_km=5.0, max_recovery_time_s=86400.0,
        )
        assert plan.within_max_recovery_time is True
        assert plan.recovery_time_s < 86400.0

    def test_exceeds_max_recovery_time(self, walker_60_6_2):
        """Max recovery time tighter than one orbit -> within = False."""
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=0, seat_idx=0, dv_avoid_m_s=0.21,
            post_maneuver_drift_km=5.0, max_recovery_time_s=60.0,
        )
        assert plan.within_max_recovery_time is False

    def test_walkerslotaddress_plan_delegation(self, walker_60_6_2):
        """WalkerSlotAddress.plan_slot_recovery delegates to geometry."""
        slot = WalkerSlotAddress(walker_60_6_2, plane_idx=1, seat_idx=2)
        plan = slot.plan_slot_recovery(
            dv_avoid_m_s=0.21, post_maneuver_drift_km=5.0
        )
        assert isinstance(plan, SlotRecoveryPlan)
        assert plan.slot_id == "P01-S02"

    def test_plan_repr_contains_status(self, walker_60_6_2):
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=0, seat_idx=0, dv_avoid_m_s=0.21,
            post_maneuver_drift_km=5.0,
        )
        r = repr(plan)
        assert "RECOVERY REQUIRED" in r
        assert "FEASIBLE" in r


# ---------------------------------------------------------------------------
# 11. Non-constellated satellite path
# ---------------------------------------------------------------------------

class TestNonConstellated:
    def test_no_recovery_zero_drift(self, walker_60_6_2):
        """Zero post-maneuver drift: no recovery, dv_return = 0."""
        plan = walker_60_6_2.plan_slot_recovery(
            plane_idx=0, seat_idx=0, dv_avoid_m_s=0.5,
            post_maneuver_drift_km=0.0, acceptable_drift_km=4.459,
        )
        assert plan.recovery_required is False
        assert plan.dv_return_m_s == pytest.approx(0.0)
        assert plan.dv_total_m_s == pytest.approx(0.5)
        assert plan.recovery_orbits == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 12. Reference architectures
# ---------------------------------------------------------------------------

class TestReferenceArchitectures:
    def test_starlink_shell_1_geometry(self):
        g = starlink_shell_1()
        assert g.total_satellites == 1584
        assert g.num_planes == 72
        assert g.phasing_parameter == 22
        assert g.inclination_deg == 53.0
        assert g.altitude_km == 482.0
        assert g.mean_motion_rev_per_day == pytest.approx(15.3020)

    def test_starlink_sats_per_plane(self):
        assert starlink_shell_1().sats_per_plane == 22

    def test_starlink_j2_rate(self):
        """Starlink Shell 1 has same J2 rate as our test fixture (same orbit)."""
        rate = starlink_shell_1().j2_raan_rate_deg_per_day
        assert abs(rate) == pytest.approx(4.663, abs=0.05)

    def test_starlink_slot_count(self):
        assert starlink_shell_1().slot_count() == 1584

    def test_leo_482km_factory(self):
        g = leo_constellation_482km()
        assert isinstance(g, WalkerDeltaGeometry)
        assert g.altitude_km == 482.0
        assert g.mean_motion_rev_per_day == pytest.approx(15.3020)

    def test_both_factories_same_j2_rate(self):
        """Both reference architectures share the same altitude/inclination."""
        r1 = starlink_shell_1().j2_raan_rate_deg_per_day
        r2 = leo_constellation_482km().j2_raan_rate_deg_per_day
        assert r1 == pytest.approx(r2, rel=1e-6)
