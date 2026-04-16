"""
APS 2.5 — Step 9.3: Constellation Architecture Reasoning
=========================================================
Module: services/planner/common/constellation_geometry.py

Implements Walker-Delta constellation geometry, slot addressing,
along-track drift tolerance, J2 RAAN perturbation, and return-to-slot
planning.  Depends on:
  - 9.1  services/planner/common/satellite_capability.py  (ConstellationSlot)
  - 9.2  services/planner/common/operator_policy.py       (OperatorPolicy)

Design decisions (APS_2_5_Research_V3.ipynb §3)
------------------------------------------------
  - LEO Walker-delta only for v2.5.  constellation_type enum reserved
    for v2.6 extensibility per §3.3.
  - Two-path methodology: Space-Track empirical defaults for large
    constellations; operator-supplied YAML for smaller operators.
    Both paths produce identical schema.
  - Neighbors are NOT explicitly modelled.  All reasoning is over the
    higher-level (plane_idx, seat_idx) slot abstraction.

Scientific gaps (APS_2_5_Research_V3.ipynb §6)
-----------------------------------------------
  - J2 RAAN drift:        IMPLEMENTED (CTO acceptance criterion, §6.1).
  - Atmospheric drag:     NOT implemented.  The TLE-derived
                          acceptable_drift_km = 4.459 km implicitly
                          encodes drag effects already present in the
                          operational TLE history.  The formal drag
                          term in the return cost model is APS 3.0 scope.
  - Covariance prop.:     NOT implemented (APS 3.0 scope).

Return burn math (APS_2_5_Research_V3.ipynb §2, corrected form)
----------------------------------------------------------------
  Gauss variational equations for a tangential impulse on circular orbit:
      delta_a       = 2a / v_c * dv_avoid
      dv_return     = n * delta_a / 2  =  dv_avoid   (exactly)
  Total cost = 2 * dv_avoid for constellation members.
  Non-constellated satellites: dv_return = 0.

References
----------
Walker, J.G. (1984). Satellite constellations. JBIS 37, 559-571.
Vallado, D.A. (2013). Fundamentals of Astrodynamics, 4e.
Brouwer, D. (1959). Solution of the problem of artificial satellite
  theory without drag. Astron. J. 64, 378-397.
APS_2_5_Research_V3.ipynb, §2, §3, §6.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_MU_KM3_S2: float   = 3.986004418e5   # Earth gravitational parameter [km³/s²]
_R_EARTH_KM: float  = 6378.137        # Earth equatorial radius [km]
_J2: float          = 1.08263e-3      # Earth oblateness coefficient (dimensionless)
_SEC_PER_DAY: float = 86400.0
_TWO_PI: float      = 2.0 * math.pi
_DEG_PER_REV: float = 360.0


# ---------------------------------------------------------------------------
# Low-level orbital mechanics helpers
# ---------------------------------------------------------------------------

def _mean_motion_to_sma_km(mean_motion_rev_per_day: float) -> float:
    """Convert mean motion [rev/day] to semi-major axis [km].

    Two-body: n = sqrt(mu / a^3)  =>  a = (mu / n^2)^(1/3).
    All units in km / s.
    """
    if mean_motion_rev_per_day <= 0:
        raise ValueError(
            f"mean_motion_rev_per_day must be positive, got {mean_motion_rev_per_day}"
        )
    n_rad_s = mean_motion_rev_per_day * _TWO_PI / _SEC_PER_DAY
    return (_MU_KM3_S2 / n_rad_s ** 2) ** (1.0 / 3.0)


def _circular_velocity_km_s(sma_km: float) -> float:
    """Circular orbital velocity [km/s]."""
    return math.sqrt(_MU_KM3_S2 / sma_km)


def _mean_motion_rad_s(sma_km: float) -> float:
    """Mean motion [rad/s] from semi-major axis [km]."""
    return math.sqrt(_MU_KM3_S2 / sma_km ** 3)


def j2_raan_rate_deg_per_day(sma_km: float, inclination_deg: float) -> float:
    """J2 nodal precession rate [deg/day] for a circular orbit.

    Brouwer (1959) first-order secular term:

        Omega_dot_J2 = -3/2 * n * J2 * (Re/a)^2 * cos(i)

    Parameters
    ----------
    sma_km : float
        Semi-major axis [km].
    inclination_deg : float
        Orbital inclination [degrees].

    Returns
    -------
    float
        Precession rate [deg/day].  Negative for prograde orbits (i < 90 deg).

    Notes
    -----
    At 482 km, i = 53 deg: Omega_dot ≈ -4.66 deg/day.
    APS_2_5_Research_V3.ipynb §6.1 states "approx -2.0 deg/day" as a
    rough illustration; the exact formula yields -4.66 deg/day, consistent
    with Vallado Table 9-1 cross-checks at comparable altitudes and
    inclinations.  The exact formula is implemented here.
    """
    n_rad_s = _mean_motion_rad_s(sma_km)
    i_rad = math.radians(inclination_deg)
    rate_rad_s = (
        -1.5 * n_rad_s * _J2
        * (_R_EARTH_KM / sma_km) ** 2
        * math.cos(i_rad)
    )
    return math.degrees(rate_rad_s) * _SEC_PER_DAY


def _along_track_drift_km(
    delta_mean_motion_rev_per_day: float,
    sma_km: float,
    elapsed_days: float,
) -> float:
    """Linearised along-track drift [km] from mean-motion difference.

    |delta_s| ≈ |delta_n * a * t|   (first-order Hill / CW approximation)

    Atmospheric drag contribution is not included.  The TLE-derived
    acceptable_drift_km = 4.459 km implicitly encodes the drag regime
    already present in the operational TLE history.
    """
    if elapsed_days < 0:
        raise ValueError("elapsed_days must be >= 0")
    delta_n_rad_s = delta_mean_motion_rev_per_day * _TWO_PI / _SEC_PER_DAY
    return abs(delta_n_rad_s) * sma_km * elapsed_days * _SEC_PER_DAY


def _return_burn_cost_m_s(dv_avoid_m_s: float, sma_km: float) -> float:
    """Two-impulse circular phasing return burn cost [m/s].

    From Gauss variational equations (APS_2_5_Research_V3.ipynb §2):

        delta_a     = 2a / v_c * dv_avoid
        dv_return   = n * delta_a / 2
                    = (n*a / v_c) * dv_avoid
                    = dv_avoid          (since n*a = v_c on circular orbit)

    The return burn costs exactly the same delta-v as the avoidance burn,
    independent of altitude and recovery orbit count.  Number of recovery
    orbits affects recovery TIME, not total delta-v cost.
    """
    if dv_avoid_m_s < 0:
        raise ValueError("dv_avoid_m_s must be >= 0")
    # Compute explicitly for numerical transparency; ratio = 1.0 exactly
    vc_km_s = _circular_velocity_km_s(sma_km)
    n_rad_s = _mean_motion_rad_s(sma_km)
    delta_a_km = 2.0 * sma_km * (dv_avoid_m_s / 1000.0) / vc_km_s
    return (n_rad_s * delta_a_km / 2.0) * 1000.0  # -> m/s


# ---------------------------------------------------------------------------
# SlotRecoveryPlan — return-target output format (AC)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlotRecoveryPlan:
    """Return-to-slot recovery plan produced after an avoidance maneuver.

    This is the output format for return-target specification required
    by the 9.3 acceptance criteria.  Consumed by 9.4 scoring
    (lambda_s * delta_S term) and 9.5 ATLAS artifact
    (PostManeuverProjection.slot_recovery_orbits).

    The target RAAN is J2-corrected to the planned recovery epoch, so
    9.4 computes the burn against the actual slot position, not a
    stale epoch-0 value.

    Attributes
    ----------
    slot_id : str
        Canonical slot identifier, e.g. 'P02-S05'.
    target_raan_deg : float
        J2-propagated RAAN at recovery epoch [deg].
        Omega_p(t) = Omega_p(0) + Omega_dot_J2 * t
    target_mean_anomaly_deg : float
        Target mean anomaly at recovery epoch [deg].
    dv_avoid_m_s : float
        Avoidance burn magnitude that created the slot offset [m/s].
    dv_return_m_s : float
        Required return burn [m/s].  Equals dv_avoid_m_s on circular
        orbit (§2 corrected formula).  Zero if recovery not required.
    dv_total_m_s : float
        Total maneuver cost: dv_avoid + dv_return [m/s].
    recovery_required : bool
        True when |post_maneuver_drift_km| > acceptable_drift_km.
    recovery_orbits : float
        Minimum recovery orbits (>= 1.0 when recovery required, else 0).
    recovery_time_s : float
        Estimated recovery time [s] = recovery_orbits * T_orbit.
    within_max_recovery_time : bool
        True if recovery_time_s <= ConstellationSlot.max_recovery_time_s.
    post_maneuver_drift_km : float
        Along-track drift from slot centre immediately post-avoidance [km].
    acceptable_drift_km : float
        Slot drift tolerance used for the recovery_required decision [km].
    return_dv_budget_m_s : float
        Budget available for the return burn [m/s].
    budget_feasible : bool
        True if dv_return_m_s <= return_dv_budget_m_s.
    """
    slot_id: str
    target_raan_deg: float
    target_mean_anomaly_deg: float
    dv_avoid_m_s: float
    dv_return_m_s: float
    dv_total_m_s: float
    recovery_required: bool
    recovery_orbits: float
    recovery_time_s: float
    within_max_recovery_time: bool
    post_maneuver_drift_km: float
    acceptable_drift_km: float
    return_dv_budget_m_s: float
    budget_feasible: bool

    def __repr__(self) -> str:
        status = "RECOVERY REQUIRED" if self.recovery_required else "NO RECOVERY NEEDED"
        feasible = "FEASIBLE" if self.budget_feasible else "OVER BUDGET"
        return (
            f"SlotRecoveryPlan({self.slot_id} | {status} | {feasible} | "
            f"dv_total={self.dv_total_m_s:.3f} m/s | "
            f"drift={self.post_maneuver_drift_km:.3f} km / "
            f"tol={self.acceptable_drift_km:.3f} km)"
        )


# ---------------------------------------------------------------------------
# WalkerDeltaGeometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkerDeltaGeometry:
    """Walker-Delta constellation geometry with J2-aware slot propagation.

    Covers slot representation, spacing rules, orbital lane membership,
    drift windows, J2-corrected time-varying slot targets, and
    return-to-slot planning.

    Parameters
    ----------
    total_satellites : int
        Total satellites (t in Walker t/p/f notation).
    num_planes : int
        Number of orbital planes (p).
    phasing_parameter : int
        Inter-plane phasing factor (f, 0 <= f < p).
    inclination_deg : float
        Orbital inclination [degrees].
    altitude_km : float
        Metadata label only.  No physics calculation in this module uses
        this field.  All orbital mechanics derive exclusively from
        mean_motion_rev_per_day -> sma_km via the two-body Kepler relation.

        Convention: follows the TLE tooling convention of mean Earth radius
        R_earth = 6371 km, NOT WGS-84 equatorial radius (6378.137 km).
        Example: 6853.352 - 6371 = 482 km (Space-Track / research doc value).
        WGS-84 geometric altitude is 475.2 km.  Callers must not pass
        WGS-84 altitudes expecting them to affect scoring -- they will not.
    mean_motion_rev_per_day : float
        Target mean motion [rev/day].  Default 15.3020 (TLE-derived,
        Space-Track 2026-03-14 to 2026-04-13, 2,555 normal-ops sats).

    Derived attributes (set in __post_init__)
    -----------------------------------------
    sma_km : float
        Semi-major axis [km] derived from mean_motion_rev_per_day.
    sats_per_plane : int
        Satellites per plane.
    raan_spacing_deg : float
        RAAN spacing between adjacent planes [deg].
    in_plane_spacing_deg : float
        Mean-anomaly spacing between adjacent seats [deg].
    j2_raan_rate_deg_per_day : float
        J2 nodal precession rate at this altitude / inclination [deg/day].
        Negative for prograde orbits.  Used in slot_raan_deg_at_time().
    """

    total_satellites: int
    num_planes: int
    phasing_parameter: int
    inclination_deg: float
    altitude_km: float
    mean_motion_rev_per_day: float = 15.3020

    sma_km: float                   = field(init=False)
    sats_per_plane: int             = field(init=False)
    raan_spacing_deg: float         = field(init=False)
    in_plane_spacing_deg: float     = field(init=False)
    j2_raan_rate_deg_per_day: float = field(init=False)

    def __post_init__(self) -> None:
        # --- validation ---
        if self.total_satellites <= 0:
            raise ValueError("total_satellites must be positive.")
        if self.num_planes <= 0:
            raise ValueError("num_planes must be positive.")
        if self.total_satellites % self.num_planes != 0:
            raise ValueError(
                f"total_satellites ({self.total_satellites}) must be divisible "
                f"by num_planes ({self.num_planes})."
            )
        if not (0 <= self.phasing_parameter < self.num_planes):
            raise ValueError(
                f"phasing_parameter must satisfy 0 <= f < num_planes "
                f"({self.num_planes}), got {self.phasing_parameter}."
            )
        if not (0.0 < self.inclination_deg <= 180.0):
            raise ValueError(
                f"inclination_deg must be in (0, 180], got {self.inclination_deg}."
            )
        if self.altitude_km <= 0:
            raise ValueError("altitude_km must be positive.")

        # --- derived ---
        sma = _mean_motion_to_sma_km(self.mean_motion_rev_per_day)
        object.__setattr__(self, "sma_km", sma)
        object.__setattr__(self, "sats_per_plane",
                           self.total_satellites // self.num_planes)
        object.__setattr__(self, "raan_spacing_deg",
                           _DEG_PER_REV / self.num_planes)
        object.__setattr__(self, "in_plane_spacing_deg",
                           _DEG_PER_REV / (self.total_satellites // self.num_planes))
        object.__setattr__(self, "j2_raan_rate_deg_per_day",
                           j2_raan_rate_deg_per_day(sma, self.inclination_deg))

    # ------------------------------------------------------------------
    # Slot addressing — epoch-0
    # ------------------------------------------------------------------

    def slot_raan_deg(self, plane_idx: int) -> float:
        """Epoch-0 RAAN [deg] for plane_idx.

        Omega_p = p * delta_Omega,   delta_Omega = 360 / num_planes.

        For the J2-corrected value at a future time use
        slot_raan_deg_at_time().
        """
        self._validate_plane_idx(plane_idx)
        return (plane_idx * self.raan_spacing_deg) % _DEG_PER_REV

    def slot_raan_deg_at_time(
        self, plane_idx: int, elapsed_seconds: float
    ) -> float:
        """J2-corrected RAAN [deg] for plane_idx at elapsed_seconds after epoch.

        Omega_p(t) = Omega_p(0) + Omega_dot_J2 * t

        This is the mandatory CTO acceptance criterion from §6.1.
        The return burn target in 9.4 must use this value, not the
        static epoch-0 RAAN, for any non-zero recovery window.

        Parameters
        ----------
        plane_idx : int
            Zero-based plane index.
        elapsed_seconds : float
            Seconds after epoch.  May be negative (past).

        Returns
        -------
        float
            J2-propagated RAAN [deg] in [0, 360).
        """
        self._validate_plane_idx(plane_idx)
        raan_0 = self.slot_raan_deg(plane_idx)
        elapsed_days = elapsed_seconds / _SEC_PER_DAY
        raan_t = raan_0 + self.j2_raan_rate_deg_per_day * elapsed_days
        return raan_t % _DEG_PER_REV

    def slot_mean_anomaly_deg(self, plane_idx: int, seat_idx: int) -> float:
        """Mean anomaly [deg] for (plane_idx, seat_idx) using Walker phasing.

        u(p, s) = s * delta_u + p * (f * delta_u / P)

        where delta_u = 360 / sats_per_plane,  f = phasing_parameter,
        P = num_planes.  Equivalent to the research doc formulation
        u_s = s * delta_u + p * delta_phi, where delta_phi = f * 360/N.
        """
        self._validate_plane_idx(plane_idx)
        self._validate_seat_idx(seat_idx)
        delta_u = self.in_plane_spacing_deg
        phase_offset = plane_idx * (self.phasing_parameter * delta_u / self.num_planes)
        return (seat_idx * delta_u + phase_offset) % _DEG_PER_REV

    def slot_id(self, plane_idx: int, seat_idx: int) -> str:
        """Canonical slot ID string, e.g. 'P02-S05'."""
        self._validate_plane_idx(plane_idx)
        self._validate_seat_idx(seat_idx)
        return f"P{plane_idx:02d}-S{seat_idx:02d}"

    # ------------------------------------------------------------------
    # Slot enumeration
    # ------------------------------------------------------------------

    def slots(self) -> Iterator["WalkerSlotAddress"]:
        """Yield all WalkerSlotAddress objects.

        Order: plane 0 seat 0, plane 0 seat 1, ..., plane p-1 seat s-1.
        """
        for p in range(self.num_planes):
            for s in range(self.sats_per_plane):
                yield WalkerSlotAddress(geometry=self, plane_idx=p, seat_idx=s)

    def slot_count(self) -> int:
        """Total slots (= total_satellites)."""
        return self.total_satellites

    # ------------------------------------------------------------------
    # Drift tolerance
    # ------------------------------------------------------------------

    def along_track_drift_km(
        self,
        actual_mean_motion_rev_per_day: float,
        elapsed_days: float,
    ) -> float:
        """Linearised along-track drift [km] relative to this geometry's slots."""
        delta_n = actual_mean_motion_rev_per_day - self.mean_motion_rev_per_day
        return _along_track_drift_km(delta_n, self.sma_km, elapsed_days)

    def is_within_drift_tolerance(
        self,
        actual_mean_motion_rev_per_day: float,
        elapsed_days: float,
        acceptable_drift_km: float = 4.459,
    ) -> bool:
        """True if along-track drift <= acceptable_drift_km."""
        return self.along_track_drift_km(
            actual_mean_motion_rev_per_day, elapsed_days
        ) <= acceptable_drift_km

    def days_until_tolerance_breach(
        self,
        actual_mean_motion_rev_per_day: float,
        acceptable_drift_km: float = 4.459,
    ) -> Optional[float]:
        """Days until along-track drift reaches acceptable_drift_km.

        Returns None if delta_n == 0 (no drift).
        """
        delta_n_rev_day = (
            actual_mean_motion_rev_per_day - self.mean_motion_rev_per_day
        )
        if delta_n_rev_day == 0.0:
            return None
        delta_n_rad_s = abs(delta_n_rev_day) * _TWO_PI / _SEC_PER_DAY
        t_s = (acceptable_drift_km * 1000.0) / (delta_n_rad_s * self.sma_km * 1000.0)
        return t_s / _SEC_PER_DAY

    # ------------------------------------------------------------------
    # Return-to-slot planning (AC: return logic in the decision)
    # ------------------------------------------------------------------

    def plan_slot_recovery(
        self,
        plane_idx: int,
        seat_idx: int,
        dv_avoid_m_s: float,
        post_maneuver_drift_km: float,
        acceptable_drift_km: float = 4.459,
        return_dv_budget_m_s: float = 4.4,
        max_recovery_time_s: float = 86400.0,
        recovery_epoch_offset_s: float = 0.0,
    ) -> SlotRecoveryPlan:
        """Produce a SlotRecoveryPlan for return to slot after an avoidance burn.

        Return-to-slot logic is integrated here as part of the maneuver
        decision, not as a post-processing afterthought.  9.4 calls this
        method before scoring to include dv_total_m_s in the utility
        function and to populate PostManeuverProjection.slot_recovery_orbits.

        The target RAAN is J2-corrected to the recovery epoch via
        slot_raan_deg_at_time(), satisfying the CTO §6.1 criterion.

        Parameters
        ----------
        plane_idx : int
            Plane index of the satellite's assigned slot.
        seat_idx : int
            Seat index of the satellite's assigned slot.
        dv_avoid_m_s : float
            Avoidance burn magnitude [m/s].
        post_maneuver_drift_km : float
            Along-track drift from slot centre immediately after the
            avoidance burn [km].  9.4 derives this from CW dynamics.
        acceptable_drift_km : float
            Drift tolerance [km].  Pass ConstellationSlot.acceptable_drift_km.
        return_dv_budget_m_s : float
            Budget for the return burn [m/s].
            Pass ConstellationSlot.return_dv_budget_m_s.
        max_recovery_time_s : float
            Maximum allowed recovery window [s].
            Pass ConstellationSlot.max_recovery_time_s.
        recovery_epoch_offset_s : float
            Seconds after epoch at which recovery begins.  Used for
            J2-corrected target RAAN.  Defaults to 0.

        Returns
        -------
        SlotRecoveryPlan
            J2-corrected target, burn magnitudes, timing, feasibility flags.
        """
        self._validate_plane_idx(plane_idx)
        self._validate_seat_idx(seat_idx)

        recovery_required = abs(post_maneuver_drift_km) > acceptable_drift_km

        target_raan = self.slot_raan_deg_at_time(plane_idx, recovery_epoch_offset_s)
        target_ma = self.slot_mean_anomaly_deg(plane_idx, seat_idx)

        dv_return = (
            _return_burn_cost_m_s(dv_avoid_m_s, self.sma_km)
            if recovery_required else 0.0
        )
        dv_total = dv_avoid_m_s + dv_return

        # Minimum recovery: 1 orbit when recovery is required
        n_orbits = 1.0 if (recovery_required and dv_avoid_m_s > 0) else 0.0
        t_orbit_s = _TWO_PI / _mean_motion_rad_s(self.sma_km)
        recovery_time_s = n_orbits * t_orbit_s

        return SlotRecoveryPlan(
            slot_id=self.slot_id(plane_idx, seat_idx),
            target_raan_deg=target_raan,
            target_mean_anomaly_deg=target_ma,
            dv_avoid_m_s=dv_avoid_m_s,
            dv_return_m_s=dv_return,
            dv_total_m_s=dv_total,
            recovery_required=recovery_required,
            recovery_orbits=n_orbits,
            recovery_time_s=recovery_time_s,
            within_max_recovery_time=(recovery_time_s <= max_recovery_time_s),
            post_maneuver_drift_km=post_maneuver_drift_km,
            acceptable_drift_km=acceptable_drift_km,
            return_dv_budget_m_s=return_dv_budget_m_s,
            budget_feasible=(dv_return <= return_dv_budget_m_s),
        )

    # ------------------------------------------------------------------
    # Internal validators
    # ------------------------------------------------------------------

    def _validate_plane_idx(self, plane_idx: int) -> None:
        if not (0 <= plane_idx < self.num_planes):
            raise IndexError(
                f"plane_idx {plane_idx} out of range [0, {self.num_planes})."
            )

    def _validate_seat_idx(self, seat_idx: int) -> None:
        if not (0 <= seat_idx < self.sats_per_plane):
            raise IndexError(
                f"seat_idx {seat_idx} out of range [0, {self.sats_per_plane})."
            )

    def __repr__(self) -> str:
        return (
            f"WalkerDeltaGeometry("
            f"{self.total_satellites}/{self.num_planes}/{self.phasing_parameter} "
            f"i={self.inclination_deg}deg h={self.altitude_km}km "
            f"n={self.mean_motion_rev_per_day:.4f}rev/day "
            f"Omega_dot_J2={self.j2_raan_rate_deg_per_day:.3f}deg/day)"
        )


# ---------------------------------------------------------------------------
# WalkerSlotAddress
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkerSlotAddress:
    """Fully-resolved slot address within a WalkerDeltaGeometry.

    Carries epoch-0 target orbital elements and delegates all
    slot-level operations — drift checking, J2 propagation, and
    recovery planning — to the parent geometry.

    Parameters
    ----------
    geometry : WalkerDeltaGeometry
        Parent constellation geometry.
    plane_idx : int
        Zero-based plane index.
    seat_idx : int
        Zero-based seat index within the plane.

    Derived attributes
    ------------------
    slot_id : str
        Canonical string ID, e.g. 'P02-S05'.
    target_raan_deg : float
        Epoch-0 RAAN [deg].  Use target_raan_at_time() for J2-corrected.
    target_mean_anomaly_deg : float
        Target mean anomaly [deg].
    """

    geometry: WalkerDeltaGeometry
    plane_idx: int
    seat_idx: int

    slot_id: str                    = field(init=False)
    target_raan_deg: float          = field(init=False)
    target_mean_anomaly_deg: float  = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id",
                           self.geometry.slot_id(self.plane_idx, self.seat_idx))
        object.__setattr__(self, "target_raan_deg",
                           self.geometry.slot_raan_deg(self.plane_idx))
        object.__setattr__(self, "target_mean_anomaly_deg",
                           self.geometry.slot_mean_anomaly_deg(
                               self.plane_idx, self.seat_idx))

    def target_raan_at_time(self, elapsed_seconds: float) -> float:
        """J2-corrected target RAAN [deg] at elapsed_seconds after epoch."""
        return self.geometry.slot_raan_deg_at_time(self.plane_idx, elapsed_seconds)

    def along_track_drift_km(
        self,
        actual_mean_motion_rev_per_day: float,
        elapsed_days: float,
    ) -> float:
        return self.geometry.along_track_drift_km(
            actual_mean_motion_rev_per_day, elapsed_days
        )

    def is_within_drift_tolerance(
        self,
        actual_mean_motion_rev_per_day: float,
        elapsed_days: float,
        acceptable_drift_km: float = 4.459,
    ) -> bool:
        return self.geometry.is_within_drift_tolerance(
            actual_mean_motion_rev_per_day, elapsed_days, acceptable_drift_km
        )

    def days_until_tolerance_breach(
        self,
        actual_mean_motion_rev_per_day: float,
        acceptable_drift_km: float = 4.459,
    ) -> Optional[float]:
        return self.geometry.days_until_tolerance_breach(
            actual_mean_motion_rev_per_day, acceptable_drift_km
        )

    def plan_slot_recovery(
        self,
        dv_avoid_m_s: float,
        post_maneuver_drift_km: float,
        acceptable_drift_km: float = 4.459,
        return_dv_budget_m_s: float = 4.4,
        max_recovery_time_s: float = 86400.0,
        recovery_epoch_offset_s: float = 0.0,
    ) -> SlotRecoveryPlan:
        """Recovery plan for this slot. Delegates to parent geometry."""
        return self.geometry.plan_slot_recovery(
            plane_idx=self.plane_idx,
            seat_idx=self.seat_idx,
            dv_avoid_m_s=dv_avoid_m_s,
            post_maneuver_drift_km=post_maneuver_drift_km,
            acceptable_drift_km=acceptable_drift_km,
            return_dv_budget_m_s=return_dv_budget_m_s,
            max_recovery_time_s=max_recovery_time_s,
            recovery_epoch_offset_s=recovery_epoch_offset_s,
        )

    def __repr__(self) -> str:
        return (
            f"WalkerSlotAddress({self.slot_id} "
            f"RAAN={self.target_raan_deg:.2f}deg "
            f"M={self.target_mean_anomaly_deg:.2f}deg)"
        )


# ---------------------------------------------------------------------------
# Reference architectures
# ---------------------------------------------------------------------------

def starlink_shell_1() -> WalkerDeltaGeometry:
    """Starlink Shell 1 reference architecture (AC: required reference model).

    Geometry from FCC filing (reliable); SMA from TLE-derived mean motion.

    FCC parameters: 1584 / 72 / 22, i = 53 deg.
    Operational altitude: ~482 km (NOT the FCC-filed 550 km).
    Source: Space-Track TLE history 2026-03-14 to 2026-04-13,
            2,555 normal-ops satellites, 238,760 TLE records.
    """
    return WalkerDeltaGeometry(
        total_satellites=1584,
        num_planes=72,
        phasing_parameter=22,
        inclination_deg=53.0,
        altitude_km=482.0,
        mean_motion_rev_per_day=15.3020,
    )


def leo_constellation_482km() -> WalkerDeltaGeometry:
    """Representative 60-satellite LEO Walker-Delta at 482 km.

    Test fixture and default for operators without full Starlink-scale
    geometry.  Parameters consistent with 9.1 ConstellationSlot defaults.
    """
    return WalkerDeltaGeometry(
        total_satellites=60,
        num_planes=6,
        phasing_parameter=2,
        inclination_deg=53.0,
        altitude_km=482.0,
        mean_motion_rev_per_day=15.3020,
    )
