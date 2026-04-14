# common/satellite_capability.py
"""
APS v2.5 — Satellite Capability Model

Defines the SatelliteCapability dataclass and supporting types
for mission-aware maneuver planning.

This schema is the engineering implementation of APS 2.5 Section 9.1.
It extends the existing satellite fields in decision_model.py
(r_sat_km, v_sat_km_s, t_burn_utc, v_remaining_m_s, a_ref_km)
with propulsion, mass, cadence, and lifetime accounting.

Design principles:
- All fields that decision_model.py already uses are preserved exactly.
- New fields have sensible defaults so existing callers are not broken.
- Fields are grouped into four sub-profiles matching the v2.5 vision:
    PropulsionProfile   — what the spacecraft can physically do
    LifetimeProfile     — fuel and mission lifetime accounting
    ManeuverCadence     — operational constraints on burn frequency
    ConstellationSlot   — orbital position recovery requirements (v2.5)

JSON schema counterpart: planner_v25.yaml #/components/schemas/SatelliteCapability
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Propulsion type enum
# ---------------------------------------------------------------------------

class PropulsionType:
    """
    Propulsion system classification.
    Determines which maneuver parameters are physically meaningful.
    """
    CHEMICAL   = "chemical"    # high-thrust, impulsive burns
    ELECTRIC   = "electric"    # low-thrust, continuous or duty-cycled
    COLD_GAS   = "cold_gas"    # very low Isp, limited delta-v
    NONE       = "none"        # no propulsion (debris, passive)


# ---------------------------------------------------------------------------
# Sub-profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropulsionProfile:
    """
    Physical propulsion capabilities of the spacecraft.

    Fields
    ------
    propulsion_type : str
        One of PropulsionType constants. Determines valid maneuver model.
    isp_s : float
        Specific impulse [seconds]. Used in rocket equation for mass accounting.
        Default 220 s (typical cold gas). Chemical ~300s, electric ~1500-3000s.
    thrust_n : float
        Thrust [Newtons]. Used to compute burn duration from dv and mass.
        For impulsive model (chemical/cold gas), this is informational.
    min_dv_m_s : float
        Minimum executable burn magnitude [m/s].
        Burns below this threshold are rounded up or rejected.
    max_dv_per_burn_m_s : float
        Maximum single-burn delta-v [m/s].
        Hard constraint; planner must not exceed this.
    """
    propulsion_type: str = PropulsionType.CHEMICAL
    isp_s: float = 220.0
    thrust_n: float = 1.0
    min_dv_m_s: float = 0.01
    max_dv_per_burn_m_s: float = 10.0

    def __post_init__(self) -> None:
        if self.isp_s <= 0:
            raise ValueError("isp_s must be > 0")
        if self.thrust_n < 0:
            raise ValueError("thrust_n must be >= 0")
        if self.min_dv_m_s < 0:
            raise ValueError("min_dv_m_s must be >= 0")
        if self.max_dv_per_burn_m_s <= 0:
            raise ValueError("max_dv_per_burn_m_s must be > 0")
        if self.min_dv_m_s > self.max_dv_per_burn_m_s:
            raise ValueError("min_dv_m_s must be <= max_dv_per_burn_m_s")


@dataclass(frozen=True)
class LifetimeProfile:
    """
    Fuel and mission lifetime accounting.

    Fields
    ------
    mass_kg : float
        Current spacecraft wet mass [kg].
        Used in rocket equation: dv = Isp * g0 * ln(m0/mf)
    v_remaining_m_s : float
        Remaining total delta-v budget [m/s].
        This is the primary fuel proxy used in the utility function.
        Carries over from v2.4 (was satellite.v_remaining_m_s).
    v_reserved_m_s : float
        Delta-v reserved for non-avoidance operations [m/s].
        e.g. deorbit, station-keeping, end-of-life.
        Planner should not consume below v_remaining - v_reserved.
    mission_lifetime_days_remaining : float
        Estimated remaining mission lifetime [days].
        Used to contextualise lifetime penalty: a burn late in mission
        life is more costly than the same burn early on.
    """
    mass_kg: float = 100.0
    v_remaining_m_s: float = 50.0
    v_reserved_m_s: float = 5.0
    mission_lifetime_days_remaining: float = 365.0

    def __post_init__(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be > 0")
        if self.v_remaining_m_s < 0:
            raise ValueError("v_remaining_m_s must be >= 0")
        if self.v_reserved_m_s < 0:
            raise ValueError("v_reserved_m_s must be >= 0")
        if self.v_reserved_m_s > self.v_remaining_m_s:
            raise ValueError("v_reserved_m_s must be <= v_remaining_m_s")

    @property
    def v_available_m_s(self) -> float:
        """Usable delta-v after reserving non-avoidance budget."""
        return max(0.0, self.v_remaining_m_s - self.v_reserved_m_s)

    @property
    def lifetime_fraction_used(self) -> float:
        """
        Proxy for mission maturity. 0 = fresh satellite, 1 = end of life.
        Requires mission_lifetime_days_total if available; approximated here
        as a placeholder until operator policy provides total lifetime.
        """
        return 0.0  # placeholder; populated by operator policy in v2.5


@dataclass(frozen=True)
class ManeuverCadence:
    """
    Operational constraints on how frequently the spacecraft can maneuver.

    Fields
    ------
    min_time_between_burns_s : float
        Minimum time between consecutive burns [seconds].
        Driven by thermal recovery, attitude control, or operator policy.
    max_burns_per_orbit : int
        Maximum number of burns per orbital period.
        Hard constraint for some propulsion types.
    attitude_restricted : bool
        If True, only prograde burns are available (attitude mode limited).
        Carries over from v2.4 hard_constraints.attitude_restricted.
    power_constrained : bool
        If True, available dv is reduced by 50% (conservative model).
        Carries over from v2.4 hard_constraints.power_constrained.
    no_burn_windows : List[dict]
        List of {"start_utc": str, "end_utc": str} blackout windows.
        Carries over from v2.4 hard_constraints.no_burn_windows.
    """
    min_time_between_burns_s: float = 0.0
    max_burns_per_orbit: int = 4
    attitude_restricted: bool = False
    power_constrained: bool = False
    no_burn_windows: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.min_time_between_burns_s < 0:
            raise ValueError("min_time_between_burns_s must be >= 0")
        if self.max_burns_per_orbit < 1:
            raise ValueError("max_burns_per_orbit must be >= 1")


@dataclass(frozen=True)
class ConstellationSlot:
    """
    Orbital position recovery requirements for constellation members.

    This is the key new addition in APS v2.5.
    If a satellite belongs to a constellation, it may need to return
    to a defined orbital relationship after an avoidance maneuver.
    The planner must account for the return burn in its utility scoring.

    Defaults below are Starlink Shell 1 empirical values derived from
    Space-Track TLE history 2026-03-14 to 2026-04-13.
    Operational altitude: ~482 km (NOT 550 km per FCC filing —
    SpaceX lowered Shell 1 from 550 km after FCC negotiations).
    2,652 satellites pulled; 97 shell-transfer satellites excluded
    (max SMA deviation >20 km); 2,555 normal-ops satellites,
    238,760 TLE records analyzed.

    These are the DEFAULT path for operators who do not supply their
    own values. Operators with known tolerances should supply
    ConstellationSlot values directly in operator_policy.yaml at
    onboarding — that is the production path for smaller constellations.

    Fields
    ------
    in_constellation : bool
        True if this satellite has slot/spacing constraints.
        If False, all other fields are ignored.
    slot_id : str
        Human-readable slot identifier (e.g. "PLANE-2-SAT-4").
    target_mean_motion_rev_per_day : float
        Nominal mean motion for the slot [rev/day].
        Default: median across 2,555 normal-ops Shell 1 satellites.
    acceptable_drift_km : float
        Maximum acceptable along-track drift from slot centre [km].
        Default: P90 of per-satellite max SMA deviation from personal
        baseline, shell-transfer satellites excluded.
    return_dv_budget_m_s : float
        Delta-v allocated for the return maneuver [m/s].
        Default: ~30% of FCC annual budget (50 m/s) divided by
        observed correction frequency (3.39 corrections/sat/yr).
        Separate from avoidance budget; planner must check feasibility.
    max_recovery_time_s : float
        Maximum acceptable time to return to slot [seconds].
    """
    in_constellation: bool                = False
    slot_id: str                          = ""
    target_mean_motion_rev_per_day: float = 15.3020  # TLE-derived: median, 2,555 normal-ops sats
    acceptable_drift_km: float            = 4.459    # TLE-derived: P90 per-satellite max SMA deviation
    return_dv_budget_m_s: float           = 4.4      # ~30% FCC annual budget / 3.39 corrections/sat/yr
    max_recovery_time_s: float            = 86400.0  # 24h default

    def __post_init__(self) -> None:
        if self.acceptable_drift_km < 0:
            raise ValueError("acceptable_drift_km must be >= 0")
        if self.return_dv_budget_m_s < 0:
            raise ValueError("return_dv_budget_m_s must be >= 0")


# ---------------------------------------------------------------------------
# Top-level SatelliteCapability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SatelliteCapability:
    """
    APS v2.5 satellite capability model.

    Combines all four sub-profiles into one object consumed by the planner.
    Extends the v2.4 satellite fields without breaking existing callers.

    Mandatory fields (carried over from v2.4):
        sat_id, a_ref_km

    New in v2.5:
        propulsion, lifetime, cadence, slot

    Usage
    -----
    From a v2.5 API request dict:
        cap = SatelliteCapability.from_request(satellite_dict)

    From defaults (backward-compatible with v2.4):
        cap = SatelliteCapability(sat_id="SAT-001", a_ref_km=6878.0)
    """
    sat_id: str
    a_ref_km: float = 7000.0
    propulsion: PropulsionProfile = field(default_factory=PropulsionProfile)
    lifetime: LifetimeProfile = field(default_factory=LifetimeProfile)
    cadence: ManeuverCadence = field(default_factory=ManeuverCadence)
    slot: ConstellationSlot = field(default_factory=ConstellationSlot)

    def __post_init__(self) -> None:
        if not self.sat_id:
            raise ValueError("sat_id must be a non-empty string")
        if self.a_ref_km <= 0:
            raise ValueError("a_ref_km must be > 0")

    @classmethod
    def from_request(cls, sat: dict) -> "SatelliteCapability":
        """
        Build from a v2.5 OpenAPI satellite dict.
        Falls back to v2.4-compatible defaults for any missing sub-profile.
        """
        sat_id   = str(sat.get("sat_id", "UNKNOWN"))
        a_ref_km = float(sat.get("a_ref_km", 7000.0))

        # PropulsionProfile
        p_raw = sat.get("propulsion", {})
        propulsion = PropulsionProfile(
            propulsion_type       = str(p_raw.get("propulsion_type", PropulsionType.CHEMICAL)),
            isp_s                 = float(p_raw.get("isp_s", 220.0)),
            thrust_n              = float(p_raw.get("thrust_n", 1.0)),
            min_dv_m_s            = float(p_raw.get("min_dv_m_s", 0.01)),
            max_dv_per_burn_m_s   = float(p_raw.get("max_dv_per_burn_m_s", 10.0)),
        )

        # LifetimeProfile — v_remaining_m_s bridges from v2.4
        l_raw = sat.get("lifetime", {})
        v_remaining = float(sat.get("v_remaining_m_s", l_raw.get("v_remaining_m_s", 50.0)))
        lifetime = LifetimeProfile(
            mass_kg                          = float(l_raw.get("mass_kg", 100.0)),
            v_remaining_m_s                  = v_remaining,
            v_reserved_m_s                   = float(l_raw.get("v_reserved_m_s", 5.0)),
            mission_lifetime_days_remaining  = float(l_raw.get("mission_lifetime_days_remaining", 365.0)),
        )

        # ManeuverCadence — bridges v2.4 hard_constraints
        c_raw = sat.get("cadence", sat.get("hard_constraints", {}))
        cadence = ManeuverCadence(
            min_time_between_burns_s = float(c_raw.get("min_time_between_burns_s", 0.0)),
            max_burns_per_orbit      = int(c_raw.get("max_burns_per_orbit", 4)),
            attitude_restricted      = bool(c_raw.get("attitude_restricted", False)),
            power_constrained        = bool(c_raw.get("power_constrained", False)),
            no_burn_windows          = list(c_raw.get("no_burn_windows", [])),
        )

        # ConstellationSlot
        s_raw = sat.get("slot", {})
        slot = ConstellationSlot(
            in_constellation                  = bool(s_raw.get("in_constellation", False)),
            slot_id                           = str(s_raw.get("slot_id", "")),
            target_mean_motion_rev_per_day    = float(s_raw.get("target_mean_motion_rev_per_day", 15.3020)),
            acceptable_drift_km               = float(s_raw.get("acceptable_drift_km", 4.459)),
            return_dv_budget_m_s              = float(s_raw.get("return_dv_budget_m_s", 4.4)),
            max_recovery_time_s               = float(s_raw.get("max_recovery_time_s", 86400.0)),
        )

        return cls(
            sat_id     = sat_id,
            a_ref_km   = a_ref_km,
            propulsion = propulsion,
            lifetime   = lifetime,
            cadence    = cadence,
            slot       = slot,
        )

    def effective_dv_limit_m_s(self, requested_dv_m_s: float) -> float:
        """
        Return the effective dv limit after applying all constraints:
        - power_constrained halves available thrust
        - max_dv_per_burn_m_s hard ceiling
        - v_available_m_s fuel ceiling
        """
        dv = min(requested_dv_m_s, self.propulsion.max_dv_per_burn_m_s)
        dv = min(dv, self.lifetime.v_available_m_s)
        if self.cadence.power_constrained:
            dv *= 0.5
        return max(0.0, dv)

    def requires_return_burn(self, delta_along_track_km: float) -> bool:
        """
        True if the post-maneuver along-track drift exceeds the slot tolerance.
        Only meaningful if in_constellation is True.
        """
        if not self.slot.in_constellation:
            return False
        return abs(delta_along_track_km) > self.slot.acceptable_drift_km