# services/planner/common/operator_policy.py
"""
APS v2.5 -- Operator Policy Layer

Defines the OperatorPolicy dataclass and supporting types.
Implements Section 9.2: operator-specific maneuver thresholds,
mission priorities, hard constraint overrides, and fleet-level
priority logic.

Design principles:
- Tier 1 (configurable): risk thresholds, scoring weights, blackout
  windows, maneuver frequency limits, fleet priority. Loaded from
  YAML at request time -- no code changes required for new operators.
- Tier 2 (embedded): CWH dynamics, Mahalanobis calculation, no-burn
  baseline, secondary conjunction check, propulsion feasibility filter.
  These never change between operators.

The key boundary: anything that differs between operators is configurable.
Anything grounded in orbital mechanics or propulsion physics is embedded.

YAML counterpart: services/planner/config/operator_policy_leo.yaml

Important: merge_blackout_windows() must always be called to combine
operator-level and satellite-level (ManeuverCadence.no_burn_windows)
blackout windows. Never apply either source in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Blackout window
# ---------------------------------------------------------------------------

@dataclass
class BlackoutWindow:
    """
    A time window during which burns are forbidden.

    Fields
    ------
    window_type : str
        Category: 'payload_operation' | 'ground_contact' |
        'satellite_cadence' | 'custom'
    start_utc : str or None
        ISO-8601 UTC start time. None for buffer-only windows.
    end_utc : str or None
        ISO-8601 UTC end time. None for buffer-only windows.
    buffer_hours_before : float
        Additional forbidden window before start_utc [hours].
    buffer_hours_after : float
        Additional forbidden window after end_utc [hours].
    source : str
        Provenance: 'operator' | 'satellite_cadence'
    """
    window_type: str
    start_utc: Optional[str]
    end_utc: Optional[str]
    buffer_hours_before: float = 0.0
    buffer_hours_after: float  = 0.0
    source: str = "operator"

    def to_dict(self) -> dict:
        return {
            "window_type":          self.window_type,
            "start_utc":            self.start_utc,
            "end_utc":              self.end_utc,
            "buffer_hours_before":  self.buffer_hours_before,
            "buffer_hours_after":   self.buffer_hours_after,
            "source":               self.source,
        }


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

@dataclass
class ScoringWeights:
    """
    Lambda weights for the APS 2.5 utility scoring function.

    U(a) = delta_C - lambda_v * delta_v
                   - lambda_L * delta_L
                   - lambda_s * delta_S

    Higher lambda_s penalises constellation slot deviation more heavily.
    Higher lambda_v penalises fuel cost more heavily.
    """
    lambda_dv: float             = 1.0
    lambda_lifetime: float       = 0.8
    lambda_slot_deviation: float = 1.2

    def __post_init__(self) -> None:
        for name, val in [
            ("lambda_dv",            self.lambda_dv),
            ("lambda_lifetime",      self.lambda_lifetime),
            ("lambda_slot_deviation", self.lambda_slot_deviation),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be >= 0")

    def to_dict(self) -> dict:
        return {
            "lambda_dv":            self.lambda_dv,
            "lambda_lifetime":      self.lambda_lifetime,
            "lambda_slot_deviation": self.lambda_slot_deviation,
        }


# ---------------------------------------------------------------------------
# OperatorPolicy
# ---------------------------------------------------------------------------

@dataclass
class OperatorPolicy:
    """
    APS v2.5 operator policy model.

    Combines all configurable operator preferences into one object
    consumed by the planner. Loaded at request time from YAML --
    no code changes required for new operators.

    Mandatory fields:
        operator_id, policy_version

    All other fields have defaults matching the DEFAULT_LEO profile.

    Usage
    -----
    From a YAML file:
        policy = OperatorPolicy.from_yaml("config/operator_policy_leo.yaml")

    From defaults (DEFAULT_LEO):
        policy = OperatorPolicy(operator_id="MY_OP", policy_version="2.5.0")
    """

    operator_id: str
    policy_version: str

    # -- Tier 1: risk thresholds (configurable) ---------------------------
    pc_maneuver_threshold: float        = 1.0e-4
    pc_monitor_threshold: float         = 1.0e-5
    min_miss_distance_km: float         = 1.0
    mahalanobis_screen_threshold: float = 4.0

    # -- Tier 1: maneuver constraints (configurable) ----------------------
    max_dv_per_event_ms: float          = 2.0
    max_maneuvers_per_week: int         = 3
    min_hours_before_tca: float         = 4.0
    max_hours_before_tca: float         = 72.0

    # -- Tier 1: blackout windows (configurable) --------------------------
    # Operator-level windows. Must be merged with satellite-level cadence
    # windows via merge_blackout_windows() before use.
    blackout_windows: List[BlackoutWindow] = field(default_factory=list)

    # -- Tier 1: scoring weights (configurable) ---------------------------
    scoring_weights: ScoringWeights     = field(default_factory=ScoringWeights)

    # -- Tier 1: operational philosophy (configurable) --------------------
    operational_philosophy: str         = "balanced"
    # Options: "risk_averse" | "balanced" | "fuel_conservative"

    # -- Tier 1: fleet priority (configurable) ----------------------------
    fleet_priority_method: str          = "highest_pc_first"
    # Options: "highest_pc_first" | "most_fuel_remaining" | "manual"

    # -- Tier 1: mission lifetime (configurable) --------------------------
    # Total mission lifetime [days] for this operator's satellites.
    # Unblocks LifetimeProfile.lifetime_fraction_used, which feeds into
    # the lambda_L lifetime penalty term in the 9.4 scoring function.
    # Required by SCRUM-280.
    # None means unknown; lifetime_fraction_used returns 0.0 as before.
    mission_lifetime_days_total: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.operator_id:
            raise ValueError("operator_id must be a non-empty string")
        if not self.policy_version:
            raise ValueError("policy_version must be a non-empty string")
        if self.pc_maneuver_threshold <= 0:
            raise ValueError("pc_maneuver_threshold must be > 0")
        if self.pc_monitor_threshold <= 0:
            raise ValueError("pc_monitor_threshold must be > 0")
        if self.pc_monitor_threshold >= self.pc_maneuver_threshold:
            raise ValueError(
                "pc_monitor_threshold must be < pc_maneuver_threshold"
            )
        if self.min_miss_distance_km < 0:
            raise ValueError("min_miss_distance_km must be >= 0")
        if self.mahalanobis_screen_threshold <= 0:
            raise ValueError("mahalanobis_screen_threshold must be > 0")
        if self.max_dv_per_event_ms <= 0:
            raise ValueError("max_dv_per_event_ms must be > 0")
        if self.max_maneuvers_per_week < 1:
            raise ValueError("max_maneuvers_per_week must be >= 1")
        if self.min_hours_before_tca < 0:
            raise ValueError("min_hours_before_tca must be >= 0")
        if self.max_hours_before_tca <= self.min_hours_before_tca:
            raise ValueError(
                "max_hours_before_tca must be > min_hours_before_tca"
            )
        valid_philosophies = {"risk_averse", "balanced", "fuel_conservative"}
        if self.operational_philosophy not in valid_philosophies:
            raise ValueError(
                f"operational_philosophy must be one of {valid_philosophies}"
            )
        valid_priority = {"highest_pc_first", "most_fuel_remaining", "manual"}
        if self.fleet_priority_method not in valid_priority:
            raise ValueError(
                f"fleet_priority_method must be one of {valid_priority}"
            )
        if (self.mission_lifetime_days_total is not None
                and self.mission_lifetime_days_total <= 0):
            raise ValueError("mission_lifetime_days_total must be > 0 if provided")

    # -- Loader -----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "OperatorPolicy":
        """
        Load policy from a YAML file at runtime.
        No code changes required for new operators.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        with open(p) as f:
            data = yaml.safe_load(f)

        rt = data.get("risk_thresholds", {})
        mc = data.get("maneuver_constraints", {})
        sw = data.get("scoring_weights", {})
        fp = data.get("fleet_priority", {})

        weights = ScoringWeights(
            lambda_dv            = float(sw.get("lambda_dv",            1.0)),
            lambda_lifetime      = float(sw.get("lambda_lifetime",      0.8)),
            lambda_slot_deviation = float(sw.get("lambda_slot_deviation", 1.2)),
        )

        windows = []
        for w in data.get("blackout_windows", []):
            windows.append(BlackoutWindow(
                window_type         = str(w.get("type", "custom")),
                start_utc           = w.get("start_utc"),
                end_utc             = w.get("end_utc"),
                buffer_hours_before = float(w.get("buffer_hours_before", 0.0)),
                buffer_hours_after  = float(w.get("buffer_hours_after",  0.0)),
                source              = "operator",
            ))

        return cls(
            operator_id                = str(data["operator_id"]),
            policy_version             = str(data["policy_version"]),
            pc_maneuver_threshold      = float(rt.get("pc_maneuver_threshold", 1.0e-4)),
            pc_monitor_threshold       = float(rt.get("pc_monitor_threshold",  1.0e-5)),
            min_miss_distance_km       = float(rt.get("min_miss_distance_km",  1.0)),
            mahalanobis_screen_threshold = float(rt.get("mahalanobis_screen_threshold", 4.0)),
            max_dv_per_event_ms        = float(mc.get("max_dv_per_event_ms",   2.0)),
            max_maneuvers_per_week     = int(mc.get("max_maneuvers_per_week",   3)),
            min_hours_before_tca       = float(mc.get("min_hours_before_tca",  4.0)),
            max_hours_before_tca       = float(mc.get("max_hours_before_tca",  72.0)),
            blackout_windows           = windows,
            scoring_weights            = weights,
            operational_philosophy     = str(data.get("operational_philosophy", "balanced")),
            fleet_priority_method      = str(fp.get("method", "highest_pc_first")),
            mission_lifetime_days_total = (
                float(data["mission_lifetime_days_total"])
                if data.get("mission_lifetime_days_total") is not None
                else None
            ),
        )

    # -- Blackout window merge --------------------------------------------

    def merge_blackout_windows(
        self, cadence_windows: List[dict]
    ) -> List[BlackoutWindow]:
        """
        Merge operator-level and satellite-level blackout windows.

        Timing data is PRESERVED from both sources:
        - Operator windows retain type, buffer hours, and start/end times.
        - Satellite cadence windows retain start_utc / end_utc from dict.

        Must be called in the planner before constraint checking.
        Do NOT apply either source in isolation -- both must be merged.

        Parameters
        ----------
        cadence_windows : list of dict
            ManeuverCadence.no_burn_windows from the satellite profile.
            Each dict must have 'start_utc' and 'end_utc' keys.

        Returns
        -------
        List[BlackoutWindow]
            Unified list with source provenance preserved on each entry.
        """
        merged: List[BlackoutWindow] = list(self.blackout_windows)

        for w in cadence_windows:
            if not isinstance(w, dict):
                continue
            merged.append(BlackoutWindow(
                window_type         = str(w.get("type", "satellite_cadence")),
                start_utc           = w.get("start_utc"),
                end_utc             = w.get("end_utc"),
                buffer_hours_before = float(w.get("buffer_hours_before", 0.0)),
                buffer_hours_after  = float(w.get("buffer_hours_after",  0.0)),
                source              = "satellite_cadence",
            ))

        return merged

    # -- Tier 2: embedded physics gates -----------------------------------

    def passes_pre_screen(self, mahalanobis_distance: float) -> bool:
        """
        Pre-screen: skip events outside the risk-relevant Mahalanobis radius.
        Runs before the optimizer to avoid scoring trivially safe events.
        Not configurable -- physics-based gate.
        """
        return mahalanobis_distance <= self.mahalanobis_screen_threshold

    def is_maneuver_required(
        self, pc: float, miss_distance_km: float
    ) -> bool:
        """
        Hard trigger: Pc threshold OR miss distance floor.
        Either condition alone is sufficient to require a maneuver.
        Not configurable -- physics-based gate.
        """
        return (
            pc >= self.pc_maneuver_threshold
            or miss_distance_km < self.min_miss_distance_km
        )

    def is_monitor_only(self, pc: float) -> bool:
        """
        True if the event is above the monitor threshold but below the
        maneuver threshold. These events are escalated to watch status
        but do not require immediate action.
        APS v2.5 Pc-threshold monitor case (was TODO in v2.4 server.py).
        """
        return self.pc_monitor_threshold <= pc < self.pc_maneuver_threshold

    def to_dict(self) -> dict:
        return {
            "operator_id":                  self.operator_id,
            "policy_version":               self.policy_version,
            "pc_maneuver_threshold":        self.pc_maneuver_threshold,
            "pc_monitor_threshold":         self.pc_monitor_threshold,
            "min_miss_distance_km":         self.min_miss_distance_km,
            "mahalanobis_screen_threshold": self.mahalanobis_screen_threshold,
            "max_dv_per_event_ms":          self.max_dv_per_event_ms,
            "max_maneuvers_per_week":       self.max_maneuvers_per_week,
            "min_hours_before_tca":         self.min_hours_before_tca,
            "max_hours_before_tca":         self.max_hours_before_tca,
            "blackout_windows":             [w.to_dict() for w in self.blackout_windows],
            "scoring_weights":              self.scoring_weights.to_dict(),
            "operational_philosophy":       self.operational_philosophy,
            "fleet_priority_method":        self.fleet_priority_method,
            "mission_lifetime_days_total":  self.mission_lifetime_days_total,
        }
