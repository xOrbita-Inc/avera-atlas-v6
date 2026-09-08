"""
SCRUM-379 -- the propagation seam around the SCRUM-378 Validity Assessor.

378 does no propagation and imports nothing from tracker, by design: John's
Option 2 decision was that the caller hands it states already at the epoch they
are needed at. This module is that caller. It is the main new code 379 adds
around 378, and it exists for one reason.

conjunction_plane_epsilon requires w_position_eci to be at the SAME epoch as
r_rel_km / v_rel_km_s. Both are at TCA. An IOD solution is at its own solution
epoch, which is earlier, often by hours. Feeding the IOD-epoch state in as the
Gramian's t0 does not raise: it returns a plausible epsilon computed against
the wrong geometry, and the validity gate silently certifies the wrong thing.
So the target state is propagated to TCA here, before build_validity_verdict_for_arc
is called, and the epoch it was propagated to is recorded on the result so an
auditor can check it rather than trust it.

Propagation uses the planner's own common/orbit_propagation.kepler_propagate.
That keeps the tracker/validity decoupling intact: nothing here imports from
services/tracker either.
"""
from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from common.decision_state_machine import ValidityRouting
from common.orbit_propagation import MU_EARTH, kepler_propagate


def _import_validity(module_name: str):
    """Import services/validity, whether running from the repo or the image.

    In the planner image services/validity is copied to /app/validity, so the
    plain import works. In a repo checkout services/ is not on sys.path, so it
    is appended -- appended, not inserted, so nothing already importable is
    shadowed by a same-named module under services/.
    """
    try:
        return importlib.import_module(f"validity.{module_name}")
    except ModuleNotFoundError:
        services_dir = str(Path(__file__).resolve().parents[2])
        if services_dir not in sys.path:
            sys.path.append(services_dir)
        return importlib.import_module(f"validity.{module_name}")


_wrapper = _import_validity("wrapper")
_routing = _import_validity("routing")

ObservationEpochState = _wrapper.ObservationEpochState
build_validity_verdict_for_arc = _wrapper.build_validity_verdict_for_arc
validity_evidence_values = _wrapper.validity_evidence_values
DEFAULT_PHENOMENOLOGIES_USED = _wrapper.DEFAULT_PHENOMENOLOGIES_USED
RoutingDecision = _routing.RoutingDecision
route_validity_verdict = _routing.route_validity_verdict

# The planner's local mirror of RoutingDecision must not drift from the real
# one, or the validity guard would compare against values that never occur.
assert {r.value for r in ValidityRouting} == {r.value for r in RoutingDecision}, (
    "decision_state_machine.ValidityRouting and services/validity's "
    "RoutingDecision have drifted apart"
)

# SCRUM-333 section 7, locked: the EARNED floor for LEO.
LEO_EPSILON_THRESHOLD: float = 0.20


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ArcObservation:
    """One observation in the tracked object's arc.

    Timestamps and observer geometry only. The target's state at this epoch is
    not supplied by the caller: this module propagates it, which is the whole
    point of the seam.
    """

    epoch_utc: datetime
    r_observer_km: np.ndarray
    ra_sigma_rad: float
    dec_sigma_rad: float
    range_sigma_km: Optional[float] = None


@dataclass(frozen=True)
class ValidityAssessment:
    """A routed validity verdict, plus proof of the epoch it was computed at.

    target_state_epoch_utc is not decoration. It is the field an auditor reads
    to confirm the Gramian's t0 was TCA and not the IOD solution epoch, which
    is the failure this module exists to prevent and which no exception would
    have caught.
    """

    verdict: Any
    routing: ValidityRouting
    evidence: Dict[str, Any]
    target_state_epoch_utc: str
    r_target_tca_km: Tuple[float, float, float]
    v_target_tca_km_s: Tuple[float, float, float]
    a_km: float
    propagated_dt_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing": self.routing.value,
            "target_state_epoch_utc": self.target_state_epoch_utc,
            "propagated_dt_s": self.propagated_dt_s,
            "a_km": self.a_km,
            **self.evidence,
        }


def semi_major_axis_km(r_km: np.ndarray, v_km_s: np.ndarray) -> float:
    """Reference semi-major axis from a state, by vis-viva.

    build_validity_verdict_for_arc takes a_km as the caller's responsibility
    and does not derive one. Deriving it from the target's own state at TCA is
    the only defensible default: the CW/state-transform assumption the Gramian
    rests on is about this orbit, not a nominal one.
    """
    r_mag = float(np.linalg.norm(r_km))
    v_sq = float(np.dot(v_km_s, v_km_s))
    denominator = 2.0 / r_mag - v_sq / MU_EARTH
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(
            "cannot derive a reference semi-major axis from an unbound state"
        )
    return 1.0 / denominator


def propagate_target_to_tca(
    r_target_km: np.ndarray,
    v_target_km_s: np.ndarray,
    state_epoch_utc: datetime,
    tca_utc: datetime,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Propagate a target state from its own epoch to TCA.

    Returns (r_tca_km, v_tca_km_s, dt_s). dt_s is positive when TCA is after
    the state's epoch, which is the normal case for an IOD solution.
    """
    epoch = _utc(state_epoch_utc, "state_epoch_utc")
    tca = _utc(tca_utc, "tca_utc")
    dt_s = (tca - epoch).total_seconds()
    r_tca, v_tca = kepler_propagate(
        np.asarray(r_target_km, dtype=float),
        np.asarray(v_target_km_s, dtype=float),
        dt_s,
    )
    return r_tca, v_tca, dt_s


def assess_validity_at_tca(
    *,
    r_target_km: np.ndarray,
    v_target_km_s: np.ndarray,
    state_epoch_utc: datetime,
    tca_utc: datetime,
    observations: Sequence[ArcObservation],
    r_rel_km_at_tca: np.ndarray,
    v_rel_km_s_at_tca: np.ndarray,
    epsilon_threshold: float = LEO_EPSILON_THRESHOLD,
    a_km: Optional[float] = None,
    phenomenologies_used: Optional[Sequence[str]] = None,
) -> ValidityAssessment:
    """Propagate to TCA, build the validity verdict, and route it.

    r_target_km / v_target_km_s are the tracked object's state at
    state_epoch_utc -- normally the IOD solution and its own solution epoch.
    They are NOT assumed to be at TCA; that is exactly what this function fixes.

    Every observation's target state is propagated too, each to its own epoch,
    and its dt_s is measured from TCA (negative for an observation that
    precedes TCA, as 378's docstring expects).

    Args:
        observations: the arc, in any order. An empty arc is rejected rather
            than producing a singular Gramian and an epsilon of zero that would
            read as a genuine NOT_EARNED.
        epsilon_threshold: the per-regime EARNED floor. Defaults to the locked
            LEO value, 0.20.
        a_km: reference semi-major axis. Derived from the target's state at TCA
            when omitted.

    Returns:
        ValidityAssessment carrying the verdict, its routing decision, the
        SCRUM-378 evidence values, and the epoch the Gramian's t0 was at.
    """
    if not observations:
        raise ValueError("an arc with no observations cannot be assessed")

    tca = _utc(tca_utc, "tca_utc")
    epoch = _utc(state_epoch_utc, "state_epoch_utc")

    r_tca, v_tca, dt_to_tca_s = propagate_target_to_tca(
        r_target_km, v_target_km_s, epoch, tca
    )

    epoch_states = []
    for obs in observations:
        obs_epoch = _utc(obs.epoch_utc, "observation epoch_utc")
        r_obs_target, v_obs_target = kepler_propagate(
            np.asarray(r_target_km, dtype=float),
            np.asarray(v_target_km_s, dtype=float),
            (obs_epoch - epoch).total_seconds(),
        )
        epoch_states.append(
            ObservationEpochState(
                r_target_km=r_obs_target,
                v_target_km_s=v_obs_target,
                # Measured from TCA, not from the solution epoch: TCA is the
                # Gramian's t0.
                dt_s=(obs_epoch - tca).total_seconds(),
                r_observer_km=np.asarray(obs.r_observer_km, dtype=float),
                ra_sigma_rad=obs.ra_sigma_rad,
                dec_sigma_rad=obs.dec_sigma_rad,
                range_sigma_km=obs.range_sigma_km,
            )
        )

    reference_a_km = a_km if a_km is not None else semi_major_axis_km(r_tca, v_tca)

    verdict = build_validity_verdict_for_arc(
        r_target_tca_km=r_tca,
        v_target_tca_km_s=v_tca,
        a_km=reference_a_km,
        observation_epochs=epoch_states,
        r_rel_km_at_tca=np.asarray(r_rel_km_at_tca, dtype=float),
        v_rel_km_s_at_tca=np.asarray(v_rel_km_s_at_tca, dtype=float),
        epsilon_threshold=epsilon_threshold,
        phenomenologies_used=(
            list(phenomenologies_used) if phenomenologies_used is not None else None
        ),
    )

    return ValidityAssessment(
        verdict=verdict,
        routing=ValidityRouting(route_validity_verdict(verdict).value),
        evidence=validity_evidence_values(verdict),
        target_state_epoch_utc=_iso(tca),
        r_target_tca_km=tuple(float(c) for c in r_tca),
        v_target_tca_km_s=tuple(float(c) for c in v_tca),
        a_km=reference_a_km,
        propagated_dt_s=dt_to_tca_s,
    )
