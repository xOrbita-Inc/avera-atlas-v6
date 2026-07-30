"""SCRUM-392: the single definition of the demo conjunction scenarios.

Why this module exists
---------------------
These presets used to be maintained as two hand-copied literals, ``defs`` in
``services/ui/app/main.py`` and ``_PRESETS`` in
``services/propagator/tests/test_demo_asset_propagation.py``, plus a third
generator in ``demo/demo_scenarios.py`` with different geometry entirely. The
two literals happened to agree and nothing enforced it, so tuning the live
preset would have left the regression baseline silently describing a demo that
no longer existed.

This module is the definition. The UI's live scenario endpoint builds from it,
the propagator regression test imports it by path, and ``demo/demo_scenarios.py``
is a thin CLI over it. It lives here rather than under ``libs/`` because the UI
is the only runtime consumer, and moving it to ``libs/`` would force the UI onto
a repo-root Docker build context that ``.dockerignore`` currently excludes.

TCA must land on the sample grid
--------------------------------
Each object is specified by the time of closest approach rather than by a start
distance, because closest approach is only *observed* at a sample instant. The
propagator finds TCA by argmin over samples at ``SAMPLE_DT_S``, so if the true
closest approach falls between samples, the reported miss is dominated by the
leftover separation along the approach axis rather than by the miss the preset
asked for.

That is what went wrong before SCRUM-392. Every object had a true TCA at 1000 s
against a 60 s grid, so the nearest sample sat 20 s away and the approach axis
contributed ``v_approach * 20 s`` to the miss. The critical preset's first object
asked for a 22 m miss and the propagator reported 400 m, 18x larger. The stated
miss geometry was decoration.

``t_star_s`` is therefore required to be an exact multiple of ``SAMPLE_DT_S``,
asserted in ``test_demo_asset_propagation.py``, and the start distance is
derived as ``t_star_s * v_approach_km_s`` rather than given independently. It is
not possible to express an off-grid preset in this format.

Confidences are fixed, not random
---------------------------------
They used to be ``np.random.uniform(0.75, 0.98, n)`` here and
``np.random.RandomState(42).uniform(...)`` in the test. So the live demo drew
fresh confidences on every run while the test used a seeded draw. Since the
propagator scales debris position uncertainty by confidence, the live demo's Pc
and risk labels varied run to run and the test's baseline could not have
described the live output even in principle. The values below are the seeded
draw, frozen, so both paths are identical and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# --- Sampling grid, shared with the propagator -------------------------------
SAMPLE_DT_S = 60.0
N_STEPS = 1440  # 24 hours

# --- Asset: circular equatorial orbit at 500 km (SCRUM-370 steps 1-2) --------
MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6371.0        # km
ASSET_ALT_KM = 500.0
ASSET_R_MAG_KM = R_EARTH + ASSET_ALT_KM
ASSET_V_CIRC_KM_S = float(np.sqrt(MU_EARTH / ASSET_R_MAG_KM))

ASSET_R_ECI_KM = [ASSET_R_MAG_KM, 0.0, 0.0]
ASSET_V_ECI_KM_S = [0.0, ASSET_V_CIRC_KM_S, 0.0]


# SCRUM-391: the 1-sigma position uncertainty of a secondary that has actually
# been tracked, rather than one known only from a TLE.
#
# Chosen on grounds, then measured, not the other way round. It is an order of
# magnitude better than DEFAULT_DEBRIS_UNCERTAINTY_M (2000 m), which is the
# TLE-grade figure the propagator falls back to, and it sits in the range a CDM
# for a well-tracked LEO object carries a day or two before TCA.
#
# The choice is not sensitive. Pc against the critical preset's own misses moves
# less than 2x across a 5x change in this value, because once sigma approaches
# the miss distances the probability saturates:
#
#     sigma     Pc at 22 m    Pc at 100 m    Pc at 316 m
#      500 m     7.36e-04       7.14e-04       5.31e-04
#      250 m     1.06e-03       1.02e-03       6.63e-04
#      100 m     1.21e-03       1.15e-03       7.08e-04
#
# So anything in that band gives the same risk labels. This is not a number tuned
# to clear PC_RED_THRESHOLD; the threshold is cleared by a wide margin either way.
TRACKED_SECONDARY_SIGMA_M = 250.0


@dataclass(frozen=True)
class DebrisSpec:
    """One demo debris object, specified by when and how close it passes.

    The relative motion is linear: ``rel(t) = rel0 + vrel * t`` with
    ``vrel = (-v_approach, 0, 0)`` and ``rel0 = (t_star_s * v_approach,
    miss_y_km, miss_z_km)``. At ``t_star_s`` the approach axis has closed
    exactly, so the separation is ``hypot(miss_y_km, miss_z_km)``, which is the
    miss this object is asking for.

    t_star_s must be a multiple of SAMPLE_DT_S, or the propagator's argmin over
    samples will not see that separation. See the module docstring.

    position_sigma_m is optional. When set, the propagator uses it directly as
    the debris 1-sigma position uncertainty and does not scale it by confidence,
    because a supplied sigma *is* how well the object is known and confidence is
    only a stand-in for the same thing. When left None, the propagator falls back
    to DEFAULT_DEBRIS_UNCERTAINTY_M / confidence, unchanged from before
    SCRUM-391.
    """

    t_star_s: float
    miss_y_km: float
    miss_z_km: float
    v_approach_km_s: float
    confidence: float
    position_sigma_m: float | None = None

    @property
    def start_dist_km(self) -> float:
        return self.t_star_s * self.v_approach_km_s

    @property
    def miss_km(self) -> float:
        return float(np.hypot(self.miss_y_km, self.miss_z_km))

    @property
    def tca_index(self) -> int:
        return int(round(self.t_star_s / SAMPLE_DT_S))


# Confidences are the frozen np.random.RandomState(42).uniform(0.75, 0.98, n)
# draw that the regression test used before SCRUM-392, kept so the only thing
# this change moves is the miss geometry.
_C = (0.836144, 0.968664, 0.918359, 0.887691, 0.785884)

# TCA indices are unchanged from the pre-SCRUM-392 baseline (17, and 20 for the
# third nominal object), so recapturing the baseline shows movement only in miss
# distance and the Pc that follows from it.
_T17 = 17 * SAMPLE_DT_S  # 1020.0 s
_T20 = 20 * SAMPLE_DT_S  # 1200.0 s

PRESETS: Dict[str, List[DebrisSpec]] = {
    # All objects tens of km away. Nothing should register.
    "nominal": [
        DebrisSpec(_T17, 50.0, 0.0, 0.20, _C[0]),
        DebrisSpec(_T17, 30.0, 10.0, 0.15, _C[1]),
        DebrisSpec(_T20, 80.0, 5.0, 0.25, _C[2]),
    ],
    # Kilometre-scale misses, close enough to screen and be dismissed.
    "warning": [
        DebrisSpec(_T17, 2.2, 0.3, 0.05, _C[0]),
        DebrisSpec(_T17, 2.8, 0.2, 0.08, _C[1]),
        DebrisSpec(_T17, 3.0, 0.1, 0.03, _C[2]),
        DebrisSpec(_T17, 2.5, 0.0, 0.06, _C[3]),
    ],
    # The closest geometry the preset format expresses.
    #
    # SCRUM-391: the first two objects carry a tracked covariance and reach RED.
    # The last two are the same close geometry at TLE-grade uncertainty and stay
    # AMBER, which is the point worth showing. Risk is not geometry alone, it is
    # geometry against how well you know where the object is. Two objects 100 m
    # and 316 m away are a lower graded risk than one 22 m away only because we
    # know less about them.
    "critical": [
        DebrisSpec(_T17, 0.02, 0.01, 0.02, _C[0], TRACKED_SECONDARY_SIGMA_M),
        DebrisSpec(_T17, 0.05, 0.00, 0.04, _C[1], TRACKED_SECONDARY_SIGMA_M),
        DebrisSpec(_T17, 0.30, 0.10, 0.06, _C[2]),
        DebrisSpec(_T17, 0.10, 0.00, 0.03, _C[3]),
    ],
    # A spread, for showing the operator list with more than one severity. The
    # first object is tracked (SCRUM-391) so this preset spans all four tiers.
    "mixed": [
        DebrisSpec(_T17, 0.05, 0.01, 0.025, _C[0], TRACKED_SECONDARY_SIGMA_M),
        DebrisSpec(_T17, 2.50, 0.20, 0.050, _C[1]),
        DebrisSpec(_T17, 3.50, 0.00, 0.080, _C[2]),
        DebrisSpec(_T17, 1.50, 0.00, 0.040, _C[3]),
        DebrisSpec(_T17, 50.0, 0.00, 0.200, _C[4]),
    ],
}

PRESET_LABELS = {"nominal": "NOM", "warning": "WRN", "critical": "CRT", "mixed": "MIX"}


# --- The curated three-object preset (SCRUM-370 step 3) ----------------------
# Specified as a target miss in the asset's RTN frame at closest approach rather
# than as an ECI offset, so the debris initial state is constructed in closed
# form against the asset's circular orbit. Kept in this module so the grid
# assertion covers it too; it was previously defined only in main.py and no test
# touched it.
@dataclass(frozen=True)
class CuratedSpec:
    name: str
    miss_rtn_km: Tuple[float, float, float]  # radial, along-track, cross-track
    t_star_s: float
    v_approach_km_s: float
    confidence: float
    position_sigma_m: float | None = None

    @property
    def miss_km(self) -> float:
        return float(np.linalg.norm(self.miss_rtn_km))

    @property
    def tca_index(self) -> int:
        return int(round(self.t_star_s / SAMPLE_DT_S))


CURATED: List[CuratedSpec] = [
    CuratedSpec("OBJ-DEMO-ALWAYS", (0.30, 0.0, 0.10), 7200.0, 0.5, 0.9),
    CuratedSpec("OBJ-DEMO-FLIP", (0.10, 0.0, 0.40), 3600.0, 0.5, 0.9),
    CuratedSpec("OBJ-DEMO-NEVER", (0.20, 0.0, 3.00), 3600.0, 0.5, 0.9),
]

SCENARIOS: Tuple[str, ...] = tuple(PRESETS) + ("demo",)
DEFAULT_SCENARIO = "mixed"


# --- State construction ------------------------------------------------------

def _build_tuple_preset(specs: Sequence[DebrisSpec], label: str):
    """Linear-relative-motion states for one of the four tuple presets."""
    asset_r = np.asarray(ASSET_R_ECI_KM, dtype=float)
    asset_v = np.asarray(ASSET_V_ECI_KM_S, dtype=float)
    obj_ids, r_list, v_list, confidences, sigmas = [], [], [], [], []
    for i, s in enumerate(specs):
        obj_ids.append(f"OBJ-{label}-{i:03d}")
        r_list.append([
            asset_r[0] + s.start_dist_km,
            asset_r[1] + s.miss_y_km,
            asset_r[2] + s.miss_z_km,
        ])
        v_list.append([asset_v[0] - s.v_approach_km_s, asset_v[1], asset_v[2]])
        confidences.append(s.confidence)
        sigmas.append(_sigma_or_nan(s))
    return obj_ids, r_list, v_list, confidences, sigmas


def _build_curated():
    """Closed-form states for the curated preset.

    Relative velocity is along the asset's transverse direction at t_star and
    the target miss has no along-track component, so the relative velocity is
    perpendicular to the target and closest approach is forced onto t_star.
    """
    asset_r = np.asarray(ASSET_R_ECI_KM, dtype=float)
    asset_v = np.asarray(ASSET_V_ECI_KM_S, dtype=float)
    n = np.sqrt(MU_EARTH / ASSET_R_MAG_KM ** 3)  # mean motion

    obj_ids, r_list, v_list, confidences, sigmas = [], [], [], [], []
    for s in CURATED:
        theta = n * s.t_star_s
        r_hat = np.array([np.cos(theta), np.sin(theta), 0.0])
        t_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
        n_hat = np.array([0.0, 0.0, 1.0])
        rot = np.column_stack([r_hat, t_hat, n_hat])
        target_eci = rot @ np.asarray(s.miss_rtn_km, dtype=float)
        vrel_eci = s.v_approach_km_s * t_hat
        rel0 = target_eci - vrel_eci * s.t_star_s
        obj_ids.append(s.name)
        r_list.append((asset_r + rel0).tolist())
        v_list.append((asset_v + vrel_eci).tolist())
        confidences.append(s.confidence)
        sigmas.append(_sigma_or_nan(s))
    return obj_ids, r_list, v_list, confidences, sigmas


def _sigma_or_nan(spec) -> float:
    """NaN is the artifact-level marker for "no covariance supplied".

    The npz carries a plain float array, so absence has to be representable as a
    number. The propagator treats non-finite or non-positive as absent and falls
    back to the confidence-scaled default.
    """
    sigma = getattr(spec, "position_sigma_m", None)
    return float("nan") if sigma is None else float(sigma)


def build_scenario(scenario: str):
    """Build one scenario's debris states.

    Returns (object_ids, r_eci_km, v_eci_km_s, confidences, position_sigma_m).
    The last is 1-sigma position uncertainty in metres per object, NaN where the
    scenario does not supply one. Unknown names fall back to DEFAULT_SCENARIO,
    matching the previous behaviour of both callers.
    """
    if scenario == "demo":
        return _build_curated()
    if scenario not in PRESETS:
        scenario = DEFAULT_SCENARIO
    return _build_tuple_preset(PRESETS[scenario], PRESET_LABELS[scenario])


def specs_for(scenario: str) -> Sequence:
    """The spec objects behind a scenario, for tests and for CLI reporting."""
    return CURATED if scenario == "demo" else PRESETS.get(scenario, PRESETS[DEFAULT_SCENARIO])


def all_specs():
    """Every spec across every scenario, for whole-set assertions."""
    for name in PRESETS:
        for s in PRESETS[name]:
            yield name, s
    for s in CURATED:
        yield "demo", s
