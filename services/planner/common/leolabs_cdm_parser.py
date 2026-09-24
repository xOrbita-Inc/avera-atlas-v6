"""services/planner/common/leolabs_cdm_parser.py

Parse a LeoLabs CCSDS CDM (delivered as field-keyed JSON) into the planner's
conjunction record (SCRUM-411, design section 6).

What is LeoLabs-specific here
-----------------------------
- The JSON is fully flattened with per-object SAT1_ and SAT2_ prefixes. There is
  no OBJECT1/OBJECT2 nesting. Every per-object field is a prefixed top-level key
  (SAT1_X, SAT1_CR_R, SAT1_COVARIANCE_METHOD, SAT1_COMMENT_OBJECT_RADIUS, ...).
- COVARIANCE_METHOD and REF_FRAME are per object, not top-level.
- The only complete covariance is the RTN 6x6 (SATn_CR_R .. SATn_CNDOT_NDOT), in
  m^2. The EME2000 covariance appears only as diagonal comments
  (SATn_COMMENT_CX_X/CY_Y/CZ_Z and the velocity diagonals). So this parser
  rotates RTN to ECI and uses those diagonal comments as a built-in regression
  check on the rotation (design section 6.2), never as a covariance source.
- SATn_OBJECT_DESIGNATOR is the LeoLabs catalog id (e.g. L2669), not NORAD.
- Object1 is not necessarily our asset. Primary/secondary is resolved by matching
  our subscribed sat's LeoLabs catalog id against SATn_OBJECT_DESIGNATOR
  (design section 6.3).

Reuse, not fork
---------------
The RTN->ECI rotation is aps_math.frames.rtn_to_eci_rotation and the Pc math is
aps_math.pc_utils.compute_pc. This module adds no second copy of either
(design section 8). If the shared math were wrong, the fix belongs there.

Output contract
---------------
to_conjunction_state() returns the same dict the TIROS-4 fixture path already
feeds the scorer (obj_id, t_ca_utc, r_rel_km, v_rel_km_s, v_rel_source,
p_rel_km2, pc_precomputed), so the scoring engine is unchanged (design section
6.1). pc_precomputed is left None on purpose: the CDM's COLLISION_PROBABILITY is
a cross-check, not our Pc (design sections 6.1 / 6.5), so the planner computes Pc
from the real geometry. The CDM value is carried in provenance for the audit
trail and the golden cross-check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from aps_math import frames
from aps_math.pc_utils import compute_pc

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tolerances and constants
# ---------------------------------------------------------------------------

# Guard tolerance for the "rotated ECI diagonal matches the CDM's EME2000
# diagonal comments" check. The design confirms agreement to ~0.000% on live
# data; 1e-3 (0.1%) leaves margin for float noise while still tripping on any
# real frame/convention regression.
_DIAG_RTOL = 1.0e-3

# Guard tolerance for "miss distance recomputed from the two ECI states matches
# the CDM's MISS_DISTANCE".
_MISS_RTOL = 1.0e-3
_MISS_ATOL_M = 1.0

# Symmetry / positive-semidefiniteness tolerances for the rotated covariance.
_SYM_RTOL = 1.0e-6
_PSD_EIG_RTOL = 1.0e-8

# SCRUM-458: the band inside which a negative eigenvalue is numerical noise
# rather than a real covariance fault, and may be repaired by clipping to zero.
#
# LeoLabs writes the covariance into a result CDM at fixed precision. Our
# submitted post-burn covariance (SCRUM-452) is strongly anisotropic over 72 h --
# kilometres along-track, metres radially -- so the round trip through that
# precision, plus the RTN->ECI rotation, drives the small axes slightly negative.
# Measured over all 1255 result CDMs of screening 603312: 1249 of them have a
# negative minimum eigenvalue, and the largest |min_eig| / max|eig| seen is
# 5.37e-3 (0.54%), against a median of 2.0e-9. 1e-2 covers every observed case
# with 1.86x in hand and is still two orders of magnitude tighter than the
# anisotropy itself, so a genuinely broken covariance does not slip through.
#
# The direction matters and is not symmetric. Clipping a negative eigenvalue to
# zero claims *more* certainty along that axis, which raises the Mahalanobis
# distance and therefore pushes toward CLEAR. That is only acceptable for a
# negative of numerical scale; beyond this band the covariance is flagged and the
# event fails the screen closed instead (see _repair_or_flag_psd).
_PSD_REPAIR_RTOL = 1.0e-2

_CALCULATED = "CALCULATED"
_EME2000 = "EME2000"

# CCSDS RTN 6x6 lower-triangle field suffixes, in row order
# R, T, N, RDOT, TDOT, NDOT. Index i>=j gives element (i, j).
_RTN_AXES = ("R", "T", "N", "RDOT", "TDOT", "NDOT")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LeoLabsParseError(ValueError):
    """The CDM could not be parsed (missing fields, bad object resolution)."""


class LeoLabsGuardError(LeoLabsParseError):
    """A guard failed: the event must not be scored.

    Covers a non-CALCULATED covariance method, a non-EME2000 frame, a covariance
    that is not symmetric / positive-semidefinite after rotation, or a rotated
    covariance whose diagonal disagrees with the CDM's own EME2000 diagonal
    comments (design section 6.4).
    """


# ---------------------------------------------------------------------------
# Parsed result
# ---------------------------------------------------------------------------

@dataclass
class ParsedObject:
    """One object's parsed state and covariance, primary or secondary."""

    role: str                      # "primary" or "secondary"
    sat_key: str                   # "SAT1" or "SAT2"
    designator: str                # LeoLabs catalog id, e.g. "L2669"
    norad_id: Optional[int]
    object_name: Optional[str]
    covariance_method: str
    ref_frame: str
    ephemeris_source: Optional[str]
    radius_m: float
    r_km: np.ndarray               # ECI position, km
    v_km_s: np.ndarray             # ECI velocity, km/s
    cov_rtn_m2: np.ndarray         # 6x6 RTN covariance, m^2
    cov_eci_m2: np.ndarray         # 6x6 ECI covariance (rotated), m^2

    # SCRUM-458: how the rotated covariance came out of the PSD check.
    # covariance_repaired: negative eigenvalues of numerical scale were clipped
    # to zero, and cov_eci_m2 below is the repaired matrix.
    # covariance_untrusted: the covariance is non-PSD beyond the repair band and
    # was left exactly as read. Covariance-derived quantities (Pc, Mahalanobis)
    # must not be believed for this object; the event is still carried so that it
    # can fail closed rather than vanish from the result.
    covariance_repaired: bool = False
    covariance_untrusted: bool = False
    covariance_min_eigenvalue_m2: Optional[float] = None

    @property
    def cov_eci_pos_m2(self) -> np.ndarray:
        """Top-left 3x3 position block of the ECI covariance, m^2."""
        return self.cov_eci_m2[:3, :3]


@dataclass
class ParsedLeoLabsCDM:
    """Everything the parser produces for one CDM."""

    primary: ParsedObject
    secondary: ParsedObject
    t_ca_utc: str
    combined_hbr_m: float
    cdm_collision_probability: Optional[float]
    cdm_collision_probability_method: Optional[str]
    miss_distance_m: Optional[float]
    provenance: Dict[str, Any] = field(default_factory=dict)

    # SCRUM-417: RTN relative geometry straight from the CDM, for the encounter viz.
    relative_position_rtn_m: Optional[List[float]] = None
    relative_velocity_rtn_m_s: Optional[List[float]] = None

    # -- covariance conditioning (SCRUM-458) -----------------------------

    @property
    def covariance_untrusted(self) -> bool:
        """True when either object's covariance is non-PSD beyond the repair band.

        The event is still a real conjunction with a real miss distance; only the
        covariance-derived limbs of the clear contract are unusable. Consumers
        must treat it as not assessable-clear rather than dropping it.
        """
        return bool(self.primary.covariance_untrusted
                    or self.secondary.covariance_untrusted)

    @property
    def covariance_repaired(self) -> bool:
        """True when either object's covariance had numerical negatives clipped."""
        return bool(self.primary.covariance_repaired
                    or self.secondary.covariance_repaired)

    def covariance_conditioning(self) -> Dict[str, Any]:
        """Per-object PSD outcome, for the audit trail and the operator note."""
        return {
            o.role: {
                "sat_key": o.sat_key,
                "designator": o.designator,
                "repaired": o.covariance_repaired,
                "untrusted": o.covariance_untrusted,
                "min_eigenvalue_m2": o.covariance_min_eigenvalue_m2,
            }
            for o in (self.primary, self.secondary)
        }

    # -- relative geometry, ECI ------------------------------------------

    def r_rel_km(self) -> np.ndarray:
        """Secondary minus primary position, ECI, km."""
        return self.secondary.r_km - self.primary.r_km

    def v_rel_km_s(self) -> np.ndarray:
        """Secondary minus primary velocity, ECI, km/s."""
        return self.secondary.v_km_s - self.primary.v_km_s

    def p_rel_eci_km2(self) -> np.ndarray:
        """Combined relative position covariance, ECI, km^2 (3x3).

        Independence assumed: the sum of the two per-object ECI position
        covariances, matching the existing ingest contract. compute_pc splits a
        combined covariance in half per object, so combined == cov1 + cov2 is the
        quantity the scorer expects.
        """
        return (self.primary.cov_eci_pos_m2 + self.secondary.cov_eci_pos_m2) / 1.0e6

    # -- output contract --------------------------------------------------

    def to_conjunction_state(self) -> Dict[str, Any]:
        """Return the ConjunctionState dict the scorer already consumes.

        Matches services/ingest/cdm_to_conjunction.cdm_to_conjunction_state and
        common/udl_client._parse_conjunction so /v1/evaluate is unchanged.
        """
        return {
            "obj_id": self.secondary.designator,
            "t_ca_utc": self.t_ca_utc,
            "r_rel_km": self.r_rel_km().tolist(),
            "v_rel_km_s": self.v_rel_km_s().tolist(),
            "v_rel_source": "state_vector_difference",
            "p_rel_km2": self.p_rel_eci_km2().flatten().tolist(),
            # Intentionally None: the CDM Pc is a cross-check, not our Pc. The
            # planner computes Pc from this real geometry.
            "pc_precomputed": None,
        }

    def to_response_conjunction(self) -> Dict[str, Any]:
        """SCRUM-417: identity and RTN relative geometry for the evaluate response.

        Additive block, populated on the live path only, so the UI can name the
        secondary object and draw the true encounter geometry instead of the
        asset-NORAD fallback.
        """
        def _obj(o: ParsedObject) -> Dict[str, Any]:
            return {
                "designator": o.designator,
                "norad_id": o.norad_id,
                "object_name": o.object_name,
            }
        miss_km = (self.miss_distance_m / 1000.0) if self.miss_distance_m is not None else None
        return {
            "primary": _obj(self.primary),
            "secondary": _obj(self.secondary),
            "tca_utc": self.t_ca_utc,
            "miss_distance_km": miss_km,
            "relative_position_rtn_m": self.relative_position_rtn_m,
            "relative_velocity_rtn_m_s": self.relative_velocity_rtn_m_s,
        }

    def to_store_cdm_dict(self) -> Dict[str, Any]:
        """SCRUM-429: this CDM in the flat shape the ingest store already takes.

        save_cdm_record consumes cdm_parser.parse_cdm_kvn()'s output -- a flat
        dict of CCSDS names prefixed OBJECT1_ / OBJECT2_. This emits that shape
        from data the parser has already validated, so a LeoLabs conjunction can
        land in the store (ADR-008's record of truth) alongside the injected
        reference CDM.

        No re-parse and no new physics. The six position covariance elements per
        object are read straight off the parsed 6x6 RTN covariance, whose axis
        order is R, T, N, RDOT, TDOT, NDOT -- so the store's CR_R / CT_R / CT_T /
        CN_R / CN_T / CN_N are the lower triangle of its top-left 3x3 block, and
        the LeoLabs SATn_ to store OBJECTn_ mapping is a rename, not a
        conversion. Units stay m^2, which is what the store holds.

        The designator field carries the NORAD id, because that is what the
        store is keyed and looked up by (GET /cdm/{primary}/{secondary}). An
        object with no NORAD id falls back to its LeoLabs catalog designator
        rather than to a placeholder, so the row still names something real.
        """
        def _object_key(obj: ParsedObject) -> str:
            return str(obj.norad_id) if obj.norad_id is not None else str(obj.designator)

        def _position_block(obj: ParsedObject, prefix: str) -> Dict[str, Any]:
            cov = obj.cov_rtn_m2
            return {
                f"{prefix}_CR_R": float(cov[0, 0]),
                f"{prefix}_CT_R": float(cov[1, 0]),
                f"{prefix}_CT_T": float(cov[1, 1]),
                f"{prefix}_CN_R": float(cov[2, 0]),
                f"{prefix}_CN_T": float(cov[2, 1]),
                f"{prefix}_CN_N": float(cov[2, 2]),
            }

        prov = self.provenance
        record: Dict[str, Any] = {
            "OBJECT1_OBJECT_DESIGNATOR": _object_key(self.primary),
            "OBJECT2_OBJECT_DESIGNATOR": _object_key(self.secondary),
            "OBJECT1_OBJECT_NAME": self.primary.object_name,
            "OBJECT2_OBJECT_NAME": self.secondary.object_name,
            "TCA": self.t_ca_utc,
            "MISS_DISTANCE": self.miss_distance_m,
            # The CDM's own Pc, not the planner's. The store column holds the
            # published Pc from the originating CDM, which is what this is.
            "COLLISION_PROBABILITY": self.cdm_collision_probability,
            # Carried for dedup: cdm_id identifies this exact message, event_id
            # the conjunction event it belongs to.
            "COMMENT_ID": prov.get("cdm_id"),
            "COMMENT_EVENT_ID": prov.get("event_id"),
        }
        record.update(_position_block(self.primary, "OBJECT1"))
        record.update(_position_block(self.secondary, "OBJECT2"))
        return record


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------

def _num(cdm: Dict[str, Any], key: str) -> Optional[float]:
    v = cdm.get(key)
    if v is None:
        return None
    return float(v)


def _require_num(cdm: Dict[str, Any], key: str) -> float:
    v = cdm.get(key)
    if v is None:
        raise LeoLabsParseError(f"CDM missing required numeric field {key!r}")
    return float(v)


def _designator(cdm: Dict[str, Any], sat_key: str) -> str:
    raw = cdm.get(f"{sat_key}_OBJECT_DESIGNATOR")
    if raw is None:
        raise LeoLabsParseError(f"CDM missing {sat_key}_OBJECT_DESIGNATOR")
    # Designators are strings like "L2669"; guard against numeric coercion.
    if isinstance(raw, float) and raw == int(raw):
        return str(int(raw))
    return str(raw)


def _iso_tca(cdm: Dict[str, Any]) -> str:
    """Prefer the pre-formatted ISO TCA; else normalise TCA to ISO-8601 Z."""
    iso = cdm.get("TCA_ISO")
    if iso:
        return str(iso)
    raw = cdm.get("TCA")
    if raw is None:
        raise LeoLabsParseError("CDM missing TCA")
    s = str(raw).strip().replace(" ", "T")
    if not s.endswith("Z"):
        s += "Z"
    return s


def build_rtn_covariance_6x6(cdm: Dict[str, Any], sat_key: str) -> np.ndarray:
    """Reconstruct the symmetric 6x6 RTN covariance (m^2) for one object.

    Fields are the CCSDS lower-triangle, row order R, T, N, RDOT, TDOT, NDOT,
    named SATn_C<row>_<col> (e.g. SAT1_CT_R is row T, col R). Missing off-diagonal
    terms default to 0; the diagonals must be present.
    """
    cov = np.zeros((6, 6), dtype=np.float64)
    for i, row in enumerate(_RTN_AXES):
        for j in range(i + 1):
            col = _RTN_AXES[j]
            key = f"{sat_key}_C{row}_{col}"
            val = cdm.get(key)
            if val is None:
                if i == j:
                    raise LeoLabsParseError(
                        f"CDM missing diagonal covariance element {key!r}"
                    )
                val = 0.0
            cov[i, j] = float(val)
            cov[j, i] = float(val)
    return cov


def _rotation_6x6(rot3: np.ndarray) -> np.ndarray:
    """Block-diagonal blockdiag(Rot, Rot) for rotating a 6x6 RTN covariance."""
    rot6 = np.zeros((6, 6), dtype=np.float64)
    rot6[:3, :3] = rot3
    rot6[3:, 3:] = rot3
    return rot6


# ---------------------------------------------------------------------------
# Covariance validation (design section 6.4)
# ---------------------------------------------------------------------------

def _assert_symmetric(cov: np.ndarray, label: str) -> None:
    """Symmetry stays a hard error.

    An asymmetric covariance is a real parse or convention fault, not precision
    noise: the RTN lower triangle is mirrored on read, so asymmetry means the
    mirroring or the rotation is wrong. Unlike a small negative eigenvalue there
    is no benign reading of it, so it is never repaired.
    """
    if not np.allclose(cov, cov.T, rtol=_SYM_RTOL, atol=0.0):
        raise LeoLabsGuardError(f"{label} covariance is not symmetric after rotation")


def _repair_or_flag_psd(
    obj: ParsedObject,
    *,
    strict: bool,
    event_id: Optional[Any] = None,
) -> None:
    """Make the rotated covariance usable, or mark it untrusted. SCRUM-458.

    Three outcomes, in order:

    1. Already PSD to the tight tolerance: nothing changes.
    2. Negative eigenvalues of numerical scale (|min_eig| <= _PSD_REPAIR_RTOL *
       max|eig|): clip them to zero, rebuild the symmetric matrix, and mark the
       object repaired. The event stays in the conjunction set and is evaluated
       normally, which is the whole point -- before SCRUM-458 this raised, and
       run_screening dropped 348 of 349 conjunctions on it.
    3. Negative beyond that band: this is not precision noise and must not be
       repaired toward clear. With strict=True (the live listing and evaluate
       paths, unchanged behaviour) it raises, so the event is not scored. With
       strict=False (the on-demand screen) the covariance is left exactly as read
       and the object is marked untrusted, so the caller can fail the event closed
       instead of omitting it from a result that then reads CLEAR.

    Case 3 never shrinks an uncertainty and never invents one: the matrix is not
    touched, only labelled.
    """
    sym = 0.5 * (obj.cov_eci_m2 + obj.cov_eci_m2.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    min_eig = float(np.min(eigvals))
    scale = float(np.max(np.abs(eigvals)) or 1.0)
    obj.covariance_min_eigenvalue_m2 = min_eig

    if min_eig >= -_PSD_EIG_RTOL * scale:
        return

    if abs(min_eig) <= _PSD_REPAIR_RTOL * scale:
        repaired = eigvecs @ np.diag(np.clip(eigvals, 0.0, None)) @ eigvecs.T
        obj.cov_eci_m2 = 0.5 * (repaired + repaired.T)
        obj.covariance_repaired = True
        log.debug(
            "clipped numerically negative covariance eigenvalues",
            extra={"event": "leolabs_covariance_psd_repaired",
                   "sat_key": obj.sat_key, "designator": obj.designator,
                   "role": obj.role, "event_id": event_id,
                   "min_eigenvalue_m2": min_eig,
                   "relative_to_max": abs(min_eig) / scale},
        )
        return

    detail = (
        f"{obj.sat_key} ({obj.designator}) covariance is not positive "
        f"semidefinite after rotation (min eigenvalue {min_eig:.3e} m^2, "
        f"{abs(min_eig) / scale:.3%} of the largest, beyond the "
        f"{_PSD_REPAIR_RTOL:.1%} repair band)"
    )
    if strict:
        raise LeoLabsGuardError(detail)

    obj.covariance_untrusted = True
    log.warning(
        "covariance is non-PSD beyond the repair band; covariance-derived "
        "limbs cannot be trusted for this event and it will fail closed",
        extra={"event": "leolabs_covariance_untrusted",
               "sat_key": obj.sat_key, "designator": obj.designator,
               "role": obj.role, "event_id": event_id,
               "min_eigenvalue_m2": min_eig,
               "relative_to_max": abs(min_eig) / scale,
               "detail": detail},
    )


def _assert_diagonal_matches_comments(
    obj: ParsedObject, cdm: Dict[str, Any]
) -> None:
    """Assert the rotated ECI diagonal matches the CDM's EME2000 comments.

    The design bakes this in so any future frame or convention regression trips
    immediately, independent of the Pc cross-check (design section 6.2). Both
    position (m^2) and velocity (m^2/s^2) diagonals are checked when present.
    """
    checks = (
        (obj.cov_eci_m2[0, 0], f"{obj.sat_key}_COMMENT_CX_X"),
        (obj.cov_eci_m2[1, 1], f"{obj.sat_key}_COMMENT_CY_Y"),
        (obj.cov_eci_m2[2, 2], f"{obj.sat_key}_COMMENT_CZ_Z"),
        (obj.cov_eci_m2[3, 3], f"{obj.sat_key}_COMMENT_CXDOT_XDOT"),
        (obj.cov_eci_m2[4, 4], f"{obj.sat_key}_COMMENT_CYDOT_YDOT"),
        (obj.cov_eci_m2[5, 5], f"{obj.sat_key}_COMMENT_CZDOT_ZDOT"),
    )
    for rotated, key in checks:
        expected = cdm.get(key)
        if expected is None:
            continue
        expected = float(expected)
        if not np.isclose(rotated, expected, rtol=_DIAG_RTOL, atol=0.0):
            rel = abs(rotated - expected) / abs(expected) if expected else float("inf")
            raise LeoLabsGuardError(
                f"rotated ECI diagonal disagrees with {key}: "
                f"got {rotated:.6g} m^2, CDM comment {expected:.6g} m^2 "
                f"(relative error {rel:.3%}, tol {_DIAG_RTOL:.1%}). "
                f"This indicates a frame or covariance-convention regression."
            )


# ---------------------------------------------------------------------------
# Per-object parse
# ---------------------------------------------------------------------------

def _parse_object(cdm: Dict[str, Any], sat_key: str, role: str) -> ParsedObject:
    cov_method = str(cdm.get(f"{sat_key}_COVARIANCE_METHOD", ""))
    ref_frame = str(cdm.get(f"{sat_key}_REF_FRAME", ""))

    r_km = np.array(
        [
            _require_num(cdm, f"{sat_key}_X"),
            _require_num(cdm, f"{sat_key}_Y"),
            _require_num(cdm, f"{sat_key}_Z"),
        ],
        dtype=np.float64,
    )
    v_km_s = np.array(
        [
            _require_num(cdm, f"{sat_key}_X_DOT"),
            _require_num(cdm, f"{sat_key}_Y_DOT"),
            _require_num(cdm, f"{sat_key}_Z_DOT"),
        ],
        dtype=np.float64,
    )

    cov_rtn = build_rtn_covariance_6x6(cdm, sat_key)

    # Rotation is built from this object's own state (design section 6.2). It only
    # depends on the direction of r and v, so km vs m does not matter here.
    rot3 = frames.rtn_to_eci_rotation(r_km, v_km_s)
    if frames.is_degenerate_state(r_km, v_km_s):
        raise LeoLabsGuardError(
            f"{sat_key} state is degenerate (zero position or v parallel to r); "
            f"no orbital frame exists, so the RTN covariance cannot be rotated."
        )
    rot6 = _rotation_6x6(rot3)
    cov_eci = rot6 @ cov_rtn @ rot6.T

    radius = cdm.get(f"{sat_key}_COMMENT_OBJECT_RADIUS")
    norad = cdm.get(f"{sat_key}_COMMENT_NORAD_ID")

    return ParsedObject(
        role=role,
        sat_key=sat_key,
        designator=_designator(cdm, sat_key),
        norad_id=int(norad) if norad is not None else None,
        object_name=cdm.get(f"{sat_key}_OBJECT_NAME"),
        covariance_method=cov_method,
        ref_frame=ref_frame,
        ephemeris_source=cdm.get(f"{sat_key}_COMMENT_EPHEMERIS_SOURCE"),
        radius_m=float(radius) if radius is not None else 0.0,
        r_km=r_km,
        v_km_s=v_km_s,
        cov_rtn_m2=cov_rtn,
        cov_eci_m2=cov_eci,
    )


# ---------------------------------------------------------------------------
# Object resolution (design section 6.3)
# ---------------------------------------------------------------------------

def _resolve_roles(cdm: Dict[str, Any], our_catalog_id: str) -> tuple[str, str]:
    """Return (primary_sat_key, secondary_sat_key) by matching our catalog id.

    Do not assume SAT1 is our asset. Match our subscribed sat's LeoLabs catalog
    id against SAT1_/SAT2_OBJECT_DESIGNATOR; the match is the primary.
    """
    ours = str(our_catalog_id).strip()
    d1 = _designator(cdm, "SAT1")
    d2 = _designator(cdm, "SAT2")
    if ours == d1:
        return "SAT1", "SAT2"
    if ours == d2:
        return "SAT2", "SAT1"
    raise LeoLabsParseError(
        f"our catalog id {ours!r} matches neither object "
        f"(SAT1={d1!r}, SAT2={d2!r}); cannot resolve primary/secondary."
    )


# ---------------------------------------------------------------------------
# Guards on both objects (design section 6.4)
# ---------------------------------------------------------------------------

def _check_guards(primary: ParsedObject, secondary: ParsedObject) -> None:
    for obj in (primary, secondary):
        # Refuse to score unless the covariance is CALCULATED. 18th Space CDMs in
        # the same feed can read DEFAULT; those must not be scored.
        if obj.covariance_method != _CALCULATED:
            raise LeoLabsGuardError(
                f"{obj.sat_key} ({obj.designator}) COVARIANCE_METHOD is "
                f"{obj.covariance_method!r}, not {_CALCULATED!r}; refusing to score."
            )
        # Reject any frame that is not EME2000 rather than assuming. 18th Space
        # CDMs can be ITRF.
        if obj.ref_frame != _EME2000:
            raise LeoLabsGuardError(
                f"{obj.sat_key} ({obj.designator}) REF_FRAME is "
                f"{obj.ref_frame!r}, not {_EME2000!r}; refusing to score."
            )
        # Symmetry is a hard error here. The PSD limb is deliberately not
        # checked yet: it runs after the diagonal-comment validation, because a
        # repair changes the diagonal and would otherwise trip that guard
        # (SCRUM-458 measured shifts past _DIAG_RTOL on 1992 of 7494 entries).
        _assert_symmetric(obj.cov_eci_m2, obj.sat_key)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_leolabs_cdm(
    cdm: Dict[str, Any],
    our_catalog_id: str,
    *,
    validate_diagonals: bool = True,
    validate_miss_distance: bool = True,
    strict_psd: bool = True,
) -> ParsedLeoLabsCDM:
    """Parse one LeoLabs CDM JSON object into a ParsedLeoLabsCDM.

    Parameters
    ----------
    cdm
        A single field-keyed LeoLabs CDM (one element of the cdms[] list).
    our_catalog_id
        Our subscribed sat's LeoLabs catalog id (e.g. "L2669"), used to resolve
        which object is the primary (design section 6.3).
    validate_diagonals
        If True (default), assert each object's rotated ECI covariance diagonal
        matches the CDM's own EME2000 diagonal comments (design section 6.2).
    validate_miss_distance
        If True (default), assert the miss distance recomputed from the two ECI
        states matches the CDM's MISS_DISTANCE (design section 6.4).
    strict_psd
        If True (default), a covariance that is non-PSD beyond the numerical
        repair band raises, so the event is not scored. This is the live listing
        and evaluate behaviour and is unchanged by SCRUM-458.

        If False, such a covariance instead marks the object untrusted and the
        parse succeeds, so the caller keeps the event and can fail it closed. The
        on-demand screen uses this: a conjunction it could not fully assess must
        make the screen NOT CLEAR, never disappear from a result that then reads
        CLEAR. Either way, negatives of numerical scale are clipped to zero and
        the parse succeeds (SCRUM-458).

    Raises
    ------
    LeoLabsGuardError
        On any guard failure (method not CALCULATED, frame not EME2000, an
        asymmetric covariance, a covariance non-PSD beyond the repair band when
        strict_psd, diagonal mismatch, miss-distance mismatch).
    LeoLabsParseError
        On missing required fields or unresolvable object identity.
    """
    primary_key, secondary_key = _resolve_roles(cdm, our_catalog_id)
    primary = _parse_object(cdm, primary_key, "primary")
    secondary = _parse_object(cdm, secondary_key, "secondary")

    _check_guards(primary, secondary)

    # Validate the diagonals against the covariance exactly as rotated, before
    # any PSD repair touches it. That keeps this guard measuring what it was
    # written to measure -- the rotation and the covariance convention -- rather
    # than measuring the repair (SCRUM-458).
    if validate_diagonals:
        _assert_diagonal_matches_comments(primary, cdm)
        _assert_diagonal_matches_comments(secondary, cdm)

    # Only now repair or flag. SCRUM-458: a small negative eigenvalue is CDM
    # precision noise and is clipped; a large one is a real fault and either
    # raises or marks the event unassessable, never silently drops it.
    event_id = cdm.get("COMMENT_EVENT_ID")
    _repair_or_flag_psd(primary, strict=strict_psd, event_id=event_id)
    _repair_or_flag_psd(secondary, strict=strict_psd, event_id=event_id)

    combined_hbr_m = primary.radius_m + secondary.radius_m
    miss_distance_m = _num(cdm, "MISS_DISTANCE")

    parsed = ParsedLeoLabsCDM(
        primary=primary,
        secondary=secondary,
        t_ca_utc=_iso_tca(cdm),
        combined_hbr_m=combined_hbr_m,
        cdm_collision_probability=_num(cdm, "COLLISION_PROBABILITY"),
        cdm_collision_probability_method=cdm.get("COLLISION_PROBABILITY_METHOD"),
        miss_distance_m=miss_distance_m,
        provenance={
            "cdm_source": cdm.get("ORIGINATOR"),
            "message_id": cdm.get("MESSAGE_ID"),
            "cdm_id": cdm.get("COMMENT_ID"),
            "event_id": cdm.get("COMMENT_EVENT_ID"),
            "primary_designator": primary.designator,
            "secondary_designator": secondary.designator,
            "primary_norad_id": primary.norad_id,
            "secondary_norad_id": secondary.norad_id,
            "primary_covariance_method": primary.covariance_method,
            "secondary_covariance_method": secondary.covariance_method,
            "primary_ephemeris_source": primary.ephemeris_source,
            "secondary_ephemeris_source": secondary.ephemeris_source,
            "ref_frame": primary.ref_frame,
            "combined_hbr_m": combined_hbr_m,
            "cdm_collision_probability": _num(cdm, "COLLISION_PROBABILITY"),
            "cdm_collision_probability_method": cdm.get("COLLISION_PROBABILITY_METHOD"),
        },
    )

    # SCRUM-417: carry the CDM's RTN relative geometry for the encounter viz.
    _rp = [_num(cdm, "RELATIVE_POSITION_R"), _num(cdm, "RELATIVE_POSITION_T"), _num(cdm, "RELATIVE_POSITION_N")]
    _rv = [_num(cdm, "RELATIVE_VELOCITY_R"), _num(cdm, "RELATIVE_VELOCITY_T"), _num(cdm, "RELATIVE_VELOCITY_N")]
    parsed.relative_position_rtn_m = _rp if all(x is not None for x in _rp) else None
    parsed.relative_velocity_rtn_m_s = _rv if all(x is not None for x in _rv) else None

    if validate_miss_distance and miss_distance_m is not None:
        recomputed_m = float(np.linalg.norm(parsed.r_rel_km())) * 1000.0
        if not np.isclose(
            recomputed_m, miss_distance_m, rtol=_MISS_RTOL, atol=_MISS_ATOL_M
        ):
            raise LeoLabsGuardError(
                f"miss distance recomputed from ECI states ({recomputed_m:.3f} m) "
                f"disagrees with CDM MISS_DISTANCE ({miss_distance_m:.3f} m); "
                f"likely a parse, unit, or frame bug."
            )

    return parsed


def parsed_pc(cdm: Dict[str, Any]) -> Optional[float]:
    """The CDM's own COLLISION_PROBABILITY as a float, or None."""
    return _num(cdm, "COLLISION_PROBABILITY")


# ---------------------------------------------------------------------------
# Golden cross-check (design section 6.5)
# ---------------------------------------------------------------------------

def cross_check_pc(parsed: ParsedLeoLabsCDM, estimation_mode: int = 64) -> float:
    """Compute Pc with aps_math.pc_utils on the parsed ECI states and covariances.

    Uses each object's own rotated ECI position covariance (m^2), states in
    meters, and the combined hard-body radius. compute_pc adds the two
    covariances, so passing them per object is exactly the combined-covariance Pc.
    Compared against the CDM's COLLISION_PROBABILITY in the golden test; agreement
    validates the parse, the units, the frame, and the rotation in one shot.
    """
    r1_m = parsed.primary.r_km * 1000.0
    v1_m_s = parsed.primary.v_km_s * 1000.0
    r2_m = parsed.secondary.r_km * 1000.0
    v2_m_s = parsed.secondary.v_km_s * 1000.0
    result = compute_pc(
        r1_m,
        v1_m_s,
        parsed.primary.cov_eci_pos_m2,
        r2_m,
        v2_m_s,
        parsed.secondary.cov_eci_pos_m2,
        parsed.combined_hbr_m,
        estimation_mode=estimation_mode,
    )
    return float(result.Pc)
