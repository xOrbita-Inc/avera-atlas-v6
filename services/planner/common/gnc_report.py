"""
SCRUM-382 -- consume the GNCReport and fold the post-burn outcome back in.

MAF v2.0 sections 9 and 10. The report is the other half of the loop: GNC says
what the burn actually did, and APS re-evaluates the risk on the state that
resulted and records it next to the decision that authorised it.

What this module does NOT compute, per the ticket's boundary: the execution-error
physics or the post-maneuver Pc. The ExecutionError block is SCRUM-365's and
arrives in the report; P_post = P_pre + P_burn is assembled here from the
report's own covariance blocks and handed to maneuver_scorer.compute_pc_post,
which is the one implementation of that calculation. Nothing here reimplements
either.

The section 10 producers this fills are the three the evidence catalogue lists
against SCRUM-382: actual_vs_predicted, post_maneuver_od and residual_risk.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

import numpy as np

from aps_math import frames
from common.maneuver_scorer import compute_pc_post

log = logging.getLogger("planner")

# A report whose execution_status is one of these did not deliver the commanded
# burn (guard doc section 3, the M3 to M4 row).
NON_NOMINAL_STATUSES = ("PARTIAL", "ABORTED")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _matrix_3x3(flat: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """A row-major flat 9 into a 3x3, or None if it is not one.

    Units are km^2 per the contract. A malformed block is None rather than a
    guess: a covariance we cannot read must not become a covariance of zeros,
    which would read as perfect knowledge.
    """
    if flat is None:
        return None
    try:
        values = [float(v) for v in flat]
    except (TypeError, ValueError):
        return None
    if len(values) != 9 or any(not math.isfinite(v) for v in values):
        return None
    return np.array(values, dtype=float).reshape(3, 3)


def post_burn_covariance_km2(
    report: Dict[str, Any], p_pre_km2: Optional[Sequence[float]] = None
) -> Optional[np.ndarray]:
    """The covariance to evaluate residual risk on: P_post = P_pre + P_burn.

    Three sources, in order of what the contract actually gives us:

    1. PostBurnState.p_post_km2, when GNC supplies it. That is GNC's own
       post-burn covariance and it is authoritative over anything assembled
       here -- it came from the flight computer's own OD.
    2. Otherwise P_pre + P_burn, per section 9's execution-error note, using
       ExecutionError.p_burn_rtn_km2 from the report.
    3. Otherwise None, which is not a covariance of zero and must not be read
       as one.

    SCRUM-428. p_burn_rtn_km2 is RTN by its name and p_pre is ECI, and summing
    them directly would silently push frame-consistency onto the caller: a
    real p_pre and the report's genuinely RTN p_burn would produce a
    plausible, wrong pc_post with no error. Case 2 is now frame-consistent on
    its own: it rotates p_burn from RTN to ECI internally using the burn-time
    r_sat/v_sat state on PostBurnState (rtn_to_eci_rotation + rotate_cw_block,
    the same pair used throughout aps_math.frames -- see that module for the
    derivation), then sums in ECI.

    If r_sat/v_sat are unavailable when case 2 would otherwise apply, this
    returns None rather than guess at a frame or fall back to the old
    mixed-frame sum: a covariance this function cannot honestly place in one
    frame is not a covariance to hand to compute_pc_post, per this module's
    own None-means-unknown-not-zero convention.

    Case 1 is the path the contract intends and the one the demo uses, which
    is why it is first.
    """
    p_post = _matrix_3x3((report.get("post_burn_state") or {}).get("p_post_km2"))
    if p_post is not None:
        return p_post

    p_burn_rtn = _matrix_3x3((report.get("execution_error") or {}).get("p_burn_rtn_km2"))
    p_pre = _matrix_3x3(p_pre_km2)
    if p_burn_rtn is None or p_pre is None:
        return None

    post_state = report.get("post_burn_state") or {}
    r_sat = post_state.get("r_sat_km")
    v_sat = post_state.get("v_sat_km_s")
    if r_sat is None or v_sat is None:
        return None

    rot = frames.rtn_to_eci_rotation(
        np.asarray(r_sat, dtype=float), np.asarray(v_sat, dtype=float)
    )
    p_burn_eci = frames.rotate_cw_block(p_burn_rtn, rot)
    return p_pre + p_burn_eci


@dataclass(frozen=True)
class PostBurnAssessment:
    """The re-evaluated risk after a burn, plus the section 10 producers.

    pc_post is None when no Pc could be established, which is not a Pc of zero.
    Everything optional here is genuinely optional: the report may be an abort
    with no post-burn state at all.
    """

    command_id: str
    conjunction_id: str
    execution_status: str
    pc_post: Optional[float]
    m2_post_estimated: Optional[float]
    replan_required: bool
    replan_note: str
    actual_vs_predicted: Dict[str, Any]
    post_maneuver_od: Dict[str, Any]
    residual_risk: Dict[str, Any]

    def evidence_values(self) -> Dict[str, Any]:
        """The three SCRUM-382 fields in the MAF section 10 catalogue.

        Named to match evidence_record.FIELD_PRODUCERS, which already lists
        actual_vs_predicted, post_maneuver_od and residual_risk against
        SCRUM-382, so these drop straight into a record's values dict.
        """
        return {
            "actual_vs_predicted": dict(self.actual_vs_predicted),
            "post_maneuver_od": dict(self.post_maneuver_od),
            "residual_risk": dict(self.residual_risk),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "conjunction_id": self.conjunction_id,
            "execution_status": self.execution_status,
            "pc_post": self.pc_post,
            "m2_post_estimated": self.m2_post_estimated,
            "replan_required": self.replan_required,
            "replan_note": self.replan_note,
            **self.evidence_values(),
        }


def assess_post_burn(
    report: Dict[str, Any],
    *,
    commanded_dv_m_s: Optional[float] = None,
    commanded_dv_rtn_m_s: Optional[Sequence[float]] = None,
    r_rel_km_at_tca: Optional[Sequence[float]] = None,
    v_rel_km_s_at_tca: Optional[Sequence[float]] = None,
    p_pre_km2: Optional[Sequence[float]] = None,
    hbr_m: Optional[float] = None,
    pc_pre: Optional[float] = None,
    pc_monitor_threshold: Optional[float] = None,
) -> PostBurnAssessment:
    """Re-evaluate risk from a GNCReport and assemble the section 10 producers.

    The Pc is computed by maneuver_scorer.compute_pc_post on the post-burn
    state, not here. This function's job is to get the right inputs to it and
    to say honestly when it could not: a missing post-burn state, an
    unreadable covariance or an absent relative velocity all produce pc_post
    None rather than a number.

    replan_required follows guard doc section 3's M3 rows: a nominal burn whose
    residual Pc is still at or above the monitor line needs a post-burn replan
    (M3 to M1), and a non-nominal execution needs one because the commanded
    burn was not delivered.
    """
    command_id = str(report.get("command_id", ""))
    conjunction_id = str(report.get("conjunction_id", ""))
    status = str(report.get("execution_status", ""))
    post_state = report.get("post_burn_state") or {}
    execution_error = report.get("execution_error") or {}
    abort_detail = report.get("abort_detail") or {}

    actual_dv = report.get("actual_dv_m_s")
    m2_post_estimated = post_state.get("m2_post_estimated")

    # -- Pc on the state that actually resulted ----------------------------
    pc_post: Optional[float] = None
    p_post = post_burn_covariance_km2(report, p_pre_km2)
    r_sat = post_state.get("r_sat_km")
    v_sat = post_state.get("v_sat_km_s")
    if (
        p_post is not None
        and r_sat is not None
        and v_sat is not None
        and r_rel_km_at_tca is not None
        and v_rel_km_s_at_tca is not None
        and hbr_m is not None
    ):
        try:
            r_sat_arr = np.asarray(r_sat, dtype=float)
            pc_post = compute_pc_post(
                r_sat_arr,
                np.asarray(v_sat, dtype=float),
                # The secondary's position relative to the post-burn primary.
                r_sat_arr + np.asarray(r_rel_km_at_tca, dtype=float),
                np.asarray(v_rel_km_s_at_tca, dtype=float),
                p_post,
                float(hbr_m),
            )
        except Exception as exc:
            log.warning(
                "post-burn Pc could not be established",
                extra={"event": "gnc_report_pc_post_failed",
                       "command_id": command_id, "exc": str(exc)},
            )
            pc_post = None

    # -- replan decision, guard doc section 3's M3 rows --------------------
    if status in NON_NOMINAL_STATUSES:
        replan_required = True
        replan_note = (
            f"execution_status {status}: the commanded burn was not delivered"
        )
        if abort_detail.get("abort_reason"):
            replan_note += f" ({abort_detail['abort_reason']})"
    elif (
        pc_post is not None
        and pc_monitor_threshold is not None
        and pc_post >= pc_monitor_threshold
    ):
        replan_required = True
        replan_note = (
            f"residual Pc {pc_post:.6g} is still at or above the monitor line "
            f"{pc_monitor_threshold:.6g}"
        )
    elif pc_post is None and status == "NOMINAL":
        # A nominal burn we cannot re-score is not a cleared conjunction.
        replan_required = True
        replan_note = (
            "burn reported NOMINAL but no post-burn Pc could be established"
        )
    else:
        replan_required = False
        replan_note = ""

    # -- section 10 producers ----------------------------------------------
    dv_delta = (
        float(actual_dv) - float(commanded_dv_m_s)
        if actual_dv is not None and commanded_dv_m_s is not None
        else None
    )
    actual_vs_predicted = {
        "commanded_dv_m_s": commanded_dv_m_s,
        "actual_dv_m_s": actual_dv,
        "dv_delta_m_s": dv_delta,
        "commanded_dv_rtn_m_s": (
            list(commanded_dv_rtn_m_s) if commanded_dv_rtn_m_s is not None else None
        ),
        "actual_dv_rtn_m_s": report.get("actual_dv_rtn_m_s"),
        "attitude_error_at_burn_deg": report.get("attitude_error_at_burn_deg"),
        "execution_status": status,
        # SCRUM-365's block, carried through verbatim. Not recomputed here.
        "execution_error": dict(execution_error) if execution_error else None,
    }

    post_maneuver_od = {
        "epoch_utc": post_state.get("epoch_utc"),
        "r_sat_km": post_state.get("r_sat_km"),
        "v_sat_km_s": post_state.get("v_sat_km_s"),
        "p_post_km2": post_state.get("p_post_km2"),
        "p_post_source": (
            "gnc_post_burn_state"
            if (post_state.get("p_post_km2") is not None)
            else ("p_pre_plus_p_burn" if p_post is not None else None)
        ),
        "m2_post_estimated": m2_post_estimated,
    }

    residual_risk = {
        "pc_pre": pc_pre,
        "pc_post": pc_post,
        "pc_monitor_threshold": pc_monitor_threshold,
        "m2_post_estimated": m2_post_estimated,
        "replan_required": replan_required,
        "replan_note": replan_note,
        "abort_detail": dict(abort_detail) if abort_detail else None,
    }

    return PostBurnAssessment(
        command_id=command_id,
        conjunction_id=conjunction_id,
        execution_status=status,
        pc_post=pc_post,
        m2_post_estimated=m2_post_estimated,
        replan_required=replan_required,
        replan_note=replan_note,
        actual_vs_predicted=actual_vs_predicted,
        post_maneuver_od=post_maneuver_od,
        residual_risk=residual_risk,
    )


def build_report_ack(assessment: PostBurnAssessment) -> Dict[str, Any]:
    """The GNCReportAck for a consumed report. Contract-shaped."""
    ack: Dict[str, Any] = {
        "command_id": assessment.command_id,
        "conjunction_id": assessment.conjunction_id,
        "received_at_utc": _utc_now_iso(),
        "replan_required": assessment.replan_required,
    }
    if assessment.replan_note:
        ack["replan_note"] = assessment.replan_note
    return ack
