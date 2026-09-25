"""services/planner/common/secondary_screen_async.py

SCRUM-456: run the LeoLabs on-demand secondary screen off the evaluate request
path, and hold its result in a process-local store a poll endpoint can read.

Why
---
The screen takes 30 s to 2 min against LeoLabs (300 s poll budget) and the
dashboard's evaluate call times out at 60 s, so a maneuver-recommended decision
failed in the UI even though the planner was working correctly. Evaluate now
returns the burn decision immediately with the secondary check PENDING and a job
id; the screen runs on a small bounded pool; GET /v1/secondary-screen/{job_id}
returns the result once it resolves.

What this module does NOT change
--------------------------------
The screen itself. The worker calls
atlas_artifact._run_on_demand_secondary_check, the same function the inline path
called, so the ephemeris (SCRUM-440/451), the covariance growth (SCRUM-452), the
seed priority (SCRUM-454), the parse and PSD repair (SCRUM-458) and the clear
contract (SCRUM-381/442) are executed by the same code as before. Only where and
when it runs moves.

The safety direction
--------------------
PENDING is not CLEAR, and it is not "not performed" either. Both distinctions are
load-bearing:

- Never CLEAR: a pending entry reports status "pending", and the artifact's
  SecondaryConflictCheck carries secondary_conjunction_clear=False. There is no
  path from pending to a clear verdict except a screen that actually returned
  CLEAR.
- Never an escalation on its own: a pending screen leaves
  secondary_check_performed False, which blocks the M1 to M2 staging AND (so the
  maneuver is not authorized as secondary-clear) without firing the M1 to M4
  escalation clause, which fires only on a *performed* check that came back not
  clear. That is the existing semantics in safety_monitor, not a new rule.
- error and timeout stay NOT CLEAR. A screen that could not run is
  indistinguishable, for safety purposes, from a screen that found a conflict:
  both fail closed. run_screening already raises rather than returning empty on
  every failure mode, and the worker records that as status "error".

In-process is deliberate for this ticket: one planner process, and a restart
losing a pending screen is safe because a lost screen reads as not performed,
which fails closed. Durable persistence is SCRUM-453.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# "planner" and not __name__: the service configures structured JSON logging on
# that logger, so a module-named logger has no handler and its records never reach
# the log. Every other planner module uses this name.
log = logging.getLogger("planner")


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------

STATUS_PENDING = "pending"
STATUS_CLEAR = "clear"
STATUS_NOT_CLEAR = "not_clear"
STATUS_ERROR = "error"

#: Statuses a decision may be finalized secondary-clear on. Exactly one.
CLEAR_STATUSES = frozenset({STATUS_CLEAR})

#: Statuses that are resolved, whether or not they are clear.
TERMINAL_STATUSES = frozenset({STATUS_CLEAR, STATUS_NOT_CLEAR, STATUS_ERROR})


def is_clear_status(status: Optional[str]) -> bool:
    """The only place a status becomes a clear boolean.

    Written as an allow-list rather than `status != "not_clear"` so that a new
    status, a typo or a None can never read as clear.
    """
    return status in CLEAR_STATUSES


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

# How long a resolved entry stays readable. The dashboard polls within seconds;
# an hour is generous and bounds memory on a long-running process.
_EVICT_AFTER_SECONDS = 3600.0

# Hard cap, so a burst cannot grow the store without bound even inside the TTL.
# Oldest entries go first.
_MAX_ENTRIES = 256

# Small pool: each job holds a LeoLabs poll for up to 300 s, and creates are
# rate-limited to 3 per 2 minutes anyway, so more workers would only queue
# against that limit. Bounded so many evaluates cannot spawn unbounded threads.
_MAX_WORKERS = 4

_lock = threading.Lock()
_entries: "Dict[str, ScreenEntry]" = {}
_executor: Optional[ThreadPoolExecutor] = None


@dataclass
class ScreenEntry:
    """One asynchronous screen, pending or resolved.

    `status` is the only thing a caller should judge clearness on, and only
    through is_clear_status.
    """

    job_id: str
    status: str = STATUS_PENDING
    created_at: float = field(default_factory=time.monotonic)
    resolved_at: Optional[float] = None
    # Set when the screen resolves.
    screening_id: Optional[str] = None     # LeoLabs' id, not our job id
    verdict: Optional[Any] = None          # ContractVerdict
    result: Optional[Any] = None           # ScreeningResult
    check: Optional[Any] = None            # SecondaryConflictCheck
    error: Optional[str] = None
    # Provenance of the covariance the screen was seeded from (SCRUM-454).
    seed_source: Optional[str] = None
    # Context, for the poll response and the audit trail.
    screening_epoch_utc: Optional[str] = None
    primary_catalog_number: Optional[str] = None

    @property
    def is_clear(self) -> bool:
        return is_clear_status(self.status)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


def _evict_locked() -> None:
    """Drop expired entries, then the oldest if still over the cap. Call locked."""
    now = time.monotonic()
    for job_id in [k for k, e in _entries.items()
                   if now - e.created_at > _EVICT_AFTER_SECONDS]:
        _entries.pop(job_id, None)
    if len(_entries) > _MAX_ENTRIES:
        for job_id, _ in sorted(_entries.items(),
                                key=lambda kv: kv[1].created_at)[
                                    :len(_entries) - _MAX_ENTRIES]:
            _entries.pop(job_id, None)


def get_entry(job_id: str) -> Optional[ScreenEntry]:
    """The entry for this job id, or None if unknown or evicted.

    None is not clear. A caller that cannot find its screen has not been told the
    screen passed; the poll endpoint returns 404 and the decision stands on its
    pending check, which fails closed.
    """
    with _lock:
        _evict_locked()
        return _entries.get(job_id)


def reset_store() -> None:
    """Drop every entry. For tests, and called from leolabs_runtime.reset_caches."""
    with _lock:
        _entries.clear()


def store_size() -> int:
    with _lock:
        return len(_entries)


def _executor_submit(fn: Callable[[], None]) -> None:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=_MAX_WORKERS,
                thread_name_prefix="secondary-screen",
            )
        executor = _executor
    executor.submit(fn)


# ---------------------------------------------------------------------------
# Starting a screen
# ---------------------------------------------------------------------------

def start_screen(
    *,
    r_post_km: Optional[List[float]],
    v_post_km_s: Optional[List[float]],
    p_post_eci_km2: Optional[Any],
    screening_epoch_utc: Optional[str],
    policy: Any,
    cap: Any,
    primary_catalog_number: Optional[str],
    dv_eci_km_s: Optional[Any] = None,
    seed_source: Optional[str] = None,
    submit: Optional[Callable[[Callable[[], None]], None]] = None,
) -> str:
    """Register a pending screen and hand the work to the pool. Returns the job id.

    Does no network and no propagation on the calling thread: the LeoLabs create
    happens in the worker, so evaluate's latency is independent of LeoLabs
    entirely. The cost is that the real screening id is not known at evaluate
    time, which is why the poll key is our own job id and the LeoLabs id is
    reported on the entry once the create returns. A create that fails surfaces as
    status "error", which fails closed.

    `submit` is injectable so tests can run the worker synchronously and keep the
    async deterministic.
    """
    job_id = uuid.uuid4().hex
    entry = ScreenEntry(
        job_id=job_id,
        seed_source=seed_source,
        screening_epoch_utc=screening_epoch_utc,
        primary_catalog_number=(
            str(primary_catalog_number) if primary_catalog_number else None),
    )
    with _lock:
        _entries[job_id] = entry
        _evict_locked()

    def _job() -> None:
        _run_job(
            job_id,
            r_post_km=r_post_km,
            v_post_km_s=v_post_km_s,
            p_post_eci_km2=p_post_eci_km2,
            screening_epoch_utc=screening_epoch_utc,
            policy=policy,
            cap=cap,
            primary_catalog_number=primary_catalog_number,
            dv_eci_km_s=dv_eci_km_s,
        )

    log.info(
        "secondary screen started in the background",
        extra={"event": "secondary_screen_started", "job_id": job_id,
               "primary_catalog_number": entry.primary_catalog_number,
               "seed_source": seed_source},
    )
    (submit or _executor_submit)(_job)
    return job_id


def _run_job(
    job_id: str,
    **kwargs: Any,
) -> None:
    """Run one screen and record the outcome. Never raises into the pool.

    The screen itself is atlas_artifact._run_on_demand_secondary_check, unchanged
    and unduplicated, so this cannot drift from what the inline path did. It
    already fails closed on every internal failure, returning a check that is not
    performed and not clear; anything it raises is caught here and recorded as
    status "error", which is also not clear.
    """
    # Imported here rather than at module scope: atlas_artifact imports plenty of
    # the planner, and this module is imported from it.
    from common import atlas_artifact as aa

    captured: Dict[str, Any] = {}

    def _sink(result: Any, verdict: Any) -> None:
        captured["result"] = result
        captured["verdict"] = verdict

    try:
        check = aa._run_on_demand_secondary_check(screening_sink=_sink, **kwargs)
    except Exception as exc:                      # never let the pool swallow it
        log.warning(
            "secondary screen raised; recording error, which fails closed",
            extra={"event": "secondary_screen_error", "job_id": job_id,
                   "reason": str(exc)},
        )
        _finish(job_id, status=STATUS_ERROR, error=str(exc))
        return

    result = captured.get("result")
    verdict = captured.get("verdict")

    if not check.secondary_check_performed:
        # The screen ran the fail-closed path: it could not complete. Not clear.
        _finish(
            job_id, status=STATUS_ERROR, check=check,
            error=check.operator_note,
            screening_id=getattr(result, "screening_id", None),
        )
        return

    status = STATUS_CLEAR if check.secondary_conjunction_clear else STATUS_NOT_CLEAR
    _finish(
        job_id, status=status, check=check, result=result, verdict=verdict,
        screening_id=getattr(result, "screening_id", None),
    )


def _finish(
    job_id: str,
    *,
    status: str,
    check: Any = None,
    result: Any = None,
    verdict: Any = None,
    error: Optional[str] = None,
    screening_id: Optional[str] = None,
) -> None:
    with _lock:
        entry = _entries.get(job_id)
        if entry is None:
            # Evicted, or reset under a test. Nothing to record, and nothing that
            # could read as clear.
            return
        entry.status = status
        entry.check = check
        entry.result = result
        entry.verdict = verdict
        entry.error = error
        entry.screening_id = screening_id
        entry.resolved_at = time.monotonic()
    log.info(
        "secondary screen resolved",
        extra={"event": "secondary_screen_resolved", "job_id": job_id,
               "status": status, "screening_id": screening_id,
               "clear": is_clear_status(status)},
    )


# ---------------------------------------------------------------------------
# Poll response
# ---------------------------------------------------------------------------

#: Conjunctions returned on the poll response, most risk-relevant first. A screen
#: can return well over a thousand events (1195 live on SWARM A in SCRUM-458), and
#: the plan requires this endpoint be fast, so the list is capped and the full
#: count reported alongside. Every breaching event is in `verdict.breaches`
#: regardless of this cap, so nothing that drove the verdict is hidden by it.
_MAX_CONJUNCTIONS_IN_RESPONSE = 50


def _conjunction_row(parsed: Any) -> Dict[str, Any]:
    secondary = parsed.secondary
    miss_m = parsed.miss_distance_m
    return {
        "object_id": (secondary.object_name or secondary.designator
                      or str(secondary.norad_id)),
        "secondary_designator": secondary.designator,
        "secondary_norad": secondary.norad_id,
        "tca_utc": parsed.t_ca_utc,
        "miss_distance_km": (miss_m / 1000.0) if miss_m is not None else None,
        "pc": parsed.cdm_collision_probability,
        "covariance_repaired": parsed.covariance_repaired,
        "covariance_untrusted": parsed.covariance_untrusted,
    }


def _verdict_dict(verdict: Any) -> Optional[Dict[str, Any]]:
    if verdict is None:
        return None
    return {
        "clear": bool(verdict.clear),
        "evaluated": int(verdict.evaluated),
        "breaches": list(verdict.breaches),
        "closest_miss_km": verdict.closest_miss_km,
        "closest_object_id": verdict.closest_object_id,
        "max_pc": verdict.max_pc,
        "flagged_objects": list(verdict.flagged_objects),
    }


def entry_to_dict(entry: ScreenEntry) -> Dict[str, Any]:
    """The poll payload: {status, clear, conjunctions, verdict, seed_source, ...}.

    `clear` is derived through is_clear_status, so a pending or error entry
    reports clear=False. It is stated explicitly rather than left for the caller
    to infer from `status`, because the inference is exactly where a reader could
    get it wrong.
    """
    result = entry.result
    conjunctions = list(getattr(result, "conjunctions", None) or [])
    # Closest first, so a truncated list keeps the events that matter. None sorts
    # last rather than raising.
    conjunctions.sort(
        key=lambda p: (p.miss_distance_m is None,
                       p.miss_distance_m if p.miss_distance_m is not None else 0.0)
    )
    rows = [_conjunction_row(p)
            for p in conjunctions[:_MAX_CONJUNCTIONS_IN_RESPONSE]]
    check = entry.check
    return {
        "job_id": entry.job_id,
        "status": entry.status,
        "clear": entry.is_clear,
        "pending": entry.status == STATUS_PENDING,
        "screening_id": entry.screening_id,
        "seed_source": entry.seed_source,
        "screening_epoch_utc": entry.screening_epoch_utc,
        "primary_catalog_number": entry.primary_catalog_number,
        "error": entry.error,
        "verdict": _verdict_dict(entry.verdict),
        "conjunctions": rows,
        "conjunctions_total": (
            len(conjunctions) if entry.is_terminal else 0),
        "conjunctions_truncated": len(conjunctions) > len(rows),
        "secondary_check_performed": (
            bool(check.secondary_check_performed) if check is not None else False),
        "secondary_conjunction_clear": (
            bool(check.secondary_conjunction_clear) if check is not None else False),
        "operator_note": (check.operator_note if check is not None else None),
        "closest_approach_km": (
            check.closest_approach_km if check is not None else None),
        "closest_object_id": (
            check.closest_object_id if check is not None else None),
        "flagged_objects": (
            list(check.flagged_objects) if check is not None else []),
    }
