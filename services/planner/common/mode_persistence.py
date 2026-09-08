"""
SCRUM-379 -- section 6.2: mode persistence and reboot recovery.

Section 6.2 is explicit: 'The state machine must persist the current flight mode
and active GNCCommand to durable onboard storage before any transition. On
reboot: read persisted state... If M2 or M3: check whether the burn window is
still open.' This module is that store, plus the pure recovery function that
decides what a restored state means.

recover_after_reboot takes a persisted record and a clock and returns a mode. It
reads nothing else, so the whole section 6.2 ladder is testable without a
filesystem or an actual reboot. Section 6.1's comms-gap table lives with the
rest of the pure section 6 logic in safety_monitor.py.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from common.decision_state_machine import (
    FlightMode,
    ManeuverCommand,
    as_mode,
    slew_lead_time_s,
)

MODE_STATE_DIR_ENV = "MODE_STATE_DIR"


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# What gets persisted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersistedMode:
    """The flight mode and active command, as written before a transition.

    latest_burn_utc is stored alongside the command because section 6.2's
    M2/M3 branch needs it after a reboot, and by then the request that carried
    it is long gone.
    """

    sat_id: str
    mode: FlightMode
    conjunction_id: str = ""
    persisted_at_utc: str = ""
    command: Optional[ManeuverCommand] = None
    latest_burn_utc: Optional[datetime] = None
    software_version: str = ""

    def __post_init__(self) -> None:
        # Normalised here rather than at each call site, so a record read back
        # off disk and one just written compare the same way.
        object.__setattr__(self, "mode", as_mode(self.mode))
        if self.latest_burn_utc is not None:
            if self.latest_burn_utc.tzinfo is None:
                raise ValueError("latest_burn_utc must be timezone-aware")
            object.__setattr__(
                self, "latest_burn_utc", self.latest_burn_utc.astimezone(timezone.utc)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sat_id": self.sat_id,
            "mode": self.mode.value,
            "conjunction_id": self.conjunction_id,
            "persisted_at_utc": self.persisted_at_utc,
            "software_version": self.software_version,
            "latest_burn_utc": (
                _iso(self.latest_burn_utc) if self.latest_burn_utc else None
            ),
            "command": (
                {
                    "dv_eci_km_s": list(self.command.dv_eci_km_s),
                    "dv_magnitude_m_s": self.command.dv_magnitude_m_s,
                    "t_burn_utc": _iso(self.command.t_burn_utc),
                    "direction": self.command.direction,
                }
                if self.command
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PersistedMode":
        command_payload = payload.get("command")
        command = None
        if command_payload:
            t_burn = _parse(command_payload.get("t_burn_utc"))
            if t_burn is not None:
                command = ManeuverCommand(
                    dv_eci_km_s=tuple(command_payload["dv_eci_km_s"]),
                    dv_magnitude_m_s=float(command_payload["dv_magnitude_m_s"]),
                    t_burn_utc=t_burn,
                    direction=str(command_payload.get("direction", "")),
                )
        return cls(
            sat_id=str(payload["sat_id"]),
            mode=as_mode(payload["mode"]),
            conjunction_id=str(payload.get("conjunction_id", "")),
            persisted_at_utc=str(payload.get("persisted_at_utc", "")),
            command=command,
            latest_burn_utc=_parse(payload.get("latest_burn_utc")),
            software_version=str(payload.get("software_version", "")),
        )


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


class ModeStore:
    """Read and write one flight mode per spacecraft."""

    def read(self, sat_id: str) -> Optional[PersistedMode]:
        raise NotImplementedError

    def write(self, state: PersistedMode) -> None:
        raise NotImplementedError


class NullModeStore(ModeStore):
    """Persists nothing, and says so.

    The default. The planner service is stateless by design, and a store that
    silently retained state between unrelated requests would be worse than no
    store: a mode left over from another operator's event would look like this
    spacecraft's. Durability is opted into by configuring MODE_STATE_DIR.
    """

    def read(self, sat_id: str) -> Optional[PersistedMode]:
        return None

    def write(self, state: PersistedMode) -> None:
        return None


class InMemoryModeStore(ModeStore):
    """Process-local store. For tests and single-process deployments."""

    def __init__(self) -> None:
        self._states: Dict[str, PersistedMode] = {}

    def read(self, sat_id: str) -> Optional[PersistedMode]:
        return self._states.get(sat_id)

    def write(self, state: PersistedMode) -> None:
        self._states[state.sat_id] = state


class FileModeStore(ModeStore):
    """One JSON file per spacecraft, written atomically.

    The stand-in for section 6.2's durable onboard storage. Written via a
    temporary file and os.replace, so a power loss mid-write leaves the previous
    state intact rather than a truncated file that would read as no state at all
    -- which recovery would then treat as a reboot with nothing persisted, and
    escalate. Correct, but a needless safehold.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, sat_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in sat_id)
        return self.directory / f"mode_{safe or 'UNKNOWN'}.json"

    def read(self, sat_id: str) -> Optional[PersistedMode]:
        path = self._path(sat_id)
        try:
            payload = json.loads(path.read_text())
            return PersistedMode.from_dict(payload)
        except (OSError, ValueError, KeyError):
            # Unreadable or corrupt. Section 6.2 has an answer for this and it
            # is not 'assume M0': recover_after_reboot escalates on None.
            return None

    def write(self, state: PersistedMode) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(state.sat_id)
        fd, tmp = tempfile.mkstemp(dir=str(self.directory), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(state.to_dict(), handle, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def build_mode_store(directory: Optional[str] = None) -> ModeStore:
    """A FileModeStore when MODE_STATE_DIR is configured, else a NullModeStore."""
    target = directory if directory is not None else os.environ.get(MODE_STATE_DIR_ENV)
    if not target:
        return NullModeStore()
    return FileModeStore(Path(target))


# ---------------------------------------------------------------------------
# Section 6.2: reboot recovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RebootRecovery:
    """Which mode to resume in after a reboot, and why.

    pre_reboot_mode is carried because section 6.2 step 5 requires the reboot
    event to be logged with the pre-reboot state, and after recovery the mode
    alone no longer says what it was.
    """

    mode: FlightMode
    pre_reboot_mode: Optional[FlightMode]
    reason: str
    burn_window_open: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "reboot_recovery",
            "mode": self.mode.value,
            "pre_reboot_mode": (
                self.pre_reboot_mode.value if self.pre_reboot_mode else None
            ),
            "reason": self.reason,
            "burn_window_open": self.burn_window_open,
        }


def recover_after_reboot(
    persisted: Optional[PersistedMode],
    t_now_utc: datetime,
) -> RebootRecovery:
    """Section 6.2's ladder, in order.

    1. Read persisted state. Nothing readable means M4 -- not knowing which mode
       the spacecraft was in is exactly the ambiguity section 1 escalates.
    2. M0 or M4: resume normally.
    3. M1: resume M1 and re-evaluate with the last known CDM.
    4. M2 or M3: only if the burn window is still open, meaning
       t_now < latest_burn_utc - (t_slew + t_settle + t_margin). At or past
       latest_burn_utc, M4. Between the two -- inside the window but with no
       time left to slew -- section 6.2 gives no branch, so it escalates rather
       than attempting a burn it cannot make.
    """
    if persisted is None:
        return RebootRecovery(
            mode=FlightMode.M4_SAFE_HOLD,
            pre_reboot_mode=None,
            reason="no persisted flight mode could be read; escalating per section 6.2",
        )

    if t_now_utc.tzinfo is None:
        raise ValueError("t_now_utc must be timezone-aware")
    now = t_now_utc.astimezone(timezone.utc)
    previous = persisted.mode

    if previous in (FlightMode.M0_NOMINAL, FlightMode.M4_SAFE_HOLD):
        return RebootRecovery(
            mode=previous,
            pre_reboot_mode=previous,
            reason=f"resumed {previous.value} normally",
        )

    if previous is FlightMode.M1_WATCH:
        return RebootRecovery(
            mode=FlightMode.M1_WATCH,
            pre_reboot_mode=previous,
            reason="re-evaluate the conjunction with the last known CDM",
        )

    latest_burn = persisted.latest_burn_utc
    if latest_burn is None:
        return RebootRecovery(
            mode=FlightMode.M4_SAFE_HOLD,
            pre_reboot_mode=previous,
            reason="no burn window was persisted; cannot confirm it is still open",
            burn_window_open=None,
        )

    if now < latest_burn - _lead_timedelta():
        return RebootRecovery(
            mode=FlightMode.M2_STAGED,
            pre_reboot_mode=previous,
            reason=(
                "burn window still open; re-present to the operator (L1) or "
                "resume the veto countdown (L2)"
            ),
            burn_window_open=True,
        )

    if now >= latest_burn:
        return RebootRecovery(
            mode=FlightMode.M4_SAFE_HOLD,
            pre_reboot_mode=previous,
            reason="burn window closed during the reboot; do not attempt the burn",
            burn_window_open=False,
        )

    return RebootRecovery(
        mode=FlightMode.M4_SAFE_HOLD,
        pre_reboot_mode=previous,
        reason=(
            "burn window closes within the slew lead time; no time to slew, and "
            "section 6.2 defines no branch for it"
        ),
        burn_window_open=False,
    )


def _lead_timedelta():
    from datetime import timedelta

    return timedelta(seconds=slew_lead_time_s())
