"""
SCRUM-382 -- the feature-flagged GNC emission adapter.

MAF v2.0 section 9. Carries a built GNCCommand to the GNC layer and handles the
GNCCommandAck, behind a flag, mirroring the LeoLabs and UDL pattern: a missing
endpoint interpolates to inert rather than to an error, so CI and the demo need
no GNC service standing.

Three modes, and the distinction between the middle two matters:

  disabled     GNC_ENABLED is false. Nothing is emitted or recorded.
  record_only  GNC_ENABLED is true but no endpoint is configured. The command
               is built, validated by construction, and recorded, and there is
               NO ack -- because nothing acknowledged it.
  live         GNC_ENABLED is true and GNC_SERVICE_URL is set. The command is
               POSTed and the real ack is returned.

record_only deliberately does not synthesize a GNCCommandAck. An ack means GNC
received the command and states the flight mode it moved to; manufacturing one
locally would put "accepted: true" in an audit trail for a burn no flight
computer ever saw. The absence of an ack is the honest record of an unconfigured
endpoint, and it is what distinguishes "we would have emitted this" from "GNC
has it".
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests as http_requests

from common.gnc_contract import GNC_COMMAND_PATH

log = logging.getLogger("planner")

# Feature flag, default false, same shape as LEOLABS_ENABLED / UDL_ENABLED.
GNC_ENABLED = os.environ.get("GNC_ENABLED", "false").lower() == "true"

# Where GNC lives. Empty is the normal state today: no GNC service exists yet,
# so emission records rather than posts. Point this at a real endpoint and the
# same code path goes live with no other change.
GNC_SERVICE_URL = os.environ.get("GNC_SERVICE_URL", "")

# A command is a small POST to a flight-adjacent service. Short, because the
# evaluate path is waiting on it and an unreachable GNC must not hold a request
# open; the additive try/except at the call site turns a timeout into a logged
# non-emission, not a failed evaluate.
GNC_TIMEOUT_S = 10.0

MODE_DISABLED = "disabled"
MODE_RECORD_ONLY = "record_only"
MODE_LIVE = "live"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def emission_mode(enabled: Optional[bool] = None, url: Optional[str] = None) -> str:
    """Which of the three modes this process is in."""
    is_enabled = GNC_ENABLED if enabled is None else enabled
    endpoint = GNC_SERVICE_URL if url is None else url
    if not is_enabled:
        return MODE_DISABLED
    if not endpoint:
        return MODE_RECORD_ONLY
    return MODE_LIVE


@dataclass(frozen=True)
class GNCEmission:
    """What happened when a command was handed to this adapter.

    ack is None unless a real GNCCommandAck came back from a real endpoint. Read
    `acknowledged` rather than `ack is not None` at a call site that only wants
    to know whether GNC has the command.
    """

    mode: str
    command_id: str
    conjunction_id: str
    emitted_at_utc: str
    posted: bool = False
    ack: Optional[Dict[str, Any]] = None
    error: str = ""
    endpoint: str = ""

    @property
    def acknowledged(self) -> bool:
        """True only when GNC itself acknowledged the command."""
        return self.ack is not None

    @property
    def accepted(self) -> Optional[bool]:
        """GNC's accepted flag, or None when nothing acknowledged the command.

        None is not False. A command GNC never saw was not rejected.
        """
        if self.ack is None:
            return None
        return bool(self.ack.get("accepted"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "command_id": self.command_id,
            "conjunction_id": self.conjunction_id,
            "emitted_at_utc": self.emitted_at_utc,
            "posted": self.posted,
            "acknowledged": self.acknowledged,
            "accepted": self.accepted,
            "ack": self.ack,
            "error": self.error or None,
            "endpoint": self.endpoint or None,
        }


def emit_gnc_command(
    command: Dict[str, Any],
    *,
    enabled: Optional[bool] = None,
    url: Optional[str] = None,
    timeout_s: float = GNC_TIMEOUT_S,
) -> GNCEmission:
    """Emit a command, or record it, depending on the mode.

    Never raises. Every failure path returns a GNCEmission carrying the reason,
    because the caller is inside /v1/evaluate and an emission problem must not
    become an evaluate problem. The additive try/except at the call site is the
    second belt; this is the first.
    """
    mode = emission_mode(enabled, url)
    endpoint = (GNC_SERVICE_URL if url is None else url) or ""
    base = dict(
        mode=mode,
        command_id=str(command.get("command_id", "")),
        conjunction_id=str(command.get("conjunction_id", "")),
        emitted_at_utc=_utc_now_iso(),
        endpoint=endpoint,
    )

    if mode == MODE_DISABLED:
        return GNCEmission(**base)

    if mode == MODE_RECORD_ONLY:
        # The command is real and contract-shaped; only the transport is
        # missing. Logged in full so the demo can show exactly what would have
        # gone to GNC, and with no ack, because nothing acknowledged it.
        log.info(
            "GNC command recorded, no endpoint configured",
            extra={
                "event": "gnc_command_recorded",
                "command_id": base["command_id"],
                "conjunction_id": base["conjunction_id"],
                "authority_level": command.get("authority_level"),
                "command": command,
            },
        )
        return GNCEmission(**base)

    try:
        response = http_requests.post(
            f"{endpoint.rstrip('/')}{GNC_COMMAND_PATH}",
            json=command,
            timeout=timeout_s,
        )
    except Exception as exc:
        log.warning(
            "GNC command POST failed",
            extra={"event": "gnc_command_post_failed",
                   "command_id": base["command_id"], "exc": str(exc)},
        )
        return GNCEmission(**base, error=f"GNC unreachable: {exc}")

    if response.status_code != 200:
        body = ""
        try:
            body = response.text[:200]
        except Exception:
            pass
        log.warning(
            "GNC rejected the command",
            extra={"event": "gnc_command_rejected",
                   "command_id": base["command_id"],
                   "status": response.status_code},
        )
        return GNCEmission(
            **base, posted=True,
            error=f"GNC returned HTTP {response.status_code}: {body}",
        )

    try:
        ack = response.json()
    except ValueError as exc:
        return GNCEmission(
            **base, posted=True, error=f"GNC ack was not JSON: {exc}"
        )

    log.info(
        "GNC command acknowledged",
        extra={"event": "gnc_command_acknowledged",
               "command_id": base["command_id"],
               "accepted": ack.get("accepted"),
               "flight_mode_after": ack.get("flight_mode_after")},
    )
    return GNCEmission(**base, posted=True, ack=ack)


def get_status() -> Dict[str, Any]:
    """Status for a dashboard badge, mirroring /leolabs-status and /udl-status."""
    mode = emission_mode()
    if mode == MODE_LIVE:
        label = "GNC LIVE"
        note = f"Emitting GNCCommand to {GNC_SERVICE_URL}."
    elif mode == MODE_RECORD_ONLY:
        label = "GNC RECORD-ONLY"
        note = (
            "GNC_ENABLED=true but GNC_SERVICE_URL is unset. Commands are built "
            "and recorded, not emitted, and carry no acknowledgement."
        )
    else:
        label = "GNC DISABLED"
        note = "GNC_ENABLED=false. No command is built or recorded."
    return {
        "enabled": GNC_ENABLED,
        "endpoint_configured": bool(GNC_SERVICE_URL),
        "endpoint": GNC_SERVICE_URL or None,
        "mode": mode,
        "label": label,
        "note": note,
    }
