from __future__ import annotations

from .models import ObjectState, Snapshot

def make_snapshot(t_seconds: float, states: dict[str, ObjectState]) -> Snapshot:
    return Snapshot(t_seconds=t_seconds, states=states)