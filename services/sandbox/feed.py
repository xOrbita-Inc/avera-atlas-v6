"""
services/sandbox/feed.py

Standalone sandbox-to-tracker feed entrypoint (SCRUM-373 AC2).

Run via:
    python -m services.sandbox.feed

Generates one minimal scenario's ObservationBundle and pushes it through
the live tracker /v1/observations interface via
SensorKnowledgeStore.push_to_tracker(). This is deliberately NOT hung off
regression_harness or canonical_scenarios (John's review): those are
benchmark/regression tooling, and coupling a benchmark driver to a live
network call is exactly the kind of call site to avoid. This module's
only job is to run one scenario and feed it live.

Scenario construction mirrors canonical_scenarios.py's Tier 1 pattern
(same public building blocks: SimObject, physical_properties_for_size_bin,
SimConfig, run_simulation, generate_observation_bundle) but does not
import canonical_scenarios.py itself, to avoid coupling this feed to
benchmark-scoring/truth-comparison logic that has nothing to do with
feeding a live tracker.
"""

from __future__ import annotations

import sys

import numpy as np

from services.sandbox.config import (
    DebrisConfig,
    HostConfig,
    IntegratorConfig,
    SimConfig,
)
from services.sandbox.models import SimObject
from services.sandbox.observation_bundle import (
    ObservationBundle,
    generate_observation_bundle,
)
from services.sandbox.scenario import physical_properties_for_size_bin
from services.sandbox.schema_emission import TrackerIngestError, TrackerIngestResult
from services.sandbox.sensor_knowledge import SensorKnowledgeStore
from services.sandbox.sensor_model import SensorConfig
from services.sandbox.simulation import run_simulation


def _circular_leo_state(
    *,
    radius_km: float = 6978.137,
    phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    speed_km_s = float(np.sqrt(3.986004418e5 / radius_km))
    position = radius_km * np.array(
        [np.cos(phase_rad), np.sin(phase_rad), 0.0],
        dtype=np.float64,
    )
    velocity = speed_km_s * np.array(
        [-np.sin(phase_rad), np.cos(phase_rad), 0.0],
        dtype=np.float64,
    )
    return position, velocity


def _make_feed_sim_object(
    *,
    object_id: str,
    kind: str,
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    size_bin: str,
) -> SimObject:
    return SimObject(
        object_id=object_id,
        kind=kind,
        r_eci_km=np.asarray(r_eci_km, dtype=np.float64).copy(),
        v_eci_km_s=np.asarray(v_eci_km_s, dtype=np.float64).copy(),
        physical=physical_properties_for_size_bin(size_bin),
        metadata={"scenario": "scrum_373_sandbox_feed"},
    )


def build_feed_bundle(*, seed: int = 42) -> ObservationBundle:
    """
    Build one minimal co-orbital single-pass scenario's ObservationBundle.

    Same physical setup as canonical_scenarios.py's Tier 1 scenario (one
    host, one co-orbital debris object, 40s pass): a known-good
    configuration that reliably produces detections, reusing only the
    public sandbox primitives rather than canonical_scenarios.py itself.
    """
    host_r0, host_v = _circular_leo_state(radius_km=6978.137)

    relative_position_km = np.array([0.0, 20.0, 0.0], dtype=np.float64)
    relative_velocity_km_s = np.array([0.0, 0.05, 0.0], dtype=np.float64)

    debris_r0 = host_r0 + relative_position_km
    debris_v = host_v + relative_velocity_km_s

    objects = {
        "host_001": _make_feed_sim_object(
            object_id="host_001",
            kind="host",
            r_eci_km=host_r0,
            v_eci_km_s=host_v,
            size_bin="10cm",
        ),
        "debris_001": _make_feed_sim_object(
            object_id="debris_001",
            kind="debris",
            r_eci_km=debris_r0,
            v_eci_km_s=debris_v,
            size_bin="5cm",
        ),
    }

    config = SimConfig(
        seed=seed,
        debris=DebrisConfig(count=1),
        hosts=HostConfig(count=1),
        integrator=IntegratorConfig(
            dt_seconds=10.0,
            duration_seconds=40.0,
            save_every_n_steps=1,
        ),
    )

    sim_result = run_simulation(objects, config)

    sensor_cfg = SensorConfig(pointing_mode="velocity_aligned")

    return generate_observation_bundle(
        sim_result=sim_result,
        sensor_cfg=sensor_cfg,
        seed=seed,
    )


def feed(*, seed: int = 42, source: str = "sandbox") -> TrackerIngestResult:
    """
    Generate one scenario's ObservationBundle and push it through the live
    tracker /v1/observations interface (SCRUM-373 AC2). This is the whole
    feed: build a bundle, push it, return what the tracker said.
    """
    bundle = build_feed_bundle(seed=seed)
    store = SensorKnowledgeStore.from_observation_bundle(bundle)
    return store.push_to_tracker(bundle, source=source)


def main() -> int:
    try:
        result = feed()
    except TrackerIngestError as e:
        print(f"[feed] failed to reach or was rejected by tracker: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[feed] scenario produced no usable observations: {e}", file=sys.stderr)
        return 1

    print(
        f"[feed] pushed {result.observation_count} observation(s), "
        f"accepted={result.accepted}, "
        f"observation_ids={list(result.observation_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
