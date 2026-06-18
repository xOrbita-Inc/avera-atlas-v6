from __future__ import annotations

import time

from .config import DebrisConfig, HostConfig, SimConfig
from .scenario import create_seeded_swarm_and_hosts
from .simulation import run_simulation


MAX_RUNTIME_SECONDS = 300.0


def main() -> None:
    config = SimConfig(
        seed=42,
        debris=DebrisConfig(count=500),
        hosts=HostConfig(count=6, mode="distributed"),
    )

    objects = create_seeded_swarm_and_hosts(config)

    start = time.perf_counter()
    result = run_simulation(objects, config)
    elapsed_seconds = time.perf_counter() - start

    print(f"objects: {len(objects)}")
    print(f"snapshots: {len(result.snapshots)}")
    print(f"elapsed_sec: {elapsed_seconds:.2f}")
    print(f"under_5_min: {elapsed_seconds < MAX_RUNTIME_SECONDS}")

    if elapsed_seconds >= MAX_RUNTIME_SECONDS:
        raise SystemExit(
            f"SCRUM-290 benchmark failed: {elapsed_seconds:.2f}s "
            f">= {MAX_RUNTIME_SECONDS:.0f}s"
        )


if __name__ == "__main__":
    main()