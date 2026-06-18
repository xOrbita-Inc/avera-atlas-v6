# Sandbox Service

Simulation core for APS 3.0 sandbox development.

## Goal

Provide an Earth-centered "god view" simulation for:

* debris swarm propagation
* host satellite constellation propagation
* deterministic state snapshots over time

This package is the foundation for later sensor/FOV modeling and schema-compliant observation emission.

## SCRUM-290 scope

This first sandbox PR includes only the simulation core:

* two-body propagation
* J2 perturbation
* simple per-object drag model
* deterministic fixed-step simulation loop
* configurable debris swarm
* configurable host constellation
* true-state snapshotting for every object
* vectorized batch propagation for the main simulation loop
* reproducible performance benchmark

Deferred to later sandbox PRs:

* sensor/FOV detection modeling
* angular observation generation
* NPZ/JSONL export
* IOD/tracker integration
* APS planner integration

## Package layout

* `config.py` — simulation configuration
* `models.py` — core data models
* `propagator.py` — scalar and vectorized RK4 orbital dynamics
* `atmosphere.py` — simple atmospheric density and drag helpers
* `scenario.py` — deterministic scenario generation
* `simulation.py` — fixed-step vectorized simulation loop
* `snapshot.py` — god-view snapshot utilities
* `benchmark.py` — reproducible SCRUM-290 performance benchmark
* `Dockerfile` — reproducible sandbox test image
* `tests/` — propagator, determinism, snapshot, and orbital-period tests

## How to run

Run from the repository root:

```bash
python -c "from services.sandbox.config import SimConfig, IntegratorConfig; from services.sandbox.scenario import create_hello_world_scenario; from services.sandbox.simulation import run_simulation; cfg=SimConfig(integrator=IntegratorConfig(dt_seconds=10.0,duration_seconds=86400.0,save_every_n_steps=360)); objects=create_hello_world_scenario(cfg); result=run_simulation(objects,cfg); print('t=0:', result.snapshots[0].states['debris_001'].r_eci_km); print('t=1hr:', result.snapshots[6].states['debris_001'].r_eci_km); print('t=24hr:', result.snapshots[-1].states['debris_001'].r_eci_km)"
```

This creates a minimal host/debris scenario and prints the debris true position at `t=0`, `t=1hr`, and `t=24hr`.

## Seeded deterministic run

Run from the repository root:

```bash
python -c "from services.sandbox.config import SimConfig, DebrisConfig, HostConfig, IntegratorConfig; from services.sandbox.scenario import create_seeded_swarm_and_hosts; from services.sandbox.simulation import run_simulation; cfg=SimConfig(seed=42,debris=DebrisConfig(count=10),hosts=HostConfig(count=3,mode='distributed'),integrator=IntegratorConfig(dt_seconds=10.0,duration_seconds=3600.0,save_every_n_steps=60)); objects=create_seeded_swarm_and_hosts(cfg); result=run_simulation(objects,cfg); print('objects:', len(objects)); print('snapshots:', len(result.snapshots)); print('first debris:', result.snapshots[-1].states['debris_0001'].r_eci_km)"
```

Using the same seed and initial configuration produces bit-identical propagated god-view trajectories. This behavior is covered by a committed test that runs the simulation twice and compares every saved position and velocity using exact array equality.

## Performance benchmark

SCRUM-290 target: a 24-hour simulation with 500 debris objects and 6 host satellites must run in under 5 minutes on a standard laptop.

Run the committed benchmark from the repository root:

```bash
python -m services.sandbox.benchmark
```

Latest local result:

```text
objects: 506
snapshots: 1441
elapsed_sec: 58.74
under_5_min: True
```

The benchmark uses:

* 500 debris objects
* 6 distributed host satellites
* 24-hour simulation duration
* 1-second fixed timestep
* J2 perturbation
* per-object drag

## Orbital-period verification

The committed propagator test initializes a host satellite at 600 km, propagates it using the actual RK4 + J2 + drag simulation path, measures one full propagated orbit, and verifies that it completes approximately 14.9 orbits per day within the required ±1% tolerance.

## Tests

Run sandbox tests from the repository root:

```bash
python -m pytest services/sandbox/tests -v
```

Run the full repository test gate:

```bash
python -m pytest
```

Build and run the sandbox tests in Docker:

```bash
docker build -f services/sandbox/Dockerfile -t avera-sandbox-scrum-290 .
docker run --rm avera-sandbox-scrum-290
```

At PR validation time:

```text
services/sandbox/tests: 9 passed
full repository: 451 passed, 10 warnings
docker sandbox run: 9 passed
performance benchmark: 58.74 seconds, under_5_min: True
```
