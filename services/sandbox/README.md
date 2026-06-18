# Sandbox Service

Simulation and sensor-model foundation for APS 3.0 sandbox development.

## Goal

Provide an Earth-centered sandbox that can:

* propagate debris and host satellite ground-truth states;
* produce deterministic state snapshots;
* apply realistic sensor visibility gates;
* emit noisy angular observations for downstream IOD processing.

The package currently includes the SCRUM-290 simulation core and the SCRUM-291 TT240-40 sensor and FOV model.

## SCRUM-290: Simulation core

The simulation core provides:

* two-body orbital propagation;
* J2 perturbation;
* simple per-object atmospheric drag;
* deterministic fixed-step RK4 integration;
* configurable debris swarms;
* configurable host constellations;
* true-state snapshotting for every object;
* vectorized batch propagation;
* reproducible performance validation.

## SCRUM-291: Sensor and FOV model

The sensor layer converts ground-truth host and debris states into simulated sensor observations.

Implemented TT240-40 and Acuros CQD-CMOS properties:

* 3.7° full conical FOV;
* 40 mm aperture;
* 240 mm focal length;
* 2.9 arcsec angular resolution;
* neuromorphic event mode;
* no frame-cycling duty-cycle penalty;
* configurable fixed pointing relative to host velocity or nadir.

Detection requires all of the following:

* target lies inside the sensor FOV;
* target lies inside the hard range cutoff for its size class;
* target is sunlit;
* the FOV does not overlap the bright Earth limb.

Hard operational range cutoffs:

* 1 cm debris: 12 km;
* 5 cm debris: 59 km;
* 10 cm debris: 117 km.

Detected observations include:

* timestamp;
* host and debris identifiers;
* noisy right ascension and declination;
* 2.9 arcsec one-sigma angular uncertainty;
* observer ECI position and velocity;
* range and off-boresight metadata;
* illumination and Earth-limb status;
* angular-rate estimates from consecutive detections.

The observation contract is validated against the existing tracker IOD interface.

## Deferred scope

The following remain outside SCRUM-290 and SCRUM-291:

* full radiometric background modeling;
* thermal-emission detection;
* production NPZ or JSONL export;
* complete admissible-region integration;
* tracker custody management;
* APS planner integration;
* UI integration.

## Package layout

* `config.py` — simulation configuration
* `models.py` — core simulation data models
* `propagator.py` — scalar and vectorized RK4 orbital dynamics
* `atmosphere.py` — atmospheric density and drag helpers
* `scenario.py` — seeded simulation and sensor-validation scenarios
* `simulation.py` — fixed-step vectorized simulation loop
* `snapshot.py` — god-view snapshot utilities
* `benchmark.py` — SCRUM-290 propagation-performance benchmark
* `sensor_model.py` — TT240-40 pointing, FOV, range, illumination, and Earth-limb gates
* `observations.py` — noisy RA/Dec observation and angular-rate generation
* `observation_bundle.py` — observation generation across simulation snapshots
* `sensor_benchmark.py` — deterministic SCRUM-291 sensor acceptance benchmark
* `Dockerfile` — reproducible sandbox test image
* `tests/` — simulation, sensor, transit, determinism, and IOD compatibility tests

## Simulation hello world

Run from the repository root:

```bash
python -c "from services.sandbox.config import SimConfig, IntegratorConfig; from services.sandbox.scenario import create_hello_world_scenario; from services.sandbox.simulation import run_simulation; cfg=SimConfig(integrator=IntegratorConfig(dt_seconds=10.0,duration_seconds=86400.0,save_every_n_steps=360)); objects=create_hello_world_scenario(cfg); result=run_simulation(objects,cfg); print('t=0:', result.snapshots[0].states['debris_001'].r_eci_km); print('t=1hr:', result.snapshots[6].states['debris_001'].r_eci_km); print('t=24hr:', result.snapshots[-1].states['debris_001'].r_eci_km)"
```

This creates a minimal host/debris scenario and prints the debris true position at `t=0`, `t=1hr`, and `t=24hr`.

## Seeded deterministic simulation

Run from the repository root:

```bash
python -c "from services.sandbox.config import SimConfig, DebrisConfig, HostConfig, IntegratorConfig; from services.sandbox.scenario import create_seeded_swarm_and_hosts; from services.sandbox.simulation import run_simulation; cfg=SimConfig(seed=42,debris=DebrisConfig(count=10),hosts=HostConfig(count=3,mode='distributed'),integrator=IntegratorConfig(dt_seconds=10.0,duration_seconds=3600.0,save_every_n_steps=60)); objects=create_seeded_swarm_and_hosts(cfg); result=run_simulation(objects,cfg); print('objects:', len(objects)); print('snapshots:', len(result.snapshots)); print('first debris:', result.snapshots[-1].states['debris_0001'].r_eci_km)"
```

Using the same seed and initial configuration produces bit-identical propagated trajectories. A committed test runs the simulation twice and compares every saved position and velocity using exact array equality.

## Generate a sensor observation

Run from the repository root:

```bash
python -c "import random, numpy as np; from services.sandbox.scenario import create_forced_detection_scenario; from services.sandbox.sensor_model import SensorConfig; from services.sandbox.observations import generate_angular_observation; objects=create_forced_detection_scenario(); obs=generate_angular_observation(host=objects['host_001'],debris=objects['debris_001'],t_seconds=0.0,sensor_cfg=SensorConfig(pointing_mode='velocity_aligned'),rng=random.Random(42),sun_dir_eci=np.array([0.0,1.0,0.0])); print(obs)"
```

This creates a deterministic in-FOV, in-range, sunlit target and emits a noisy angular observation.

## SCRUM-290 performance benchmark

The SCRUM-290 target is a 24-hour simulation with 500 debris objects and 6 host satellites in under 5 minutes on a standard laptop.

Run:

```bash
python -m services.sandbox.benchmark
```

Validated result:

```text
objects: 506
snapshots: 1441
elapsed_sec: 61.52
under_5_min: True
```

The benchmark uses:

* 500 debris objects;
* 6 distributed host satellites;
* 24-hour duration;
* 1-second fixed timestep;
* J2 perturbation;
* per-object drag.

## SCRUM-291 sensor acceptance benchmark

Run:

```bash
python -m services.sandbox.sensor_benchmark
```

Validated result:

```text
1-sensor configuration
  scheduled new encounters: 4
  detected encounters: 4
  unique targets detected: 4
  detections_per_day: 4.00
  target_per_day: 3.0–6.0
  within_acceptance_band: True

6-sensor configuration
  scheduled new encounters: 18
  detected encounters: 18
  unique targets detected: 18
  detections_per_day: 18.00
  target_per_day: 12.0–25.0
  within_acceptance_band: True

outside_fov_control: True
outside_range_control: True
eclipse_control: True
earth_limb_control: True
```

This benchmark uses a deterministic Section 4.3 acceptance population. It validates that the production sensor gates correctly accept and reject representative encounters and that the resulting daily counts lie inside the required reference bands.

It is not intended to claim that a uniformly sampled 500-object debris swarm independently reproduces the operational debris population density.

## Transit-time verification

The sensor acceptance tests validate representative Section 5.2 transit cases through the production FOV gate:

```text
co-orbital: approximately 7.63 seconds
moderate crossing: approximately 0.95 seconds
high crossing: approximately 0.38 seconds
```

The tests also verify that transit duration decreases as relative crossing speed increases.

## Orbital-period verification

A committed propagator test initializes a host satellite at 600 km, propagates it through the actual RK4, J2, and drag simulation path, measures one complete orbit, and verifies approximately 14.9 orbits per day within ±1%.

## IOD compatibility

Committed compatibility tests verify that detected sandbox observations can be mapped into the tracker IOD interface with:

* timezone-aware timestamps;
* RA and Dec in radians;
* angular uncertainties;
* observer ECI position in kilometres;
* observer ECI velocity in kilometres per second.

Three converted observations are accepted by the existing IOD solver interface. Single-observation angular rates are retained for future partial-tracklet and admissible-region integration.

## Tests

Run the sandbox suite:

```bash
python -m pytest services/sandbox/tests -v
```

Run the full repository gate:

```bash
python -m pytest
```

Build and run the sandbox suite in Docker:

```bash
docker build -f services/sandbox/Dockerfile -t avera-sandbox-scrum-291 .
docker run --rm avera-sandbox-scrum-291
```

Latest validation:

```text
sensor acceptance benchmark: passed
services/sandbox/tests: 97 passed
full repository: 539 passed, 10 warnings
docker sandbox run: 97 passed
```
