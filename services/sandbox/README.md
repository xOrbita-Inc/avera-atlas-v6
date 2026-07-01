# Sandbox Service

Simulation, sensor-model, tasking, and canonical regression-harness foundation for APS 3.0 sandbox development.

## Goal

Provide an Earth-centered sandbox that can:

* propagate debris and host satellite ground-truth states;
* produce deterministic state snapshots;
* apply realistic sensor visibility gates;
* emit noisy angular observations for downstream IOD processing;
* export sensor-visible observations using the tracker-compatible `observations_multi.npz` schema;
* enforce strict god-view / sensor-knowledge isolation;
* execute sandbox-side re-observation tasking commands;
* run the canonical three-tier IOD regression harness.

The package currently includes the SCRUM-290 simulation core, the SCRUM-291 TT240-40 sensor and FOV model, the SCRUM-292 observation schema emission and isolation layer, the SCRUM-293 re-observation tasking interface, and the SCRUM-294 canonical regression harness.

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

## SCRUM-292: Observation schema emission and sensor-knowledge isolation

The schema-emission layer converts detected sandbox observations into the tracker-compatible `observations_multi.npz` contract.

The emitted artifact includes:

* UTC timestamps;
* stable observer IDs and target IDs;
* observer and target index arrays;
* observation time index arrays;
* observation type;
* observer ECI position in metres;
* observer ECI velocity in metres per second;
* RA and Dec in radians;
* angular uncertainty in arcseconds;
* observation quality labels;
* optional range and range-rate fields using `NaN` when unavailable;
* metadata describing frame, units, schema version, sensor type, and god-view isolation.

The emitted `observations_multi.npz` artifact is loaded directly through the existing tracker observation loader and converted to tracker `IODObservation` records without sandbox-specific adapter code.

The sensor-knowledge layer exposes only the observations actually emitted by each sensor. It deliberately blocks access to simulation objects, snapshots, and truth states from the sensor-facing interface. Attempts to access god-view state from sensor knowledge raise an error.

Deterministic replay is also validated: the same simulation seed and sensor seed produce the same loaded observation stream.

## SCRUM-293: Re-observation tasking interface

The sandbox supports a minimum viable re-observation tasking interface for Tier 2 testing. APS-side tasking logic remains out of scope; this module only executes valid tasking commands inside the sandbox.

Implemented tasking features:

- `TaskingCommand` with `task_id`, `sensor_id`, `start_time`, `end_time`, `target`, and `priority`.
- `predicted_point` target support using an explicit commanded ECI boresight.
- Survey-mode detection remains unchanged; fixed pointing still flows through `detect_object(...)`.
- Tasked detection uses `detect_object_with_boresight(...)`, which applies the same FOV, range, sunlight, and Earth-limb gates as survey mode.
- Tasked observations are returned as normal `AngularObservation` records and can be emitted through the Sandbox 3 `observations_multi.npz` schema path.
- Explicit task rejection is returned for unknown sensors, non-host sensors, missing task-window snapshots, unsupported admissible-region execution, invalid time windows, and blackout-window constraint conflicts.

The end-to-end APS tasking-service-to-IOD closure remains part of Sandbox 5. SCRUM-293 provides the sandbox execution boundary needed for that test.

## SCRUM-294: Canonical regression harness

The sandbox includes a canonical regression harness for exercising the three-tier IOD scenario structure and reporting which parts are fully validated versus scaffolded or deferred.

Run all canonical scenarios:

```bash
python -m services.sandbox.regression_harness
```

Write the JSON report to disk:

```bash
python -m services.sandbox.regression_harness --output artifacts/sandbox_regression_report.json
```

The harness emits a deterministic JSON report with suite status, top-level acceptance status, per-scenario pass/fail status, IOD closure metrics, measured or deferred error metrics, tracked-object counts, and human-readable summaries.

Implemented canonical scenarios:

Scenario A — Tier 1 co-orbital single-pass
* 1 host satellite.
* 1 co-orbital debris object.
* 5 observations over a 40 second pass.
* Tracker IOD closes from the single-pass tracklet.
* Along-track error is measured against scenario truth at the solver-selected epoch.
* This scenario is acceptance-validated when the measured along-track error is under 10 km.
  
Scenario B — Tier 2 high-crossing tasked re-observation scaffold
* 3 host satellites.
* 1 high-crossing debris object.
* Initial 2-observation partial tracklet fails IOD as expected.
* Additional re-observation samples are combined with the partial tracklet and the final tracker IOD closes.
* The current implementation does not yet route the re-observation through the SCRUM-293 execute_tasking_command(...) interface.
* This scenario is retained as a closure scaffold only.
* Tasking-interface-driven closure and final accuracy validation remain future work.

Scenario C — Tier 3 custody maintenance deferred scaffold
* 6 host satellites.
* 50 debris objects.
* 7 simulated days are represented in the report shape.
* The reference custody-uncertainty timeline is recorded for traceability.
* Emergent uncertainty computation is not performed in this harness.
* Full-scale swarm runs and emergent custody-uncertainty validation are deferred to SCRUM-343.

The harness can pass as an execution/reporting scaffold while the top-level `acceptance_validated` field remains `false` until all acceptance-level validation stories are complete. Scenario C must not be cited as uncertainty validation; it points to SCRUM-343 for that work.

## SCRUM-358: Emergent encounter-rate validation

The sandbox includes an emergent encounter-rate benchmark for the SCRUM-291 AC2 follow-up. Unlike the original deterministic sensor benchmark, this benchmark does not schedule a fixed number of reference encounters. It builds a density-calibrated 500-object debris population, propagates the full swarm and host constellation through the sandbox simulation core, screens the resulting geometry through the production TT240-40 sensor gates, and measures detections per day from the propagated run.

Run the benchmark:

```bash
python -m services.sandbox.emergent_encounter_benchmark
```

Run the 24-hour validation configuration used for SCRUM-358:

```bash
python -m services.sandbox.emergent_encounter_benchmark --debris-count 500 --duration-seconds 86400 --dt-seconds 10 --save-every-n-steps 6
```

The benchmark reports:

* whether the result was derived from a propagated swarm;
* whether the initial population used Section 4.2 density calibration;
* the density multiplier and local eligible-density fraction;
* 1-sensor, 6-sensor same-host, and 6-sensor distributed results;
* total sensor samples, in-FOV samples, detected samples, unique detected targets, and detections per day;
* target Section 4.3 bands and ±30% acceptance bands;
* rejection-reason diagnostics for FOV, range, sunlight, and Earth-limb gates.

Current 24-hour validation result:

```text
1-sensor configuration:
  detections_per_day: 4.0
  target_per_day: 3.0–6.0
  within_acceptance_band: True

6-sensor same-host configuration:
  detections_per_day: 5.0
  scaling behavior: sub-linear relative to distributed coverage

6-sensor distributed configuration:
  detections_per_day: 18.0
  target_per_day: 12.0–25.0
  within_acceptance_band: True
```

The measured rates are emergent from propagation plus the production FOV, range, sunlight, and Earth-limb gates. The benchmark uses a density-calibrated initial population rather than a hardcoded encounter list, so detections-per-day values are no longer fixture constants.


## Deferred scope

The following remain outside SCRUM-290 through SCRUM-294:

* full radiometric background modeling;
* thermal-emission detection;
* JSONL debug mirror export;
* production admissible-region integration;
* Scenario B tasking-interface-driven closure through `execute_tasking_command(...)`;
* Scenario B final accuracy validation from SCRUM-293-produced tasked observations;
* production covariance propagation for custody management;
* emergent custody-uncertainty validation, owned by SCRUM-343;
* APS planner integration;
* UI integration;
* emergent encounter-rate validation from an operationally representative debris population.

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
* `schema_emission.py` — tracker-compatible `observations_multi.npz` emission
* `sensor_knowledge.py` — sensor-visible observation streams with god-view access blocked
*  `tasking.py` — re-observation tasking command execution, explicit-boresight detection, and task rejection handling
* `canonical_scenarios.py` — canonical three-tier IOD scenario definitions and report models
* `regression_harness.py` — CLI entry point for the SCRUM-294 canonical regression harness
* `Dockerfile` — reproducible sandbox test image
* `tests/` — simulation, sensor, transit, schema, isolation, tasking, regression-harness, determinism, and IOD compatibility tests

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

## Observation schema emission

The sandbox can emit tracker-compatible observation artifacts from detected sensor observations.

The primary artifact is:

```text
observations_multi.npz
```

This file is generated at runtime and should not normally be committed.

The emitted NPZ follows the existing tracker loader contract:

```text
times_utc
observer_ids
target_ids
obs_observer_idx
obs_target_idx
obs_time_idx
obs_type
observer_eci_m
observer_eci_m_s
obs_ra_rad
obs_dec_rad
obs_sigma_ra_arcsec
obs_sigma_dec_arcsec
obs_quality
obs_range_m
obs_range_rate_m_s
meta
```

For TT240-40 optical observations, `obs_type` is `angles`, and unavailable range/range-rate values are emitted as `NaN`.

The emitted artifact is validated by loading it through:

```text
services/tracker/observation_loader.py
```

and converting it to tracker `IODObservation` records without a sandbox-specific adapter.

## God-view / sensor-knowledge isolation

The sandbox maintains two separate views of the same run:

* **God view:** full simulation truth, including every object state at every saved time step.
* **Sensor knowledge:** only the ordered observations actually emitted by each sensor.

IOD-facing code should consume only the sensor-knowledge stream or the emitted observation artifact. The sensor-knowledge interface intentionally raises an error if code attempts to access simulation objects, snapshots, or truth states.

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

Committed compatibility tests verify that detected sandbox observations can be emitted to the tracker-compatible `observations_multi.npz` schema and then loaded into the tracker IOD path with:

* timezone-aware timestamps;
* RA and Dec in radians;
* angular uncertainties;
* observer ECI position in kilometres after loader conversion;
* observer ECI velocity in kilometres per second after loader conversion.

The schema-emission tests verify:

* emitted artifacts load with the tracker observation loader;
* emitted artifacts convert to tracker `IODObservation` records without sandbox-specific adapter code;
* the converted observation stream reaches `IODSolver().solve(...)`;
* same-seed replay produces the same loaded observation stream;
* different seeds preserve schema identity while changing noisy angles.

Single-observation angular rates are retained in sandbox observations for future partial-tracklet and admissible-region integration, but the current tracker `IODObservation` contract does not consume angular-rate fields directly.

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
docker build -f services/sandbox/Dockerfile -t avera-sandbox-scrum-294 .
docker run --rm avera-sandbox-scrum-294
```

Latest validation:

```text
SCRUM-293 tasking tests: 10 passed
SCRUM-294 regression harness tests: 6 passed
SCRUM-358 emergent encounter benchmark tests: 6 passed
services/sandbox/tests: 128 passed
full repository: 576 passed, 10 warnings
```

Current SCRUM-358 acceptance status:

```text
Encounter-rate source: propagated density-calibrated swarm
1-sensor measured rate: 4.0/day, target 3.0–6.0/day
6-sensor same-host measured rate: 5.0/day, sub-linear relative to distributed
6-sensor distributed measured rate: 18.0/day, target 12.0–25.0/day
Top-level benchmark passed: true
```
