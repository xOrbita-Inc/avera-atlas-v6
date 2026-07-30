# Demo Scenarios

Synthetic conjunction scenarios for testing and demonstration of the AVERA-ATLAS pipeline.

## Why synthetic scenarios

The pipeline is designed to ingest real data from SWIR sensors and Space-Track CDMs. During development and demonstration we need controllable, repeatable cases that exercise specific conjunction geometries. Synthetic scenarios give deterministic outcomes, cover geometries that real data rarely produces, need no Space-Track account or live sensor, and run in seconds. Once real CDM ingestion is operational they stay useful as regression cases.

## Where the scenarios are defined

`services/ui/app/demo_presets.py`, and nowhere else.

That module is the single definition. Three things build from it: the UI endpoint `POST /api/scenarios/run`, the CLI in `demo/demo_scenarios.py`, and the regression test `services/propagator/tests/test_demo_asset_propagation.py`.

Before SCRUM-392 the geometry existed three times, as a literal in the UI, a hand-copied literal in that test, and an independent generator in the CLI with different object names and different miss distances. Nothing kept them in step, so the test's baseline could describe a demo nobody was running. If you change a scenario, change it in `demo_presets.py`.

## The five scenarios

Every object declares a time of closest approach and the miss it passes at. Those are what the propagator produces, asserted per object in the regression test. Risk level is computed downstream from Pc by `pc_to_risk_level`, so it is recorded here as observed output rather than as a design target.

### nominal

Tens of km of separation. Confirms the pipeline classifies low-risk conjunctions without triggering maneuver evaluation.

| Object | Miss at TCA | TCA | Observed risk |
|---|---|---|---|
| OBJ-NOM-000 | 50.0 km | T+17 min | NOMINAL |
| OBJ-NOM-001 | 31.6 km | T+17 min | NOMINAL |
| OBJ-NOM-002 | 80.2 km | T+20 min | NOMINAL |

### warning

Kilometre-scale misses, close enough to screen and then be dismissed.

| Object | Miss at TCA | TCA | Observed risk |
|---|---|---|---|
| OBJ-WRN-000 | 2220 m | T+17 min | AMBER |
| OBJ-WRN-001 | 2807 m | T+17 min | GREEN |
| OBJ-WRN-002 | 3002 m | T+17 min | GREEN |
| OBJ-WRN-003 | 2500 m | T+17 min | GREEN |

### critical

The closest geometry the preset format expresses.

| Object | Miss at TCA | TCA | Observed risk |
|---|---|---|---|
| OBJ-CRT-000 | 22.4 m | T+17 min | AMBER |
| OBJ-CRT-001 | 50.0 m | T+17 min | AMBER |
| OBJ-CRT-002 | 316.2 m | T+17 min | AMBER |
| OBJ-CRT-003 | 100.0 m | T+17 min | AMBER |

### mixed

A spread, for showing the operator list with more than one severity.

| Object | Miss at TCA | TCA | Observed risk |
|---|---|---|---|
| OBJ-MIX-000 | 51.0 m | T+17 min | AMBER |
| OBJ-MIX-001 | 2508 m | T+17 min | GREEN |
| OBJ-MIX-002 | 3500 m | T+17 min | GREEN |
| OBJ-MIX-003 | 1500 m | T+17 min | AMBER |
| OBJ-MIX-004 | 50.0 km | T+17 min | NOMINAL |

### demo

The curated three-object preset from SCRUM-370 step 3, specified as a target miss in the asset's RTN frame and constructed in closed form.

| Object | Miss at TCA | TCA | Observed risk |
|---|---|---|---|
| OBJ-DEMO-ALWAYS | 316.2 m | T+120 min | AMBER |
| OBJ-DEMO-FLIP | 412.3 m | T+60 min | AMBER |
| OBJ-DEMO-NEVER | 3007 m | T+60 min | GREEN |

## No scenario reaches RED, and none can

Read this before adding a scenario in the hope of producing a RED alert.

`PC_RED_THRESHOLD` is 1e-4 and `HBR_M` is 15 m. Debris position uncertainty comes from `DEFAULT_DEBRIS_UNCERTAINTY_M`, 2000 m, divided by the object's confidence, so it can only be inflated above 2 km and never reduced. That puts the encounter-plane sigma between 1.03 and 1.37 km, and for a near-zero miss Pc is about `HBR^2 / (2 sigma_x sigma_z)`, which tops out near 9.7e-5. An object placed exactly on the asset does not reach RED.

This is a covariance ceiling, not a geometry shortfall, so moving an object closer cannot change it. `test_red_is_unreachable_at_the_demo_covariance` asserts the ceiling so the next person to try finds out in a second. **SCRUM-391** is the fix: a per-object covariance in the scenario spec, so a scenario can describe a well-tracked secondary the way a genuine RED conjunction has one.

Until then the demo's top severity tier is AMBER. That is honest output from correct Pc, not a regression. Before SCRUM-390 the demo showed RED on Pc values that were 4.26x too high.

## Usage

### From the dashboard

Select a scenario, click **RUN SCENARIO**, and the pipeline nodes light up as each stage processes. Conjunctions appear in the Active Conjunctions table.

### From the command line

```bash
python demo/demo_scenarios.py critical
python demo/demo_scenarios.py            # default: mixed
python demo/demo_scenarios.py demo       # the curated three-object preset
```

The CLI writes `states_multi.npz` and prints the geometry. It does not print a risk level, because it has no basis for one; the propagator computes Pc and assigns the band.

### From Docker

```bash
docker compose exec ui python /app/demo/demo_scenarios.py critical
```

## Adding a scenario

Add an entry to `PRESETS` in `services/ui/app/demo_presets.py`, plus a label in `PRESET_LABELS`.

```python
"your_scenario": [
    DebrisSpec(t_star_s=17 * SAMPLE_DT_S, miss_y_km=0.1, miss_z_km=0.0,
               v_approach_km_s=0.03, confidence=0.9),
],
```

| Field | Meaning |
|---|---|
| `t_star_s` | Time of closest approach, seconds. **Must be a multiple of `SAMPLE_DT_S` (60 s).** |
| `miss_y_km` | Miss component along ECI y, which is the asset's along-track direction at t=0 |
| `miss_z_km` | Miss component along ECI z, cross-track |
| `v_approach_km_s` | Closing speed along the approach axis |
| `confidence` | Scales debris position uncertainty: `2000 m / confidence` |

Start distance is derived as `t_star_s * v_approach_km_s` rather than given, so an off-grid preset cannot be expressed.

### Why TCA has to be on the grid

The propagator finds closest approach by taking the argmin of separation over samples spaced `SAMPLE_DT_S` apart. If true TCA falls between two samples, the nearest sample still carries separation along the approach axis, and that residual is what gets reported as the miss.

Every preset was off-grid before SCRUM-392, all at a true TCA of 1000 s against a 60 s grid, so the nearest sample sat 20 s away. The critical preset's first object asked for a 22 m miss and the propagator reported 400 m, 18 times larger. The stated `miss_y` and `miss_z` values were decoration. Two tests now guard this, one on the multiple and one comparing stated miss against produced miss per object.

### Physics

The asset rides a circular equatorial orbit at 500 km, propagated with `kepler_propagate`. Each debris object is carried as that Keplerian asset track plus a constant relative offset, `rel0 + vrel * t`, so miss distance and TCA depend only on relative motion. Objects start offset along the approach axis and converge, which is what produces a usable encounter view rather than a static separation.

Confidences are fixed values, not drawn. They used to be `np.random.uniform` in the UI and a seeded draw in the test, so the live demo's Pc varied between runs and the test's baseline could not have described it.

## Demo tips

- **Maneuver planning workflow**: use `critical`, which puts four objects in the same band and forces the planner to rank candidates.
- **Multi-event triage**: use `mixed`, which spans three bands and fills the operator table.
- **Full spectrum**: run `nominal` then `critical` back to back.
- **Policy demonstration**: run `critical`, select a conjunction, then raise the lambda_v slider and re-plan. Prograde's utility drops as fuel cost is penalised more heavily, which is the intelligence-first point.

When presenting, say that the demo covariance is a surrogate and the top tier is AMBER by construction. It is a stronger position than showing a RED that the numbers do not support.
