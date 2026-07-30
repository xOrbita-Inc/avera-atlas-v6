# Demo Scenarios

Synthetic conjunction scenarios for testing and demonstration of the AVERA-ATLAS pipeline.

## Why synthetic scenarios

The pipeline is designed to ingest real data from SWIR sensors and Space-Track CDMs. During development and demonstration we need controllable, repeatable cases that exercise specific conjunction geometries. Synthetic scenarios give deterministic outcomes, cover geometries that real data rarely produces, need no Space-Track account or live sensor, and run in seconds. Once real CDM ingestion is operational they stay useful as regression cases.

## Where the scenarios are defined

`services/ui/app/demo_presets.py`, and nowhere else.

That module is the single definition. Three things build from it: the UI endpoint `POST /api/scenarios/run`, the CLI in `demo/demo_scenarios.py`, and the regression test `services/propagator/tests/test_demo_asset_propagation.py`.

Before SCRUM-392 the geometry existed three times, as a literal in the UI, a hand-copied literal in that test, and an independent generator in the CLI with different object names and different miss distances. Nothing kept them in step, so the test's baseline could describe a demo nobody was running. If you change a scenario, change it in `demo_presets.py`.

## The five scenarios

Every object declares a time of closest approach, the miss it passes at, and optionally its position uncertainty. Those are what the propagator produces, asserted per object in the regression test. Risk level is computed downstream from Pc by `pc_to_risk_level`, so it is recorded here as observed output rather than as a design target.

The "known as" column is what the object's covariance comes from. **tracked** means the scenario supplies a real position sigma, the way a CDM would. **TLE** means it does not, and the propagator falls back to 2000 m divided by confidence. That distinction moves Pc by more than a decade at the same miss distance, which is why it is shown.

### nominal

Tens of km of separation. Confirms the pipeline classifies low-risk conjunctions without triggering maneuver evaluation.

| Object | Miss at TCA | TCA | Known as | Observed risk |
|---|---|---|---|---|
| OBJ-NOM-000 | 50.0 km | T+17 min | TLE | NOMINAL |
| OBJ-NOM-001 | 31.6 km | T+17 min | TLE | NOMINAL |
| OBJ-NOM-002 | 80.2 km | T+20 min | TLE | NOMINAL |

### warning

Kilometre-scale misses, close enough to screen and then be dismissed.

| Object | Miss at TCA | TCA | Known as | Observed risk |
|---|---|---|---|---|
| OBJ-WRN-000 | 2220 m | T+17 min | TLE | AMBER |
| OBJ-WRN-001 | 2807 m | T+17 min | TLE | GREEN |
| OBJ-WRN-002 | 3002 m | T+17 min | TLE | GREEN |
| OBJ-WRN-003 | 2500 m | T+17 min | TLE | GREEN |

### critical

The closest geometry the preset format expresses, and the scenario that makes the covariance point.

| Object | Miss at TCA | TCA | Known as | Observed risk |
|---|---|---|---|---|
| OBJ-CRT-000 | 22.4 m | T+17 min | tracked, 250 m | RED |
| OBJ-CRT-001 | 50.0 m | T+17 min | tracked, 250 m | RED |
| OBJ-CRT-002 | 316.2 m | T+17 min | TLE | AMBER |
| OBJ-CRT-003 | 100.0 m | T+17 min | TLE | AMBER |

Objects 1 and 3 are worth pausing on. They sit 50 m and 100 m from the asset, distances of the same order, and they are a decade apart in Pc. The difference is not geometry, it is that one has been tracked and the other is known from a TLE. Risk is what you know about where something is, not only where it is. A test asserts that this pair stays split, so if the demo ever collapses them into one band it fails loudly.

### mixed

A spread, for showing the operator list with more than one severity.

| Object | Miss at TCA | TCA | Known as | Observed risk |
|---|---|---|---|---|
| OBJ-MIX-000 | 51.0 m | T+17 min | tracked, 250 m | RED |
| OBJ-MIX-001 | 2508 m | T+17 min | TLE | GREEN |
| OBJ-MIX-002 | 3500 m | T+17 min | TLE | GREEN |
| OBJ-MIX-003 | 1500 m | T+17 min | TLE | AMBER |
| OBJ-MIX-004 | 50.0 km | T+17 min | TLE | NOMINAL |

All four tiers in one screen.

### demo

The curated three-object preset from SCRUM-370 step 3, specified as a target miss in the asset's RTN frame and constructed in closed form.

| Object | Miss at TCA | TCA | Known as | Observed risk |
|---|---|---|---|---|
| OBJ-DEMO-ALWAYS | 316.2 m | T+120 min | TLE | AMBER |
| OBJ-DEMO-FLIP | 412.3 m | T+60 min | TLE | AMBER |
| OBJ-DEMO-NEVER | 3007 m | T+60 min | TLE | GREEN |

## How RED is reached, and why it was not before

Read this before adding a scenario in the hope of producing a RED alert, because geometry alone will not do it.

`PC_RED_THRESHOLD` is 1e-4 and `HBR_M` is 15 m. An object with no supplied covariance inherits `DEFAULT_DEBRIS_UNCERTAINTY_M`, 2000 m, divided by its confidence, so its uncertainty can only ever be worse than 2 km. That puts the encounter-plane sigma between 1.03 and 1.37 km, and for a near-zero miss Pc is about `HBR^2 / (2 sigma_x sigma_z)`, which tops out near 9.7e-5. An object placed exactly on the asset does not reach RED. Moving objects closer cannot change that; it is a covariance ceiling, not a geometry shortfall.

SCRUM-391 did not move the thresholds or the hard-body radius to recover the colour. Those are physical and policy quantities that exist for reasons unrelated to the demo, and a test pins them. What changed is that a scenario can now state what it actually knows about an object. `TRACKED_SECONDARY_SIGMA_M` is 250 m, an order of magnitude better than the TLE default and in the range a CDM for a well-tracked LEO object carries a day or two before TCA. At that covariance the same geometry clears 1e-4 by a factor of ten.

The value is not finely tuned. Pc against the critical preset's own misses moves less than 2x across a 5x change in it, because once sigma approaches the miss distances the probability saturates:

| Debris sigma | Pc at 22 m | Pc at 100 m | Pc at 316 m |
|---|---|---|---|
| 500 m | 7.36e-04 | 7.14e-04 | 5.31e-04 |
| 250 m | 1.06e-03 | 1.02e-03 | 6.63e-04 |
| 100 m | 1.21e-03 | 1.15e-03 | 7.08e-04 |

Anything in that band gives the same risk labels, so the demo's RED does not hinge on the number chosen.

`test_red_is_still_unreachable_without_a_supplied_covariance` keeps the ceiling pinned for the fallback path, so an object known only from a TLE still cannot reach RED however close it passes.

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
    # Known from a TLE: uncertainty comes from 2000 m / confidence.
    DebrisSpec(t_star_s=17 * SAMPLE_DT_S, miss_y_km=0.1, miss_z_km=0.0,
               v_approach_km_s=0.03, confidence=0.9),
    # Tracked: uncertainty is stated and used as given.
    DebrisSpec(t_star_s=17 * SAMPLE_DT_S, miss_y_km=0.05, miss_z_km=0.0,
               v_approach_km_s=0.03, confidence=0.9,
               position_sigma_m=TRACKED_SECONDARY_SIGMA_M),
],
```

| Field | Meaning |
|---|---|
| `t_star_s` | Time of closest approach, seconds. **Must be a multiple of `SAMPLE_DT_S` (60 s).** |
| `miss_y_km` | Miss component along ECI y, which is the asset's along-track direction at t=0 |
| `miss_z_km` | Miss component along ECI z, cross-track |
| `v_approach_km_s` | Closing speed along the approach axis |
| `confidence` | Scales debris position uncertainty when no sigma is supplied: `2000 m / confidence` |
| `position_sigma_m` | Optional. 1-sigma position uncertainty in metres, used as given. **Not** scaled by confidence, because a supplied sigma is how well the object is known and confidence is only a stand-in for the same thing. Omit for a TLE-grade object. |

Start distance is derived as `t_star_s * v_approach_km_s` rather than given, so an off-grid preset cannot be expressed.

### Why TCA has to be on the grid

The propagator finds closest approach by taking the argmin of separation over samples spaced `SAMPLE_DT_S` apart. If true TCA falls between two samples, the nearest sample still carries separation along the approach axis, and that residual is what gets reported as the miss.

Every preset was off-grid before SCRUM-392, all at a true TCA of 1000 s against a 60 s grid, so the nearest sample sat 20 s away. The critical preset's first object asked for a 22 m miss and the propagator reported 400 m, 18 times larger. The stated `miss_y` and `miss_z` values were decoration. Two tests now guard this, one on the multiple and one comparing stated miss against produced miss per object.

### Physics

The asset rides a circular equatorial orbit at 500 km, propagated with `kepler_propagate`. Each debris object is carried as that Keplerian asset track plus a constant relative offset, `rel0 + vrel * t`, so miss distance and TCA depend only on relative motion. Objects start offset along the approach axis and converge, which is what produces a usable encounter view rather than a static separation.

Confidences are fixed values, not drawn. They used to be `np.random.uniform` in the UI and a seeded draw in the test, so the live demo's Pc varied between runs and the test's baseline could not have described it.

The propagator records which covariance each Pc was computed against, in `debris_sigma_m` and `covariance_sources` on `prop_multi.npz`. That is deliberately visible rather than internal, in the same spirit as SCRUM-369's covariance source marker: an operator looking at a Pc should be able to tell whether it rests on a measurement or on a default.

## Demo tips

- **Maneuver planning workflow**: use `critical`. Two RED objects force the planner to rank candidates, and the two AMBER objects at comparable miss distances give you the covariance point to make out loud.
- **Multi-event triage**: use `mixed`, which spans all four bands and fills the operator table.
- **Full spectrum**: run `nominal` then `critical` back to back.
- **Policy demonstration**: run `critical`, select a conjunction, then raise the lambda_v slider and re-plan. Prograde's utility drops as fuel cost is penalised more heavily, which is the intelligence-first point.

When presenting, say plainly which objects carry a supplied covariance and which are TLE-grade. The RED cases rest on a stated 250 m sigma, not on a measurement from a real CDM, and the surrogate is still a surrogate. Volunteering that is a stronger position than being asked.
