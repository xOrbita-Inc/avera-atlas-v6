# Demo Scenarios

Synthetic conjunction scenario generator for testing and demonstration of the AVERA-ATLAS pipeline.

## Why Synthetic Scenarios?

The AVERA-ATLAS pipeline is designed to ingest real data from SWIR sensors and Space-Track CDMs. However, during development and demonstration, we need controllable, repeatable test cases that exercise specific risk levels and conjunction geometries. Synthetic scenarios provide:

- **Deterministic outcomes** — each scenario produces known risk levels, so you can verify the pipeline classifies and responds correctly
- **Full risk spectrum** — real operational data is dominated by NOMINAL events; synthetic scenarios guarantee RED/AMBER alerts for testing maneuver planning
- **No external dependencies** — no Space-Track account, no live sensor, no CDM feed required
- **Fast iteration** — generate and run in seconds rather than waiting for real conjunction events

Once real CDM ingestion (V2.2) and Space-Track integration are operational, these scenarios serve as regression tests to verify the pipeline still handles known cases correctly.

## Available Scenarios

### Nominal

All objects at safe distances. Validates that the pipeline correctly classifies low-risk conjunctions and does not trigger unnecessary maneuver evaluations.

| Object | Miss Distance | TCA | Expected Risk |
|--------|--------------|-----|---------------|
| SAT_001 | 50.0 km | T+200 min | NOMINAL |
| DEB_001 | 75.0 km | T+400 min | NOMINAL |
| SAT_002 | 100.0 km | T+600 min | NOMINAL |

### Warning

Mix of AMBER and GREEN alerts. Tests the intermediate risk tier where heightened monitoring is appropriate but maneuver may not be required.

| Object | Miss Distance | TCA | Expected Risk |
|--------|--------------|-----|---------------|
| DEB_001 | 500 m | T+150 min | AMBER |
| DEB_002 | 800 m | T+300 min | AMBER |
| SAT_001 | 25.0 km | T+500 min | NOMINAL |
| DEB_003 | 1.2 km | T+700 min | GREEN |

### Critical

RED alert with imminent collision risk. This is the primary demo scenario for the LeoLabs capabilities discussion — it forces the planner to evaluate maneuver candidates and produce a GO recommendation.

| Object | Miss Distance | TCA | Expected Risk |
|--------|--------------|-----|---------------|
| DEB_CRIT | 50 m | T+30 min | RED |
| DEB_002 | 300 m | T+120 min | RED |
| SAT_001 | 2.0 km | T+400 min | GREEN |

### Mixed

Realistic operational scenario with all risk levels represented. Best for demonstrating the full dashboard — multiple conjunction rows, risk-sorted table, and the ability to click between events and see different planner responses.

| Object | Miss Distance | TCA | Expected Risk |
|--------|--------------|-----|---------------|
| Cosmos_DEB | 80 m | T+45 min | RED |
| Fengyun_DEB | 400 m | T+180 min | RED |
| CubeSat_012 | 1.5 km | T+350 min | GREEN |
| Starlink_42 | 15.0 km | T+500 min | NOMINAL |
| Iridium_DEB | 600 m | T+720 min | AMBER |
| Unknown_001 | 45.0 km | T+900 min | NOMINAL |

## Usage

### From the Dashboard

1. Select a scenario button (Nominal / Warning / Critical / Mixed)
2. Click **RUN SCENARIO**
3. Pipeline nodes light up green as each stage processes
4. Conjunctions appear in the Active Conjunctions table

### From the Command Line

```bash
# Generate a specific scenario
python demo/demo_scenarios.py critical

# Default (mixed)
python demo/demo_scenarios.py
```

### From Docker

```bash
docker compose exec ui python -c "
import sys; sys.path.insert(0, '/app')
exec(open('demo/demo_scenarios.py').read())
write_scenario(generate_scenario('critical'))
"
```

## Creating Custom Scenarios

To add a new scenario, edit `demo_scenarios.py` and add an entry to the `scenarios` dict inside `generate_scenario()`:

```python
scenarios = {
    # ... existing scenarios ...
    "your_scenario": [
        {"name": "OBJ_001", "miss_km": 0.1,  "tca_step": 60,  "type": "debris"},
        {"name": "OBJ_002", "miss_km": 5.0,  "tca_step": 300, "type": "satellite"},
    ],
}
```

### Object Definition Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Object identifier displayed in the dashboard |
| `miss_km` | float | Closest approach distance in kilometers |
| `tca_step` | int | Time step at which TCA occurs (each step = 60 seconds, so `tca_step: 30` = T+30 min) |
| `type` | string | Object class: `"debris"`, `"satellite"`, or `"unknown"` |

### How Miss Distance Maps to Risk

The propagator computes Pc based on miss distance, relative velocity, and covariance. Approximate mapping with default parameters:

| Miss Distance | Expected Pc | Risk Level |
|--------------|-------------|------------|
| < 200 m | > 1×10⁻⁴ | RED |
| 200 m – 1 km | 1×10⁻⁵ – 1×10⁻⁴ | AMBER |
| 1 km – 10 km | 1×10⁻⁷ – 1×10⁻⁵ | GREEN |
| > 10 km | < 1×10⁻⁷ | NOMINAL |

These are approximate — actual Pc depends on relative velocity vector orientation and the covariance matrix. The propagator uses a default 2 km position uncertainty for debris when no CDM covariance is available.

### Physics Behind Scenario Generation

The generator uses a linear relative motion model to create debris states that produce predictable close approaches:

1. An asset orbit is defined (ISS-like: 420 km altitude, 51.6° inclination)
2. For each object, a debris state is computed such that the linear propagation produces the specified miss distance at the specified TCA time step
3. The approach geometry uses the local LVLH frame (radial, along-track, cross-track) to position debris on a realistic approach vector
4. Position and velocity vectors are transformed to ECI for downstream processing

This means objects don't start at the miss distance — they start offset and converge to it over time, producing realistic approach trajectories in the encounter view.

### Demo Tips

- **For LeoLabs**: Use "Critical" to show the maneuver planning workflow, then switch to "Mixed" to show multi-event triage
- **For investors**: Use "Mixed" — it fills the dashboard with activity and demonstrates scale
- **For engineering review**: Use "Nominal" then "Critical" back-to-back to show the pipeline handles the full spectrum
- **Policy demonstration**: Run "Critical", select the RED conjunction, then adjust the λv slider up to 0.05 and RE-PLAN — watch prograde's utility drop as fuel cost is penalized more heavily. This demonstrates the "intelligence-first" value proposition
