# Propagator Service

Keplerian orbit propagation with conjunction screening and probability of collision (Pc) calculation.

## Role in Pipeline

```
SWIR Sensor → Ingest → Detect → Track → [PROPAGATE] → Plan → ATLAS
```

The Propagator reads assembled state vectors from the Ingest service, propagates each object's orbit forward using two-body Keplerian dynamics, and screens for close approaches to the asset. For each conjunction event, it computes the probability of collision using NASA-standard methods.

## Process

1. **Load states** — reads `states_multi.npz` from shared volume
2. **Asset propagation** — SGP4 propagation from TLE for the protected asset (ISS-like orbit)
3. **Debris propagation** — Keplerian two-body propagation for each tracked object
4. **Conjunction screening** — identifies close approaches within the screening threshold (100 km default)
5. **Pc calculation** — computes collision probability at each TCA using covariance-based methods
6. **Risk classification** — assigns RED/AMBER/GREEN/NOMINAL based on Pc thresholds
7. **Write output** — saves `prop_multi.npz` with propagated trajectories and conjunction data

## Risk Thresholds

| Level | Pc Threshold | Action |
|-------|-------------|--------|
| RED | Pc ≥ 1×10⁻⁴ | Maneuver evaluation required |
| AMBER | 1×10⁻⁵ ≤ Pc < 1×10⁻⁴ | Heightened monitoring |
| GREEN | 1×10⁻⁷ ≤ Pc < 1×10⁻⁵ | Standard tracking |
| NOMINAL | Pc < 1×10⁻⁷ | No action needed |

## Output Artifact

Writes `prop_multi.npz` to `/data/planner_artifacts/` containing propagated trajectories, conjunction events with TCA times, miss distances, Pc values, and risk levels.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DATA_DIR` | `/data/planner_artifacts` | Shared volume path |

### Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| `HBR_M` | 15.0 | Combined hard body radius (meters) |
| `SCREENING_THRESHOLD_KM` | 100.0 | Close approach screening distance |
| `DEFAULT_DEBRIS_UNCERTAINTY_M` | 2000.0 | Default position uncertainty (2 km) |

## Docker

```bash
docker build -t avera/propagator:v6 .
```

The propagator runs as a batch process (not a web server). It executes propagation when `states_multi.npz` is present, writes results, and sleeps until new data arrives.

## Dependencies

- `sgp4` — SGP4/SDP4 orbit propagator for the asset TLE
- `numpy` — Keplerian mechanics and matrix operations
- `pc_utils.py` — NASA-standard Pc calculation utilities
