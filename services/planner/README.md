# Planner Service (APS v2.4)

Avoidance decision model implementing Clohessy-Wiltshire dynamics with Mahalanobis confidence-gain utility optimization.

## Role in Pipeline

```
SWIR Sensor → Ingest → Detect → Track → Propagate → [PLAN] → ATLAS
```

The Planner is the core intelligence layer of the APS stack. Given a conjunction event and operator policy parameters, it evaluates candidate maneuver directions and recommends the optimal burn vector to reduce collision risk while minimizing fuel expenditure and satellite lifetime impact.

## V2.4 Decision Model

### Mathematical Framework

For each candidate burn direction d̂ (prograde, radial, cross-track):

1. **Burn vector**: Δv = d̂ · dv_mag (fixed magnitude from operator policy)
2. **CW mapping**: Δr = Φ_rv · Δv (Clohessy-Wiltshire state transition maps velocity change to position change at TCA)
3. **Post-burn position**: r_post = r_rel − Δr
4. **Mahalanobis metric**: m²_post = r_post^T · P_rel^{-1} · r_post (covariance-weighted miss distance)
5. **Confidence gain**: ΔC = m²_pre − m²_post (improvement in miss confidence)
6. **Utility**: U = ΔC − λv·dv_mag − λL·lifetime_penalty

The candidate with highest U is recommended.

### Candidate Directions

| Direction | Unit Vector | Physical Meaning |
|-----------|------------|-----------------|
| **Prograde** | û(v_sat) | Along velocity — shifts along-track timing |
| **Radial** | û(r_sat) | Along position — raises/lowers orbit |
| **Cross-track** | û(r_sat × v_sat) | Normal to orbital plane — out-of-plane shift |

If the satellite is attitude-restricted, only prograde is evaluated.

### Operator Policy Parameters

| Parameter | Field | Range | Default | Effect |
|-----------|-------|-------|---------|--------|
| Fuel weight | `lambda_v` | 0–0.1 | 0.01 | Higher → penalize fuel cost more |
| Lifetime weight | `lambda_L` | 0–0.05 | 0.005 | Higher → penalize satellite lifetime impact |
| Δv limit | `dv_mag_limit_m_s` | 0.1–5.0 | 1.0 | Maximum burn magnitude in m/s |

## API

All endpoints defined in `openapi/planner-v2.4.yaml`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health and version |
| `/v1/evaluate` | POST | Evaluate single conjunction |
| `/v1/evaluate/batch` | POST | Evaluate multiple conjunctions |

### Evaluate Request

```json
{
  "conjunction_id": "eval-OBJ-MIX-000",
  "satellite": {
    "sat_id": "XORB-001",
    "r_sat_km": [6878.0, 0.0, 0.0],
    "v_sat_km_s": [0.0, 7.668, 0.0],
    "t_burn_utc": "2024-01-15T12:00:00Z",
    "v_remaining_m_s": 25.0
  },
  "conjunction": {
    "obj_id": "DEB_001",
    "t_ca_utc": "2024-01-15T12:17:00Z",
    "r_rel_km": [0.5, 0.0, 0.0],
    "p_rel_km2": [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01],
    "pc_precomputed": 3.57e-04
  },
  "policy": {
    "lambda_v": 0.01,
    "lambda_L": 0.005,
    "dv_mag_limit_m_s": 1.0,
    "a_ref_km": 6878.0
  }
}
```

### Evaluate Response

```json
{
  "conjunction_id": "eval-OBJ-MIX-000",
  "recommendation": {
    "direction": "prograde",
    "dv_eci_km_s": [0.0, 0.001, 0.0],
    "dv_magnitude_m_s": 1.0,
    "t_burn_utc": "2024-01-15T12:00:00Z",
    "utility": -0.004
  },
  "metrics": {
    "delta_C": 0.006,
    "m2_pre": 25.10,
    "m2_post": 25.09,
    "fuel_cost_m_s": 1.0,
    "lifetime_penalty": 0.04,
    "risk_surrogate_post": 3.57e-04,
    "all_candidates": [
      {"direction": "prograde", "dv_eci_km_s": [0, 0.001, 0], "delta_C": 0.006, "utility": -0.004},
      {"direction": "radial", "dv_eci_km_s": [0.001, 0, 0], "delta_C": -0.381, "utility": -0.391},
      {"direction": "cross-track", "dv_eci_km_s": [0, 0, 0.001], "delta_C": 0.001, "utility": -0.009}
    ]
  }
}
```

## File Structure

```
planner/
├── server.py              # FastAPI wrapper — thin HTTP layer
├── avoid/
│   ├── __init__.py
│   └── decision_model.py  # Core math — CW, Mahalanobis, utility optimization
├── Dockerfile
└── requirements.txt
```

The separation between `server.py` and `decision_model.py` is intentional. The decision model is a pure Python module with no web framework dependency. It can be tested from the command line, imported into notebooks, or wrapped by any HTTP server.

## Known Limitations

- **Surrogate covariance**: The demo uses an identity matrix (P_rel = 0.01·I₃). Real CDM covariance data will significantly change m² values and candidate rankings.
- **CW assumptions**: The Clohessy-Wiltshire framework assumes near-circular orbits and short time spans. For eccentric orbits or multi-orbit propagation, Yamanaka-Ankersen or J2-corrected STMs are needed.
- **Fixed magnitude**: All candidates share the same |Δv|. Variable-magnitude optimization is a V2.5 feature.

## Docker

```bash
docker build -t avera/planner:v6 .
docker run -p 8060:8060 avera/planner:v6
```

## CLI Testing

The decision model supports standalone execution:

```bash
cd services/planner
python -m avoid.decision_model --help
echo '{ ... }' | python -m avoid.decision_model --stdin
```
