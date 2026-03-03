# Viz Service

Trajectory visualization generator producing MP4 animations and PNG summary charts for conjunction events.

## Role in Pipeline

```
SWIR Sensor → Ingest → Detect → Track → Propagate → [VIZ] → ATLAS
```

The Viz service reads propagated trajectory data and generates visual artifacts for operator review. It produces a 3D orbital animation (MP4) and a static conjunction summary chart (PNG) that the ATLAS dashboard can display.

## Outputs

| File | Format | Description |
|------|--------|-------------|
| `planner_output.mp4` | MP4 (H.264) | 3D orbital view animation with risk-colored trajectory segments |
| `conjunction_summary.png` | PNG | Static chart showing conjunction timeline, miss distances, and risk levels |

## Visualization Features

- 3D orbital view with wireframe Earth at proper scale
- Asset and debris trajectories color-coded by risk level (RED/AMBER/GREEN/NOMINAL)
- TCA markers with miss distance annotations
- Risk-colored trajectory segments that highlight the close approach window
- Proper orbital mechanics scaling (not schematic — actual km distances)

## Risk Colors

| Level | Color |
|-------|-------|
| RED | `#ff4757` |
| AMBER | `#ffa502` |
| GREEN | `#2ed573` |
| NOMINAL | `#66fcf1` |
| Asset | `#00d4ff` |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DATA_DIR` | `/data/planner_artifacts` | Shared volume path for input/output |

## Docker

```bash
docker build -t avera/viz:v6 .
```

The Viz service runs as a batch process. It watches for `prop_multi.npz` in the shared volume, generates visualizations, and writes output files. Requires `ffmpeg` for MP4 encoding (installed in the Dockerfile).

## Dependencies

- `matplotlib` — 3D plotting and animation
- `ffmpeg` — Video encoding (system package)
- `numpy` — Trajectory data handling
