# AVERA-ATLAS v6

**Autonomous Conjunction Assessment & Maneuver Planning for Space Situational Awareness**

AVERA-ATLAS is xOrbita's intelligence-first SSA platform. It fuses CCSDS 508.0-B-1 Conjunction Data Messages from Space-Track with on-orbit SWIR detections through a containerized microservice pipeline, producing operator-ready maneuver recommendations. Target edge compute is the NVIDIA Jetson Orin Nano aboard CubeSat-class spacecraft.

This repository contains the v6 stack running at [averaatlas.space](https://averaatlas.space).

---

## Architecture

AVERA-ATLAS follows a strict separation of concerns between two layers:

- **APS (Analytical Planning Stack)** — computational engine that produces machine-readable outputs. All decision logic, orbital mechanics, and risk assessment live here. APS never generates UI content.
- **ATLAS (Advanced Tracking and Location Analysis System)** — presentation layer that renders APS outputs for human operators. ATLAS never runs analytics.

This boundary is non-negotiable. Violations are architectural defects, not style preferences.

### Pipeline

The system has two parallel data sources that converge at the Planner:

```
Space-Track CDM feed ──► Ingest ───────────────────────────────┐
                         (8001)                                 │
                                                                ▼
SWIR frames ──► Detector ──► Tracker ──► Propagator ──► Planner ──► ATLAS
                (8000)       (8002)      (internal)     (8060)      (8080)
```

- The **CDM path** (Ingest) persists conjunction covariance from Space-Track so the Planner can retrieve real operator-grade uncertainty at decision time.
- The **detection path** (Detector → Tracker) converts image-plane detections into tracked orbital states for screening and eventual handoff to the Planner.

Each stage is an independent Docker container. Services communicate via REST APIs and a shared Docker volume for bulk artifacts (`states_multi.npz`, `prop_multi.npz`).

### Service Map

| Service | Port | Role | Layer |
|---------|------|------|-------|
| **ingest** | 8001 | Persists Space-Track CDMs; sole writer to the CDM store per ADR-001 | APS |
| **detector** | 8000 | YOLOv8 SWIR debris classification (11 spacecraft classes); forwards to tracker | APS |
| **tracker** | 8002 | Pixel → ECI transform, cross-sensor correlation, angles-only IOD, track lifecycle | APS |
| **physics-classifier** | 8003 | DINOv2 spectrogram classifier (ACTIVE_SAT, DEAD_SAT, DEBRIS, etc.); parallel to detector | APS |
| **propagator** | — | Keplerian orbit propagation, Pc calculation, conjunction screening | APS |
| **planner** | 8060 | V2.4 avoidance decision model (CW + Mahalanobis utility) | APS |
| **viz** | — | Trajectory visualization, MP4/PNG generation | APS |
| **ui** | 8080 | ATLAS dashboard — renders planner output, never computes | ATLAS |

### Data Flow

**CDM path (operational):**

1. Operator or scheduled job calls `POST /cdm/poll` on Ingest with a NORAD ID or Pc threshold.
2. Ingest fetches matching CDMs from Space-Track in CCSDS 508.0-B-1 KVN format, parses them, persists covariance to SQLite.
3. Planner retrieves covariance via `GET /cdm/{primary}/{secondary}` before evaluating a conjunction.

**Detection path (operational as of SCRUM-325):**

1. SWIR sensor (or uploaded image via the UI) submits a frame to Detector `POST /predict`.
2. Detector runs YOLOv8 inference, returns bounding boxes synchronously to the UI, and asynchronously forwards the batch to Tracker `POST /detections`.
3. Tracker transforms pixel coordinates to ECI line-of-sight, correlates with existing uncorrelated-track buffers, and attempts IOD once enough observations have accumulated (≥3 angular observations spanning ≥0.5° of arc).
4. Tracker exports confirmed tracks to `states_multi.npz` via `POST /export/states` for the propagator.

**Convergence at Planner:**

5. Propagator reads states, runs Keplerian propagation, writes `prop_multi.npz` with conjunction data.
6. UI reads propagated data, displays conjunction table.
7. Operator selects a conjunction. UI proxies the evaluation request to Planner.
8. Planner retrieves the matching CDM covariance from Ingest, evaluates burn candidates (prograde, radial, cross-track), returns a recommendation.
9. ATLAS renders encounter geometry, burn vector, risk metrics, and the decision card.

### Shared Volume

All services share a Docker volume mounted at `/data/planner_artifacts/`. Key artifacts:

| File | Producer | Consumer |
|------|----------|----------|
| `states_multi.npz` | tracker (primary), demo_scenarios | propagator |
| `prop_multi.npz` | propagator | viz, ui |
| `planner_output.mp4` | viz | ui |
| `conjunction_summary.png` | viz | ui |

A separate volume at `/data/cdm_store/` holds the SQLite database owned by Ingest.

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- ARM64 host (Jetson Orin Nano) or Apple Silicon Mac with Rancher Desktop
- 4GB+ available memory
- Space-Track account for live CDM polling (optional; `/cdm/inject` supports manual injection for demos)

### Build and Run

```bash
cd avera-atlas-v6

# Build all containers
docker compose build

# Start the stack
docker compose up -d

# Open dashboard
open http://localhost:8080
```

### Verify Services

```bash
# All containers running
docker compose ps

# Planner healthy
curl http://localhost:8060/health

# Detector healthy and pointing at the right tracker URL
curl http://localhost:8000/health

# Pipeline status from the UI
curl http://localhost:8080/api/pipeline/status
```

### End-to-End Smoke Test

After any change to detector or tracker, verify the live pipeline:

```bash
./scripts/smoke_test_pipeline.sh
```

The script posts a synthetic image to Detector and polls Tracker for the resulting UCT. Expected output ends with `[smoke] SMOKE TEST PASSED`.

### Run a Demo Scenario

From the dashboard:
1. Select a scenario type (Nominal, Warning, Critical, Mixed)
2. Click **RUN SCENARIO**
3. Wait for pipeline nodes to turn green
4. Click a conjunction row to evaluate it
5. Adjust operator policy sliders and click **RE-PLAN** to see ranking changes

---

## V2.4 Decision Model

The planner implements Sreerjit Sarma's physics-based optimization model:

**Inputs:** Relative position/covariance at TCA, satellite inertial state at burn time, operator policy parameters (λv fuel weight, λL lifetime weight, Δv magnitude limit), optional precomputed Pc.

**Process:**
1. Generate candidate burn directions (prograde, radial, cross-track) from satellite state vectors.
2. Map each Δv through the Clohessy-Wiltshire state transition matrix (Φ_rv) to predict post-burn relative position at TCA.
3. Compute Mahalanobis distance (m²) as a confidence-weighted miss metric.
4. Calculate utility: U = ΔC − λv·dv − λL·lifetime_penalty.
5. Select highest-utility candidate as recommendation.

**Outputs:** Recommended burn vector (3D ECI km/s), burn time, utility score, confidence gain ΔC, fuel cost, lifetime penalty, post-maneuver risk surrogate, and all candidate evaluations.

### Known Limitations

- **CW framework**: Assumes near-circular orbits; Yamanaka-Ankersen or J2-corrected alternatives needed for eccentric orbits
- **Single-impulse burns**: No multi-burn optimization
- **No attitude dynamics**: Burns assumed instantaneous along candidate directions

### OpenAPI Specification

API contracts are defined in `openapi/planner.yaml` (APS Planner v2.4.3) and `openapi/ingest.yaml` (Ingest Service v1.0.0). Tracker's schema is defined inline at `services/tracker/schemas.py` pending formal OpenAPI extraction. All services code against these interface contracts.

---

## Dashboard Features

The ATLAS dashboard provides real-time conjunction assessment visualization:

- **Encounter Geometry View** — Canvas 2D visualization at meter/kilometer scale showing asset position, debris approach trajectory, miss distance, HBR (15m), covariance ellipse, and burn vector
- **Maneuver Decision Card** — GO/STANDBY/NO ACTION recommendation with risk classification
- **Burn Candidates Table** — Ranked comparison of prograde/radial/cross-track burns with ΔC, dv, and utility scores. Click rows to preview burn vectors in the encounter view
- **Fuel vs Confidence Chart** — Visual comparison of fuel cost against risk reduction for each candidate
- **Operator Policy Controls** — Interactive sliders for λv (fuel weight), λL (lifetime weight), and Δv limit. Changes trigger re-planning with visible ranking shifts
- **Zoom/Pan** — Scroll to zoom (0.15x–25x), drag to pan, double-click to reset. Default 2.5x zoom
- **Out-of-plane rendering** — Cross-track burns (perpendicular to orbital plane) display engineering notation (⊙/⊗) instead of zero-length arrows

---

## Project Structure

```
avera-atlas-v6/
├── docker-compose.yaml          # Service orchestration
├── conf/                        # Configuration files
├── data/                        # Runtime artifacts (gitignored)
├── demo/                        # Demo scenario generator
│   ├── demo_scenarios.py        # Scenario definitions and generator
│   └── README.md                # Scenario documentation
├── scripts/                     # Operational scripts
│   └── smoke_test_pipeline.sh   # End-to-end Detector → Tracker smoke test
├── k8s/                         # Kubernetes manifests
│   ├── 00-namespace.yaml
│   ├── 01-detector.yaml
│   ├── 02-planner.yaml
│   ├── 03-viz.yaml
│   └── 04-ui.yaml
├── openapi/
│   ├── planner.yaml             # APS Planner interface contract (v2.4.3)
│   └── ingest.yaml              # Ingest Service interface contract (v1.0.0)
└── services/
    ├── ingest/                  # CDM persistence (primary); legacy detection buffer
    ├── detector/                # YOLOv8 SWIR detection, forwards to tracker
    ├── tracker/                 # Pixel→ECI, correlation, IOD, track lifecycle
    ├── physics-classifier/      # DINOv2 spectrogram classifier
    ├── propagator/              # Keplerian orbit propagation
    ├── planner/                 # V2.4 avoidance decision model
    ├── viz/                     # Trajectory visualization
    └── ui/                      # ATLAS dashboard
```

---

## Deployment

### Development (Docker Compose)

```bash
docker compose up -d --build
```

After source changes, use `--force-recreate` to ensure running containers pick up the new image:

```bash
docker compose up -d --build --force-recreate
```

### Edge Compute (Jetson Orin Nano)

The ARM64 architecture is shared between Apple Silicon development machines and the Jetson Orin Nano target hardware. Docker images built on M4 Max transfer without cross-compilation.

```bash
# Build on Mac
docker compose build

# Export images
docker save avera/ui:v6 avera/planner:v6 | gzip > atlas-images.tar.gz

# Transfer and load on Jetson
scp atlas-images.tar.gz jetson:~/
ssh jetson "docker load < atlas-images.tar.gz"
```

### Kubernetes

Manifests in `k8s/` provide production deployment templates. Apply in order:

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/
```

---

## Demo Framing

When presenting to technical audiences (for example, LeoLabs), this system should be framed as **pipeline validation with live Space-Track CDM covariance**. Real CCSDS 508.0-B-1 covariance is integrated through the Ingest service; conjunction assessment quality reflects actual CDM data when available, and surrogate identity covariance when not (always labelled in the UI).

What the demo proves:
- End-to-end pipeline from detection to maneuver recommendation works
- V2.4 decision model correctly evaluates burn candidates using CW dynamics
- Operator policy parameters visibly change planner rankings
- Architecture supports real-time edge deployment

What the demo does not prove:
- Operational Pc accuracy at scale (depends on CDM availability for the specific object pair)
- Multi-orbit propagation fidelity (CW is short-arc only)
- SWIR detection performance against real on-orbit imagery (current validation uses synthetic scenes and uploaded ground images)

---

## Roadmap

| Version | Milestone |
|---------|-----------|
| V2.2 | CDM parser (CCSDS 508.0-B-1), Space-Track integration for real ephemeris ✅ |
| V2.3 | Real covariance data replaces identity matrix surrogate ✅ |
| V2.4 | CW + Mahalanobis utility decision model, in production ✅ |
| V2.5 | Mission-aware maneuver planning (slot definitions, station-keeping rules, return-to-slot logic) |
| V3.x | Sensor-constrained on-demand autonomy; Pc derived from xOrbita CQD-CMOS sensor |
| V4.0 | Mission-optimized orbital positioning; multi-constraint optimization |

See the Strategic Timeline document for the full milestone schedule.

---

## License

Proprietary — xOrbita Inc. All rights reserved.
