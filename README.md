# AVERA-ATLAS v6

**Autonomous Conjunction Assessment & Maneuver Planning for Space Situational Awareness**

AVERA-ATLAS is xOrbita's intelligence-first SSA platform. It processes SWIR sensor observations through a containerized microservice pipeline — from raw detection to maneuver recommendation — running on edge compute (NVIDIA Jetson Orin Nano) aboard CubeSat-class spacecraft.

This repository contains the v6 demonstration stack used to validate the APS (Analytical Planning Stack) v2.4 decision model against the ATLAS presentation layer.

---

## Architecture

AVERA-ATLAS follows a strict separation of concerns between two layers:

- **APS (Analytical/Autonomous Planning Stack)** — computational engine that produces machine-readable outputs. All decision logic, orbital mechanics, and risk assessment live here. APS never generates UI content.
- **ATLAS (Advanced Tracking and Location Analysis System)** — presentation layer that renders APS outputs for human operators. ATLAS never runs analytics.

This boundary is non-negotiable. Violations are architectural defects, not style preferences.

### Pipeline

```
SWIR Sensor → Ingest → Detect → Track → Propagate → Plan → ATLAS UI
              (8001)   (8000)   (8002)              (8060)   (8080)
```

Each stage is an independent Docker container communicating via REST APIs and shared volume artifacts.

### Service Map

| Service | Port | Role | Layer |
|---------|------|------|-------|
| **ingest** | 8001 | Buffers detection frames, writes `states_multi.npz` | APS |
| **detector** | 8000 | YOLOv8 SWIR debris classification (11 spacecraft classes) | APS |
| **tracker** | 8002 | Multi-sensor fusion, angles-only IOD, track correlation | APS |
| **propagator** | — | Keplerian orbit propagation, Pc calculation, conjunction screening | APS |
| **planner** | 8060 | V2.4 avoidance decision model (CW + Mahalanobis utility) | APS |
| **viz** | — | Trajectory visualization, MP4/PNG generation | APS |
| **ui** | 8080 | ATLAS dashboard — renders planner output, never computes | ATLAS |

### Data Flow

1. SWIR sensor captures frames → Detector classifies objects with YOLOv8
2. Ingest buffers detections, assembles state vectors → writes `states_multi.npz`
3. Propagator reads states, runs Keplerian propagation → writes `prop_multi.npz` with conjunction data
4. UI reads propagated data, displays conjunction table
5. Operator selects conjunction → UI proxies request to Planner service
6. Planner evaluates burn candidates (prograde/radial/cross-track), returns recommendation
7. ATLAS renders encounter geometry, burn vector, risk metrics, and decision card

### Shared Volume

All services share a Docker volume mounted at `/data/planner_artifacts/`. Key artifacts:

| File | Producer | Consumer |
|------|----------|----------|
| `states_multi.npz` | ingest, demo_scenarios | propagator |
| `prop_multi.npz` | propagator | viz, ui |
| `planner_output.mp4` | viz | ui |
| `conjunction_summary.png` | viz | ui |

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- ARM64 host (Jetson Orin Nano) or Apple Silicon Mac with Rancher Desktop
- 4GB+ available memory

### Build and Run

```bash
# Clone and enter project
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
# Check all containers are running
docker compose ps

# Check planner health
curl http://localhost:8060/health

# Check pipeline status
curl http://localhost:8080/api/pipeline/status
```

### Run a Demo Scenario

From the dashboard:
1. Select a scenario type (Nominal, Warning, Critical, Mixed)
2. Click **RUN SCENARIO**
3. Wait for pipeline nodes to turn green
4. Click a conjunction row to evaluate it
5. Adjust operator policy sliders and click **RE-PLAN** to see ranking changes

Or from the command line:
```bash
docker compose exec ui python -c "
import sys; sys.path.insert(0, '/app')
from demo.demo_scenarios import generate_scenario, write_scenario
write_scenario(generate_scenario('critical'))
"
```

---

## V2.4 Decision Model

The planner implements Sreejit Sarma's physics-based optimization model:

**Inputs:** Relative position/covariance at TCA, satellite inertial state at burn time, operator policy parameters (λv fuel weight, λL lifetime weight, Δv magnitude limit), optional precomputed Pc.

**Process:**
1. Generate candidate burn directions (prograde, radial, cross-track) from satellite state vectors
2. Map each Δv through the Clohessy-Wiltshire state transition matrix (Φ_rv) to predict post-burn relative position at TCA
3. Compute Mahalanobis distance (m²) as a confidence-weighted miss metric
4. Calculate utility: U = ΔC − λv·dv − λL·lifetime_penalty
5. Select highest-utility candidate as recommendation

**Outputs:** Recommended burn vector (3D ECI km/s), burn time, utility score, confidence gain ΔC, fuel cost, lifetime penalty, post-maneuver risk surrogate, and all candidate evaluations.

### Known Limitations

- **Live covariance**: CCSDS 508.0-B-1 CDM covariance from Space-Track (integrated V2.5)
- **CW framework**: Assumes near-circular orbits; Yamanaka-Ankersen or J2-corrected alternatives needed for eccentric orbits
- **Single-impulse burns**: No multi-burn optimization
- **No attitude dynamics**: Burns assumed instantaneous along candidate directions

### OpenAPI Specification

API contracts are defined in `openapi/planner.yaml` (APS Planner v2.4.2) and `openapi/ingest.yaml` (Ingest Service v1.0.0), published on SwaggerHub. All services code against these interface contracts.

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
├── k8s/                         # Kubernetes manifests
│   ├── 00-namespace.yaml
│   ├── 01-detector.yaml
│   ├── 02-planner.yaml
│   ├── 03-viz.yaml
│   └── 04-ui.yaml
├── openapi/
│   ├── planner.yaml             # APS Planner interface contract (v2.4.2)
│   └── ingest.yaml              # Ingest Service interface contract (v1.0.0)
└── services/
    ├── ingest/                   # Frame buffer and state assembly
    ├── detector/                 # YOLOv8 SWIR detection
    ├── tracker/                  # Multi-sensor fusion and IOD
    ├── propagator/               # Keplerian orbit propagation
    ├── planner/                  # V2.4 avoidance decision model
    ├── viz/                      # Trajectory visualization
    └── ui/                       # ATLAS dashboard
```

---

## Deployment

### Development (Docker Compose)

```bash
docker compose up -d --build
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

When presenting to technical audiences (e.g., LeoLabs), this system should be framed as **pipeline validation with live Space-Track CDM covariance**. Real CCSDS 508.0-B-1 covariance is now integrated; conjunction assessment quality reflects actual CDM data rather than identity-matrix placeholders.

What the demo proves:
- End-to-end pipeline from detection to maneuver recommendation works
- V2.4 decision model correctly evaluates burn candidates using CW dynamics
- Operator policy parameters visibly change planner rankings
- Architecture supports real-time edge deployment

What the demo does not prove:
- Operational Pc accuracy (requires real CDM covariance data)
- Multi-orbit propagation fidelity (CW is short-arc only)
- Actual SWIR detection performance (demo uses synthetic scenarios)

---

## Roadmap

| Version | Milestone |
|---------|-----------|
| V2.2 | CDM parser (CCSDS 508.0-B-1), Space-Track integration for real ephemeris |
| V2.3 | Real covariance data replaces identity matrix surrogate ✅ |
| V2.5 | Multi-burn optimization, J2-corrected propagation |
| V3.0 | Autonomous closed-loop: sensor → plan → execute without ground-in-the-loop |

---

## License

Proprietary — xOrbita / Avera Enterprises Inc. All rights reserved.
