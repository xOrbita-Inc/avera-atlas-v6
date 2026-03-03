# UI Service (ATLAS Dashboard)

Presentation layer for the AVERA-ATLAS conjunction assessment system. Renders APS planner output for human operators. **ATLAS never runs analytics.**

## Role in Pipeline

```
SWIR Sensor → Ingest → Detect → Track → Propagate → Plan → [ATLAS]
```

The UI service is the ATLAS presentation layer. It serves the single-page dashboard, proxies API requests to upstream APS services, and reads propagated data from the shared volume. All decision logic, orbital mechanics, and risk computation live in the APS services — ATLAS only renders their outputs.

## APS/ATLAS Boundary

This boundary is the most important architectural constraint in the system:

- **ATLAS may**: Display data, format numbers, render charts, proxy requests to APS services, manage UI state
- **ATLAS must not**: Compute Pc, evaluate burn candidates, classify risk levels, run orbital mechanics, make GO/NO-GO decisions

If you find analytics code in this service, it is an architectural defect.

## Dashboard Panels

### Left Sidebar
- **Scenario** — Select and run demo scenarios (Nominal, Warning, Critical, Mixed)
- **Operator Policy** — Interactive sliders for λv (fuel weight), λL (lifetime weight), and Δv limit. Changes trigger re-planning via the Planner service
- **Tracking Summary** — Object count, risk level breakdown, closest approach, maximum Pc
- **APS V2.4 Output** — Raw planner metrics: m² pre/post, fuel cost, lifetime penalty, burn vector (ECI km/s)

### Center
- **Encounter Geometry** — Canvas 2D visualization showing asset, debris approach, miss distance, HBR (15m combined), 1σ covariance envelope, burn vector arrow, and post-burn miss envelope. Supports scroll zoom (0.15x–25x), click-drag pan, and double-click reset

### Right Sidebar
- **Maneuver Decision** — GO/STANDBY/NO ACTION card with target, TCA countdown, burn direction, and Δv magnitude
- **Burn Candidates** — Table of prograde/radial/cross-track candidates ranked by utility. Click rows to preview burn vectors in the encounter view (amber arrow for previewed, cyan for recommended, ghost arrow shows recommended when previewing alternatives)
- **Fuel vs Confidence Gain** — Three-column chart comparing fuel cost against ΔC risk reduction
- **Active Conjunctions** — Clickable table of all conjunction events with miss distance, Pc, TCA, and risk classification

### Bottom Bar
- **Risk metrics** — Pc pre→post, m² pre→post, utility score (live-updating)

## API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Serve dashboard HTML |
| `/api/planner/evaluate` | POST | Proxy to Planner `/v1/evaluate` |
| `/api/planner/evaluate/batch` | POST | Proxy to Planner `/v1/evaluate/batch` |
| `/api/planner/health` | GET | Check Planner service health |
| `/api/conjunctions` | GET | Read conjunction data from `prop_multi.npz` |
| `/api/pipeline/status` | GET | Aggregate health of all upstream services |
| `/api/scenarios/run` | POST | Generate and run a demo scenario |
| `/api/video` | GET | Serve visualization MP4 |
| `/api/summary-image` | GET | Serve conjunction summary PNG |

## File Structure

```
ui/
├── app/
│   ├── main.py                # FastAPI backend — proxies to APS services
│   └── templates/
│       └── index.html         # Single-page dashboard (HTML + JS + CSS)
├── Dockerfile
└── requirements.txt
```

The dashboard is a single HTML file containing all CSS, JavaScript, and inline assets (including the xOrbita logo as base64). No build toolchain, no npm, no bundler. This keeps the container image small and eliminates frontend build dependencies on the Jetson.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PLANNER_SERVICE_URL` | `http://planner:8060` | APS Planner service endpoint |
| `TRACKER_SERVICE_URL` | `http://tracker:8000` | Tracker service endpoint |
| `SWIR_SERVICE_URL` | `http://detector:8000/predict` | Detector service endpoint |
| `DATA_DIR` | `/data/planner_artifacts` | Shared volume path |

## Docker

```bash
docker build -t avera/ui:v6 .
docker run -p 8080:8000 \
  -e PLANNER_SERVICE_URL=http://planner:8060 \
  -v planner_data:/data/planner_artifacts \
  avera/ui:v6
```

## Development

The dashboard is assembled from four Python source files in the build toolchain (not shipped in the container):

| File | Content |
|------|---------|
| `dash_html.py` | HTML structure and panel layout |
| `dash_css.py` | All CSS styles |
| `dash_logic.py` | JavaScript — state management, API calls, planner integration |
| `dash_renderer.py` | JavaScript — Canvas 2D encounter geometry rendering |

These are combined by `asm.py` into the final `index.html`. To modify the dashboard:

```bash
# Edit source files
vim dash_logic.py

# Reassemble
python asm.py

# Rebuild container
docker compose build ui && docker compose up ui
```
