"""
AVERA-ATLAS Dashboard Backend (v6)

ATLAS presentation layer. All analytics delegated to APS services.
Decision logic lives in the Planner service (port 8060), not here.
"""

import os
import json
import numpy as np
import requests
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# SCRUM-392: single source of truth for the demo conjunction scenarios.
from app import demo_presets

app = FastAPI(title="AVERA-ATLAS Dashboard", version="6.0.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

APS_VERSION = "2.5"

PLANNER_SERVICE_URL = os.getenv("PLANNER_SERVICE_URL", "http://planner:8060")
TRACKER_SERVICE_URL = os.getenv("TRACKER_SERVICE_URL", "http://tracker:8000")
DETECTOR_SERVICE_URL = os.getenv("SWIR_SERVICE_URL", "http://detector:8000/predict")
PHYSICS_CLASSIFIER_URL = os.getenv(
    "PHYSICS_CLASSIFIER_URL",
    "http://physics-classifier:8000/predict"
)
PHYSICS_CLASSIFIER_HEALTH_URL = os.getenv(
    "PHYSICS_CLASSIFIER_URL",
    "http://physics-classifier:8000"
).replace("/predict", "") + "/health"
INGEST_SERVICE_URL = os.getenv("INGEST_SERVICE_URL", "http://ingest:8000")
DATA_DIR = os.getenv("DATA_DIR", "/data/planner_artifacts")
PROP_ARTIFACT_PATH = os.path.join(DATA_DIR, "prop_multi.npz")

templates = Jinja2Templates(directory="app/templates")

# SCRUM-421: the Live Asset dropdown's source of truth.
#
# The LeoLabs Catalog Object Subscriptions available during the trial (10
# objects, per the SCRUM-421 comment thread). NORAD ID is what the planner needs
# as primary_norad; name is what the operator reads. Trial-scoped and expected to
# change when the subscription set changes, which is why it lives here as one
# named constant rather than as literals scattered through the template.
#
# Fast-follow swap point: replace the body of leolabs_subscribed_assets() with a
# call to a LeoLabs subscriptions endpoint (the planner already builds an
# AssetRegistry from list_subscribed_objects). Nothing in index.html changes --
# it renders whatever this function returns.
LEOLABS_SUBSCRIBED_ASSETS = [
    {"name": "CRYOSAT 2",   "norad_id": 36508, "leolabs_catalog": "L2669"},
    {"name": "GRACE-FO 1",  "norad_id": 43476, "leolabs_catalog": "L21219"},
    {"name": "GRACE-FO 2",  "norad_id": 43477, "leolabs_catalog": "L21221"},
    {"name": "LARETS",      "norad_id": 27944, "leolabs_catalog": "L2486"},
    {"name": "SENTINEL 3A", "norad_id": 41335, "leolabs_catalog": "L1159"},
    {"name": "SENTINEL 3B", "norad_id": 43437, "leolabs_catalog": "L21073"},
    {"name": "STARLETTE",   "norad_id": 7646,  "leolabs_catalog": "L1768"},
    {"name": "SWARM A",     "norad_id": 39452, "leolabs_catalog": "L3972"},
    {"name": "SWARM B",     "norad_id": 39451, "leolabs_catalog": "L5429"},
    {"name": "SWARM C",     "norad_id": 39453, "leolabs_catalog": "L3969"},
]


def leolabs_subscribed_assets():
    """The subscribed assets the Live Asset dropdown offers."""
    return LEOLABS_SUBSCRIBED_ASSETS


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "aps_version": APS_VERSION,
            "leolabs_assets": leolabs_subscribed_assets(),
        },
    )


# ---------- Planner proxy (ATLAS never runs analytics) ----------

@app.post("/api/planner/evaluate")
async def planner_evaluate(request: Request):
    """Proxy a single conjunction evaluation to the APS Planner service."""
    body = await request.json()
    try:
        resp = requests.post(
            f"{PLANNER_SERVICE_URL}/v1/evaluate",
            json=body, timeout=60  # SCRUM-412: live LeoLabs evaluate (cold registry build + API round trips) exceeds 10s
        )
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except requests.exceptions.ConnectionError:
        return JSONResponse(status_code=503, content={
            "error": "Planner service unavailable"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/planner/evaluate/batch")
async def planner_batch(request: Request):
    """Proxy a batch evaluation to the APS Planner service."""
    body = await request.json()
    try:
        resp = requests.post(
            f"{PLANNER_SERVICE_URL}/v1/evaluate/batch",
            json=body, timeout=30
        )
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except requests.exceptions.ConnectionError:
        return JSONResponse(status_code=503, content={
            "error": "Planner service unavailable"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/planner/v1/leolabs/conjunctions")
async def planner_leolabs_conjunctions(request: Request):
    """Proxy the SCRUM-422 conjunction listing to the planner.

    Deliberately relays the upstream status code verbatim, unlike the two status
    proxies above which collapse any non-200 to 502. The dashboard has to tell
    these apart: 404 means the asset is not in the LeoLabs subscription, 422
    means the window was bad, and 503 means the feed is off -- which must not be
    shown to an operator as "no conjunctions", because an empty sky and an
    unavailable feed look identical once the distinction is lost.

    Timeout matches the evaluate proxy: a cold registry build plus the CDM search
    is the same round-trip cost.
    """
    try:
        resp = requests.get(
            f"{PLANNER_SERVICE_URL}/v1/leolabs/conjunctions",
            params=dict(request.query_params),
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        return JSONResponse(status_code=503, content={
            "error": "Planner service unavailable"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    try:
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except ValueError:
        return JSONResponse(status_code=502, content={
            "error": f"planner conjunction list returned invalid JSON "
                     f"(HTTP {resp.status_code})"
        })


@app.get("/api/planner/health")
async def planner_health():
    """Check planner service health."""
    try:
        resp = requests.get(f"{PLANNER_SERVICE_URL}/health", timeout=3)
        return JSONResponse(content=resp.json())
    except Exception:
        return JSONResponse(status_code=503, content={
            "status": "unreachable", "version": "unknown"
        })


@app.get("/api/planner/udl-status")
async def planner_udl_status():
    """Passthrough proxy to the planner UDL status endpoint (SCRUM-347).

    The dashboard badge (loadSourceMode) calls this on page load to decide
    the data source label. The planner computes the status from its env;
    ATLAS relays it verbatim and never reshapes or caches it.

    On any upstream failure (unreachable, timeout, or an error response) we
    return 502 with a JSON error body so the badge JS falls through cleanly
    on resp.ok being false rather than seeing a 500 traceback or a hang.
    """
    try:
        resp = requests.get(f"{PLANNER_SERVICE_URL}/udl-status", timeout=5)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})
    if not resp.ok:
        return JSONResponse(status_code=502, content={
            "error": f"planner udl-status returned {resp.status_code}"
        })
    try:
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except ValueError:
        return JSONResponse(status_code=502, content={
            "error": "planner udl-status returned invalid JSON"
        })


@app.get("/api/planner/leolabs-status")
async def planner_leolabs_status():
    """Passthrough proxy to the planner LeoLabs status endpoint (SCRUM-412).

    Mirrors the UDL status proxy. The dashboard badge (loadSourceMode) calls this
    first, since LeoLabs takes precedence over UDL as the live conjunction source.
    Relayed verbatim; any upstream failure returns 502 so the badge JS falls
    through to the UDL / reference source cleanly.
    """
    try:
        resp = requests.get(f"{PLANNER_SERVICE_URL}/leolabs-status", timeout=5)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})
    if not resp.ok:
        return JSONResponse(status_code=502, content={
            "error": f"planner leolabs-status returned {resp.status_code}"
        })
    try:
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except ValueError:
        return JSONResponse(status_code=502, content={
            "error": "planner leolabs-status returned invalid JSON"
        })


# ---------- Ingest CDM proxy ----------

@app.post("/api/ingest/poll")
async def ingest_poll(request: Request):
    """Proxy a CDM poll request to the ingest service.

    Triggers a Space-Track fetch and persists results to the CDM store.
    Body: { "norad_id": int } or { "pc_threshold": float, "days_lookahead": int }
    """
    body = await request.json()
    try:
        resp = requests.post(
            f"{INGEST_SERVICE_URL}/cdm/poll",
            json=body, timeout=30
        )
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except requests.exceptions.ConnectionError:
        return JSONResponse(status_code=503, content={
            "error": "Ingest service unavailable"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/ingest/cdm/source_mode")
async def ingest_cdm_source_mode():
    """Passthrough proxy to the ingest CDM source-mode endpoint (SCRUM-347).

    The dashboard badge calls this after the planner UDL check to label the
    CDM data source (live Space-Track polling vs reference CDM). ATLAS relays
    the ingest response verbatim and never reshapes or caches it.

    Defined BEFORE the /cdm/{primary_norad}/{secondary_norad} route as
    defensive ordering so the literal source_mode path can never be shadowed
    by the two-segment record lookup.

    On any upstream failure (unreachable, timeout, or an error response) we
    return 502 with a JSON error body so the badge JS falls through cleanly
    on resp.ok being false.
    """
    try:
        resp = requests.get(f"{INGEST_SERVICE_URL}/cdm/source_mode", timeout=5)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})
    if not resp.ok:
        return JSONResponse(status_code=502, content={
            "error": f"ingest cdm/source_mode returned {resp.status_code}"
        })
    try:
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except ValueError:
        return JSONResponse(status_code=502, content={
            "error": "ingest cdm/source_mode returned invalid JSON"
        })


@app.get("/api/ingest/cdm/{primary_norad}/{secondary_norad}")
async def ingest_cdm_lookup(primary_norad: str, secondary_norad: str):
    """Proxy a CDM record lookup to the ingest service.

    Returns the most recent CDM record for the object pair including
    assembled RTN covariance and covariance_source provenance.
    """
    try:
        resp = requests.get(
            f"{INGEST_SERVICE_URL}/cdm/{primary_norad}/{secondary_norad}",
            timeout=5
        )
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except requests.exceptions.ConnectionError:
        return JSONResponse(status_code=503, content={
            "error": "Ingest service unavailable"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/ingest/inject")
async def ingest_inject(request: Request):
    """Proxy a CDM inject request to the ingest service.

    Used for demo and testing when Space-Track CDM access is unavailable.
    Injects the real TIROS 4 / IRIDIUM 33 DEB example CDM from CCSDS 508.0-B-1.
    """
    body = await request.json()
    try:
        resp = requests.post(
            f"{INGEST_SERVICE_URL}/cdm/inject",
            json=body, timeout=10
        )
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except requests.exceptions.ConnectionError:
        return JSONResponse(status_code=503, content={
            "error": "Ingest service unavailable"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/ingest/decision_log/{log_id}")
async def ingest_decision_log(log_id: str):
    """Proxy to the ingest DecisionLog retrieval endpoint (SCRUM-351).

    Passes a 404 through unchanged so the dashboard can show a clear
    not-found message rather than a generic error. Any other upstream
    failure becomes a 502 so the client can handle it cleanly.
    """
    try:
        resp = requests.get(
            f"{INGEST_SERVICE_URL}/store/decision_log/{log_id}", timeout=5
        )
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})
    if resp.status_code == 404:
        try:
            return JSONResponse(status_code=404, content=resp.json())
        except ValueError:
            return JSONResponse(status_code=404, content={"error": "not found"})
    if not resp.ok:
        return JSONResponse(status_code=502, content={
            "error": f"ingest decision_log returned {resp.status_code}"
        })
    try:
        return JSONResponse(status_code=200, content=resp.json())
    except ValueError:
        return JSONResponse(status_code=502, content={
            "error": "ingest decision_log returned invalid JSON"
        })


@app.get("/api/ingest/store")
async def ingest_store():
    """Fetch all CDM records and planner outputs from the ingest SQLite store.

    Returns both tables for display in the CDM Store viewer modal.
    """
    try:
        resp_cdm = requests.get(
            f"{INGEST_SERVICE_URL}/store/cdm_records",
            timeout=5
        )
        resp_outputs = requests.get(
            f"{INGEST_SERVICE_URL}/store/planner_outputs",
            timeout=5
        )
        cdm_data = resp_cdm.json() if resp_cdm.ok else []
        output_data = resp_outputs.json() if resp_outputs.ok else []
        return JSONResponse(content={
            "cdm_records": cdm_data,
            "planner_outputs": output_data
        })
    except requests.exceptions.ConnectionError:
        return JSONResponse(status_code=503, content={
            "error": "Ingest service unavailable"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/api/ingest/store/deduplicate")
async def ingest_deduplicate():
    """Proxy deduplicate request to ingest service."""
    try:
        resp = requests.delete(
            f"{INGEST_SERVICE_URL}/store/cdm_records/duplicates",
            timeout=10
        )
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except requests.exceptions.ConnectionError:
        return JSONResponse(status_code=503, content={"error": "Ingest service unavailable"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/api/ingest/store/clear")
async def ingest_clear():
    """Proxy clear all request to ingest service."""
    try:
        resp = requests.delete(
            f"{INGEST_SERVICE_URL}/store/cdm_records/all",
            timeout=10
        )
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except requests.exceptions.ConnectionError:
        return JSONResponse(status_code=503, content={"error": "Ingest service unavailable"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------- Detector proxy ----------

@app.post("/api/detect")
async def detect_image(request: Request):
    """Proxy an image detection request to the Detector service."""
    body = await request.json()
    try:
        resp = requests.post(DETECTOR_SERVICE_URL, json=body, timeout=30)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except requests.exceptions.ConnectionError:
        return JSONResponse(status_code=503, content={
            "error": "Detector service unavailable"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/detect/physics")
async def classify_physics(request: Request):
    """Proxy a physics spectrogram classification
    request to the physics-classifier service.

    Accepts base64-encoded 256x256 PNG spectrogram.
    Returns 5-class orbital object classification
    with confidence scores per class.

    Modality should be set by caller: EM or THERMAL.
    """
    body = await request.json()
    try:
        resp = requests.post(
            PHYSICS_CLASSIFIER_URL,
            json=body,
            timeout=10
        )
        return JSONResponse(
            status_code=resp.status_code,
            content=resp.json()
        )
    except requests.exceptions.ConnectionError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Physics classifier unavailable"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ---------- Conjunction data from propagator artifacts ----------

@app.get("/api/conjunctions")
async def get_conjunctions():
    """Read conjunction assessment data from propagator artifacts.

    Returns raw screening data. Decision logic is NOT applied here -
    the dashboard calls the planner API separately for recommendations.
    """
    if not os.path.exists(PROP_ARTIFACT_PATH):
        return JSONResponse(content={
            "status": "no_data",
            "conjunctions": [],
            "summary": {}
        })

    try:
        data = np.load(PROP_ARTIFACT_PATH, allow_pickle=True)

        obj_ids = data["obj_ids"]
        ca_table = data["ca_table"]
        pc_values = data.get("pc_values", np.zeros(len(obj_ids)))
        risk_levels = data.get("risk_levels", np.array(["NOMINAL"] * len(obj_ids)))
        tca_indices = data.get("tca_indices", np.zeros(len(obj_ids), dtype=int))
        rel_velocities = data.get("relative_velocities", np.zeros(len(obj_ids)))

        n_red = int(data.get("n_red_alerts", 0))
        n_amber = int(data.get("n_amber_alerts", 0))

        # Read screening params for dt_sec
        sp = data.get("screening_params", None)
        dt_sec = 60.0
        if sp is not None:
            try:
                sp_dict = json.loads(str(sp))
                dt_sec = float(sp_dict.get("dt_sec", 60.0))
            except Exception:
                pass

        # Read asset state at epoch if available
        r_asset = data.get("r_asset", None)
        v_asset = data.get("v_asset", None)
        r_objects = data.get("r_objects", None)
        # SCRUM-398: the propagator has always saved v_objects alongside
        # r_objects. Nothing read it, so the demo path could not supply a
        # relative velocity and every demo event reached the planner with no
        # encounter plane and therefore no Pc.
        v_objects = data.get("v_objects", None)

        conjunctions = []
        for i in range(len(obj_ids)):
            tca_idx = int(tca_indices[i])
            time_to_tca_s = tca_idx * dt_sec
            pc = float(pc_values[i])
            miss_km = float(ca_table[i])

            conj = {
                "object_id": str(obj_ids[i]),
                "miss_distance_km": round(miss_km, 4),
                "miss_distance_m": round(miss_km * 1000, 1),
                "pc": pc,
                "pc_display": f"{pc:.2e}" if pc > 0 else "< 1e-10",
                "risk_level": str(risk_levels[i]),
                "time_to_tca_s": time_to_tca_s,
                "time_to_tca_min": round(time_to_tca_s / 60, 1),
                "relative_velocity_km_s": round(float(rel_velocities[i]), 3),
                "tca_index": tca_idx,
            }

            # Include orbital positions for 3D viz if available
            if r_objects is not None and r_asset is not None:
                conj["r_obj_km"] = r_objects[i, tca_idx].tolist()
                conj["r_asset_km"] = r_asset[tca_idx].tolist()
                # SCRUM-370 Step 2: pass the real asset velocity at TCA so the
                # planner orients the surrogate covariance in the asset's true
                # orbital frame instead of against a hardcoded velocity.
                if v_asset is not None:
                    conj["v_asset_km_s"] = v_asset[tca_idx].tolist()
                    # SCRUM-398: secondary minus primary at TCA, which is the
                    # sense compute_pc_from_geometry expects. Both propagated
                    # from the same states over the same grid, so this is the
                    # relative velocity at the sample the miss distance is
                    # measured at, not an interpolation.
                    if v_objects is not None:
                        conj["v_obj_km_s"] = v_objects[i, tca_idx].tolist()
                        conj["v_rel_km_s"] = (
                            v_objects[i, tca_idx] - v_asset[tca_idx]
                        ).tolist()

            conjunctions.append(conj)

        # Sort by risk severity then Pc
        risk_order = {"RED": 0, "AMBER": 1, "GREEN": 2, "NOMINAL": 3}
        conjunctions.sort(key=lambda x: (risk_order.get(x["risk_level"], 4), -x["pc"]))

        # Build orbit tracks for 3D visualization (sampled)
        orbit_data = None
        if r_asset is not None:
            sample_step = max(1, len(r_asset) // 200)
            orbit_data = {
                "asset_track": r_asset[::sample_step].tolist(),
            }
            if r_objects is not None:
                orbit_data["object_tracks"] = {}
                for i in range(min(len(obj_ids), 8)):
                    orbit_data["object_tracks"][str(obj_ids[i])] = (
                        r_objects[i, ::sample_step].tolist()
                    )

        return JSONResponse(content={
            "status": "active",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_tracked": len(obj_ids),
                "red_alerts": n_red,
                "amber_alerts": n_amber,
                "green_alerts": int(np.sum(risk_levels == "GREEN")),
                "highest_pc": float(np.max(pc_values)) if len(pc_values) > 0 else 0,
                "closest_approach_km": float(np.min(ca_table)) if len(ca_table) > 0 else None,
            },
            "conjunctions": conjunctions,
            "orbits": orbit_data,
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error", "error": str(e), "conjunctions": []
        })


@app.get("/api/orbits")
async def get_orbits():
    """Propagate full Keplerian orbital arcs for globe visualization.

    Reads initial state vectors from states_multi.npz and propagates
    each object through one complete orbit using two-body dynamics.
    Returns ~90 points per object at ~60-second intervals covering
    one full LEO revolution (~92 minutes).

    This is independent of the short-window prop_multi.npz artifact
    and is used exclusively by the 3D globe view.
    """
    STATES_PATH = os.path.join(DATA_DIR, "states_multi.npz.processed")
    if not os.path.exists(STATES_PATH):
        STATES_PATH = os.path.join(DATA_DIR, "states_multi.npz")
    MU = 398600.4418  # km^3/s^2

    def propagate_two_body(r0, v0, n_steps=90, dt_s=None):
        # Compute exact orbital period from vis-viva so the track closes perfectly.
        # T = 2π * sqrt(a³/μ) where a = |r0| for near-circular orbit.
        a = float(np.linalg.norm(r0))
        period_s = 2 * np.pi * np.sqrt(a**3 / MU)
        if dt_s is None:
            dt_s = period_s / n_steps
        """Simple two-body RK4 propagation. Returns list of [x,y,z] km."""
        r = np.array(r0, dtype=float)
        v = np.array(v0, dtype=float)
        points = [r.tolist()]

        def accel(r):
            rmag = np.linalg.norm(r)
            return -MU / rmag**3 * r

        for _ in range(n_steps):
            # RK4
            k1v = accel(r)
            k1r = v
            k2v = accel(r + 0.5*dt_s*k1r)
            k2r = v + 0.5*dt_s*k1v
            k3v = accel(r + 0.5*dt_s*k2r)
            k3r = v + 0.5*dt_s*k2v
            k4v = accel(r + dt_s*k3r)
            k4r = v + dt_s*k3v
            r = r + (dt_s/6.0)*(k1r + 2*k2r + 2*k3r + k4r)
            v = v + (dt_s/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
            points.append(r.tolist())

        return points

    if not os.path.exists(STATES_PATH):
        return JSONResponse(content={"status": "no_data", "asset": [], "objects": {}})

    try:
        data = np.load(STATES_PATH, allow_pickle=True)

        # Asset state from metadata if available, else use first object as fallback
        asset_r = [6871.0, 0.0, 0.0]
        asset_v = [0.0, 7.6246, 0.0]
        try:
            meta = json.loads(str(data.get("metadata", "{}")))
            if "asset_state" in meta:
                asset_r = meta["asset_state"]["r_eci_km"]
                asset_v = meta["asset_state"]["v_eci_km_s"]
        except Exception:
            pass

        asset_track = propagate_two_body(asset_r, asset_v)

        obj_ids = data["object_ids"]
        r_objects = data["r_eci_km"]
        v_objects = data["v_eci_km_s"]

        # Inclinations applied here for visualization only.
        # The scenario generator uses equatorial orbits for correct
        # propagator conjunction geometry. The globe applies different
        # inclinations per object so orbit tracks cross visually.
        VIZ_INCLINATIONS = [97.8, 51.6, 28.0, 135.0, 72.0, 45.0, 63.0, 98.0]

        object_tracks = {}
        for i in range(min(len(obj_ids), 8)):
            oid = str(obj_ids[i])
            try:
                r0 = r_objects[i].tolist()
                v0 = v_objects[i].tolist()

                # Rotate velocity around the x-axis by the assigned inclination.
                # This tilts the orbital plane for visual variety while keeping
                # the orbital speed consistent with the altitude.
                inc_rad = np.radians(VIZ_INCLINATIONS[i % len(VIZ_INCLINATIONS)])
                v_mag = float(np.linalg.norm(v0))
                # Preserve speed, apply inclination rotation
                v_viz = [
                    v0[0],
                    v_mag * np.cos(inc_rad),
                    v_mag * np.sin(inc_rad),
                ]

                track = propagate_two_body(r0, v_viz)
                object_tracks[oid] = track
            except Exception:
                pass

        return JSONResponse(content={
            "status": "ok",
            "asset": asset_track,
            "objects": object_tracks,
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error", "error": str(e)
        })


# ---------- Artifacts (video, images) ----------

@app.get("/api/video")
async def get_video():
    p = os.path.join(DATA_DIR, "planner_output.mp4")
    if os.path.exists(p):
        return FileResponse(p, media_type="video/mp4",
                            headers={"Cache-Control": "no-cache"})
    return JSONResponse(status_code=404, content={"error": "not ready"})


@app.get("/api/summary-image")
async def get_summary():
    p = os.path.join(DATA_DIR, "conjunction_summary.png")
    if os.path.exists(p):
        return FileResponse(p, media_type="image/png",
                            headers={"Cache-Control": "no-cache"})
    return JSONResponse(status_code=404, content={"error": "not ready"})


# ---------- Pipeline status (simplified) ----------

@app.get("/api/pipeline/status")
async def pipeline_status():
    """Lightweight pipeline health for status bar."""
    services = {}

    # Planner
    try:
        r = requests.get(f"{PLANNER_SERVICE_URL}/health", timeout=2)
        services["planner"] = {"status": "online", "version": r.json().get("version")}
    except Exception:
        services["planner"] = {"status": "offline"}

    # Tracker
    try:
        r = requests.get(f"{TRACKER_SERVICE_URL}/status", timeout=2)
        d = r.json()
        services["tracker"] = {
            "status": "online",
            "tracks": d.get("active_tracks", 0),
            "detections": d.get("detections_processed", 0),
        }
    except Exception:
        services["tracker"] = {"status": "offline"}

    # Physics Classifier
    try:
        r = requests.get(
            PHYSICS_CLASSIFIER_HEALTH_URL,
            timeout=2
        )
        d = r.json()
        services["physics_classifier"] = {
            "status":       "online",
            "model_loaded": d.get(
                "model_loaded", False
            ),
            "version":      d.get("version"),
        }
    except Exception:
        services["physics_classifier"] = {
            "status": "offline"
        }

    # Propagator (check if artifacts exist and are recent)
    prop_exists = os.path.exists(PROP_ARTIFACT_PATH)
    services["propagator"] = {
        "status": "online" if prop_exists else "idle",
        "artifact": "prop_multi.npz" if prop_exists else None,
    }

    return JSONResponse(content={
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": services,
    })


# ---------- Demo scenario trigger ----------

@app.post("/api/scenarios/run")
async def run_scenario(request: Request):
    """Trigger a demo scenario with controlled conjunction geometry.

    Always uses the synthetic scenario generator which writes
    states_multi.npz with precise close-approach offsets per scenario.
    The tracker demo endpoint generates random debris positions that
    don't produce meaningful conjunctions, so we bypass it here.
    """
    body = await request.json()
    scenario = body.get("scenario", "mixed")

    try:
        _generate_synthetic_scenario(scenario)
        return JSONResponse(content={
            "status": "started",
            "scenario": scenario,
            "message": f"Scenario \'{scenario}\' generated",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error", "error": str(e)
        })


def _generate_synthetic_scenario(scenario):
    """Write states_multi.npz for one demo scenario.

    SCRUM-392: the preset definitions live in demo_presets, which is the single
    source the propagator regression test also imports. They used to be a
    literal here and a hand-copied literal in that test, with nothing keeping
    them in step. Geometry and confidences both come from that module now, so
    the live demo and the test describe the same scenarios and both are
    deterministic.
    """
    asset_r = list(demo_presets.ASSET_R_ECI_KM)
    asset_v = list(demo_presets.ASSET_V_ECI_KM_S)
    obj_ids, r_list, v_list, confidences, position_sigma_m = (
        demo_presets.build_scenario(scenario)
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "states_multi.npz")

    np.savez(
        out,
        object_ids=np.array(obj_ids),
        r_eci_km=np.array(r_list),
        v_eci_km_s=np.array(v_list),
        confidences=np.array(confidences),
        # SCRUM-391: per-object 1-sigma position uncertainty in metres, NaN
        # where the scenario supplies none. The propagator falls back to its
        # confidence-scaled default for those.
        position_sigma_m=np.array(position_sigma_m, dtype=float),
        t_window=np.array([demo_presets.SAMPLE_DT_S, demo_presets.N_STEPS]),
        metadata=json.dumps({
            "source": "demo_scenario",
            "scenario": scenario,
            "t0": datetime.utcnow().isoformat(),
            "asset_state": {
                "r_eci_km": asset_r,
                "v_eci_km_s": asset_v,
            },
        }),
    )
