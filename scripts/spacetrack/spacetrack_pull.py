"""
spacetrack_pull.py
==================
Pulls 30-day TLE history for Starlink Shell 1 satellites from Space-Track,
derives the three ConstellationSlot default values, and writes audit outputs.

This script was used to derive the ConstellationSlot defaults currently
committed in common/satellite_capability.py. The parameters below match
the exact run that produced those values.

Published run parameters (2026-03-14 to 2026-04-13):
    SHELL1_ALT_MIN  = 440 km   (empirical: Shell 1 at ~482 km, NOT 550 km per FCC)
    SHELL1_ALT_MAX  = 480 km
    SHELL1_INCL_MIN = 52.9 deg
    SHELL1_INCL_MAX = 53.2 deg
    CORRECTION_JUMP_KM = 0.5  (filters SGP4 noise; catches real corrections)
    Outlier threshold  = 20 km (excludes shell-transfer satellites)
    Window             = 30 days

Published results:
    Satellites pulled          : 2,652
    Shell-transfer excluded    : 97 (max_dev > 20 km)
    Normal-ops analyzed        : 2,555 satellites, 238,760 TLE records
    target_mean_motion_rev_per_day : 15.3020
    acceptable_drift_km            : 4.459  (P90)
    return_dv_budget_m_s           : 4.4    (~30% FCC annual / 3.39 corrections/sat/yr)

Usage:
    python3 spacetrack_pull.py

Requires:
    .env file in the same directory with:
        SPACETRACK_USER=your_email
        SPACETRACK_PASS=your_password

Output:
    - shell1_tles.json       raw TLE history (audit trail)
    - slot_defaults.json     derived ConstellationSlot values
    - Console summary        ready-to-paste satellite_capability.py update
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

MU_EARTH        = 3.986004418e5   # km³/s²
EARTH_RADIUS_KM = 6371.0

SHELL1_INCL_MIN = 52.9            # deg — Shell 1 inclination band
SHELL1_INCL_MAX = 53.2
SHELL1_ALT_MIN  = 440             # km — perigee floor
SHELL1_ALT_MAX  = 480             # km — apogee ceiling
WINDOW_DAYS     = 30
CORRECTION_JUMP_KM = 0.5         # SMA jump threshold for maneuver detection

BASE_URL   = "https://www.space-track.org"
LOGIN_URL  = f"{BASE_URL}/ajaxauth/login"
LOGOUT_URL = f"{BASE_URL}/ajaxauth/logout"
GP_HISTORY = f"{BASE_URL}/basicspacedata/query/class/gp_history"

# ── Auth ──────────────────────────────────────────────────────────────────────

def login(session: requests.Session, user: str, password: str) -> None:
    resp = session.post(LOGIN_URL, data={"identity": user, "password": password})
    resp.raise_for_status()
    if "Login" in resp.text and "Failed" in resp.text:
        raise RuntimeError("Space-Track login failed — check credentials in .env")
    print("Logged in to Space-Track.")


def logout(session: requests.Session) -> None:
    session.get(LOGOUT_URL)
    print("Logged out.")


# ── TLE Pull ──────────────────────────────────────────────────────────────────

def pull_shell1_norad_ids(session: requests.Session) -> list[str]:
    """
    Query gp class for active Starlink Shell 1 satellites.
    Filters: OBJECT_NAME LIKE STARLINK%, DECAY_DATE null, inclination 52.9-53.2 deg.
    """
    url = (
        f"{BASE_URL}/basicspacedata/query/class/gp"
        f"/OBJECT_NAME/~~STARLINK%25"
        f"/INCLINATION/{SHELL1_INCL_MIN}--{SHELL1_INCL_MAX}"
        f"/DECAY_DATE/null-val"
        f"/orderby/NORAD_CAT_ID asc"
        f"/format/json"
        f"/emptyresult/show"
    )
    print("Fetching active Shell 1 NORAD IDs...")
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()

    # Filter further by altitude
    ids = []
    for sat in data:
        try:
            apogee  = float(sat.get("APOAPSIS",  0))
            perigee = float(sat.get("PERIAPSIS", 0))
            if SHELL1_ALT_MIN <= perigee and apogee <= SHELL1_ALT_MAX:
                ids.append(str(sat["NORAD_CAT_ID"]))
        except (ValueError, KeyError):
            continue

    print(f"Found {len(ids)} active Shell 1 satellites.")
    return ids


def pull_tle_history(
    session: requests.Session,
    norad_ids: list[str],
    days: int = WINDOW_DAYS,
    batch_size: int = 50,
) -> list[dict]:
    """
    Pull gp_history for each NORAD ID over the last `days` days.
    Batches requests to avoid hitting rate limits.
    """
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    epoch_range = (
        f"{start_dt.strftime('%Y-%m-%d')}--{end_dt.strftime('%Y-%m-%d')}"
    )

    all_records = []
    total_batches = (len(norad_ids) + batch_size - 1) // batch_size

    for i in range(0, len(norad_ids), batch_size):
        batch = norad_ids[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Pulling TLE history batch {batch_num}/{total_batches} "
              f"({len(batch)} satellites)...")

        id_str = ",".join(batch)
        url = (
            f"{GP_HISTORY}"
            f"/NORAD_CAT_ID/{id_str}"
            f"/EPOCH/{epoch_range}"
            f"/orderby/NORAD_CAT_ID asc,EPOCH asc"
            f"/format/json"
            f"/emptyresult/show"
        )
        resp = session.get(url)
        resp.raise_for_status()
        records = resp.json()
        all_records.extend(records)
        time.sleep(0.3)   # be polite to the API

    print(f"Total TLE records retrieved: {len(all_records)}")
    return all_records


# ── Derivation ────────────────────────────────────────────────────────────────

def mean_motion_to_sma(mm_rev_per_day: float) -> float:
    """Convert TLE mean motion [rev/day] to semi-major axis [km]."""
    n = mm_rev_per_day * 2 * np.pi / 86400.0
    return (MU_EARTH / n**2) ** (1.0 / 3.0)


def derive_constellation_slot_defaults(records: list[dict]) -> dict:
    """
    Derive the three ConstellationSlot placeholder values from TLE history.

    target_mean_motion_rev_per_day:
        Median mean motion across all records in the window.

    acceptable_drift_km:
        P90 of per-satellite maximum SMA deviation from personal baseline,
        excluding outliers with max_dev > 20 km (shell transfers, debris event).
        Uses raw deviation directly — no correction detection needed.
        This cleanly separates normal station-keeping from relocating satellites.

    return_dv_budget_m_s:
        FCC-estimated annual delta-v (50 m/s) divided by annualized
        correction frequency per satellite.
    """

    by_sat: dict[str, list[dict]] = {}
    for rec in records:
        nid = str(rec.get("NORAD_CAT_ID", ""))
        if nid:
            by_sat.setdefault(nid, []).append(rec)

    # ── target_mean_motion ────────────────────────────────────────────────────
    all_mm = []
    for recs in by_sat.values():
        for r in recs:
            try:
                all_mm.append(float(r["MEAN_MOTION"]))
            except (KeyError, ValueError):
                pass

    if not all_mm:
        raise RuntimeError("No MEAN_MOTION values found in TLE records.")

    median_mm = float(np.median(all_mm))
    shell_sma = mean_motion_to_sma(median_mm)
    print(f"\nShell median mean motion : {median_mm:.4f} rev/day")
    print(f"Shell median SMA         : {shell_sma:.2f} km  "
          f"({shell_sma - EARTH_RADIUS_KM:.1f} km altitude)")

    # ── acceptable_drift_km ───────────────────────────────────────────────────
    raw_max_devs         = []
    station_keeping_devs = []
    correction_events    = 0
    skipped              = 0

    for nid, recs in by_sat.items():
        recs_sorted = sorted(recs, key=lambda r: r.get("EPOCH", ""))
        smas = []
        for r in recs_sorted:
            try:
                smas.append(mean_motion_to_sma(float(r["MEAN_MOTION"])))
            except (KeyError, ValueError):
                pass

        if len(smas) < 5:
            skipped += 1
            continue

        smas = np.array(smas)
        personal_baseline = float(np.median(smas))
        max_dev = float(np.max(np.abs(smas - personal_baseline)))
        raw_max_devs.append(max_dev)

        # Station-keeping only: exclude shell transfers and debris outliers
        if max_dev <= 20.0:
            station_keeping_devs.append(max_dev)

        # Count corrections for return_dv calculation
        for j in range(1, len(smas)):
            delta = smas[j] - smas[j - 1]
            moving_toward = (
                (smas[j - 1] > personal_baseline and delta < 0) or
                (smas[j - 1] < personal_baseline and delta > 0)
            )
            if moving_toward and abs(delta) > CORRECTION_JUMP_KM:
                correction_events += 1

    n_sats     = len(by_sat) - skipped
    n_outliers = len(raw_max_devs) - len(station_keeping_devs)

    print(f"Satellites analyzed      : {n_sats}")
    print(f"Satellites skipped (<5 TLEs): {skipped}")
    print(f"Outliers excluded (max_dev > 20 km): {n_outliers}")
    print(f"Satellites in station-keeping distribution: {len(station_keeping_devs)}")

    if not station_keeping_devs:
        raise RuntimeError("No station-keeping satellites after filtering.")

    devs = np.array(station_keeping_devs)
    acceptable_drift_km = float(np.percentile(devs, 90))
    print(f"Max deviation P50        : {np.percentile(devs, 50):.3f} km")
    print(f"Max deviation P75        : {np.percentile(devs, 75):.3f} km")
    print(f"Max deviation P90        : {np.percentile(devs, 90):.3f} km  <- acceptable_drift_km")

    # ── return_dv_budget_m_s ──────────────────────────────────────────────────
    fcc_annual_dv_ms             = 50.0
    annual_corrections           = correction_events * (365.0 / WINDOW_DAYS)
    corrections_per_sat_per_year = annual_corrections / max(1, n_sats)
    return_dv_budget             = fcc_annual_dv_ms / max(1.0, corrections_per_sat_per_year)
    print(f"Correction events detected: {correction_events}")
    print(f"Corrections per sat/year : {corrections_per_sat_per_year:.2f}")
    print(f"return_dv_budget_m_s     : {return_dv_budget:.4f} m/s")

    return {
        "source": "Space-Track TLE history",
        "epoch_start": (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d"),
        "epoch_end": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "n_satellites_analyzed": n_sats,
        "n_outliers_excluded": n_outliers,
        "n_tle_records": len(records),
        "n_correction_events": correction_events,
        "corrections_per_sat_per_year": round(corrections_per_sat_per_year, 2),
        "target_mean_motion_rev_per_day": round(median_mm, 4),
        "shell_sma_km": round(shell_sma, 2),
        "shell_altitude_km": round(shell_sma - EARTH_RADIUS_KM, 1),
        "acceptable_drift_km": round(acceptable_drift_km, 3),
        "return_dv_budget_m_s": round(return_dv_budget, 4),
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_satellite_capability_update(defaults: dict) -> None:
    print("\n" + "=" * 70)
    print("READY TO PASTE INTO common/satellite_capability.py")
    print("Replace the three PLACEHOLDER lines in ConstellationSlot with:")
    print("=" * 70)
    print(f"""
@dataclass(frozen=True)
class ConstellationSlot:
    \"\"\"
    Defaults below are Starlink Shell 1 empirical values derived from
    Space-Track TLE history {defaults['epoch_start']} to {defaults['epoch_end']}
    ({defaults['n_satellites_analyzed']} active satellites, 
    {defaults['n_tle_records']} TLE records,
    {defaults['n_correction_events']} correction events detected).
    \"\"\"
    in_constellation: bool                = False
    slot_id: str                          = ""
    target_mean_motion_rev_per_day: float = {defaults['target_mean_motion_rev_per_day']}
    acceptable_drift_km: float            = {defaults['acceptable_drift_km']}
    return_dv_budget_m_s: float           = {defaults['return_dv_budget_m_s']}
    max_recovery_time_s: float            = 86400.0
""")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load credentials
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
    user     = os.getenv("SPACETRACK_USER")
    password = os.getenv("SPACETRACK_PASS")

    if not user or not password:
        print("ERROR: SPACETRACK_USER and SPACETRACK_PASS must be set in .env")
        sys.exit(1)

    out_dir = Path(__file__).parent
    tles_path     = out_dir / "shell1_tles.json"
    defaults_path = out_dir / "slot_defaults.json"

    with requests.Session() as session:
        try:
            login(session, user, password)

            # Step 1: Get active Shell 1 NORAD IDs
            norad_ids = pull_shell1_norad_ids(session)
            if not norad_ids:
                print("No Shell 1 satellites found. Check inclination filter.")
                sys.exit(1)

            # Step 2: Pull TLE history
            print(f"\nPulling {WINDOW_DAYS}-day TLE history for {len(norad_ids)} satellites...")
            records = pull_tle_history(session, norad_ids, days=WINDOW_DAYS)

            # Save raw data
            with open(tles_path, "w") as f:
                json.dump(records, f, indent=2)
            print(f"\nRaw TLE history saved to: {tles_path}")

        finally:
            logout(session)

    # Step 3: Derive ConstellationSlot defaults
    print("\nDeriving ConstellationSlot defaults...")
    defaults = derive_constellation_slot_defaults(records)

    # Save derived values
    with open(defaults_path, "w") as f:
        json.dump(defaults, f, indent=2)
    print(f"Slot defaults saved to: {defaults_path}")

    # Print update
    print_satellite_capability_update(defaults)


if __name__ == "__main__":
    main()