from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from services.sandbox.observation_bundle import ObservationBundle
from services.sandbox.observations import AngularObservation

SCHEMA_NAME = "observations_multi"
SCHEMA_VERSION = 1
OBS_TYPE_ANGLES = "angles"
OBS_QUALITY_NOMINAL = "nominal"
TRACKER_RANGE_SIGMA_KM = 0.05


@dataclass(frozen=True)
class EmittedObservationArtifact:
    path: Path
    observation_count: int
    observer_count: int
    target_count: int
    schema_name: str = SCHEMA_NAME
    schema_version: int = SCHEMA_VERSION

TRACKER_CONTRACT_NAME = "tracker_observation_ingest"
TRACKER_CONTRACT_VERSION = "0.1.0"


@dataclass(frozen=True)
class TrackerObservationContractArtifact:
    path: Path
    observation_count: int
    schema_name: str = TRACKER_CONTRACT_NAME
    schema_version: str = TRACKER_CONTRACT_VERSION

def _datetime64_from_seconds(t_seconds: float) -> np.datetime64:
    # Sandbox time is relative seconds from simulation start. For deterministic
    # replay artifacts, anchor the stream at a fixed UTC epoch.
    epoch = np.datetime64("2026-01-01T00:00:00.000000000", "ns")
    offset_ns = int(round(t_seconds * 1_000_000_000))
    return epoch + np.timedelta64(offset_ns, "ns")


def _arcsec_from_rad(value_rad: float) -> float:
    return float(np.rad2deg(value_rad) * 3600.0)


def _stable_unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def _observation_sort_key(obs: AngularObservation) -> tuple[float, str, str]:
    return (float(obs.t_seconds), obs.host_id, obs.debris_id)


def _meta(sensor_type: str, extra_meta: dict[str, Any] | None = None) -> str:
    meta: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "frame": "ECI",
        "time_scale": "UTC",
        "position_units": "m",
        "velocity_units": "m/s",
        "angle_units": "rad",
        "sigma_units": "arcsec",
        "range_units": "m",
        "range_rate_units": "m/s",
        "sensor_type": sensor_type,
        "obs_type_values": [OBS_TYPE_ANGLES],
        "optional_numeric_null": "NaN",
        "god_view_isolation": True,
    }

    if extra_meta:
        meta.update(extra_meta)

    return json.dumps(meta, sort_keys=True)


def detected_observations_for_schema(
    bundle: ObservationBundle,
) -> list[AngularObservation]:
    return sorted(bundle.detected_observations, key=_observation_sort_key)


def write_observations_multi_npz(
    bundle: ObservationBundle,
    output_path: str | Path,
    *,
    sensor_type: str = "optical",
    extra_meta: dict[str, Any] | None = None,
) -> EmittedObservationArtifact:
    observations = detected_observations_for_schema(bundle)

    if not observations:
        raise ValueError("Cannot emit observations_multi.npz with zero detections")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    observer_ids = _stable_unique([obs.host_id for obs in observations])
    target_ids = _stable_unique([obs.debris_id for obs in observations])
    times_utc = np.array(
        sorted({_datetime64_from_seconds(obs.t_seconds) for obs in observations}),
        dtype="datetime64[ns]",
    )

    observer_index = {observer_id: i for i, observer_id in enumerate(observer_ids)}
    target_index = {target_id: i for i, target_id in enumerate(target_ids)}
    time_index = {time: i for i, time in enumerate(times_utc)}

    obs_observer_idx = np.array(
        [observer_index[obs.host_id] for obs in observations],
        dtype=np.int64,
    )
    obs_target_idx = np.array(
        [target_index[obs.debris_id] for obs in observations],
        dtype=np.int64,
    )
    obs_time_idx = np.array(
        [time_index[_datetime64_from_seconds(obs.t_seconds)] for obs in observations],
        dtype=np.int64,
    )

    observer_eci_m = np.full(
        (len(observer_ids), len(times_utc), 3),
        np.nan,
        dtype=np.float64,
    )
    observer_eci_m_s = np.full(
        (len(observer_ids), len(times_utc), 3),
        np.nan,
        dtype=np.float64,
    )

    for obs in observations:
        oi = observer_index[obs.host_id]
        ti = time_index[_datetime64_from_seconds(obs.t_seconds)]
        observer_eci_m[oi, ti, :] = np.asarray(obs.observer_eci_m, dtype=np.float64)
        observer_eci_m_s[oi, ti, :] = np.asarray(
            obs.observer_eci_m_s,
            dtype=np.float64,
        )

    obs_ra_rad = np.array([obs.ra_rad for obs in observations], dtype=np.float64)
    obs_dec_rad = np.array([obs.dec_rad for obs in observations], dtype=np.float64)
    obs_sigma_ra_arcsec = np.array(
        [_arcsec_from_rad(obs.ra_sigma_rad) for obs in observations],
        dtype=np.float64,
    )
    obs_sigma_dec_arcsec = np.array(
        [_arcsec_from_rad(obs.dec_sigma_rad) for obs in observations],
        dtype=np.float64,
    )

    obs_range_m = np.array(
        [
            np.nan if obs.range_km is None else float(obs.range_km) * 1000.0
            for obs in observations
        ],
        dtype=np.float64,
    )
    obs_range_rate_m_s = np.full(len(observations), np.nan, dtype=np.float64)

    np.savez(
        output_path,
        times_utc=times_utc,
        observer_ids=np.array(observer_ids, dtype=str),
        target_ids=np.array(target_ids, dtype=str),
        obs_observer_idx=obs_observer_idx,
        obs_target_idx=obs_target_idx,
        obs_time_idx=obs_time_idx,
        obs_type=np.array([OBS_TYPE_ANGLES] * len(observations), dtype=str),
        observer_eci_m=observer_eci_m,
        observer_eci_m_s=observer_eci_m_s,
        obs_ra_rad=obs_ra_rad,
        obs_dec_rad=obs_dec_rad,
        obs_sigma_ra_arcsec=obs_sigma_ra_arcsec,
        obs_sigma_dec_arcsec=obs_sigma_dec_arcsec,
        obs_quality=np.array([OBS_QUALITY_NOMINAL] * len(observations), dtype=str),
        obs_range_m=obs_range_m,
        obs_range_rate_m_s=obs_range_rate_m_s,
        meta=np.array(_meta(sensor_type=sensor_type, extra_meta=extra_meta)),
    )

    return EmittedObservationArtifact(
        path=output_path,
        observation_count=len(observations),
        observer_count=len(observer_ids),
        target_count=len(target_ids),
    )

def _observation_id(obs: AngularObservation, sequence: int) -> str:
    return (
        f"obs-{obs.host_id}-{obs.debris_id}-"
        f"{int(round(float(obs.t_seconds) * 1_000_000)):012d}-{sequence:06d}"
    )


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: bool | None) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("Observer ECI vectors must contain exactly three values.")
    return [float(component) for component in array]


def observation_to_tracker_record(
    obs: AngularObservation,
    *,
    sequence: int,
    source: str = "sandbox",
) -> dict[str, Any]:
    """Convert one sandbox observation to the published tracker contract.

    This intentionally emits sensor-facing observation data only. It does not
    include truth state, future debris state, scenario placement data, or any
    other god-view fields.
    """
    return {
        "observation_id": _observation_id(obs, sequence),
        "sensor_id": obs.host_id,
        "target_id": obs.debris_id,
        "t_seconds": float(obs.t_seconds),
        "timestamp_utc": str(_datetime64_from_seconds(obs.t_seconds)).replace(
            ".000000000",
            "Z",
        ),
        "detected": bool(obs.detected),
        "reason": obs.reason,
        "ra_rad": _optional_float(obs.ra_rad),
        "dec_rad": _optional_float(obs.dec_rad),
        "ra_sigma_rad": _optional_float(obs.ra_sigma_rad),
        "dec_sigma_rad": _optional_float(obs.dec_sigma_rad),
        "ra_rate_rad_s": _optional_float(obs.ra_rate_rad_s),
        "dec_rate_rad_s": _optional_float(obs.dec_rate_rad_s),
        "range_km": _optional_float(obs.range_km),
        "range_sigma_km": (
            TRACKER_RANGE_SIGMA_KM if obs.range_km is not None else None
        ),
        "range_rate_km_s": None,
        "observer_eci_m": _optional_vector(obs.observer_eci_m),
        "observer_eci_m_s": _optional_vector(obs.observer_eci_m_s),
        "off_boresight_deg": _optional_float(obs.off_boresight_deg),
        "sunlit": _optional_bool(obs.sunlit),
        "earth_limb_blocked": _optional_bool(obs.earth_limb_blocked),
        "sensor_mode": None,
        "source": source,
    }


def bundle_to_tracker_ingest_request(
    bundle: ObservationBundle,
    *,
    source: str = "sandbox",
) -> dict[str, Any]:
    observations = detected_observations_for_schema(bundle)

    if not observations:
        raise ValueError("Cannot emit tracker observation contract with zero detections")

    return {
        "observations": [
            observation_to_tracker_record(
                obs,
                sequence=index,
                source=source,
            )
            for index, obs in enumerate(observations)
        ]
    }


def write_tracker_observations_json(
    bundle: ObservationBundle,
    output_path: str | Path,
    *,
    source: str = "sandbox",
) -> TrackerObservationContractArtifact:
    request_body = bundle_to_tracker_ingest_request(bundle, source=source)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(request_body, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return TrackerObservationContractArtifact(
        path=output_path,
        observation_count=len(request_body["observations"]),
    )