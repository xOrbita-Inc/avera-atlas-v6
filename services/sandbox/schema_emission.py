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


@dataclass(frozen=True)
class EmittedObservationArtifact:
    path: Path
    observation_count: int
    observer_count: int
    target_count: int
    schema_name: str = SCHEMA_NAME
    schema_version: int = SCHEMA_VERSION


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