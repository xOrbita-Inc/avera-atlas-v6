from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import json

import numpy as np

from iod import IODObservation


@dataclass
class ObservationRecord:
    observer_id: str
    target_id: str
    timestamp_utc: np.datetime64
    obs_type: str
    observer_eci_m: np.ndarray
    obs_ra_rad: float
    obs_dec_rad: float
    obs_sigma_ra_arcsec: float
    obs_sigma_dec_arcsec: float
    obs_quality: str
    observer_eci_m_s: np.ndarray | None = None
    obs_range_m: float | None = None
    obs_range_rate_m_s: float | None = None


@dataclass
class LoadedObservationBatch:
    times_utc: np.ndarray
    observer_ids: np.ndarray
    target_ids: np.ndarray
    observations: list[ObservationRecord]
    meta: dict[str, Any]


REQUIRED_FIELDS = {
    "times_utc",
    "observer_ids",
    "target_ids",
    "obs_observer_idx",
    "obs_target_idx",
    "obs_time_idx",
    "obs_type",
    "observer_eci_m",
    "obs_ra_rad",
    "obs_dec_rad",
    "obs_sigma_ra_arcsec",
    "obs_sigma_dec_arcsec",
    "obs_quality",
    "meta",
}


def _decode_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_meta(raw_meta: Any) -> dict[str, Any]:
    if isinstance(raw_meta, np.ndarray):
        if raw_meta.shape == ():
            raw_meta = raw_meta.item()
        elif raw_meta.size == 1:
            raw_meta = raw_meta.reshape(()).item()

    if isinstance(raw_meta, bytes):
        raw_meta = raw_meta.decode("utf-8")

    if isinstance(raw_meta, str):
        try:
            return json.loads(raw_meta)
        except json.JSONDecodeError:
            return {"raw_meta": raw_meta}

    if isinstance(raw_meta, dict):
        return raw_meta

    return {"raw_meta": raw_meta}


def _datetime64_to_datetime(value: np.datetime64) -> datetime:
    timestamp_ns = value.astype("datetime64[ns]").astype(np.int64)
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc)


def load_observations_multi(
    npz_path: str | Path,
    target_id: str | None = None,
) -> LoadedObservationBatch:
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"Observation artifact not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        missing = REQUIRED_FIELDS - set(data.files)
        if missing:
            raise ValueError(
                f"observations_multi.npz is missing required fields: {sorted(missing)}"
            )

        times_utc = data["times_utc"]
        observer_ids = data["observer_ids"]
        target_ids = data["target_ids"]

        obs_observer_idx = data["obs_observer_idx"]
        obs_target_idx = data["obs_target_idx"]
        obs_time_idx = data["obs_time_idx"]
        obs_type = data["obs_type"]

        observer_eci_m = data["observer_eci_m"]
        obs_ra_rad = data["obs_ra_rad"]
        obs_dec_rad = data["obs_dec_rad"]
        obs_sigma_ra_arcsec = data["obs_sigma_ra_arcsec"]
        obs_sigma_dec_arcsec = data["obs_sigma_dec_arcsec"]
        obs_quality = data["obs_quality"]
        meta = _load_meta(data["meta"])

        observer_eci_m_s = (
            data["observer_eci_m_s"] if "observer_eci_m_s" in data.files else None
        )
        obs_range_m = data["obs_range_m"] if "obs_range_m" in data.files else None
        obs_range_rate_m_s = (
            data["obs_range_rate_m_s"] if "obs_range_rate_m_s" in data.files else None
        )

        observations: list[ObservationRecord] = []

        for i in range(len(obs_time_idx)):
            current_target_id = _decode_str(target_ids[int(obs_target_idx[i])])

            if target_id is not None and current_target_id != target_id:
                continue

            current_observer_id = _decode_str(observer_ids[int(obs_observer_idx[i])])
            current_time = times_utc[int(obs_time_idx[i])]

            record = ObservationRecord(
                observer_id=current_observer_id,
                target_id=current_target_id,
                timestamp_utc=current_time,
                obs_type=_decode_str(obs_type[i]),
                observer_eci_m=np.asarray(
                    observer_eci_m[int(obs_observer_idx[i]), int(obs_time_idx[i])]
                ),
                obs_ra_rad=float(obs_ra_rad[i]),
                obs_dec_rad=float(obs_dec_rad[i]),
                obs_sigma_ra_arcsec=float(obs_sigma_ra_arcsec[i]),
                obs_sigma_dec_arcsec=float(obs_sigma_dec_arcsec[i]),
                obs_quality=_decode_str(obs_quality[i]),
                observer_eci_m_s=(
                    np.asarray(
                        observer_eci_m_s[
                            int(obs_observer_idx[i]), int(obs_time_idx[i])
                        ]
                    )
                    if observer_eci_m_s is not None
                    else None
                ),
                obs_range_m=(
                    float(obs_range_m[i])
                    if obs_range_m is not None and not np.isnan(obs_range_m[i])
                    else None
                ),
                obs_range_rate_m_s=(
                    float(obs_range_rate_m_s[i])
                    if obs_range_rate_m_s is not None
                    and not np.isnan(obs_range_rate_m_s[i])
                    else None
                ),
            )
            observations.append(record)

    observations.sort(key=lambda obs: obs.timestamp_utc)

    return LoadedObservationBatch(
        times_utc=times_utc,
        observer_ids=observer_ids,
        target_ids=target_ids,
        observations=observations,
        meta=meta,
    )


def to_iod_observations(batch: LoadedObservationBatch) -> list[IODObservation]:
    iod_observations: list[IODObservation] = []

    for obs in batch.observations:
        if obs.observer_eci_m_s is None:
            raise ValueError(
                "observer_eci_m_s is required to convert to IODObservation"
            )

        iod_obs = IODObservation(
            timestamp=_datetime64_to_datetime(obs.timestamp_utc),
            ra=obs.obs_ra_rad,
            dec=obs.obs_dec_rad,
            ra_sigma=np.deg2rad(obs.obs_sigma_ra_arcsec / 3600.0),
            dec_sigma=np.deg2rad(obs.obs_sigma_dec_arcsec / 3600.0),
            observer_position_km=np.asarray(obs.observer_eci_m, dtype=np.float64)
            / 1000.0,
            observer_velocity_km_s=np.asarray(obs.observer_eci_m_s, dtype=np.float64)
            / 1000.0,
        )
        iod_observations.append(iod_obs)

    return iod_observations