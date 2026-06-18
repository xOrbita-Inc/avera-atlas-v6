from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import SimConfig
from .models import ObjectState, SimObject, SimulationResult
from .propagator import rk4_step_batch
from .snapshot import make_snapshot


def _build_state_map_from_arrays(
    object_ids: list[str],
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    t_seconds: float,
) -> dict[str, ObjectState]:
    return {
        object_id: ObjectState(
            object_id=object_id,
            t_seconds=t_seconds,
            r_eci_km=r_eci_km[idx].copy(),
            v_eci_km_s=v_eci_km_s[idx].copy(),
        )
        for idx, object_id in enumerate(object_ids)
    }


def run_simulation(
    objects: dict[str, SimObject],
    config: SimConfig,
) -> SimulationResult:
    """
    Run a deterministic fixed-step god-view simulation.

    Objects are propagated in a vectorized batch so the SCRUM-290 default
    24-hour / 500-debris / 6-host scenario can run on a standard laptop.
    """
    dt = config.integrator.dt_seconds
    duration = config.integrator.duration_seconds
    save_every_n_steps = config.integrator.save_every_n_steps

    if dt <= 0:
        raise ValueError("dt_seconds must be positive.")
    if duration < 0:
        raise ValueError("duration_seconds must be non-negative.")
    if save_every_n_steps <= 0:
        raise ValueError("save_every_n_steps must be positive.")

    object_ids = list(objects.keys())
    object_templates = [objects[object_id] for object_id in object_ids]

    r_eci_km = np.vstack(
        [obj.r_eci_km.astype(np.float64, copy=True) for obj in object_templates]
    )
    v_eci_km_s = np.vstack(
        [obj.v_eci_km_s.astype(np.float64, copy=True) for obj in object_templates]
    )

    area_m2 = np.array([obj.physical.area_m2 for obj in object_templates], dtype=np.float64)
    mass_kg = np.array([obj.physical.mass_kg for obj in object_templates], dtype=np.float64)
    cd = np.array([obj.physical.cd for obj in object_templates], dtype=np.float64)

    snapshots = []
    total_steps = int(round(duration / dt))

    snapshots.append(
        make_snapshot(
            0.0,
            _build_state_map_from_arrays(
                object_ids=object_ids,
                r_eci_km=r_eci_km,
                v_eci_km_s=v_eci_km_s,
                t_seconds=0.0,
            ),
        )
    )

    t_seconds = 0.0
    for step in range(1, total_steps + 1):
        t_seconds = step * dt

        r_eci_km, v_eci_km_s = rk4_step_batch(
            r_eci_km=r_eci_km,
            v_eci_km_s=v_eci_km_s,
            dt_s=dt,
            area_m2=area_m2,
            mass_kg=mass_kg,
            cd=cd,
            drag_cfg=config.drag,
        )

        if step % save_every_n_steps == 0 or step == total_steps:
            snapshots.append(
                make_snapshot(
                    t_seconds=t_seconds,
                    states=_build_state_map_from_arrays(
                        object_ids=object_ids,
                        r_eci_km=r_eci_km,
                        v_eci_km_s=v_eci_km_s,
                        t_seconds=t_seconds,
                    ),
                )
            )

    final_objects = {
        object_id: replace(
            obj,
            r_eci_km=r_eci_km[idx].copy(),
            v_eci_km_s=v_eci_km_s[idx].copy(),
        )
        for idx, (object_id, obj) in enumerate(zip(object_ids, object_templates))
    }

    return SimulationResult(
        objects=final_objects,
        snapshots=snapshots,
    )