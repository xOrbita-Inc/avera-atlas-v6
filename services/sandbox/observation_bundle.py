from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .models import SimObject, SimulationResult
from .observations import AngularObservation, generate_angular_observation
from .sensor_model import SensorConfig


DEFAULT_SUN_DIR_ECI = np.array(
    [1.0, 0.0, 0.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ObservationBundle:
    """Sensor-model output generated from simulation snapshots."""

    observations: list[AngularObservation]

    @property
    def detected_observations(self) -> list[AngularObservation]:
        """Return only observations that passed every detection gate."""
        return [
            observation
            for observation in self.observations
            if observation.detected
        ]

    @property
    def rejected_observations(self) -> list[AngularObservation]:
        """Return observations rejected by one or more sensor gates."""
        return [
            observation
            for observation in self.observations
            if not observation.detected
        ]

    def reason_counts(self) -> dict[str, int]:
        """Count observation outcomes by detection reason."""
        counts: dict[str, int] = {}

        for observation in self.observations:
            counts[observation.reason] = (
                counts.get(observation.reason, 0) + 1
            )

        return counts


def _state_to_sim_object(
    base_object: SimObject,
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
) -> SimObject:
    """
    Create a SimObject at one saved simulation state.

    Physical properties and metadata are retained from the original object.
    Position and velocity arrays are copied so emitted observations do not
    share mutable state with the simulation snapshots.
    """
    return SimObject(
        object_id=base_object.object_id,
        kind=base_object.kind,
        r_eci_km=np.asarray(
            r_eci_km,
            dtype=np.float64,
        ).copy(),
        v_eci_km_s=np.asarray(
            v_eci_km_s,
            dtype=np.float64,
        ).copy(),
        physical=base_object.physical,
        metadata=base_object.metadata,
    )


def generate_observation_bundle(
    sim_result: SimulationResult,
    sensor_cfg: SensorConfig,
    seed: int = 42,
    sun_dir_eci: np.ndarray | None = None,
) -> ObservationBundle:
    """
    Apply the sensor model to every host-debris pair at every snapshot.

    The same seed, simulation result, and sensor configuration produce
    deterministic noisy observations.

    Angular rates are calculated using the previous successful detection for
    each host-target pair. Rejected observations do not replace the previous
    successful detection.
    """
    if sun_dir_eci is None:
        sun_dir_eci = DEFAULT_SUN_DIR_ECI

    normalized_sun_direction = np.asarray(
        sun_dir_eci,
        dtype=np.float64,
    )

    if normalized_sun_direction.shape != (3,):
        raise ValueError("sun_dir_eci must be a three-element vector.")

    if float(np.linalg.norm(normalized_sun_direction)) < 1e-12:
        raise ValueError("sun_dir_eci cannot be a near-zero vector.")

    rng = random.Random(seed)

    base_objects = sim_result.objects

    host_ids = sorted(
        object_id
        for object_id, sim_object in base_objects.items()
        if sim_object.kind == "host"
    )
    debris_ids = sorted(
        object_id
        for object_id, sim_object in base_objects.items()
        if sim_object.kind == "debris"
    )

    observations: list[AngularObservation] = []

    previous_detected_observations: dict[
        tuple[str, str],
        AngularObservation,
    ] = {}

    for snapshot in sim_result.snapshots:
        for host_id in host_ids:
            host_state = snapshot.states[host_id]
            host_object = _state_to_sim_object(
                base_object=base_objects[host_id],
                r_eci_km=host_state.r_eci_km,
                v_eci_km_s=host_state.v_eci_km_s,
            )

            for debris_id in debris_ids:
                debris_state = snapshot.states[debris_id]
                debris_object = _state_to_sim_object(
                    base_object=base_objects[debris_id],
                    r_eci_km=debris_state.r_eci_km,
                    v_eci_km_s=debris_state.v_eci_km_s,
                )

                pair_key = (host_id, debris_id)

                observation = generate_angular_observation(
                    host=host_object,
                    debris=debris_object,
                    t_seconds=snapshot.t_seconds,
                    sensor_cfg=sensor_cfg,
                    rng=rng,
                    previous_observation=(
                        previous_detected_observations.get(pair_key)
                    ),
                    sun_dir_eci=normalized_sun_direction,
                )

                observations.append(observation)

                if observation.detected:
                    previous_detected_observations[pair_key] = observation

    return ObservationBundle(observations=observations)