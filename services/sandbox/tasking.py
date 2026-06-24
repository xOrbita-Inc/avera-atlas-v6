from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .models import SimObject, SimulationResult
from .observation_bundle import ObservationBundle, _state_to_sim_object
from .observations import AngularObservation, make_noisy_ra_dec
from .sensor_model import (
    DEFAULT_SUN_DIR_ECI,
    DetectionResult,
    SensorConfig,
    detect_object_with_boresight,
    unit,
)


TaskTargetType = Literal[
    "predicted_point",
    "admissible_region",
]

TaskStatus = Literal[
    "accepted",
    "rejected",
]

@dataclass(frozen=True)
class TaskingConstraints:
    """Sandbox-side host constraints for task execution."""

    blackout_windows: tuple[tuple[float, float], ...] = ()

@dataclass(frozen=True)
class TaskingTarget:
    """Target specification for a sandbox re-observation task."""

    type: TaskTargetType
    params: dict[str, Any]


@dataclass(frozen=True)
class TaskingCommand:
    """Minimum viable sandbox tasking command."""

    task_id: str
    sensor_id: str
    start_time: float
    end_time: float
    target: TaskingTarget
    priority: int = 0


@dataclass(frozen=True)
class TaskingResult:
    """Result of attempting to execute one tasking command."""

    task_id: str
    sensor_id: str
    status: TaskStatus
    reason: str
    observations: list[AngularObservation]

    @property
    def detected_observations(self) -> list[AngularObservation]:
        return [
            observation
            for observation in self.observations
            if observation.detected
        ]


def predicted_point_target(
    position_eci_km: np.ndarray | list[float] | tuple[float, float, float],
) -> TaskingTarget:
    """Build a predicted-point task target from an ECI position."""
    return TaskingTarget(
        type="predicted_point",
        params={
            "position_eci_km": list(
                np.asarray(position_eci_km, dtype=np.float64)
            ),
        },
    )


def _validate_command(command: TaskingCommand) -> None:
    if not command.task_id:
        raise ValueError("task_id must be non-empty.")

    if not command.sensor_id:
        raise ValueError("sensor_id must be non-empty.")

    if command.end_time < command.start_time:
        raise ValueError("end_time must be greater than or equal to start_time.")

    if command.target.type not in ("predicted_point", "admissible_region"):
        raise ValueError(f"Unsupported target type: {command.target.type}")


def _snapshot_in_window(
    snapshot_time: float,
    command: TaskingCommand,
) -> bool:
    return command.start_time <= snapshot_time <= command.end_time


def _commanded_boresight_for_target(
    host: SimObject,
    target: TaskingTarget,
) -> np.ndarray:
    if target.type == "admissible_region":
        raise ValueError(
            "admissible_region task execution is not implemented in the "
            "single-sensor MVP."
        )

    if target.type != "predicted_point":
        raise ValueError(f"Unsupported target type: {target.type}")

    if "position_eci_km" not in target.params:
        raise ValueError("predicted_point target requires position_eci_km.")

    target_position = np.asarray(
        target.params["position_eci_km"],
        dtype=np.float64,
    )

    if target_position.shape != (3,):
        raise ValueError("position_eci_km must contain exactly three values.")

    return unit(target_position - host.r_eci_km)


def _observation_from_tasked_detection(
    host: SimObject,
    debris: SimObject,
    t_seconds: float,
    detection: DetectionResult,
    sensor_cfg: SensorConfig,
    rng: random.Random,
    previous_observation: AngularObservation | None = None,
) -> AngularObservation:
    observer_eci_m = host.r_eci_km.astype(np.float64, copy=True) * 1000.0
    observer_eci_m_s = host.v_eci_km_s.astype(np.float64, copy=True) * 1000.0

    if not detection.detected:
        return AngularObservation(
            host_id=host.object_id,
            debris_id=debris.object_id,
            t_seconds=t_seconds,
            detected=False,
            reason=f"tasked_{detection.reason}",
            ra_rad=None,
            dec_rad=None,
            ra_sigma_rad=None,
            dec_sigma_rad=None,
            ra_rate_rad_s=None,
            dec_rate_rad_s=None,
            range_km=detection.range_km,
            off_boresight_deg=detection.off_boresight_deg,
            sunlit=detection.sunlit,
            earth_limb_blocked=detection.earth_limb_blocked,
            observer_eci_m=observer_eci_m,
            observer_eci_m_s=observer_eci_m_s,
        )

    relative_position_km = debris.r_eci_km - host.r_eci_km
    relative_distance_km = float(np.linalg.norm(relative_position_km))

    if relative_distance_km < 1e-12:
        raise ValueError(
            "Detected host and debris states cannot be co-located."
        )

    los_eci = relative_position_km / relative_distance_km

    (
        ra_rad,
        dec_rad,
        ra_sigma_rad,
        dec_sigma_rad,
    ) = make_noisy_ra_dec(
        los_eci=los_eci,
        angular_sigma_arcsec=sensor_cfg.angular_resolution_arcsec,
        rng=rng,
    )

    observation = AngularObservation(
        host_id=host.object_id,
        debris_id=debris.object_id,
        t_seconds=t_seconds,
        detected=True,
        reason="tasked_detected",
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        ra_sigma_rad=ra_sigma_rad,
        dec_sigma_rad=dec_sigma_rad,
        ra_rate_rad_s=None,
        dec_rate_rad_s=None,
        range_km=detection.range_km,
        off_boresight_deg=detection.off_boresight_deg,
        sunlit=detection.sunlit,
        earth_limb_blocked=detection.earth_limb_blocked,
        observer_eci_m=observer_eci_m,
        observer_eci_m_s=observer_eci_m_s,
    )

    if previous_observation is None:
        return observation

    from .observations import estimate_angular_rates
    from dataclasses import replace

    ra_rate_rad_s, dec_rate_rad_s = estimate_angular_rates(
        previous_observation=previous_observation,
        current_observation=observation,
    )

    return replace(
        observation,
        ra_rate_rad_s=ra_rate_rad_s,
        dec_rate_rad_s=dec_rate_rad_s,
    )


def execute_tasking_command(
    sim_result: SimulationResult,
    command: TaskingCommand,
    sensor_cfg: SensorConfig,
    seed: int = 42,
    sun_dir_eci: np.ndarray | None = None,
    constraints: TaskingConstraints | None = None,
) -> TaskingResult:
    """
    Execute a re-observation task inside the sandbox.

    The sandbox repoints the named host sensor during the commanded time
    window, evaluates the normal sensor detection gates using the commanded
    boresight, and returns observations in the same AngularObservation form
    used by survey-mode observations.
    """
    _validate_command(command)

    if constraints is None:
        constraints = TaskingConstraints()

    constraint_violation = _violates_constraints(
        command=command,
        constraints=constraints,
    )

    if constraint_violation is not None:
        return TaskingResult(
            task_id=command.task_id,
            sensor_id=command.sensor_id,
            status="rejected",
            reason=constraint_violation,
            observations=[],
        )

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

    if command.sensor_id not in sim_result.objects:
        return TaskingResult(
            task_id=command.task_id,
            sensor_id=command.sensor_id,
            status="rejected",
            reason="unknown_sensor",
            observations=[],
        )

    sensor_object = sim_result.objects[command.sensor_id]

    if sensor_object.kind != "host":
        return TaskingResult(
            task_id=command.task_id,
            sensor_id=command.sensor_id,
            status="rejected",
            reason="sensor_is_not_host",
            observations=[],
        )

    if command.target.type == "admissible_region":
        return TaskingResult(
            task_id=command.task_id,
            sensor_id=command.sensor_id,
            status="rejected",
            reason="admissible_region_not_implemented",
            observations=[],
        )

    rng = random.Random(seed)
    observations: list[AngularObservation] = []
    previous_detected_observations: dict[str, AngularObservation] = {}

    base_objects = sim_result.objects
    debris_ids = sorted(
        object_id
        for object_id, sim_object in base_objects.items()
        if sim_object.kind == "debris"
    )

    matching_snapshots = [
        snapshot
        for snapshot in sim_result.snapshots
        if _snapshot_in_window(snapshot.t_seconds, command)
    ]

    if not matching_snapshots:
        return TaskingResult(
            task_id=command.task_id,
            sensor_id=command.sensor_id,
            status="rejected",
            reason="no_snapshots_in_task_window",
            observations=[],
        )

    for snapshot in matching_snapshots:
        if command.sensor_id not in snapshot.states:
            return TaskingResult(
                task_id=command.task_id,
                sensor_id=command.sensor_id,
                status="rejected",
                reason="sensor_state_missing",
                observations=[],
            )

        host_state = snapshot.states[command.sensor_id]
        host_object = _state_to_sim_object(
            base_object=base_objects[command.sensor_id],
            r_eci_km=host_state.r_eci_km,
            v_eci_km_s=host_state.v_eci_km_s,
        )

        try:
            commanded_boresight = _commanded_boresight_for_target(
                host=host_object,
                target=command.target,
            )
        except ValueError as exc:
            return TaskingResult(
                task_id=command.task_id,
                sensor_id=command.sensor_id,
                status="rejected",
                reason=str(exc),
                observations=[],
            )

        for debris_id in debris_ids:
            if debris_id not in snapshot.states:
                continue

            debris_state = snapshot.states[debris_id]
            debris_object = _state_to_sim_object(
                base_object=base_objects[debris_id],
                r_eci_km=debris_state.r_eci_km,
                v_eci_km_s=debris_state.v_eci_km_s,
            )

            detection = detect_object_with_boresight(
                host=host_object,
                debris=debris_object,
                sensor_cfg=sensor_cfg,
                boresight_eci=commanded_boresight,
                sun_dir_eci=normalized_sun_direction,
            )

            previous_observation = previous_detected_observations.get(
                debris_id
            )

            observation = _observation_from_tasked_detection(
                host=host_object,
                debris=debris_object,
                t_seconds=snapshot.t_seconds,
                detection=detection,
                sensor_cfg=sensor_cfg,
                rng=rng,
                previous_observation=previous_observation,
            )

            observations.append(observation)

            if observation.detected:
                previous_detected_observations[debris_id] = observation

    return TaskingResult(
        task_id=command.task_id,
        sensor_id=command.sensor_id,
        status="accepted",
        reason="executed",
        observations=observations,
    )


def tasking_result_to_bundle(result: TaskingResult) -> ObservationBundle:
    """Expose tasked observations through the standard sandbox bundle type."""
    return ObservationBundle(observations=result.observations)

def _windows_overlap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> bool:
    return first_start <= second_end and second_start <= first_end


def _violates_constraints(
    command: TaskingCommand,
    constraints: TaskingConstraints,
) -> str | None:
    for blackout_start, blackout_end in constraints.blackout_windows:
        if _windows_overlap(
            command.start_time,
            command.end_time,
            blackout_start,
            blackout_end,
        ):
            return "blackout_window"

    return None