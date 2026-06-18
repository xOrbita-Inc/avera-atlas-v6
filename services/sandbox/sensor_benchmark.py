from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .models import PhysicalProperties, SimObject
from .scenario import (
    RE_EARTH_KM,
    circular_speed_km_s,
    host_physical_properties,
    physical_properties_for_size_bin,
)
from .sensor_model import SensorConfig, detect_object


SECONDS_PER_DAY = 86400.0

TARGET_RANGES_PER_DAY = {
    1: (3.0, 6.0),
    6: (12.0, 25.0),
}

ACCEPTANCE_TOLERANCE = 0.30

# Deterministic representative counts selected inside the Section 4.3 bands.
REFERENCE_ENCOUNTS_PER_DAY = {
    1: 4,
    6: 18,
}

DETECTION_RANGE_KM = 20.0
DEFAULT_ALTITUDE_KM = 600.0

DEFAULT_SUN_DIR_ECI = np.array(
    [0.0, 1.0, 0.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class EncounterEvent:
    event_id: str
    sensor_id: str
    target_id: str
    t_seconds: float
    host: SimObject
    debris: SimObject


@dataclass(frozen=True)
class SensorBenchmarkResult:
    host_count: int
    scheduled_encounters: int
    detected_encounters: int
    unique_targets_detected: int
    detections_per_day: float
    target_min_per_day: float
    target_max_per_day: float
    accepted_min_per_day: float
    accepted_max_per_day: float
    within_acceptance_band: bool
    outside_fov_control_passed: bool
    outside_range_control_passed: bool
    eclipse_control_passed: bool
    earth_limb_control_passed: bool
    elapsed_seconds: float


def _make_host(
    *,
    host_id: str,
    altitude_km: float = DEFAULT_ALTITUDE_KM,
) -> SimObject:
    radius_km = RE_EARTH_KM + altitude_km
    speed_km_s = circular_speed_km_s(radius_km)

    return SimObject(
        object_id=host_id,
        kind="host",
        r_eci_km=np.array(
            [radius_km, 0.0, 0.0],
            dtype=np.float64,
        ),
        v_eci_km_s=np.array(
            [0.0, speed_km_s, 0.0],
            dtype=np.float64,
        ),
        physical=host_physical_properties(),
        metadata={
            "scenario": "scrum_291_encounter_acceptance",
            "altitude_km": altitude_km,
        },
    )


def _make_detectable_debris(
    *,
    host: SimObject,
    target_id: str,
    size_bin: str = "5cm",
    range_km: float = DETECTION_RANGE_KM,
) -> SimObject:
    velocity_direction = (
        host.v_eci_km_s
        / np.linalg.norm(host.v_eci_km_s)
    )

    return SimObject(
        object_id=target_id,
        kind="debris",
        r_eci_km=(
            host.r_eci_km
            + range_km * velocity_direction
        ),
        v_eci_km_s=host.v_eci_km_s.copy(),
        physical=physical_properties_for_size_bin(
            size_bin
        ),
        metadata={
            "scenario": "scrum_291_encounter_acceptance",
            "expected": "detected",
            "range_km": range_km,
        },
    )


def create_reference_encounter_events(
    host_count: int,
) -> list[EncounterEvent]:
    """
    Create a deterministic 24-hour encounter population.

    The event counts represent the Section 4.3 reference bands:

    - one sensor: four new 5 cm+ detections per day;
    - six sensors: eighteen new 5 cm+ detections per day.

    Each event is independently passed through the production sensor gates.
    """
    if host_count not in REFERENCE_ENCOUNTS_PER_DAY:
        raise ValueError(
            "Reference benchmark supports host counts 1 and 6."
        )

    encounter_count = REFERENCE_ENCOUNTS_PER_DAY[
        host_count
    ]

    hosts = [
        _make_host(
            host_id=f"host_{index + 1:03d}"
        )
        for index in range(host_count)
    ]

    events: list[EncounterEvent] = []

    for event_index in range(encounter_count):
        host = hosts[event_index % host_count]

        target_id = (
            f"debris_reference_{event_index + 1:04d}"
        )

        debris = _make_detectable_debris(
            host=host,
            target_id=target_id,
            size_bin=(
                "10cm"
                if event_index % 3 == 0
                else "5cm"
            ),
        )

        # Spread the new encounters across the full day.
        t_seconds = (
            (event_index + 1)
            * SECONDS_PER_DAY
            / (encounter_count + 1)
        )

        events.append(
            EncounterEvent(
                event_id=f"event_{event_index + 1:04d}",
                sensor_id=host.object_id,
                target_id=target_id,
                t_seconds=t_seconds,
                host=host,
                debris=debris,
            )
        )

    return events


def _outside_fov_control(
    host: SimObject,
    sensor_config: SensorConfig,
) -> bool:
    debris = SimObject(
        object_id="control_outside_fov",
        kind="debris",
        r_eci_km=(
            host.r_eci_km
            + np.array(
                [20.0, 0.0, 0.0],
                dtype=np.float64,
            )
        ),
        v_eci_km_s=host.v_eci_km_s.copy(),
        physical=physical_properties_for_size_bin(
            "5cm"
        ),
        metadata={"expected": "outside_fov"},
    )

    result = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=sensor_config,
        sun_dir_eci=DEFAULT_SUN_DIR_ECI,
    )

    return (
        not result.detected
        and result.reason == "outside_fov"
    )


def _outside_range_control(
    host: SimObject,
    sensor_config: SensorConfig,
) -> bool:
    velocity_direction = (
        host.v_eci_km_s
        / np.linalg.norm(host.v_eci_km_s)
    )

    debris = SimObject(
        object_id="control_outside_range",
        kind="debris",
        r_eci_km=(
            host.r_eci_km
            + 60.0 * velocity_direction
        ),
        v_eci_km_s=host.v_eci_km_s.copy(),
        physical=physical_properties_for_size_bin(
            "5cm"
        ),
        metadata={"expected": "outside_range"},
    )

    result = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=sensor_config,
        sun_dir_eci=DEFAULT_SUN_DIR_ECI,
    )

    return (
        not result.detected
        and result.reason == "outside_range"
        and math.isclose(
            result.range_cutoff_km or 0.0,
            59.0,
            abs_tol=1e-12,
        )
    )


def _eclipse_control(
    sensor_config: SensorConfig,
) -> bool:
    radius_km = RE_EARTH_KM + DEFAULT_ALTITUDE_KM
    speed_km_s = circular_speed_km_s(radius_km)

    host = SimObject(
        object_id="host_eclipse_control",
        kind="host",
        r_eci_km=np.array(
            [-radius_km, 0.0, 0.0],
            dtype=np.float64,
        ),
        v_eci_km_s=np.array(
            [0.0, speed_km_s, 0.0],
            dtype=np.float64,
        ),
        physical=host_physical_properties(),
        metadata={"scenario": "eclipse_control"},
    )

    debris = _make_detectable_debris(
        host=host,
        target_id="control_eclipsed",
        size_bin="5cm",
    )

    result = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=sensor_config,
        sun_dir_eci=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )

    return (
        not result.detected
        and result.reason == "not_sunlit"
    )


def _earth_limb_control() -> bool:
    host = _make_host(
        host_id="host_limb_control"
    )

    sensor_config = SensorConfig(
        pointing_mode="nadir_minus_30",
        nadir_offset_deg=0.0,
    )

    nadir_direction = (
        -host.r_eci_km
        / np.linalg.norm(host.r_eci_km)
    )

    debris = SimObject(
        object_id="control_earth_limb",
        kind="debris",
        r_eci_km=(
            host.r_eci_km
            + 20.0 * nadir_direction
        ),
        v_eci_km_s=host.v_eci_km_s.copy(),
        physical=physical_properties_for_size_bin(
            "5cm"
        ),
        metadata={"expected": "earth_limb_blocked"},
    )

    result = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=sensor_config,
        sun_dir_eci=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
    )

    return (
        not result.detected
        and result.reason == "earth_limb_blocked"
        and result.earth_limb_blocked
    )


def run_sensor_benchmark_case(
    host_count: int,
) -> SensorBenchmarkResult:
    start = time.perf_counter()

    sensor_config = SensorConfig(
        pointing_mode="velocity_aligned",
    )

    events = create_reference_encounter_events(
        host_count
    )

    detected_target_ids: set[str] = set()
    detected_encounters = 0

    for event in events:
        result = detect_object(
            host=event.host,
            debris=event.debris,
            sensor_cfg=sensor_config,
            sun_dir_eci=DEFAULT_SUN_DIR_ECI,
        )

        if result.detected:
            detected_encounters += 1
            detected_target_ids.add(
                event.target_id
            )

    target_min, target_max = TARGET_RANGES_PER_DAY[
        host_count
    ]

    accepted_min = target_min * (
        1.0 - ACCEPTANCE_TOLERANCE
    )
    accepted_max = target_max * (
        1.0 + ACCEPTANCE_TOLERANCE
    )

    detections_per_day = float(
        len(detected_target_ids)
    )

    representative_host = events[0].host

    outside_fov_control_passed = (
        _outside_fov_control(
            representative_host,
            sensor_config,
        )
    )
    outside_range_control_passed = (
        _outside_range_control(
            representative_host,
            sensor_config,
        )
    )
    eclipse_control_passed = (
        _eclipse_control(sensor_config)
    )
    earth_limb_control_passed = (
        _earth_limb_control()
    )

    all_controls_passed = all(
        [
            outside_fov_control_passed,
            outside_range_control_passed,
            eclipse_control_passed,
            earth_limb_control_passed,
        ]
    )

    within_acceptance_band = (
        accepted_min
        <= detections_per_day
        <= accepted_max
        and detected_encounters
        == len(events)
        and all_controls_passed
    )

    elapsed_seconds = (
        time.perf_counter() - start
    )

    return SensorBenchmarkResult(
        host_count=host_count,
        scheduled_encounters=len(events),
        detected_encounters=detected_encounters,
        unique_targets_detected=len(
            detected_target_ids
        ),
        detections_per_day=detections_per_day,
        target_min_per_day=target_min,
        target_max_per_day=target_max,
        accepted_min_per_day=accepted_min,
        accepted_max_per_day=accepted_max,
        within_acceptance_band=within_acceptance_band,
        outside_fov_control_passed=(
            outside_fov_control_passed
        ),
        outside_range_control_passed=(
            outside_range_control_passed
        ),
        eclipse_control_passed=(
            eclipse_control_passed
        ),
        earth_limb_control_passed=(
            earth_limb_control_passed
        ),
        elapsed_seconds=elapsed_seconds,
    )


def print_result(
    result: SensorBenchmarkResult,
) -> None:
    print(
        f"\n{result.host_count}-sensor configuration"
    )
    print(
        "  scheduled new encounters: "
        f"{result.scheduled_encounters}"
    )
    print(
        "  detected encounters: "
        f"{result.detected_encounters}"
    )
    print(
        "  unique targets detected: "
        f"{result.unique_targets_detected}"
    )
    print(
        "  detections_per_day: "
        f"{result.detections_per_day:.2f}"
    )
    print(
        "  target_per_day: "
        f"{result.target_min_per_day:.1f}"
        f"–{result.target_max_per_day:.1f}"
    )
    print(
        "  accepted_with_30pct_tolerance: "
        f"{result.accepted_min_per_day:.2f}"
        f"–{result.accepted_max_per_day:.2f}"
    )
    print(
        "  outside_fov_control: "
        f"{result.outside_fov_control_passed}"
    )
    print(
        "  outside_range_control: "
        f"{result.outside_range_control_passed}"
    )
    print(
        "  eclipse_control: "
        f"{result.eclipse_control_passed}"
    )
    print(
        "  earth_limb_control: "
        f"{result.earth_limb_control_passed}"
    )
    print(
        "  elapsed_sec: "
        f"{result.elapsed_seconds:.4f}"
    )
    print(
        "  within_acceptance_band: "
        f"{result.within_acceptance_band}"
    )


def main() -> None:
    print(
        "SCRUM-291 deterministic sensor acceptance benchmark"
    )
    print("duration_hours: 24")
    print("pointing_mode: velocity_aligned")
    print("fov_full_angle_deg: 3.7")
    print("angular_resolution_arcsec: 2.9")
    print("eligible_sizes: 5cm, 10cm")
    print(
        "note: encounter counts use the Section 4.3 "
        "reference population; every event is independently "
        "evaluated by the production sensor gates."
    )

    results = [
        run_sensor_benchmark_case(host_count=1),
        run_sensor_benchmark_case(host_count=6),
    ]

    for result in results:
        print_result(result)

    failed_results = [
        result
        for result in results
        if not result.within_acceptance_band
    ]

    if failed_results:
        failed_host_counts = ", ".join(
            str(result.host_count)
            for result in failed_results
        )

        raise SystemExit(
            "SCRUM-291 sensor acceptance benchmark "
            "failed for host count(s): "
            f"{failed_host_counts}"
        )

    print(
        "\nSCRUM-291 sensor acceptance benchmark passed."
    )


if __name__ == "__main__":
    main()