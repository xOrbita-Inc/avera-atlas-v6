from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from services.sandbox.observation_bundle import ObservationBundle
from services.sandbox.observations import AngularObservation
from services.sandbox.schema_emission import (
    TrackerIngestResult,
    TrackerObservationContractArtifact,
    push_observations_to_tracker,
    write_tracker_observations_json,
)


class GodViewAccessError(PermissionError):
    """Raised when sensor-knowledge code attempts to access sandbox truth state."""


@dataclass(frozen=True)
class SensorObservationStream:
    sensor_id: str
    observations: tuple[AngularObservation, ...]

    def __iter__(self) -> Iterator[AngularObservation]:
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def truth_states(self):
        raise GodViewAccessError(
            "Sensor knowledge cannot access god-view truth states."
        )

    @property
    def objects(self):
        raise GodViewAccessError(
            "Sensor knowledge cannot access god-view simulation objects."
        )

    @property
    def snapshots(self):
        raise GodViewAccessError(
            "Sensor knowledge cannot access god-view snapshots."
        )


@dataclass(frozen=True)
class SensorKnowledgeStore:
    streams_by_sensor: dict[str, SensorObservationStream]

    @classmethod
    def from_observation_bundle(
        cls,
        bundle: ObservationBundle,
    ) -> "SensorKnowledgeStore":
        grouped: dict[str, list[AngularObservation]] = {}

        for obs in bundle.detected_observations:
            grouped.setdefault(obs.host_id, []).append(obs)

        streams = {
            sensor_id: SensorObservationStream(
                sensor_id=sensor_id,
                observations=tuple(
                    sorted(
                        observations,
                        key=lambda obs: (obs.t_seconds, obs.debris_id),
                    )
                ),
            )
            for sensor_id, observations in grouped.items()
        }

        return cls(streams_by_sensor=streams)

    @property
    def sensor_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.streams_by_sensor))

    def stream_for_sensor(self, sensor_id: str) -> SensorObservationStream:
        if sensor_id not in self.streams_by_sensor:
            raise KeyError(f"Unknown sensor_id: {sensor_id}")
        return self.streams_by_sensor[sensor_id]

    def write_schema_artifact(
        self,
        bundle: ObservationBundle,
        output_path: str | Path,
    ) -> TrackerObservationContractArtifact:
        """
        DEPRECATED as the primary path (SCRUM-373 AC2): use push_to_tracker
        to feed observations through the live interface instead. Kept
        working for offline debugging/dry-run use only.
        """
        return write_tracker_observations_json(bundle, output_path)

    def push_to_tracker(
        self,
        bundle: ObservationBundle,
        *,
        source: str = "sandbox",
        timeout_s: float = 5.0,
    ) -> TrackerIngestResult:
        """
        Feed observations through the live tracker /v1/observations
        interface (SCRUM-373 AC2), the primary ingest path. Operates on
        the same ObservationBundle as write_schema_artifact and goes
        through the same sensor-facing conversion
        (bundle_to_tracker_ingest_request), so SCRUM-292 sensor-knowledge
        isolation is preserved identically: this class has no access to
        truth_states, objects, or snapshots (see the GodViewAccessError
        properties below), so there is nothing god-view available to leak
        into the live call even by accident.
        """
        return push_observations_to_tracker(
            bundle, source=source, timeout_s=timeout_s
        )

    @property
    def truth_states(self):
        raise GodViewAccessError(
            "Sensor knowledge cannot access god-view truth states."
        )

    @property
    def objects(self):
        raise GodViewAccessError(
            "Sensor knowledge cannot access god-view simulation objects."
        )

    @property
    def snapshots(self):
        raise GodViewAccessError(
            "Sensor knowledge cannot access god-view snapshots."
        )
