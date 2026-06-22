from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from services.sandbox.observation_bundle import ObservationBundle
from services.sandbox.observations import AngularObservation
from services.sandbox.schema_emission import (
    EmittedObservationArtifact,
    write_observations_multi_npz,
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
    ) -> EmittedObservationArtifact:
        return write_observations_multi_npz(bundle, output_path)

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