from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ObjectKind = Literal["debris", "host"]
SizeBin = Literal["1cm", "5cm", "10cm"]


@dataclass(frozen=True)
class PhysicalProperties:
    size_bin: SizeBin
    characteristic_size_m: float
    mass_kg: float
    area_m2: float
    cd: float = 2.2


@dataclass(frozen=True)
class SimObject:
    object_id: str
    kind: ObjectKind
    r_eci_km: np.ndarray
    v_eci_km_s: np.ndarray
    physical: PhysicalProperties
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    t_seconds: float
    r_eci_km: np.ndarray
    v_eci_km_s: np.ndarray


@dataclass(frozen=True)
class Snapshot:
    t_seconds: float
    states: dict[str, ObjectState]


@dataclass(frozen=True)
class SimulationResult:
    objects: dict[str, SimObject]
    snapshots: list[Snapshot]