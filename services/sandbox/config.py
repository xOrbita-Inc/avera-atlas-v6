from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SizeBin = Literal["1cm", "5cm", "10cm"]
ConstellationMode = Literal["same_host", "distributed"]


@dataclass(frozen=True)
class DebrisConfig:
    count: int = 500
    altitude_min_km: float = 400.0
    altitude_max_km: float = 800.0
    inclination_min_deg: float = 0.0
    inclination_max_deg: float = 98.0

    # Fractions should sum to 1.0
    size_bin_weights: dict[SizeBin, float] = field(
        default_factory=lambda: {
            "1cm": 0.6,
            "5cm": 0.3,
            "10cm": 0.1,
        }
    )


@dataclass(frozen=True)
class HostConfig:
    count: int = 6
    mode: ConstellationMode = "distributed"
    altitude_km: float = 600.0
    inclination_deg: float = 97.5


@dataclass(frozen=True)
class IntegratorConfig:
    dt_seconds: float = 1.0
    duration_seconds: float = 24.0 * 3600.0
    save_every_n_steps: int = 60


@dataclass(frozen=True)
class DragConfig:
    enabled: bool = True
    cd: float = 2.2
    atmosphere_scale_height_km: float = 60.0
    reference_density_kg_m3: float = 3.5e-12
    reference_altitude_km: float = 400.0


@dataclass(frozen=True)
class SimConfig:
    seed: int = 42
    debris: DebrisConfig = field(default_factory=DebrisConfig)
    hosts: HostConfig = field(default_factory=HostConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    drag: DragConfig = field(default_factory=DragConfig)