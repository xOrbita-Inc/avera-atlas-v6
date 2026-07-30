"""
aps_math -- numerical routines shared across AVERA-ATLAS services.

SCRUM-388. Code in here is imported by more than one service, so it lives in
exactly one place. Do not copy a module out of this package into a service.

Why this package exists
-----------------------
`pc_utils` implements NASA-standard probability-of-collision calculation and
lived in services/propagator, where the planner could not reach it. The
planner's operator policy expresses risk in Pc, `pc_maneuver_threshold` and
`pc_monitor_threshold`, but the planner had no way to compute one and fell
back to whatever an external CDM or UDL record happened to supply.

The alternative to a shared package was copying roughly seven hundred lines of
safety-relevant numerics into a second service. SCRUM-386 is a recent and
expensive demonstration of what one undetected numerical defect costs, and two
diverging copies of a Pc routine is the same failure waiting to happen.

Build implications
------------------
Docker build context for any service importing this package must be the
repository root, not the service directory, because a build cannot COPY from
outside its context. See services/planner/Dockerfile and
services/propagator/Dockerfile, and the `context: .` entries in
docker-compose.yaml.

Modules
-------
pc_utils      probability of collision, Alfano 2005 and Frisbee
conventions   shared numerical conventions (ADR-010). Values every service must
              agree on, each carrying why it was chosen and what changing it
              would affect.
"""

from . import conventions  # noqa: F401
from .pc_utils import (  # noqa: F401
    PcResult,
    compute_pc,
    compute_pc_batch,
    default_covariance_from_uncertainty,
    frisbee_max_pc,
    pc_circle,
)

__all__ = [
    "conventions",
    "PcResult",
    "compute_pc",
    "compute_pc_batch",
    "default_covariance_from_uncertainty",
    "frisbee_max_pc",
    "pc_circle",
]
