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
frames        reference-frame transforms (SCRUM-397) and the CW state
              transition matrix (SCRUM-378). RTN to ECI, expressing an
              RTN-ordered linear map in ECI, and Phi(tau, t0).
observability measurement Jacobians for the Fisher-information observability
              gate (SCRUM-378, MAF v2.0 Sec 7).
"""

from . import conventions  # noqa: F401
from . import frames  # noqa: F401
from . import observability  # noqa: F401
from .frames import (  # noqa: F401
    MU_EARTH,
    cw_phi_full,
    is_degenerate_state,
    rotate_cw_block,
    rtn_to_eci_rotation,
)
from .observability import (  # noqa: F401
    observation_jacobian,
)
# pc_utils is imported lazily (PEP 562) because it is the only module here that
# needs scipy. SCRUM-447 gave the ui image a reason to import aps_math.orbits --
# forty lines of RK4 over numpy -- and an eager import chain would have made a
# dashboard container carry the whole scipy stack to get it, or else forced a
# second copy of the propagator, which the note at the top of this file exists to
# forbid. Every name below still resolves exactly as it did; it just resolves on
# first use. Services that use Pc already depend on scipy and see no change.
_PC_UTILS_EXPORTS = frozenset({
    "PcResult",
    "compute_pc",
    "compute_pc_batch",
    "default_covariance_from_uncertainty",
    "frisbee_max_pc",
    "pc_circle",
})


def __getattr__(name):  # noqa: D401 - module-level lazy attribute hook
    # importlib.import_module, not `from . import pc_utils`: the latter resolves
    # the submodule by looking up the attribute on this package, which lands back
    # in this function and recurses until the stack ends.
    if name in _PC_UTILS_EXPORTS or name == "pc_utils":
        import importlib

        module = importlib.import_module(".pc_utils", __name__)
        return module if name == "pc_utils" else getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _PC_UTILS_EXPORTS | {"pc_utils"})

__all__ = [
    "conventions",
    "frames",
    "observability",
    "MU_EARTH",
    "cw_phi_full",
    "is_degenerate_state",
    "rotate_cw_block",
    "rtn_to_eci_rotation",
    "observation_jacobian",
    "PcResult",
    "compute_pc",
    "compute_pc_batch",
    "default_covariance_from_uncertainty",
    "frisbee_max_pc",
    "pc_circle",
]
