"""
SCRUM-382 -- the APS to GNC interface contract, as code.

openapi/gnc_interface.yaml is final and is the authority for every schema on
this path. This module does not restate it. It holds the two things code needs
that a schema cannot express, plus a loader so tests validate against the file
itself rather than against a second copy of the field names.

The two mappings are here because they are real mismatches between the
planner's internal vocabulary and the contract's enums, and a silent mismatch
would produce a command that fails schema validation at the GNC boundary:

  * direction. The scorer names candidates with hyphens ("anti-radial",
    "cross-track", "anti-cross"); the contract's ManeuverSpec.direction enum
    uses underscores and spells the cross-track pair out in full.
  * covariance_source. The planner's surrogate is labelled
    "surrogate_elliptical" (its actual shape, per SCRUM-369); the contract's
    RiskInputs.covariance_source enum admits only "real_cdm" or
    "surrogate_identity". Anything that is not a real CDM maps to the
    surrogate value, because the contract's distinction is real-versus-not and
    the ellipse's shape is not something GNC acts on.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, Optional

# Contract paths. Kept here so the client and the server routes cannot drift
# from each other or from the file.
GNC_COMMAND_PATH = "/v1/gnc/command"
GNC_APPROVE_PATH = "/v1/gnc/approve"
GNC_VETO_PATH = "/v1/gnc/veto"
GNC_REPORT_PATH = "/v1/gnc/report"
GNC_HEALTH_PATH = "/health"

CONTRACT_RELATIVE_PATH = Path("openapi") / "gnc_interface.yaml"

# -- direction, scorer vocabulary to ManeuverSpec.direction ------------------

_DIRECTION_TO_CONTRACT: Dict[str, str] = {
    "prograde": "prograde",
    "retrograde": "retrograde",
    "radial": "radial",
    "anti-radial": "anti_radial",
    "anti_radial": "anti_radial",
    "cross-track": "cross_track",
    "cross_track": "cross_track",
    "anti-cross": "anti_cross_track",
    "anti-cross-track": "anti_cross_track",
    "anti_cross_track": "anti_cross_track",
}

CONTRACT_DIRECTIONS = (
    "prograde", "retrograde", "radial",
    "anti_radial", "cross_track", "anti_cross_track",
)


class GNCContractError(ValueError):
    """A value cannot be expressed in the contract's vocabulary."""


def contract_direction(direction: str) -> str:
    """Map a scorer direction onto ManeuverSpec.direction.

    Raises rather than guessing. A direction the contract has no word for means
    the planner recommended something GNC cannot be told to do, and emitting a
    command with a bad enum would be rejected at the boundary anyway -- better
    to fail where the cause is visible. 'no-burn' lands here too, which is
    correct: there is no command to build for a burn that was not recommended.
    """
    key = str(direction or "").strip().lower()
    mapped = _DIRECTION_TO_CONTRACT.get(key)
    if mapped is None:
        raise GNCContractError(
            f"direction {direction!r} has no ManeuverSpec.direction equivalent; "
            f"the contract admits {list(CONTRACT_DIRECTIONS)}"
        )
    return mapped


# -- covariance_source, planner label to RiskInputs.covariance_source --------

CONTRACT_REAL_CDM = "real_cdm"
CONTRACT_SURROGATE = "surrogate_identity"


def contract_covariance_source(covariance_source: Optional[str]) -> str:
    """Map a planner covariance label onto RiskInputs.covariance_source.

    The contract's enum is a two-way distinction: the covariance either came
    from a real CDM or it did not. The planner's "surrogate_elliptical" is a
    more precise statement about the same not-a-real-CDM case, and GNC does not
    act on the ellipse's shape, so it maps to the contract's surrogate value.

    Everything unrecognised also maps to the surrogate value, deliberately. A
    covariance whose provenance we cannot name is not a real CDM, and claiming
    real_cdm on an unknown label is the one direction this must never fail in.
    """
    value = str(covariance_source or "").strip().lower()
    if value in ("real_cdm", "real"):
        return CONTRACT_REAL_CDM
    return CONTRACT_SURROGATE


# -- safe action, SafeActionRef.safe_action_type -----------------------------

SAFE_ACTION_CANNED_NUDGE = "canned_nudge"
SAFE_ACTION_PASSIVE_HOLD = "passive_hold"
CONTRACT_SAFE_ACTION_TYPES = (SAFE_ACTION_CANNED_NUDGE, SAFE_ACTION_PASSIVE_HOLD)


# -- the contract file itself -------------------------------------------------


def contract_path(repo_root: Optional[Path] = None) -> Path:
    """Locate openapi/gnc_interface.yaml.

    Walks up from this module to find the repository root, so it resolves the
    same way whether a test runs from the repo root or from the service
    directory. Not used at runtime: the planner image does not ship the
    contract, and a builder that validated only when a file happened to be
    present would behave differently in the container than in CI. Tests
    validate; the builder is written to conform.
    """
    if repo_root is not None:
        return Path(repo_root) / CONTRACT_RELATIVE_PATH
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / CONTRACT_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    raise GNCContractError(
        f"could not locate {CONTRACT_RELATIVE_PATH} above {here}"
    )


@functools.lru_cache(maxsize=1)
def _contract_document() -> Dict[str, Any]:
    import yaml

    return yaml.safe_load(contract_path().read_text())


def load_gnc_schema(name: str) -> Dict[str, Any]:
    """One component schema from the contract, with $refs left intact.

    Callers validate with a resolver rooted at the whole document, so nested
    $ref targets resolve to the contract's own definitions rather than to
    copies.
    """
    schemas = _contract_document()["components"]["schemas"]
    if name not in schemas:
        raise GNCContractError(
            f"{name!r} is not a schema in the contract; it defines "
            f"{sorted(schemas)}"
        )
    return schemas[name]


def validate_against_contract(name: str, payload: Any) -> None:
    """Validate a payload against a contract schema, raising on a violation.

    Test-facing. The whole contract document is handed to the validator as the
    resolution root so $ref targets inside a schema resolve.
    """
    import jsonschema

    document = _contract_document()
    schema = dict(load_gnc_schema(name))
    # Resolve $refs against the contract document itself.
    schema["components"] = document["components"]
    jsonschema.validate(instance=payload, schema=schema)
