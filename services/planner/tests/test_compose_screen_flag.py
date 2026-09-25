"""tests/test_compose_screen_flag.py

SCRUM-455: .env must be the single control for the on-demand secondary screen, in
both composes.

The bug this guards against left no trace at runtime. Neither compose forwarded
SECONDARY_SCREEN_ENABLED into the planner container, so the .env line did nothing
and the screen ran on whatever the code default happened to be. The 2026-09-24 KVM
promotion is what exposed it: setting the var in .env changed nothing, because
nothing carried it across.

That is invisible to every other kind of test -- the flag works, the screen works,
the .env is right, and the wiring between them is missing. So the guard is a plain
YAML parse and a membership check. No stack, no containers.

Run from repo root:
    python -m pytest services/planner/tests/test_compose_screen_flag.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import yaml
except ImportError:                                    # pragma: no cover
    yaml = None

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSES = {
    "dev": _REPO_ROOT / "docker-compose.yaml",
    "prod": _REPO_ROOT / "docker-compose.prod.yaml",
}
_FLAG = "SECONDARY_SCREEN_ENABLED"

pytestmark = pytest.mark.skipif(yaml is None, reason="PyYAML is not installed")


def _services(path: Path) -> dict:
    assert path.exists(), f"missing compose file {path}"
    loaded = yaml.safe_load(path.read_text())
    assert isinstance(loaded, dict), f"{path} did not parse to a mapping"
    return loaded.get("services") or {}


def _env_names(service: dict) -> set:
    """The env var names a service declares, in either compose env syntax."""
    env = service.get("environment") or []
    if isinstance(env, dict):
        return set(env.keys())
    return {str(item).partition("=")[0] for item in env}


@pytest.mark.parametrize("which", sorted(_COMPOSES))
def test_the_planner_forwards_the_screen_flag(which):
    """Without this the .env line is inert and the screen runs on the default."""
    services = _services(_COMPOSES[which])
    assert "planner" in services, f"{which} compose has no planner service"
    assert _FLAG in _env_names(services["planner"]), (
        f"{_COMPOSES[which].name} does not forward {_FLAG} to the planner, so "
        f".env cannot control the secondary screen in {which}."
    )


@pytest.mark.parametrize("which", sorted(_COMPOSES))
def test_the_flag_is_forwarded_from_the_environment_not_hardcoded(which):
    """It must interpolate from .env, not pin a value into the compose file.

    `SECONDARY_SCREEN_ENABLED=true` written literally here would be the same bug
    wearing a different hat: the compose file would decide, not .env.
    """
    services = _services(_COMPOSES[which])
    env = services["planner"].get("environment") or []
    entries = (list(env.items()) if isinstance(env, dict)
               else [str(e).partition("=")[::2] for e in env])
    value = dict(entries)[_FLAG]
    assert "${" in str(value), (
        f"{_COMPOSES[which].name} pins {_FLAG}={value!r} instead of "
        f"interpolating it from .env"
    )


def test_both_composes_agree_on_the_ui_build_context():
    """SCRUM-455/447: the prod ui context was stale and failed the prod ui build.

    services/ui/Dockerfile copies libs/aps_math, so the build context has to be the
    repo root. The prod compose used ./services/ui, which cannot see libs/, so a
    clean prod ui build failed. Worth a test because the failure only shows up on a
    build from scratch -- an existing image keeps working.
    """
    dev = _services(_COMPOSES["dev"])["ui"].get("build")
    prod = _services(_COMPOSES["prod"])["ui"].get("build")
    assert prod == dev, (
        f"the ui build differs between composes: dev={dev} prod={prod}"
    )
    assert dev.get("context") == ".", (
        "the ui build context must be the repo root so libs/aps_math is visible"
    )


def test_every_service_needing_the_root_context_has_it_in_both_composes():
    """The same class of bug, checked across the board rather than for ui alone.

    Any service whose Dockerfile copies libs/ needs the root context. This walks
    the Dockerfiles rather than hardcoding the list, so a new service that copies
    libs/ and forgets the context is caught here instead of on a prod build.
    """
    offenders = []
    for which, path in _COMPOSES.items():
        for name, service in _services(path).items():
            build = service.get("build")
            if not isinstance(build, dict):
                continue
            context = str(build.get("context", ""))
            dockerfile = build.get("dockerfile")
            if dockerfile:
                candidate = _REPO_ROOT / dockerfile
            else:
                candidate = _REPO_ROOT / context.lstrip("./") / "Dockerfile"
            if not candidate.is_file():
                continue                                # external or generated
            text = candidate.read_text()
            needs_root = "libs/" in text
            if needs_root and context != ".":
                offenders.append(f"{which}:{name} context={context!r}")
    assert not offenders, (
        "these services copy libs/ but do not build from the repo root: "
        + ", ".join(offenders)
    )
