"""tests/test_leolabs_live_smoke.py

SCRUM-411: opt-in live smoke test against the real LeoLabs API.

This is the only test that touches the network. It is skipped unless BOTH of
these are true, so CI never needs a credential and it never runs by accident:

  - LEOLABS_ACCESS_KEY and LEOLABS_SECRET_KEY are set, and
  - LEOLABS_LIVE_SMOKE=1 is set (explicit opt-in).

It exercises the client -> parser -> Pc cross-check path end to end on a real
LeoLabs CDM for a subscribed sat, which is the parser half of AC7. The full
/v1/evaluate wiring and the source badge (AC6) land in a later step, after this
parser work is reviewed.

Run it deliberately, from repo root:

    export LEOLABS_ACCESS_KEY=...   # from Manage API Keys on the platform
    export LEOLABS_SECRET_KEY=...
    export LEOLABS_LIVE_SMOKE=1
    python -m pytest services/planner/tests/test_leolabs_live_smoke.py -v -s \
        --leolabs-object L2669
"""

from __future__ import annotations

import datetime as dt
import os

import pytest

_HAVE_KEYS = bool(
    os.environ.get("LEOLABS_ACCESS_KEY") and os.environ.get("LEOLABS_SECRET_KEY")
)
_OPTED_IN = os.environ.get("LEOLABS_LIVE_SMOKE") == "1"

pytestmark = pytest.mark.skipif(
    not (_HAVE_KEYS and _OPTED_IN),
    reason=(
        "live LeoLabs smoke test is opt-in: set LEOLABS_ACCESS_KEY, "
        "LEOLABS_SECRET_KEY and LEOLABS_LIVE_SMOKE=1 to run it."
    ),
)

# The subscribed sat to query. Override with LEOLABS_SMOKE_OBJECT.
_OBJECT = os.environ.get("LEOLABS_SMOKE_OBJECT", "L2669")


def test_live_search_parse_and_cross_check():
    from common.leolabs_client import LeoLabsClient
    from common.leolabs_cdm_parser import cross_check_pc, parse_leolabs_cdm

    client = LeoLabsClient()

    # LeoLabs caps the minTca..maxTca span at 30 days. Use a 7-day lookahead and
    # a 23-day lookback so the whole window stays within the cap.
    now = dt.datetime.now(dt.timezone.utc)
    min_tca = (now - dt.timedelta(days=23)).strftime("%Y-%m-%dT%H:%M:%SZ")
    max_tca = (now + dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

    cdms = client.search_conjunction_cdms(
        object1=_OBJECT, min_tca=min_tca, max_tca=max_tca, cdm_source="LeoLabs"
    )
    if not cdms:
        pytest.skip(
            f"no LeoLabs CDMs for {_OBJECT} in the window; widen the range or "
            f"submit an on-demand screening."
        )

    parsed = parse_leolabs_cdm(cdms[0], _OBJECT)

    # Golden check 2 on live data: our Pc agrees with the CDM's.
    ours = cross_check_pc(parsed)
    theirs = parsed.cdm_collision_probability
    print(f"\nlive CDM {parsed.provenance.get('cdm_id')}: "
          f"our Pc={ours:.6e}  CDM Pc={theirs}")
    if theirs:
        assert abs(ours - theirs) / theirs < 5.0e-3
