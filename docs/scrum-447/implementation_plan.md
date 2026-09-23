# SCRUM-447 — implementation plan: 3D globe on live LeoLabs data

Ticket: https://xorbita.atlassian.net/browse/SCRUM-447
Design: project doc claude/3d-globe-live-data-design-2026-09-23.md
Visual target: the "Live Conjunction Globe" mockup (artifact v3).

## What this is

Re-enable the encounter panel's 3D globe, which SCRUM-446 disabled because it drew
surrogate propagator orbits, and drive it from live LeoLabs state vectors so the
globe shows the selected Live Asset and its current conjunctions in orbit, colored
by collision risk.

## What already exists in the tree, so this is mostly plumbing

Two pieces the design treated as work already exist and should be reused, not
rebuilt.

- State fetch is done. LeoLabsClient.get_states(catalog_number, latest=True) is
  implemented in services/planner/common/leolabs_client.py at line 445 and returns
  the object's latest state vector. No new client method is needed.
- Two-body propagation is done. services/ui/app/main.py get_orbits() at line 592
  already contains a working RK4 two-body propagator, propagate_two_body(r0, v0),
  that closes one full LEO revolution from a state vector and returns [x,y,z] km
  points. The live path reuses this exact propagator and only changes where r0 and
  v0 come from, from the states_multi.npz artifact to live get_states.

The globe rendering is intact and dormant: initGlobe, updateGlobeOrbits
(index.html:3440), startGlobeLoop, globeFocusOnConj, eciToScene.
fetchAndUpdateGlobeOrbits (index.html:3423) fetches GET /api/orbits and hands
{asset, objects} to updateGlobeOrbits. setView('3d') (index.html:3265) already
calls fetchAndUpdateGlobeOrbits and startGlobeLoop. btn3D is hidden at
index.html:500 by the SCRUM-446 one-line display:none. currentView defaults to
'2d' at index.html:3241.

So the build is to source the tracks from live states, wire a live endpoint, add
the haze, labels and risk styling to the renderer, and unhide the button.

## Seams

1. New planner route GET /v1/leolabs/orbits, taking asset (NORAD) and the same
   in-volume conjunction context the SCRUM-445 conjunctions route already builds.
   For the asset and each in-volume secondary NORAD it gets the latest state via
   get_states, propagates one revolution, and returns
   {status, asset_track, object_tracks:{norad:[[x,y,z]...]}, risk:{norad:band}}.
   Reuse the exact risk banding the conjunctions route computes per row so the
   globe colors match the 2D panel. Keep the 200/404/422/503 contract, feed-off is
   503 and never an empty 200, the same as the conjunctions route.

2. Move propagate_two_body out of services/ui/app/main.py into a small shared
   helper the planner can import, for example services/planner/common/orbits.py or
   libs/aps_math, rather than duplicating it. The UI /api/orbits keeps calling the
   same helper.

3. State-fetch cost. A dense asset such as SWARM C with 96 in-volume conjunctions
   is up to 96 get_states calls. Cache the result per query fingerprint of asset
   plus in-volume set. Also decide in the build whether to source each secondary
   state from the CDM already parsed for the row (SAT2 state at TCA), which avoids
   the per-secondary API calls, with get_states as the fallback. Record the choice
   in this plan doc.

4. New UI proxy GET /api/orbits/live in services/ui/app/main.py that proxies the
   planner route, matching how the conjunctions proxy already works.

5. Frontend source switch. fetchAndUpdateGlobeOrbits (index.html:3423) fetches
   /api/orbits/live?asset=<liveAssetNorad> when isLiveMode() and liveAssetNorad are
   set, and keeps /api/orbits for scenario mode. Keep the response shape as
   {asset_track, object_tracks, risk} so updateGlobeOrbits needs only small
   changes.

6. Renderer changes in updateGlobeOrbits (index.html:3440): draw all of the asset's
   conjunctions by default, nominal objects as a dim toggleable haze and red, amber
   and green at full strength, a Nominal control that drops the haze, labels on the
   asset and the red and amber objects only, risk coloring tied to the same banding
   as the 2D panel, and keep the moving markers, the dashed closest-approach link to
   the worst conjunction, the covariance toggle, and focus-on-worst. This matches
   the mockup.

7. Re-enable last. Remove style="display:none" from btn3D at index.html:500 only
   once the live path renders. 2D stays the default view.

## Fetch timing

Fetch the live orbits only when the operator switches to the 3D view, not on every
Live Asset load, so an unused globe costs nothing. setView('3d') already triggers
fetchAndUpdateGlobeOrbits, so this holds by construction as long as nothing calls
it on load.

## Confirm-live step, do this first

Before building the endpoint, confirm inside the planner container that get_states
returns a usable state for a real asset and a real secondary. On the local stack:

    docker compose exec -T planner python - <<'PY'
    from common.leolabs_client import LeoLabsClient
    c = LeoLabsClient()
    print(c.get_states("39453"))
    PY

Confirm the response carries position and velocity so propagate_two_body can run.
If some secondaries have no state, that is the case the endpoint handles by
skipping that object while still showing its risk on the row, not by failing.

## Acceptance criteria

1. With Live Asset SWARM C selected and conjunctions fetched, switching to the 3D
   globe shows the asset orbit plus its conjunctions colored by the same risk bands
   as the 2D panel, sourced from live LeoLabs states and not from any npz artifact.
2. Nominal conjunctions render as a dim haze by default and the Nominal toggle drops
   them. Labels show on the asset and the red and amber objects only.
3. The worst conjunction carries the dashed closest-approach link and focus-on-worst
   frames it.
4. 2D stays the default view and the globe is fetched only on switching to 3D.
5. Feed-off returns 503 through the live endpoint rather than an empty globe shown
   as success.
6. Offline tests cover the planner route: state to track, an empty in-volume set, a
   secondary with no state so it is skipped, and feed-off 503. Run
   python3 -m pytest services/planner and confirm green. The live render is a
   manual browser check on the local stack.

## Out of scope

The separate full globe redesign, with its own requirements now being drafted, is a
later sprint and Tylen's. This ticket is the live-data drive of the globe as it
renders today.

---

# Confirm-live results and the decisions they forced (2026-09-23)

Run against the real LeoLabs API inside the planner container before any endpoint
code was written. Three of the plan's assumptions above did not survive it.

## 1. The plan's own confirm-live snippet is wrong

`c.get_states("39453")` returns **HTTP 404**. `get_states` takes a LeoLabs
*catalog number*, not a NORAD id. The working call is `c.get_states("L3969")`;
the NORAD-to-catalog mapping already exists as
`AssetRegistry.leolabs_for_norad()`.

## 2. The response shape, confirmed rather than assumed

```
{"states": [ {
    "catalogNumber": "L3969", "noradCatalogNumber": 39453,
    "timestamp": "2026-09-23T19:00:17Z",
    "frames": {"EME2000": {"position": [x,y,z], "velocity": [vx,vy,vz], ...}}
} ]}
```

- `states` is a **list**; the latest is `states[0]`.
- Position and velocity live under `frames.EME2000`, not at the top level.
- **Units are metres and metres/second**, not km. Measured on SWARM C:
  `|r| = 6.79e6` m (6,790 km, a sane LEO radius) and `|v| = 7,662` m/s.
  `propagate_two_body` works in km and km/s, so the endpoint converts. Building
  against the assumed km would have produced tracks 1000x too large, which is
  exactly the failure the confirm-live step exists to prevent.

## 3. get_states is 403 for secondaries, which settles plan item 3

`get_states` on a conjunction secondary (e.g. `L140659` / STARLINK-30994) returns
**HTTP 403**: the subscription covers our own assets, not arbitrary catalog
objects. So sourcing secondary states from `get_states` is not possible on this
account, and the plan's "up to 96 get_states calls" cost concern does not arise.

**Decision (plan item 3): secondary states come from the already-parsed CDM.**
`ParsedLeoLabsCDM.secondary` carries `r_km` and `v_km_s` straight from the CDM's
SAT2 block, already parsed for the SCRUM-445 row and already in km. Spot-checked
against three real secondaries: `|r|` 6,788-6,795 km and `|v|` 7.663-7.670 km/s,
both physically sane. The asset state still comes from `get_states`, which is the
live source the ticket asks for and the one a reviewer is told to confirm.

Consequences:

- **Zero extra API calls per request.** One `get_states` for the asset, and the
  secondaries ride along on the conjunction fetch that already happened. The
  per-query state cache the plan proposed is therefore not needed and is not
  built; adding cross-request cache state for a single call would be cost without
  benefit.
- **`get_states` returning 403 or 404 for the asset is a real failure** (503), but
  a secondary lacking a usable CDM state is skipped, not fatal, exactly as the
  plan requires.

## 4. Epochs are not aligned, and the globe does not claim they are

The asset state is at *now*; a secondary's CDM state is at *its TCA*, up to the
lookahead window away. The tracks are therefore each drawn from that object's own
epoch and are not a simultaneous snapshot.

This is not a regression and is not papered over. The existing renderer already
places every object marker at a **random phase offset** along its ring
(`debrisPhaseMap[objId]=Math.random()`, index.html), so the globe has never
claimed simultaneity: what it conveys is the set of orbital paths and their risk
colouring. Cross-propagating secondaries to a common epoch was considered and
rejected -- two-body with no J2 over a multi-day TCA offset would add more error
than it removes, and the full globe redesign is explicitly out of scope here. The
response carries `epoch_utc` per object so the limitation is visible to any
consumer rather than implicit.

## 5. The UI container cannot import libs/ as it stands

Plan item 2 wants one shared propagator. The UI image builds from
`./services/ui`, so `libs/` is outside its context; planner and propagator build
from the repository root for exactly this reason (SCRUM-388). So the shared helper
lands in `libs/aps_math/orbits.py` and the UI service moves to a root build
context with `COPY libs/aps_math/`, mirroring the planner Dockerfile rather than
inventing a third arrangement.
