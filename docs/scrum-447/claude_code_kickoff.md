# SCRUM-447 — kickoff for Claude Code

Drive the 3D globe encounter view from live LeoLabs data, re-enabling the globe
SCRUM-446 disabled. Ticket: https://xorbita.atlassian.net/browse/SCRUM-447

Read docs/scrum-447/implementation_plan.md first. It names every seam with file and
line, and it records that two pieces already exist in the tree and must be reused:
LeoLabsClient.get_states (services/planner/common/leolabs_client.py:445) for the
state vectors, and the two-body RK4 propagator propagate_two_body inside
services/ui/app/main.py get_orbits (line 592) for the tracks. This is plumbing, not
a rebuild.

Run in order:

1. Read docs/scrum-447/implementation_plan.md.
2. Confirm-live first. On the local stack, run the get_states check in the plan
   inside the planner container and confirm a real asset returns a usable state
   vector. Do not build the endpoint against an assumption about the response shape.
3. Move propagate_two_body into a shared helper the planner can import, and have the
   UI /api/orbits keep using it, so there is one propagator.
4. Add the planner route GET /v1/leolabs/orbits that returns asset_track,
   object_tracks and per-object risk for the selected asset and its in-volume
   conjunctions, reusing the conjunctions route's risk banding. Keep the
   200/404/422/503 status contract, feed-off is 503 and never an empty 200.
5. Add the UI proxy GET /api/orbits/live and point fetchAndUpdateGlobeOrbits at it
   in Live Asset mode, keeping /api/orbits for scenario mode.
6. Extend updateGlobeOrbits for the nominal haze and its toggle, labels on the asset
   and red and amber only, and the risk styling, per the mockup described in the
   plan.
7. Unhide btn3D at index.html:500 as the last step, once the live path renders. 2D
   stays the default view.
8. Tests: add planner route tests for state to track, empty in-volume set, a
   secondary with no state (skipped), and feed-off 503. Run the planner suite from
   the repo root and confirm green: python3 -m pytest services/planner. The live
   globe render is a manual browser check on the local stack.
9. Commit on a branch named scrum-447-globe-live-data, including this
   docs/scrum-447/ folder in the commit. No AI attribution in the commit message or
   PR body.
10. Push and open the PR against main, then report the PR number and head SHA, and
    post the verification-first review verdict on the PR itself, not only in chat.

Cowork will do the verification-first review before John merges: read the committed
diff, run the planner suite on the branch, confirm the live endpoint sources states
from get_states and not the npz, confirm the status contract, and check the globe on
the rebuilt local stack against the SWARM C acceptance. John is the only one who
merges. Development and testing stay on the local stack.
