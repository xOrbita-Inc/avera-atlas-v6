# SCRUM-461 implementation plan: dashboard shows a truncated conjunction list as partial

Ticket: https://xorbita.atlassian.net/browse/SCRUM-461
Branch off main. UI only. Safety-relevant. Part of SCRUM-433. Relates to SCRUM-459.

## What this is

SCRUM-459 made the planner honest about a truncated listing: the list response
(/api/planner/v1/leolabs/conjunctions) and the globe response
(/api/planner/v1/leolabs/orbits) now carry complete (bool), partial (bool), and,
when partial, a truncation block {cdms_pulled, cdms_in_window, truncated_by}. The
dashboard does not read those fields yet, so a truncated SWARM B listing renders as
an ordinary, complete-looking list. The truncated set is a prefix in LeoLabs order,
not Pc order, so the worst conjunction can be beyond the cut. An operator triaging
worst-first would not know the view is partial. This ticket makes the dashboard say
so, on the table and on the globe.

## The honesty rule, because it is the whole point

A short or uncluttered view that looks complete reads as a quiet sky, and that is the
direction that gets someone hurt. So partial must be shown unmistakably, and the
banner must say the worst conjunction may not be in view. This is the same safety
direction as the planner side of SCRUM-459; the dashboard is the last place it can be
lost.

## The change, all in the ui service

1. The Active Conjunctions table (SCRUM-421/445 area of index.html). After a list
   response, read complete / partial / truncation. When partial, render a clear
   banner above or on the table: the view is incomplete, showing the worst of
   cdms_pulled pulled from cdms_in_window in the window, and the worst conjunction may
   not be shown. Use the denominator the backend provides; do not invent one. When
   complete, render exactly as today, no banner.

2. The globe (SCRUM-447 orbits view). The orbits response carries the same fields.
   An uncluttered globe reads as a quiet sky even more than a short table does, so
   when partial show the same banner on the globe view. When complete, unchanged.

3. One shared piece of copy and one shared style for the banner, used by both, so the
   table and the globe say the same thing. Make it unmistakable, not a subtle tint or
   a small grey note.

## Do not

- Do not change the planner, the responses, or the fetch. SCRUM-459 already sends
  everything the banner needs; this only reads and renders it.
- Do not render a partial list or globe as complete anywhere, and do not drop the
  rows that did come back. A partial view is useful; it just must announce itself.
- Do not gate the rows behind the banner. Show the rows and the banner together.

## Tests

- A ui proxy or template test in the pattern of the existing ones: a list response
  with complete false and a truncation block renders the banner with the counts; a
  complete response renders no banner. Assert the banner text carries the
  cdms_pulled / cdms_in_window numbers.
- The globe equivalent: a partial orbits response shows the banner.
- The JS rendering is verified live where there is no offline harness, as SCRUM-457
  did; cover the visual there.

Run the existing ui tests and confirm green.

## Live acceptance

Rebuild the ui. Select SWARM B (once SCRUM-460 lets it list and evaluate without
taking the planner down; until then a fixture or a forced partial response is enough
to drive the banner): the table shows the partial banner with the real counts, and the
globe shows it too. Select a sparse asset and confirm no banner appears on a complete
view. Record in docs/scrum-461/live_verification.md, ideally with a screenshot of the
banner. Do not deploy to the KVM.

## Commit

Branch off main. Include docs/scrum-461/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms a
partial list and a partial globe both show the banner with the counts, that a complete
view shows nothing new, and that no partial view can render as complete, then reviews
the live banner.
