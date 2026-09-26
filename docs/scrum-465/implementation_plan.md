# SCRUM-465 implementation plan: globe bottom dock to match the concept

Ticket: https://xorbita.atlassian.net/browse/SCRUM-465
Branch off main. UI only. Relates to SCRUM-447, SCRUM-449. Pairs with SCRUM-464.

## What this is

The Live Conjunction Globe concept has a stacked bottom-right dock: an asset switch, a
worst-conjunction detail card, then the control row. The shipped globe replaced it with
a bare control row. Rebuild the dock to match the concept.

## The concept, for reference

From the mockup: a vertical `#dock` (flex column, ~300px wide, bottom-right) holding
three panels in order:

1. `#scenrow`: two asset tab buttons ("Swarm C · 96", "CryoSat-2 · 3"),
   aria-pressed for the active one.
2. `#info`: the worst-conjunction card. A header row with the object name and a RED
   risk badge, a muted "Worst of 96 conjunctions in the reporting volume" line, then a
   two-column `<dl>`: Asset, Miss, Pc (LeoLabs), TCA, Covariance (real CDM).
3. `#controls`: Auto-rotate, Labels, Nominal, Covariance, Focus worst.

Panel styling: JetBrains Mono, translucent panel background with a subtle blur, cyan
pressed state on buttons, tabular-nums on the values.

## The change, all in services/ui/app/templates/index.html

1. Asset switch. The concept hardcodes two tabs; prod has the subscribed LeoLabs assets
   (the LIVE ASSET list, about ten). Build a compact asset switcher in the dock, each
   entry showing the asset and its conjunction count in the concept style
   ("SWARM B · 96"), the current one highlighted, wrapping or horizontally scrolling if
   there are many. It must drive the SAME selection the LIVE ASSET dropdown drives, one
   source of truth: selecting in the dock reselects the asset and refetches the globe
   exactly as the dropdown does, and switching via the dropdown updates the dock's
   highlight. Do not create a second, independent asset state.

   DESIGN POINT to settle at review: tabs versus a compact scrolling strip, given about
   ten assets rather than the concept's two. Propose a single-row horizontally
   scrollable strip of chips, current asset first, so it reads like the concept without
   wrapping to many rows.

2. Worst-conjunction card. Populate from the current worst conjunction, the same object
   FOCUS WORST frames and the same data the Active Conjunctions panel already has: name
   plus risk badge, "Worst of N conjunctions in the reporting volume", then Asset,
   Miss, Pc (LeoLabs), TCA, Covariance. Covariance shows real_cdm versus surrogate
   honestly, reusing the existing cov-badge styling. Update it on asset switch and on
   fetch; when a view is partial (SCRUM-461/462), the "N" is the honest count already
   computed, not a fabricated total.

3. Control row. Auto-rotate, Labels, Nominal, Covariance, Focus worst, restyled to the
   concept. This is the same set of toggles already wired; only the styling and its
   place in the dock change.

## Keep

- The SCRUM-461/462 partial-view chip and caption stay intact and legible inside or
  beside the dock. A truncated view must still announce itself; the dock must not hide
  or replace that signal.
- One asset selection state shared with the LIVE ASSET dropdown.
- No row is gated behind the dock; the globe and its data render as today.

## Do not

- Do not change the planner or the orbits response. This is layout and wiring over data
  the page already has.
- Do not add a second asset selector that can drift from the dropdown.
- Do not regress the partial-view honesty.

## Tests

- Template-wiring / harness in the existing pattern: the dock contains the asset
  switch, the worst card and the control row; the asset switch is driven from the
  subscribed-asset list and its selection calls the same handler as the LIVE ASSET
  dropdown; the worst card fields are bound to the worst-conjunction data; the
  partial-view chip still renders when a response is partial.
- Assert there is one asset-selection code path, not two.
- Run the existing ui tests and confirm green.

## Live acceptance

Rebuild the ui. The globe view shows the dock: a working asset switch that reselects
the asset and refreshes the globe (and stays in sync with the dropdown), a
worst-conjunction card populated from live data and updating on switch, and the control
row, all matching the concept layout and styling. A partial response still shows the
chip. Capture a screenshot next to the concept dock. Record in
docs/scrum-465/live_verification.md. Do not deploy to the KVM.

## Commit

Branch off main. Include docs/scrum-465/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms the dock
matches the concept, that the asset switch shares one selection state with the dropdown,
that the worst card is bound to live data and updates on switch, and that the
partial-view honesty is intact, then reviews the live dock against the concept.
