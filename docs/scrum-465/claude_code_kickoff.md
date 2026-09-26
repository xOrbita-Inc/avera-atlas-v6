# SCRUM-465 kickoff for Claude Code

Rebuild the globe's bottom-right dock to match the Live Conjunction Globe concept: an
asset switch, a worst-conjunction detail card, then the control row. The shipped globe
has only a bare control row. Ticket:
https://xorbita.atlassian.net/browse/SCRUM-465

Branch off main. UI only. No planner changes. Pairs with SCRUM-464 (the covariance
dome); they ship together.

Read docs/scrum-465/implementation_plan.md first. The point: one asset-selection state
shared with the existing LIVE ASSET dropdown, a worst card bound to live data, and the
SCRUM-461/462 partial-view honesty kept intact.

Run in order:

1. Read docs/scrum-465/implementation_plan.md.

2. Find in services/ui/app/templates/index.html: the globe control cluster
   (#globeControls and the toggle buttons), the LIVE ASSET selection handler and the
   subscribed-asset list behind the dropdown, the worst-conjunction data (what FOCUS
   WORST and the Active Conjunctions panel use), and the SCRUM-461/462 partial-view chip
   and caption.

3. Build the dock as a bottom-right stacked panel styled to the concept (JetBrains
   Mono, translucent panel background, cyan pressed state), holding, top to bottom:
   the asset switch, the worst-conjunction card, the control row.

4. Asset switch: a compact strip of chips built from the subscribed-asset list, each
   "NAME · count", current highlighted. Selecting one calls the SAME handler the LIVE
   ASSET dropdown calls, and the dropdown's selection updates the strip highlight. One
   selection state, not two. Given ~10 assets, propose a single-row horizontally
   scrollable strip rather than wrapping tabs; flag as the review design point.

5. Worst card: name + risk badge, "Worst of N conjunctions in the reporting volume",
   then Asset / Miss / Pc (LeoLabs) / TCA / Covariance, bound to the current worst
   conjunction and updated on switch and fetch. Covariance shows real_cdm vs surrogate
   with the existing cov-badge. N is the honest partial-aware count, not a fabricated
   total.

6. Control row: Auto-rotate, Labels, Nominal, Covariance, Focus worst, restyled into
   the dock. Same toggles, same handlers.

7. Keep the SCRUM-461/462 partial-view chip and caption intact and legible in or beside
   the dock. Do not add a second asset selector that can drift from the dropdown, do not
   change the planner or the orbits response, do not gate any rows behind the dock.

8. Tests: template-wiring that the dock holds the three parts; that the asset switch is
   built from the subscribed list and its selection calls the one asset handler (assert
   a single selection path, not two); that the worst-card fields are bound to the
   worst-conjunction data; that the partial-view chip still renders on a partial
   response. Run the existing ui tests and confirm green.

9. Rebuild the ui. The dock shows the asset switch (reselects and refreshes, stays in
   sync with the dropdown), the worst card (populated, updates on switch), and the
   control row, matching the concept. A partial response still shows the chip.
   Screenshot next to the concept dock. Record in docs/scrum-465/live_verification.md.
   Do not deploy to the KVM.

10. Commit off main, include docs/scrum-465/, push, open a PR against main. No
    attribution trailers, author John Avera <javera@xorbita.com>. Report the PR number
    and head SHA.

Then Cowork runs the verification-first review before John merges.
