# SCRUM-436 implementation plan

Stop the dashboard logging "Planner call failed: t_burn_utc must be before
t_ca_utc" on load. Small UI fix in services/ui/app/templates/index.html.

## The bug (confirmed in code, main 5bd7057)

On load, DOMContentLoaded (around line 2837) runs refreshPipeline() and
refreshData(), then auto-evaluates the top conjunction:
`evaluateConjunction(conjunctionData.conjunctions[idx] || [0], idx)`.

In evaluateConjunction the scenario (non-live) branch builds the request with:
- satellite.t_burn_utc = new Date(Date.now() + 2*60*60*1000)  // now + 2h
- conjunction.t_ca_utc = new Date(Date.now() + conj.time_to_tca_s*1000)

When the top scenario conjunction is under two hours to TCA -- which the default
scenario on load is -- t_ca_utc is earlier than t_burn_utc, so the maneuver
scorer raises "t_burn_utc must be before t_ca_utc" (maneuver_scorer.py:968) and
the evaluate 422s. The live branch is fine: it omits t_burn_utc and the planner
defaults it to six hours before the CDM's TCA. The scenario branch cannot omit
it, because the surrogate path requires it, so it must send a valid value.

The same fixed now+2h burn is in getPayloadContent (around line 2526), which
builds the payload-preview modal, so the previewed payload can also show a burn
after the TCA.

## The fix

Place the burn before TCA instead of at a fixed offset from now, in both
places.

- In evaluateConjunction's scenario branch, compute the TCA once and derive the
  burn from it:
    const tcaMs = Date.now() + conj.time_to_tca_s * 1000;
    // SCRUM-436: the burn must be strictly before TCA. Six hours before, or
    // halfway between now and TCA when the conjunction is nearer than that, so
    // it is always after now and before TCA regardless of how close the
    // conjunction is.
    const burnMs = tcaMs - Math.min(6*60*60*1000, (tcaMs - Date.now()) / 2);
  Then t_burn_utc = new Date(burnMs).toISOString() and
  t_ca_utc = new Date(tcaMs).toISOString().
- Guard the degenerate case: if conj.time_to_tca_s is missing or <= 0 (TCA at or
  in the past), do not fire the evaluate for that row rather than sending an
  invalid window. The auto-evaluate on load should pick the first row with a
  positive time_to_tca_s, or skip if none.
- Apply the same tcaMs / burnMs computation in getPayloadContent so the preview
  matches what is sent.

Keep everything else, the live branch and all other request fields, unchanged.

## Acceptance

- Loading the dashboard produces no "t_burn_utc must be before t_ca_utc" in the
  console, and the on-load auto-evaluate of the top conjunction returns a
  decision rather than EVALUATE FAILED.
- A scenario preset whose top conjunction is under two hours to TCA evaluates
  cleanly.
- The payload-preview modal shows a burn before the TCA.

## Verification

Browser: load the dashboard, confirm zero console errors on load and a decision
card rather than EVALUATE FAILED, then select a near-TCA scenario conjunction
and confirm it evaluates. This is the exact symptom the ticket describes.

## Files

- services/ui/app/templates/index.html (evaluateConjunction scenario branch;
  getPayloadContent)
