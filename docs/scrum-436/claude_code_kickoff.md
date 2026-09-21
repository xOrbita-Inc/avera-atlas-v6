# SCRUM-436 kickoff for Claude Code

Fix the on-load "t_burn_utc must be before t_ca_utc" poller error. Full spec in
docs/scrum-436/implementation_plan.md. Read it first.

Branch: scrum-436-onload-burn-window off current main (5bd7057).

Do:
1. In services/ui/app/templates/index.html evaluateConjunction, in the scenario
   (non-live) branch, stop hardcoding t_burn_utc to now+2h. Compute
   tcaMs = Date.now() + conj.time_to_tca_s*1000 once, set
   burnMs = tcaMs - Math.min(6*60*60*1000, (tcaMs - Date.now())/2), and send
   t_burn_utc from burnMs and t_ca_utc from tcaMs, so the burn is always after
   now and strictly before TCA.
2. Guard time_to_tca_s missing or <= 0: do not fire the evaluate with an invalid
   window; the on-load auto-evaluate should pick the first row with a positive
   time_to_tca_s or skip.
3. Apply the same tcaMs/burnMs computation in getPayloadContent so the
   payload-preview modal matches what is sent.
4. Leave the live branch and all other fields unchanged.

Verify in the browser: load the dashboard, confirm no "t_burn_utc must be before
t_ca_utc" console error and a decision card rather than EVALUATE FAILED, and that
a near-TCA scenario conjunction evaluates cleanly. Run the planner suite from the
repo root (should be unchanged, this is UI-only). No AI attribution. Do not
merge; open the PR for John.

Report: files changed with line counts, the before/after of the on-load console,
and confirmation a near-TCA scenario evaluates.
