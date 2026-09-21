Implement SCRUM-419 in this repo. Space-Track is retired as a data source; this retires the Space-Track poll control in the CDM Operations panel and relabels the panel to match the LeoLabs-live architecture. UI only.
Ticket: https://xorbita.atlassian.net/browse/SCRUM-419

READ FIRST: docs/scrum-419/implementation_plan.md.

Branch off main: feature/scrum-419-retire-spacetrack-poll.

Do, in services/ui/app/templates/index.html:
1. Remove the #cdmNoradRow block (the "Asset NORAD ID (for Space-Track CDM poll)" label, #assetNoradInput, and the #pollBtn POLL button), and remove triggerCdmPoll() and the POST /api/ingest/poll call it drives. Tidy the poll-only status copy ("No poll run yet"); keep the pollLastResult line since injectExampleCdm also writes it, and repoint its label to the inject result.
2. Keep assetNoradId and its non-poll uses. It is set in injectExampleCdm and read by updatePublishedPc, so do not remove the variable globally; remove only the poll input binding and the POLL button.
3. Relabel the CDM source-mode line (#cdmSourceMode / loadSourceMode) to drop the Space-Track live tier, and update the CDM Operations info modal (openCdmOpsInfoModal) to drop the Space-Track poll description, so nothing implies a Space-Track poll is part of the LeoLabs live path. Keep the line consistent with the top LEOLABS LIVE badge.
4. Leave the CDM store, INJECT EXAMPLE CDM (TIROS 4), VIEW CDM STORE, RUN TIROS 4 E2E TEST and the Decision Log Lookup as-is; they are the offline reference and audit tooling.

Do not touch: the evaluate path, the scenario or live-asset selectors, the store, or the decision-log path. The ui /api/ingest/poll proxy and the ingest /cdm/poll endpoint can stay dormant; a deeper Space-Track code retirement is a separate ticket, and the credential revoke is SCRUM-430.

Constraints:
- UI-only change, no backend behavior change.
- NO AI attribution in commits or the PR (repo rule).
- Verify in the browser that the panel no longer shows a POLL control or Space-Track-poll copy, and that inject, VIEW CDM STORE, the TIROS 4 test, both scenario and live-asset modes, and the decision-log lookup all still work.

When done, open a PR against main and report the URL, the exact index.html changes, and a note or screenshot confirming the panel reads LeoLabs-live with no Space-Track poll, so I can review.
