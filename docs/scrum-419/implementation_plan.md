# SCRUM-419 Implementation Plan — Retire the Space-Track poll control and relabel the CDM Operations panel

## Intent
Space-Track is retired as a data source. Remove the Space-Track CDM poll control from the CDM Operations panel and relabel the panel so it reflects the actual architecture: LeoLabs is the live source (fetched by the planner during /v1/evaluate), and the inject path is the offline reference. UI-only change, no backend behavior change. Low priority, not a demo blocker.

## Context (what these things are)
- The Space-Track poll (the POLL button, POST /api/ingest/poll to the ingest /cdm/poll) was the original intended live CDM source, gated behind SPACETRACK_LIVE_POLLING_ENABLED (default off, SCRUM-329). On prod it 503s because the gate is off, not because of credentials. LeoLabs replaced it as the live source and does not use this poll.
- The CDM store (ADR-008), INJECT EXAMPLE CDM (TIROS 4), VIEW CDM STORE, RUN TIROS 4 E2E TEST and the Decision Log Lookup all stay. They are the offline reference and audit tooling, independent of Space-Track.

## Scope — the delta

### Remove (services/ui/app/templates/index.html)
- The #cdmNoradRow block: the "Asset NORAD ID (for Space-Track CDM poll)" label, the #assetNoradInput field, and the #pollBtn POLL button.
- triggerCdmPoll() and the POST /api/ingest/poll call it drives.
- The pollBtn-enable wiring (onNoradChange) and the poll-only copy ("No poll run yet"). The pollLastResult line is also written by injectExampleCdm, so keep that line and repoint its label to the inject result rather than a poll.

### Preserve carefully
- assetNoradId is used beyond the poll: it is set to '226' in injectExampleCdm and read by updatePublishedPc (around line 1374). Do NOT remove assetNoradId globally. Keep the variable and its inject and updatePublishedPc uses; remove only the poll input binding and the POLL button. The evaluate path already ignores assetNoradId (it reads the Scenario | Live Asset selector), so removing the poll does not change evaluate.

### Relabel (the source-mode line and info modal)
- #cdmSourceMode / loadSourceMode currently resolves UDL live, then Space-Track live, then reference CDM. Drop the Space-Track live tier. Show the reference CDM (TIROS 4) state when inject is active, and keep the line consistent with the top LEOLABS LIVE badge so the panel reads LeoLabs-live for the evaluate source. Remove any language that implies a Space-Track poll is needed.
- Update openCdmOpsInfoModal (the CDM Operations field reference) to drop the Space-Track poll description.

### Backend (leave, note)
- The ui /api/ingest/poll proxy and the ingest /cdm/poll endpoint can stay dormant (gated off) for this UI-scoped change. A full Space-Track code retirement (the poll endpoint, spacetrack_client, the source-mode live tier, SPACETRACK_* handling) is out of scope here and would be its own ticket. The credential revoke is SCRUM-430.

## Boundaries
- UI only. Do not change evaluate, the store, inject, the TIROS test, or the decision-log path.
- Do not remove assetNoradId where inject and updatePublishedPc use it.
- Do not touch the LeoLabs live path or the scenario path.

## Test and verification
- The panel no longer shows a POLL button or a Space-Track-poll NORAD field, and no copy implies a Space-Track poll.
- INJECT EXAMPLE CDM, VIEW CDM STORE, RUN TIROS 4 E2E TEST and Decision Log Lookup all still work.
- The source-mode line no longer offers a Space-Track live tier and reads consistently with the LEOLABS LIVE badge.
- Load the dashboard and click through scenario and live-asset modes to confirm nothing regressed (browser check).

## Build order
1. Remove the #cdmNoradRow and the POLL button, keeping assetNoradId and its inject and updatePublishedPc uses.
2. Remove triggerCdmPoll and the /api/ingest/poll call; tidy the poll-only copy.
3. Relabel the source-mode line and the info modal to drop the Space-Track tier.
4. Browser-verify the panel and both modes.
