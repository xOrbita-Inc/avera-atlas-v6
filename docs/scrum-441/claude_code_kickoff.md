# SCRUM-441 kickoff for Claude Code

Build the LeoLabs on-demand screening client and result parser: submit the SCRUM-440
post-burn ephemeris for a full-catalog screening, poll, retrieve, and parse the
covariance-bearing results. Wiring it into the decision guard is SCRUM-442, not this.
Ticket: https://xorbita.atlassian.net/browse/SCRUM-441

Branch scrum-441-ondemand-screening-client off main. Backend only, planner.

Read docs/scrum-441/implementation_plan.md first. The transport already exists on the
client and the parser already exists, so this is composition, and the one open item
is confirming the request field names and status enum on the first live submit.

Run in order:

1. Read docs/scrum-441/implementation_plan.md, and skim the SCRUM-439 spike note in
   the project (claude/SCRUM-439_spike_findings_2026-09-22.md) for the confirmed
   lifecycle, rate limits, and the body-field mapping.
2. Add services/planner/common/leolabs_screening.py with two functions:
   build_screening_request(ephemeris, thresholds, primary_object), returning the
   create_screening body (Screen Against All Objects, Use File Uncertainty, the
   clear-contract thresholds of max miss / min PoC / max Mahalanobis, and the 15 m
   combined HBR override), with every LeoLabs JSON field name kept in this one place;
   and run_screening(ephemeris, ...) that submits with create_screening, waits with
   wait_for_screening, retrieves with get_screening_cdms, parses each result with
   parse_leolabs_cdm, dedupes to one row per event, and returns the parsed
   conjunctions with covariance.
3. Read the clear-contract threshold values from the existing SCRUM-381 path that
   guard_secondary_clear uses; do not invent them. Gate the module behind
   LEOLABS_ENABLED, matching leolabs_runtime.
4. Handle the cases honestly: an empty result is a clear sky and returns an empty
   set; a poll timeout, transport error, rate-limit refusal, or no on-demand access
   is a typed failure the caller can fail closed on, never a silent empty result.
   Do not turn the pagination count-mismatch warning into an error.
5. Do not touch guard_secondary_clear, the decision path, the 440 builder, or the
   internal secondary_horizon screen. 441 exposes a clean call and stops.
6. Add offline tests with a mocked client: the request body shape, the
   submit-poll-retrieve-parse happy path on a fixture result CDM, empty result,
   timeout, transport error, rate-limit refusal, and a non-default status vocabulary
   through is_complete. Run python3 -m pytest services/planner and confirm green.
7. Run one live on-demand screening on the stack with a real asset. Record in
   docs/scrum-441/live_verification.md: the confirmed request-body field names, the
   confirmed status enum, whether LeoLabs accepted the 440 ephemeris and how it
   interpolates at the 300 s step, and whether the result looks sane given the
   constant position-only covariance floor. Flag the covariance-floor finding for
   John or Sreejit; do not decide it here.
8. Commit on scrum-441-ondemand-screening-client, include docs/scrum-441/, push, open
   a PR against main. No attribution trailers, author John Avera. Report the PR
   number and head SHA.

Then Cowork runs the verification-first review before John merges: it checks the
request body and the parse against real code, runs the offline suite, and reviews the
live first-submit result and the two SCRUM-440 questions.
