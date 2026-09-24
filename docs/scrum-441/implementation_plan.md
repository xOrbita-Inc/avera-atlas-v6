# SCRUM-441 implementation plan: LeoLabs on-demand screening client and result parser

Ticket: https://xorbita.atlassian.net/browse/SCRUM-441
Branch off main. Backend only, planner. Part of the SCRUM-433 on-demand secondary
screen. Depends on SCRUM-440 (the ephemeris, now merged) and the SCRUM-439 spike
(the lifecycle, done). Wiring the result into the maneuver decision is SCRUM-442,
not this.

## What this is

A runtime that takes the post-burn ephemeris SCRUM-440 builds, submits it for an
on-demand screening against the full LeoLabs catalog, polls to completion,
retrieves the result CDMs, and parses them into our covariance-bearing conjunction
model. It handles the empty result and the error cases. The first live submit is
also where the request-body field names and the status enum are confirmed, and
where the two questions SCRUM-440 left open get answered.

## What already exists, so this is composition not new transport

- The client has the whole lifecycle: create_screening(body),
  get_screening_status(ids), get_screening_cdms(id) (paginated through the proven
  token path), and wait_for_screening(id, timeout, is_complete) with a jittered
  poll and the 3-per-2-minute screening limiter. Do not reimplement any of it.
- SCRUM-440 leolabs_ephemeris.build_screening_ephemeris produces the file to
  submit.
- leolabs_cdm_parser.parse_leolabs_cdm parses a LeoLabs CDM into a ParsedLeoLabsCDM
  with EME2000 covariance, and leolabs_conjunction_list gives the dedupe and row
  helpers. Screening results come back as CDMs on
  /screenings/{id}/cdms, so they parse the same way the live conjunction list does.
- The guard that will consume this, guard_secondary_clear in safety_monitor.py, and
  the SECONDARY_SCREEN_ENABLED flag in server.py, already exist behind the deferred
  switch. 441 produces what that guard reads; 442 connects them.

## The confirmed lifecycle (SCRUM-439, do not re-derive)

Create POST /catalog/conjunctions/screenings, poll GET .../screenings/{ids},
retrieve GET .../screenings/{id}/cdms. Rate limits are 3 screening creates per 2
minutes on top of the org-wide 4 requests/second, both already enforced in the
client. Latency is 30 s to 2 min, so poll, never block a request inline. The one
open item is the exact request-body JSON field names and the terminal status enum,
both isolated in the client behind the body dict and the injectable is_complete and
both confirmed on the first live submit here.

## Seam

New module services/planner/common/leolabs_screening.py, following the
leolabs_runtime / leolabs_evaluate pattern:

- build_screening_request(ephemeris, thresholds, primary_object) returning the
  create_screening body. Screen Against All Objects (the full LEO catalog, which is
  the wide secondary-screen volume), Use File Uncertainty (our uploaded covariance
  is the one screened), and the thresholds from our clear-definition contract: max
  miss distance, min probability of collision, max Mahalanobis distance, and the
  hard-body-radius override to our 15 m combined. Read those threshold values from
  the existing SCRUM-381 clear contract that guard_secondary_clear already uses,
  rather than inventing them. Keep every LeoLabs JSON field name in this one
  function, since these are the names confirmed on the first live submit.
- run_screening(ephemeris, ...) that submits with create_screening, waits with
  wait_for_screening, retrieves with get_screening_cdms, parses each result with
  parse_leolabs_cdm, dedupes to one row per event, and returns the parsed
  conjunctions with covariance. An empty result is a real and unremarkable clear
  sky, not an error. A timeout, a transport failure, a rate-limit refusal, or no
  on-demand access is a typed failure the caller can fail closed on, never a silent
  empty result, since a screen that could not run must not read as a clear screen.
- Behind LEOLABS_ENABLED, matching leolabs_runtime. The live submit additionally
  needs on-demand access, which is in the trial and extended for the Disrupt window.

## The two SCRUM-440 questions answered here

The first live submit is where these resolve, so build the live path to surface
them and record them in docs/scrum-441/live_verification.md:

- The 300 s ephemeris step. Confirm LeoLabs accepts the 440 file and, from the
  returned conjunction geometry, whether its interpolation between submitted states
  is fine at 300 s or wants a finer step. If finer, it is a one-line change in 440.
- The constant position-only covariance floor. Look at whether the screen result is
  sane given a covariance that does not grow over 72 h. This is the physics call
  flagged in 440. Do not decide it silently; record what the live result shows and
  hand it to John or Sreejit before 442 makes the screen load-bearing.

## Do not

- Do not wire the result into guard_secondary_clear or the decision path; that is
  442. 441 exposes a clean call and stops there.
- Do not change the 440 ephemeris builder or the internal secondary_horizon screen.
- Do not turn the pagination retrieved-vs-total count-mismatch warning into a hard
  error; the live catalog moves under a long pull and the warning is correct
  behaviour (SCRUM-439 finding).

## Tests

Offline, with a mocked client (no live LeoLabs in CI):
- build_screening_request produces the full-catalog, use-file-uncertainty body with
  the clear-contract thresholds and the 15 m HBR override.
- run_screening submits, polls to a terminal status, retrieves, and parses result
  CDMs into conjunctions with covariance, on a fixture screening-result CDM.
- An empty result returns an empty clear set; a poll timeout, a transport error, and
  a rate-limit refusal each raise the typed failure rather than returning empty.
- Injectable is_complete is exercised with a non-default status vocabulary so the
  enum is not hard-wired.

Run python3 -m pytest services/planner and confirm green.

The live acceptance is a real on-demand screening of a post-burn ephemeris returning
parsed conjunctions with covariance. Run it once on the stack with a real asset and
record the outcome, the confirmed field names, and the confirmed status enum in the
live-verification doc.

## Verify

Rebuild the planner container. The offline suite runs anywhere; the one live submit
needs LeoLabs on-demand access on the stack and is rate-limited to 3 creates per 2
minutes, so do it deliberately, not in a loop.

## Commit

Branch scrum-441-ondemand-screening-client off main. Include docs/scrum-441/ in the
commit. No attribution trailers, author John Avera <javera@xorbita.com>. PR against
main; report the PR number and head SHA.

Cowork runs the verification-first review before John merges, checks the request
body and the parse against real code, runs the offline suite, and reviews the live
first-submit result including the two SCRUM-440 questions.
