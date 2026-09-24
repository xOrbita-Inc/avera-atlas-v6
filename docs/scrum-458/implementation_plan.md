# SCRUM-458 implementation plan: stop the screen dropping conjunctions on non-PSD covariance

Ticket: https://xorbita.atlassian.net/browse/SCRUM-458
Branch off main. Backend only, planner. Bug fix, safety-relevant. Part of
SCRUM-433. Relates to SCRUM-452, SCRUM-454, SCRUM-442.

## The bug

On the local stack (screening 603312) the on-demand screen returned 1255 result
CDMs and skipped 348 of 349 conjunctions, parsing only 1. Every skip is
leolabs_screening_cdm_skipped, reason "SAT1 covariance is not positive
semidefinite after rotation", min eigenvalues from -0.88 to -1.0e5 m^2.

SAT1 is our uploaded primary. The SCRUM-452 covariance grows strongly anisotropic
over 72 h, kilometres along-track and metres radially. LeoLabs writes it into the
result CDM at fixed precision, and when leolabs_cdm_parser reads it back and
rotates it, the small axes go slightly negative. _assert_symmetric_psd uses a very
tight relative tolerance (_PSD_EIG_RTOL 1e-8) and raises, so parse_leolabs_cdm
fails and run_screening skips the event.

## Why this is a safety bug, not cosmetic

Skipping conjunctions makes the screen under-report conflicts. A screen that drops
348 of 349 events and then reports on the 1 that survived can read CLEAR when the
truth is NOT CLEAR. That is the dangerous direction. No conjunction may be silently
dropped from a screen that then certifies clear.

## The fix, with the safety direction kept straight

The repair must never shrink a real uncertainty toward a clear. Clipping a negative
eigenvalue to zero makes the covariance less uncertain in that direction, which
raises the Mahalanobis distance and pushes toward CLEAR. That is fine for a
negative that is pure numerical or CDM-precision noise, and dangerous for a
negative that is genuinely large. So the fix is threshold-based.

1. Small, numerical non-PSD: repair and keep. When the most negative eigenvalue is
   within a small relative band of zero (propose repair when
   |min_eig| <= REPAIR_RTOL * max_eig, REPAIR_RTOL on the order of 1e-2; the
   observed negatives are about 0.5 percent of max, so this covers them; confirm
   the exact value at review), clip the negative eigenvalues to zero, rebuild the
   symmetric covariance, and return a valid parse so the event stays in the
   conjunction set and is evaluated by the clear contract. Log at debug or info.

2. Genuinely non-PSD, beyond the band: do not repair toward clear and do not
   silently drop. Log a warning with the event id and the eigenvalue. The event
   still carries a geometric miss distance, which is covariance-free, so the miss
   limb of the clear contract still evaluates it. The covariance-dependent limbs
   (Pc, Mahalanobis) cannot be trusted, so the event is treated as not
   assessable-clear: it fails the screen closed rather than being dropped. The net
   property is that a conjunction the screen could not fully assess makes the
   screen NOT CLEAR, never a silent omission from a CLEAR result.

Put the repair on the covariance the consumers use, in the parse path where
_assert_symmetric_psd is today. Keep the symmetry check as a hard error (an
asymmetric matrix is a real parse fault, not precision noise); only the PSD limb
becomes repair-or-flag.

Confirm, and record in the live note, whether our primary is consistently SAT1 in
screening results, so the log messages name the right object.

## Out of scope

Do not change the SCRUM-452 covariance growth here. Reducing the anisotropy we
submit so the CDM round-trip loses less precision is a separate, optional
follow-up; the parser repair is the load-bearing fix and is safe regardless of
whose covariance is marginal, ours or a catalog secondary's.

Do not change the async work (SCRUM-456/457) or the seed priority (SCRUM-454).

## Tests

Offline, with fixtures:
- A CDM whose covariance is slightly non-PSD (a tiny negative eigenvalue) parses,
  is repaired to PSD, and the event is kept and evaluated.
- A CDM whose covariance is badly non-PSD (a large negative) is not silently
  dropped: it is flagged, and the screen fails closed on that event rather than
  omitting it from a clear result.
- The repair never turns a would-breach event into a clear one: a slightly
  non-PSD covariance on an event that breaches on miss distance still breaches.
- Regression: a screening fixture that today skips hundreds of events now returns
  the real conjunction set with repaired covariance, and the skip count drops to
  near zero for numerical non-PSD.
- The existing live conjunction-list parse still passes unchanged.

Run python3 -m pytest services/planner and confirm green.

## Live acceptance

Rebuild the planner. Run one on-demand screen on the same asset that produced
screening 603312 and confirm the skip count is near zero, the screen returns the
real conjunction set rather than one event, and any genuinely non-PSD event is
logged and fails closed rather than dropped. Respect 3 creates per 2 minutes.
Record in docs/scrum-458/live_verification.md, including the confirmed SAT1
identity and the chosen REPAIR_RTOL.

## Commit

Branch off main. Include docs/scrum-458/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it checks the
repair-versus-flag threshold and the safety direction (no repair toward clear on a
real non-PSD), confirms no conjunction is silently dropped from a clear result,
runs the offline suite, and reviews the live skip-count drop.
