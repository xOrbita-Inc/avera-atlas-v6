# SCRUM-440 kickoff for Claude Code

Build the post-burn trajectory ephemeris for LeoLabs on-demand screening. This is
the LeoLabs JSON builder only; submission and polling are SCRUM-441. Ticket:
https://xorbita.atlassian.net/browse/SCRUM-440

Branch scrum-440-post-burn-ephemeris off main. Backend only, planner.

Read docs/scrum-440/implementation_plan.md first. It records what the SCRUM-439
spike already confirmed (the ephemeris format and that the client already has the
screening lifecycle), the inputs the planner already produces, and the one physics
decision.

Run in order:

1. Read docs/scrum-440/implementation_plan.md, and skim the SCRUM-439 spike note in
   the project (claude/SCRUM-439_spike_findings_2026-09-22.md) for the confirmed
   ephemeris format and the LeoLabs EME2000 covariance representation.
2. Add services/planner/common/leolabs_ephemeris.py with
   build_screening_ephemeris(epoch_utc, r_sat_km, v_sat_km_s, p_post_eci_km2,
   horizon, step) -> dict. It propagates the post-burn state with kepler_propagate
   (common.orbit_propagation, the propagator secondary_horizon.py already uses)
   across the SCRUM-381 screen window that secondary_horizon.py defines, and emits
   the LeoLabs JSON: frame and covarianceFrame EME2000, a states array of
   { timestamp (ISO8601 Z), position (m), velocity (m/s), 6x6 covariance }.
3. Build the 6x6 covariance from the 3x3 ECI position P_post in one small,
   documented helper: position block (top-left 3x3) = P_post converted km^2 to m^2,
   velocity block and cross terms zero, same covariance on each state. Name the two
   better options (a real 6x6 from GNC, or propagating P_post via
   aps_math.cw_phi_full) in the comment as future work, do not build them. This is
   the flagged decision; keep it swappable.
4. Convert units carefully: position km to m (x1000), velocity km/s to m/s (x1000),
   covariance km^2 to m^2 (x1e6). Match the exact JSON key names and covariance
   nesting to LeoLabs' own get_states EME2000 shape so what we emit equals what
   LeoLabs returns.
5. Do not submit, poll, or call create_screening; that is 441. Do not change
   kepler_propagate.
6. Add services/planner/tests/test_leolabs_ephemeris.py, offline: frame and
   covarianceFrame EME2000; a states array across the horizon at the step; position
   and velocity are the km inputs times 1000; the covariance top-left 3x3 equals
   P_post times 1e6 to tolerance and the velocity block is zero; the first state is
   the post-burn r/v at epoch_utc; timestamps ISO8601 Z and monotonic; a
   missing/degenerate P_post is handled honestly, not fabricated. Run
   python3 -m pytest services/planner and confirm green.
7. Commit on scrum-440-post-burn-ephemeris, include docs/scrum-440/ in the commit,
   push, open a PR against main. No attribution trailers, author John Avera. Report
   the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it reads the
emitted JSON against the confirmed LeoLabs format, checks the unit conversions, and
raises the 6x6 covariance construction for John or Sreejit to confirm before 441
relies on it.
