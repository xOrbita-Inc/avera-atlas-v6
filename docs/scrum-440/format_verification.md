# SCRUM-440 verification — the emitted ephemeris against the real LeoLabs shape

440 stops at the JSON, so there is no live acceptance to report; that arrives on
the first real submit in SCRUM-441. What *can* be checked now is whether what we
emit is the same object LeoLabs itself hands back, and that is checked here
against the wire rather than against the format note.

## The target shape, confirmed live

The spike note (`claude/SCRUM-439_spike_findings_2026-09-22.md`) is a project
document and is not in this repo, so its format claim was re-confirmed directly:
a live `GET /catalog/objects/L3969/states` on 2026-09-24 from inside the planner
container.

`frames.EME2000` carries four keys:

    position            (3,)    metres
    velocity            (3,)    metres/second
    covariance          (6, 6)  m^2 and (m/s)^2
    covarianceExtended  (8, 8)

So the plan's claim holds, and with a detail worth stating: LeoLabs exposes
**both** a 6x6 `covariance` and an 8x8 `covarianceExtended`, and the 6x6 is
exactly the top-left 6x6 of the extended form (verified with `np.allclose`). The
6x6 under the key `covariance` is therefore LeoLabs' own representation, and it
is what this builder emits. The extra two rows of the extended form are not ours
to fill and are not emitted.

Measured structure of the live 6x6, which is what the builder mirrors:

    symmetric                     True
    PSD (min eigenvalue)          1.55e-06
    position block sigmas         5.78, 3.70, 6.40      -> metres
    velocity block sigmas         8.19e-3, 5.89e-3, 5.37e-3  -> metres/second
    ordering                      [r, v]: position top-left, velocity bottom-right

The sigma magnitudes are how the units were pinned: metres against a position
given in metres, m/s against a velocity in m/s.

## What the builder emits

    build_screening_ephemeris('2026-09-24T12:00:00Z',
                              [6792.0, 0.0, 0.0], [0.0, 5.0, 5.8], P)

with `P = [[0.04, 0.01, 0], [0.01, 0.09, 0.02], [0, 0.02, 0.16]]` km^2:

```json
{
 "frame": "EME2000",
 "covarianceFrame": "EME2000",
 "states": [
  {
   "timestamp": "2026-09-24T12:00:00Z",
   "position": [6792000.0, 0.0, 0.0],
   "velocity": [0.0, 5000.0, 5800.0],
   "covariance": [[ 40000.0,  10000.0,       0.0, 0.0, 0.0, 0.0],
                  [ 10000.0,  90000.0,   20000.0, 0.0, 0.0, 0.0],
                  [     0.0,  20000.0,  160000.0, 0.0, 0.0, 0.0],
                  [     0.0,      0.0,       0.0, 0.0, 0.0, 0.0],
                  [     0.0,      0.0,       0.0, 0.0, 0.0, 0.0],
                  [     0.0,      0.0,       0.0, 0.0, 0.0, 0.0]]
  }
 ]
}
```

Unit conversions, each done in exactly one place and each tested:

| quantity | in | out | factor |
|---|---|---|---|
| position | 6792.0 km | 6 792 000.0 m | x1e3 |
| velocity | 5.0 km/s | 5000.0 m/s | x1e3 |
| covariance | 0.04 km^2 | 40 000.0 m^2 | x1e6 |

0.04 km^2 reads back as a 200 m one-sigma, which is the sanity check a human can
do at a glance and which the suite asserts.

At the defaults the file is **865 states, 365 KB**, built and serialised inside
the rebuilt planner container.

## The window

- **Horizon 72 h.** The SCRUM-381 screen runs across
  `OperatorPolicy.max_hours_before_tca`, which `authorization_envelope` locks at
  `LOCKED_MAX_TCA_HOURS = 72.0`. `secondary_horizon.py` does not define a horizon
  of its own -- it takes `horizon_hours` from `atlas_artifact.py`, which passes
  the policy value -- so the constant here is that value, and a caller holding a
  live policy should pass its own rather than rely on the two staying in step.
- **Step 300 s, a judgement rather than a measurement.**
  `secondary_horizon._TCA_BRACKET_STEP_S` is 60 s but is explicitly a root-finding
  bracket for stationary points of relative distance, not a statement about how
  densely a trajectory should be described, so it is deliberately not reused.
  300 s over 72 h gives roughly 18 states per LEO revolution.

  **Open question for 441:** the right cadence depends on how LeoLabs interpolates
  between submitted states, which is not in anything available here. The plan
  asserts a coarse step is fine because LeoLabs re-screens continuously across the
  submitted states; that is from the spike and is not verified here. If it turns
  out they chord linearly between states, 300 s is too coarse for LEO and the step
  should drop. It is a parameter, so that is a one-line change.

## The flagged decision: 3x3 position covariance into a 6x6

Implemented as recommended and kept in one small function,
`covariance_6x6_from_position_3x3`: position block from P_post converted km^2 to
m^2, velocity block and cross terms zero, the same covariance on every state.

This is a floor, not a model, and it is the thing to confirm before 441 relies on
it. Two better options are named in the code and deliberately not built: a real
6x6 if GNC's OD ever supplies one, or propagating P_post along the trajectory
with `aps_math.cw_phi_full` so the emitted covariance grows per state. Over a 72 h
horizon position uncertainty genuinely does not stay constant, so repeating one
covariance understates it at the far end of the window.

**For John or Sreejit to confirm or override.**

## Refusing to fabricate

`post_burn_covariance_km2`'s own documented rule is that `None` means unknown and
must not be read as zero. The builder honours that by refusing: a `None`,
all-zero, non-finite, asymmetric, non-PSD or wrong-shaped covariance raises
`LeoLabsEphemerisError` rather than emitting a zero block. A zero covariance would
submit a screening against a trajectory declared exactly known, and it would look
like a successful submission. Validation runs before any propagation, so there is
no half-built file.

## Tests

    python3 -m pytest services/planner    1509 passed, 2 skipped   (+40)

The new file covers the emitted shape and key names, each unit conversion, the
zero velocity block, the first state being the post-burn state unpropagated,
ISO-8601 Z and monotonic timestamps, states matching `kepler_propagate` exactly
(so the ephemeris describes the same trajectory the internal screen does rather
than a second opinion), and every refusal path.

## Not verified here

- **That LeoLabs accepts the file.** That is 441's first live submit, by design.
- **The screening request body and status enum** -- the doc gap the spike
  identified, and a 441 concern.
- **The interpolation question** above, which decides whether 300 s is right.
