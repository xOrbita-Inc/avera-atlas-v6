# The authorized-to-execute payload, SCRUM-379 to SCRUM-382

What 379 emits when the decision state machine reaches M3, and what 382 consumes
to assemble the GNC command. This is the gap Sreerjit flagged: nothing in the
repo assembled it.

Defined in `services/planner/common/decision_state_machine.py` as
`AuthorizedExecution`. Emitted on `MonitorDecision.authorized_execution`, and
returned by `/v1/evaluate` as the top-level `authorized_execution` key. The key
is absent, not null, on any evaluation that did not reach M3.

## Shape

```json
{
  "conjunction_id": "conj-2026-0302-001",
  "dv_eci_km_s": [0.0, -0.00035, 0.0],
  "dv_magnitude_m_s": 0.35,
  "t_burn_utc": "2026-04-01T15:45:00Z",
  "mode": "M3",
  "authority_level": "L2",
  "envelope_version": "env-v1-sha256-abc123456789",
  "approval_basis": "l2_veto_window_expired",
  "validity_status": "EARNED",
  "validity_epsilon": 0.73,
  "covariance_source": "real_cdm",
  "authorized_at_utc": "2026-04-01T13:45:00Z"
}
```

| Field | Source | Notes |
|---|---|---|
| `conjunction_id` | `ManeuverScoringResult.conjunction_id` | |
| `dv_eci_km_s` | `ManeuverScoringResult.dv_eci_km_s` | The scorer's burn, unchanged. 379 never modifies a maneuver. |
| `dv_magnitude_m_s` | `ManeuverScoringResult.dv_magnitude_m_s` | Already checked against the envelope's effective cap for the granted level. |
| `t_burn_utc` | `ManeuverScoringResult.t_burn_utc` | ISO-8601 UTC, Z-suffixed. |
| `mode` | The state machine | Always `"M3"`. Present so a consumer never has to infer it. |
| `authority_level` | Compiled envelope (SCRUM-375) | `"L1"` or `"L2"`. |
| `envelope_version` | Compiled envelope | The version the decision rested on. |
| `approval_basis` | The state machine | `"l1_operator_approval"` or `"l2_veto_window_expired"`. Section 3's two M2 to M3 rows, and nothing else. |
| `validity_status` | SCRUM-378 verdict | Always `"EARNED"` here; recorded rather than assumed. |
| `validity_epsilon` | SCRUM-378 verdict | |
| `covariance_source` | The planner's own resolution | `"real_cdm"` for an L2 auto-execute unless the operator explicitly acknowledged a surrogate. |
| `authorized_at_utc` | The monitor's clock | When the decision was made, not when the burn runs. |

## Rules the shape encodes

Nothing optional and nothing derived. A field that is not known at authorisation
time does not belong here, because a consumer cannot tell a missing value from a
real one. Every field is populated on every emission.

The dv travels as the scorer produced it. 379 decides whether a burn may be
executed; it does not re-plan one.

`approval_basis` is why the burn is lawful, not merely that it is. An operator
reading an audit trail should not have to reconstruct whether a burn was
approved by a person or by a veto window that closed.

## What is deliberately not here

Command IDs, acknowledgements, GNC framing, and the post-burn report are 382's,
per section 10's `commands_and_acknowledgments`, `actual_vs_predicted`,
`post_maneuver_od` and `residual_risk` field producers. 379 stops at authorising.
