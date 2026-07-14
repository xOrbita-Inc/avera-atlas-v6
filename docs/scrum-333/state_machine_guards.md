# Decision State Machine: Guards, Veto Windows and Recovery
**SCRUM-333 Deliverable 5 · xOrbita Inc · Internal / Confidential**  
Owner: Sreejit · Reviewer: Minh · APS 3.0

---

## 1. Overview

The decision state machine governs how the satellite moves between flight
modes (M0 to M4) in response to conjunction events, operator commands, and
monitor certification results.

The planner service (`server.py`) is stateless; each evaluation call is
independent. The state machine is a net new component in APS 3.0. It owns
the current flight mode, persists state across events, and drives the GNC
command flow.

Any undefined or ambiguous transition defaults to ESCALATE to M4.
Full audit logging on every transition per MAF §10.

---

## 2. States

| State | Meaning |
|---|---|
| M0 | Nominal: no active conjunction |
| M1 | Watch: conjunction flagged, being assessed |
| M2 | Staged: burn ready, awaiting approval or inside veto window |
| M3 | Executing: burn in progress |
| M4 | Safehold / Abort: reversionary, escalate to ground |

---

## 3. Transition Table

| From | To | Guard conditions (ALL must be true) |
|---|---|---|
| M0 | M1 | Pc >= `pc_monitor_threshold` (1e-5) |
| M1 | M0 | Pc < `pc_monitor_threshold` for 2 consecutive evaluations |
| M1 | M2 | Pc >= `pc_maneuver_threshold` (1e-4) AND IOD confidence CONFIDENT AND validity EARNED AND envelope satisfied AND secondary conflict clear AND slew feasible AND authority L1 or L2 |
| M1 | M4 | IOD confidence DEGRADED or REJECTED OR validity NOT_EARNED OR secondary conflict not clear OR TCA < `min_hours_before_tca` (4.0 hrs) with Pc still above monitor threshold |
| M2 | M3 | L1: GNCApprovalCommand received AND approval_accepted = true AND slew still feasible AND validity still EARNED |
| M2 | M3 | L2: veto_window_close_utc passed with no GNCVetoCommand AND slew still feasible AND validity still EARNED |
| M2 | M2 | L2: GNCVetoCommand received AND veto_accepted = true AND Pc still >= `pc_maneuver_threshold` AND new CDM available with `data_age_s` within freshness bound AND validity still EARNED on new CDM. If no fresh CDM or validity fails on new CDM, go to M4. |
| M2 | M0 | Pc drops below `pc_monitor_threshold` while staged (event resolved before burn) |
| M2 | M4 | Slew infeasible at veto_window_close_utc OR validity gate fails mid-staging OR envelope violated OR L1 approval_accepted = false OR watchdog expired |
| M3 | M0 | GNCReport received AND execution_status = NOMINAL AND m2_post_estimated > Mahalanobis safe threshold |
| M3 | M1 | GNCReport received AND execution_status = NOMINAL AND Pc still >= `pc_monitor_threshold` after post-burn replan (replan required) |
| M3 | M4 | GNCReport received AND execution_status = ABORTED or PARTIAL AND residual Pc still elevated |
| M4 | M0 | Ground command only: explicit operator clearance via ARBITER |

---

## 4. Guard Condition Definitions

### 4.1 Envelope Satisfied
All of the following hold:
- Orbit regime matches the active envelope
- Authority level matches the operator grant
- Pc >= operator-configured `pc_maneuver_threshold` and <= max_dv cap for the authority level (L1: 2.0 m/s, L2: 0.5 m/s)
- `data_age_s` is within the freshness bound (recommended: 86400s / 24 hrs for Space-Track CDM; Minh sets floor)
- `covariance_source` is `real_cdm` (surrogate_identity triggers a warning and requires explicit operator acknowledgement before L2 auto-execute)
- `v_remaining_m_s` > `dv_magnitude_m_s` + `v_reserved_m_s`
- Envelope version is cryptographically valid and not expired

### 4.2 Secondary Conflict Clear
`SecondaryConflictCheck.secondary_conjunction_clear = true` from the
`atlas_artifact` in the planner response. If the check was not performed
(`secondary_check_performed = false`), treat as NOT CLEAR and escalate.

### 4.3 Slew Feasible
Current time satisfies:

$$
t_{now} \leq t_{burn} - (t_{slew} + t_{settle} + t_{margin})
$$

where:
- $t_{slew} + t_{settle}$ = 120s (OPEN ASSUMPTION: bus spec pending)
- $t_{margin}$ = 30s
- $t_{burn}$ <= `latest_burn_utc`

Checked at the M1 to M2 transition and continuously monitored in M2.

### 4.4 Validity Still EARNED (M2 continuous check)
The validity gate is re-evaluated on each new CDM update while in M2.
If a new CDM arrives and validity drops to NOT_EARNED, transition to M4
immediately regardless of veto window status.

### 4.5 Mahalanobis Safe Threshold
Post-burn M3 to M0 transition requires:

$$
m^2_{post} > m^2_{safe}
$$

Recommended: $m^2_{safe} = 25$ (5-sigma separation in the conjunction
plane). Minh sets the final value.

---

## 5. Veto Window Durations Per Authority Level

| Level | Veto window | Auto-execute trigger |
|---|---|---|
| L0 | N/A (advisory only, no GNC command issued) | N/A |
| L1 | No veto window. Holds until GNCApprovalCommand or TCA closes. | Operator approval only |
| L2 | $t_{veto} = \max(t_{review},\; t_{slew} + t_{settle} + t_{margin})$ | veto_window_close_utc passes with no veto |
| L3 | No veto window (gated to later phase) | Pre-authorized, single-shot |

**L2 veto window placeholders:**

| Parameter | Value | Status |
|---|---|---|
| $t_{review}$ | 300 s (5 min) | Operator minimum review time |
| $t_{slew} + t_{settle}$ | 120 s | OPEN ASSUMPTION: bus spec pending |
| $t_{margin}$ | 30 s | Safety buffer |
| **Minimum veto window** | **300 s** | Driven by $t_{review}$ |

**L1 timeout:** If no GNCApprovalCommand is received before
`latest_burn_utc`, GNC transitions to M4 and notifies ARBITER. The
operator missed the window; the event is escalated, not silently dropped.

---

## 6. Persistence and Recovery

### 6.1 Comms Gap

A comms gap is defined as no uplink from ground for longer than
`comms_gap_threshold` (recommended: 600s / 10 min; Minh sets floor).

| State at gap start | Behaviour during gap | Recovery on comms restore |
|---|---|---|
| M0 | Continue monitoring. No action required. | Resume normal. |
| M1 | Continue monitoring with last known CDM. Flag staleness if data_age_s exceeds freshness bound. | Re-evaluate with fresh CDM on restore. |
| M2 (L1) | Hold staged. Cannot execute without operator approval. If TCA closes before comms restore, go to M4. | On restore: if still within window, re-present to operator. |
| M2 (L2) | Continue veto countdown with onboard clock. Auto-execute at veto_window_close_utc if no veto received. This is the intended L2 behaviour; autonomy was pre-granted for exactly this case. | Post-burn: send GNCReport on restore. |
| M3 | Continue burn execution. Burns are short; a comms gap during the burn is expected to resolve before the post-burn report timeout. | Send GNCReport on restore. |
| M4 | Hold safehold. Do not exit M4 autonomously. | Operator clears via ARBITER on restore. |

### 6.2 Reboot Mid-Event

The state machine must persist the current flight mode and active
GNCCommand to durable onboard storage before any transition. On reboot:

1. Read persisted state.
2. If M0 or M4: resume normally.
3. If M1: re-evaluate conjunction with last known CDM. Treat as a fresh
   M0 to M1 trigger if Pc is still above the monitor threshold.
4. If M2 or M3: check whether the burn window is still open.
   - If `t_now < latest_burn_utc - (t_slew + t_settle + t_margin)`:
     re-enter M2 and re-present to operator (L1) or resume veto countdown (L2).
   - If `t_now >= latest_burn_utc`: transition to M4. Window closed
     during reboot; do not attempt the burn. Escalate to ground.
5. Log the reboot event with pre-reboot state in the evidence package.

### 6.3 Partial Subsystem Failure Mid-Event

| Failure | State | Action |
|---|---|---|
| Navigation filter degraded (EKF divergence) | Any | Go to M4. Cannot certify state estimate. |
| Thruster fault (no ignition confirmation from IMU) | M3 | execution_status = ABORTED. Go to M4. Send GNCReport. |
| Partial burn (IMU confirms cutoff before commanded delta-v) | M3 | execution_status = PARTIAL. Go to M4 if residual Pc elevated. Send GNCReport with dv_achieved_before_abort_m_s. |
| Attitude control failure (slew error exceeds threshold) | M2 or M3 | Go to M4. abort_reason = ATTITUDE_ERROR_EXCEEDED. |
| Validity assessor service unavailable | M1 or M2 | Treat as NOT_EARNED. Go to M4. Cannot certify Pc. |
| Ingest service unavailable (no fresh CDM) | M1 | Continue monitoring with last CDM if data_age_s is within freshness bound. If stale, go to M4. |
| Monitor certification failure (any reason) | M2 | Go to M4. Do not execute an unverified burn. |

Any failure that prevents the monitor from certifying safety AND
probability-validity at commit AND at execution goes to M4. The system
does not degrade gracefully into an uncertified burn.

---

## 7. Audit Logging

Every transition produces a tamper-evident append-only log entry
consistent with MAF §10. Minimum fields per entry:

```json
{
    "event": "state_transition",
    "from_mode": "M1",
    "to_mode": "M2",
    "trigger": "pc_above_maneuver_threshold",
    "conjunction_id": "conj-2026-0302-001",
    "timestamp_utc": "2026-04-01T13:45:00Z",
    "pc_at_transition": 0.000312,
    "validity_status": "EARNED",
    "epsilon": 0.73,
    "authority_level": "L2",
    "envelope_version": "env-v1.2-sha256-abc123",
    "software_version": "aps-3.0.0"
}
```

Undefined or ambiguous transitions log as:

```json
{
    "event": "state_transition",
    "from_mode": "M2",
    "to_mode": "M4",
    "trigger": "undefined_transition_escalate",
    "detail": "Guard condition not matched, defaulting to ESCALATE per MAF §8"
}
```

---

## 8. Open Items for Minh

| Item | Recommendation | Decision |
|---|---|---|
| Data freshness bound | 86400s (24 hrs) for Space-Track CDM | Minh sets floor |
| Mahalanobis safe threshold for M3 to M0 | $m^2_{safe} = 25$ | Minh sets floor |
| Comms gap threshold | 600s (10 min) | Minh sets floor |
| Slew + settle time | 120s placeholder | Bus spec pending (Blackwing) |
| Re-veto behaviour in M2 | Hold staged and re-enter veto window only if fresh CDM is available within freshness bound and validity is still EARNED. Otherwise go to M4. | Confirm with Minh |
