# Candidate Threshold Values and Authorization Envelope Tunables
**SCRUM-333 Deliverable 4 · xOrbita Inc · Internal / Confidential**  
Owner: Sreejit · Reviewer: Minh · APS 3.0

---

## 1. Purpose

Engineering-recommended candidate threshold values for the authorization
envelope (MAF §5). These are the tunables Minh sets the final action lines
against. Values marked **RECOMMEND** are engineering inputs. Values marked
**CONFIRM** are already in the stack and carried forward. Values marked
**OPEN** require Minh's decision before they can be locked.

All numeric values assume LEO operations on Space-Track / UDL conjunction
data. Per-regime extension to MEO/GEO is out of scope for APS 3.0.

---

## 2. Collision Probability Thresholds

Pc thresholds are per operator, not per authority level. The authority level
controls who approves action at whatever threshold that operator has
configured. Different operators can set different action lines within the
bounds Minh sets in the MAF as non-negotiable floors.

The values below are defaults from `operator_policy_leo.yaml`, carried
forward as the engineering-recommended starting point. They are
operator-configurable at onboarding. The MAF floor is the lower bound
no operator may go below.

| Parameter | Default value | Source | Notes |
|---|---|---|---|
| `pc_maneuver_threshold` | 1e-4 | CONFIRM (`operator_policy_leo.yaml`) | Action line. Triggers M1 to M2 transition. Operator-configurable; MAF sets non-negotiable floor. |
| `pc_monitor_threshold` | 1e-5 | CONFIRM (`operator_policy_leo.yaml`) | Watch line. Triggers M0 to M1 transition. Operator-configurable; must stay one order of magnitude below `pc_maneuver_threshold`. |
| `min_miss_distance_km` | 1.0 km | CONFIRM (`operator_policy_leo.yaml`) | Hard floor on acceptable miss distance. Operator-configurable upward; MAF sets floor. |
| `mahalanobis_screen_threshold` | 4.0 | CONFIRM (`operator_policy_leo.yaml`) | Skip events with Mahalanobis distance above this value as they are too far to be actionable. Operator-configurable. |
| `m2_safe_threshold` | 25.0 | RECOMMEND | Post-burn M3 to M0 condition. Equivalent to 5-sigma separation in the conjunction plane. Minh sets floor. |

---

## 3. TCA Horizon

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `min_hours_before_tca` | 4.0 hrs | CONFIRM (`operator_policy_leo.yaml`) | Latest acceptable burn ignition. Hard constraint enforced in `latest_burn_utc` in `TimingBudget`. |
| `max_hours_before_tca` | 72.0 hrs | CONFIRM (`operator_policy_leo.yaml`) | Earliest burn window open. Beyond this, Pc uncertainty is too high for meaningful action. |

---

## 4. Delta-v Caps Per Authority Level

Per MAF §2 and §6, max delta-v is defined per authority level. This is an
engineering input (MAF §11 "fixed floor + YOUR INPUT feasibility").

| Level | Max delta-v | Justification |
|---|---|---|
| L0 | N/A | Advisory only. No burn commanded. |
| L1 | 2.0 m/s | Operator explicitly approves every burn, so the full budget is available. Matches existing `max_dv_per_event_ms` in `operator_policy_leo.yaml`. Typical operational burns are 0.1 to 0.5 m/s; the 2.0 m/s ceiling covers worst-case geometries without forcing operator overrides. |
| L2 | 0.5 m/s | System auto-executes if no veto is received. This cap reflects what is safe to execute without explicit approval. Covers the vast majority of LEO CA maneuvers; PRISMA demonstrated effective avoidance well under this value. At typical small-sat power levels, 0.5 m/s is already a multi-minute burn, giving the operator meaningful veto time. Anything above 0.5 m/s at L2 should require either L1 approval or an explicit operator-raised cap at onboarding. |

**Open item:** Whether operators can raise the L2 cap at onboarding, and
what the ceiling on that raise is, is a product decision for Minh.

---

## 5. IOD Confidence Floor

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `iod_rms_residual_floor_arcsec` | 300 arcsec | RECOMMEND | One third of the hard solver rejection threshold (`MAX_ACCEPTED_RMS_RESIDUAL_ARCSEC = 900.0` in `services/tracker/iod.py`). Solutions between 300 and 900 arcsec are solver-accepted but operationally degraded; escalate to M1 Watch and do not act autonomously. Minh sets final floor. |

---

## 6. Validity Gate Threshold

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `epsilon_min_leo` | 0.20 | RECOMMEND | Condition number of the observability Gramian projected onto the conjunction plane. Rejects degenerate geometries and short tracking arcs. Conservative for APS 3.0 on Space-Track data. Tighten when onboard IOD matures. Minh sets final floor. |

---

## 7. Data Freshness Bound

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `data_freshness_bound_s` | 86400 s (24 hrs) | RECOMMEND | Maximum acceptable CDM data age at the time of autonomous action. Space-Track CDMs are typically updated every 8 to 24 hrs. Beyond 24 hrs, density model error and atmospheric uncertainty accumulate to a point where the Pc estimate is unreliable for autonomous action. Minh sets final floor. |

---

## 8. Secondary Conflict Gate

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `secondary_conflict_clear` | Must be `true` | CONFIRM (MAF §6 safety floor) | `SecondaryConflictCheck.secondary_conjunction_clear` from `atlas_artifact`. If the check was not performed (`secondary_check_performed = false`), treat as NOT CLEAR and escalate. Binary gate with no numeric tunable. |

---

## 9. Maneuver Frequency Limits

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `max_maneuvers_per_week` | 3 | CONFIRM (`operator_policy_leo.yaml`) | Fleet-level frequency limit. Prevents propellant depletion from over-reactive autonomy. |
| `v_reserved_m_s` | 5.0 m/s | CONFIRM (`satellite_capability.py`) | Reserved delta-v budget for non-avoidance operations. GNC avoidance layer never touches this. |

---

## 10. Envelope Compilation Rule

The authorization envelope is machine-generated from operator profiles
but never goes live automatically. Required before activation:

1. Deterministic compilation from controlled inputs
2. Validation against safety floors (MAF §6)
3. Human review and explicit approval
4. Cryptographic signing and versioning
5. Defined activation time and rollback path
6. Re-approval on any change

Any change to the envelope or monitor logic drops the system to L0 until
ground re-validates.

---

## 11. Summary Table

| Parameter | Value | Status |
|---|---|---|
| `pc_maneuver_threshold` | 1e-4 | CONFIRM |
| `pc_monitor_threshold` | 1e-5 | CONFIRM |
| `min_miss_distance_km` | 1.0 km | CONFIRM |
| `mahalanobis_screen_threshold` | 4.0 | CONFIRM |
| `m2_safe_threshold` | 25.0 | RECOMMEND (Minh sets floor) |
| `min_hours_before_tca` | 4.0 hrs | CONFIRM |
| `max_hours_before_tca` | 72.0 hrs | CONFIRM |
| Max delta-v L0 | N/A | CONFIRM |
| Max delta-v L1 | 2.0 m/s | RECOMMEND |
| Max delta-v L2 | 0.5 m/s | RECOMMEND |
| `iod_rms_residual_floor_arcsec` | 300 arcsec | RECOMMEND (Minh sets floor) |
| `epsilon_min_leo` | 0.20 | RECOMMEND (Minh sets floor) |
| `data_freshness_bound_s` | 86400 s | RECOMMEND (Minh sets floor) |
| `secondary_conflict_clear` | true (binary gate) | CONFIRM |
| `max_maneuvers_per_week` | 3 | CONFIRM |
| `v_reserved_m_s` | 5.0 m/s | CONFIRM |
