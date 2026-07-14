# Validity Method and IOD Confidence Floor
**SCRUM-333 Deliverables 3 and 6 · xOrbita Inc · Internal / Confidential**  
Owner: Sreejit · Reviewer: Minh · APS 3.0

---

## 1. Validity Method

### 1.1 Purpose

Before any Pc threshold is consulted, the system checks whether the
collision probability was earned by the observation geometry. If not
earned, the event cannot be actioned autonomously and drops to M4.

This is the safety primitive that prevents the satellite from acting on
a confident-looking Pc that the tracking geometry never actually supported.

### 1.2 Method Recommendation: Observability Consistency Check

**Recommended for v1:** Fisher-information observability consistency check.

**Rejected for v1:** Possibility-theory / ESPF. Too computationally heavy
for onboard use at this stage. Requires multiple phenomenologies not yet
available. Revisit when onboard IOD matures and sensor fusion is in place.

### 1.3 Mathematical Definition

The observability Gramian over the tracking arc $[t_0, t_f]$:

$$
\mathbf{W}(t_0, t_f) = \int_{t_0}^{t_f}
\mathbf{\Phi}^T(\tau, t_0)\,\mathbf{H}^T(\tau)\,\mathbf{R}^{-1}\,
\mathbf{H}(\tau)\,\mathbf{\Phi}(\tau, t_0)\, d\tau
$$

where:
- $\mathbf{\Phi}(\tau, t_0)$ is the STM from $t_0$ to $\tau$ (CW dynamics)
- $\mathbf{H}(\tau)$ is the measurement Jacobian at epoch $\tau$ (reused from EKF)
- $\mathbf{R}$ is the measurement noise covariance

Project $\mathbf{W}$ onto the conjunction plane to get $\mathbf{W}_{CP}$,
then compute its condition number:

$$
\epsilon = \frac{\lambda_{\min}(\mathbf{W}_{CP})}{\lambda_{\max}(\mathbf{W}_{CP})}
$$

Validity decision:

$$
\text{validity} =
\begin{cases}
\texttt{EARNED} & \text{if } \epsilon \geq \epsilon_{\min} \\
\texttt{NOT\_EARNED} & \text{if } \epsilon < \epsilon_{\min}
\end{cases}
$$

### 1.4 Per-Regime Epsilon Threshold

| Orbit regime | $\epsilon_{\min}$ | Rationale |
|---|---|---|
| LEO | 0.20 | Conservative. Rejects degenerate geometries and short tracking arcs without escalating well-tracked conjunctions. Tighten when onboard IOD matures. |
| MEO / GEO | YOUR INPUT (Minh sets floor) | Out of scope for APS 3.0. |

The epsilon threshold is an engineering recommendation. Minh sets the
final floor value per MAF §5.

### 1.5 Weak Direction Reporting

When $\epsilon < \epsilon_{\min}$, the validity assessor identifies which
RTN directions have poor observability using the eigenvectors of
$\mathbf{W}_{CP}$ corresponding to small eigenvalues. These are reported
in the `weak_directions` field so APS can surface them to the operator
and inform replanning.

Typical failure mode for CDM-based LEO conjunctions: poor radial
observability from short tracking arcs. Cross-track is typically the
best-resolved direction.

### 1.6 Architecture Note

The Validity Assessor is its own AVERA microservice, separate from the
monitor (enforce). It is a distinct trust layer; the monitor enforces
the safety floors, and the Validity Assessor certifies the probability.

The validity verdict is carried into the GNC command via the
`ValidityVerdict` field in `gnc_interface.yaml`.

### 1.7 Interface Field

```json
"validity": {
    "status": "EARNED" | "NOT_EARNED",
    "epsilon": 0.73,
    "epsilon_threshold": 0.20,
    "weak_directions": ["radial"],
    "phenomenologies_used": ["TLE"]
}
```

`phenomenologies_used` reflects current data sources. For APS 3.0 on
Space-Track / UDL data this will typically be `["TLE"]`. Expands to
`["optical", "RF", "LIDAR"]` as onboard sensing matures.

---

## 2. IOD Confidence Floor

### 2.1 Purpose

A separate quality gate on the IOD solution itself. Even if the validity
gate passes (good observation geometry), a poor IOD fit means the state
vector is unreliable. The IOD confidence floor rejects solutions where
fit quality is insufficient before they propagate into Pc computation.

### 2.2 Existing Quality Infrastructure

The IOD solver (`services/tracker/iod.py`) already implements hard
rejection thresholds:

```python
MAX_ACCEPTED_RMS_RESIDUAL_ARCSEC = 900.0
MAX_ACCEPTED_TOTAL_RESIDUAL_ARCSEC = 1800.0
```

These are outer bounds; solutions outside them are rejected entirely by
the solver. The IOD confidence floor sits inside these bounds as a
tighter operational gate.

### 2.3 Recommended Floor

$$
\text{IOD confidence floor} : \text{rms\_residual\_arcsec} \leq 300 \text{ arcsec}
$$

One third of the hard solver rejection threshold, providing a meaningful
quality band. Solutions between 300 and 900 arcsec are technically
accepted by the solver but are operationally too uncertain to act on
autonomously.

| RMS residual | Status | Action |
|---|---|---|
| <= 300 arcsec | CONFIDENT | Proceed to validity gate |
| 300 to 900 arcsec | DEGRADED | Escalate to M1 Watch, do not act autonomously |
| > 900 arcsec | REJECTED | Solver rejects outright |

### 2.4 Relationship to Validity Gate

Two separate checks in sequence:

```
IOD solution produced
    |
    v
IOD confidence check
rms_residual <= 300 arcsec?
    | YES                    | NO
    v                        v
Validity gate            Escalate to M1
epsilon >= 0.20?         (do not act)
    | YES      | NO
    v          v
Proceed    Drop to M4
to Pc      escalate
```

The IOD confidence floor is the earlier gate (fit quality).
The validity gate is the later gate (geometric diversity).
Both must pass before Pc is consulted.

### 2.5 Interface Field

The IOD confidence result is carried as a field in the APS risk inputs:

```json
"iod_confidence": {
    "status": "CONFIDENT" | "DEGRADED" | "REJECTED",
    "rms_residual_arcsec": 187.3,
    "floor_arcsec": 300.0,
    "observations_used": 3,
    "method_used": "gauss_herrick_gibbs"
}
```

### 2.6 Note for Minh

The numeric floor value (300 arcsec) is an engineering recommendation.
Minh sets the final floor value per MAF §5 and §6. The floor should be
revisited once real observation data from the sensor suite is available
to calibrate against actual IOD performance.

---

## 3. Open Assumptions

| Item | Assumption | Resolution |
|---|---|---|
| $\epsilon_{\min}$ for MEO/GEO | Not defined | Out of scope for APS 3.0 |
| IOD floor value | 300 arcsec recommended | Minh sets final value |
| Multiple phenomenologies | TLE only for APS 3.0 | Expands with onboard sensing |
| Epsilon calibration | Not yet calibrated against historical CDM outcomes | APS 3.0 gap 6.5 |
