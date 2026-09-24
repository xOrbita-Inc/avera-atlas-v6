# SCRUM-449 live verification — encounter selection race

Run 2026-09-24 against the running local stack (ui container rebuilt from this
branch; planner untouched). Live Asset SWARM C (39453), 100 rows loaded.

The repository has no JS test harness, so the verification is a driven browser
check. Rather than eyeball the header, the repaint functions were wrapped to
record *which conjunction each post-await paint was for* — that makes
"the stale one never painted" an assertion rather than an impression.

    ['updateRiskBar','drawEncounter','updateDecisionCard'] wrapped to log {fn, object, norad}
    showEvaluateError wrapped to log every message

## 1. The reported bug, reproduced as a race and shown fixed

The plan's exact scenario: the worst-row evaluate is in flight, then the operator
picks a different row. Row 0 is STARLINK-30994 (58483, RED, the worst); row 3 is
STARLINK-38092 (100012).

    evaluateConjunction(rows[0], 0)   // the auto-evaluate, as it fires today
    ...300 ms...
    selectConj(3)                     // operator clicks another row

Result after both round-trips settled (~35 s):

| | |
|---|---|
| **stale worst-row ever painted** | **false** |
| paints recorded | 4, all STARLINK-38092 / 100012 |
| header | `Encounter: STARLINK-38092 (100012) — Miss 22589m` |
| `currentConj` | STARLINK-38092 / 100012 |
| EVALUATE FAILED messages | none |

Before the fix this is precisely the case that repainted 58483 over the
selection.

## 2. Rapid selection — last one wins

Four selections ~250 ms apart, each a distinct secondary, far faster than a
~13 s evaluate:

    STARLINK-30994 (58483) -> STARLINK-38092 (100012)
    -> STARLINK-38296 (100376) -> ELECTRON KICK STAGE R/B (68293)

| | |
|---|---|
| scored paints (`updateRiskBar`, only a scored result reaches it) | **exactly one**, for ELECTRON KICK STAGE R/B |
| header | `Encounter: ELECTRON KICK STAGE R/B (68293) — Miss 20390m` |
| `selectedConjunction` | 27, the last pick |
| EVALUATE FAILED from the three aborted requests | none |

The four `drawEncounter` calls that *did* fire early are the pre-score row
geometry from `highlightRowGeometry` — the instant draw the plan explicitly says
to keep. They draw the row the operator just clicked, which is correct; it is the
post-await repaint that needed guarding.

## 3. 2D / 3D toggle

Toggling to 3D and back left `currentConj`, `selectedConjunction` and the header
unchanged on ELECTRON KICK STAGE R/B, with one `drawEncounter` for the same
object (the redraw on view change) and no errors.

## 4. Whole-panel consistency

Screenshot after the burst: header, decision-card TARGET, the 2D geometry labels
and the risk summary all read ELECTRON KICK STAGE R/B (68293), Pc 3.10e-9, miss
20390 m, covariance GOOD from live LeoLabs CDM covariance. Nothing on screen
belongs to a previously selected object.

## Regression

    python3 -m pytest services/planner    1456 passed, 2 skipped   (unchanged)

The diff is `services/ui/app/templates/index.html` only.

## 5. The stale alert chip — found during this work, then fixed

The `#encounterAlert` chip kept the previous object's text whenever a scored
result was neither RED nor AMBER. `.encounter-alert` is `display:none` unless it
also carries `.red` or `.amber`, and **three** places reset the class to hide the
chip without touching `textContent`: `updateRiskBar`, the source-switch clear,
and `clearAll`. After the rapid-selection burst it still read
`AMBER — STARLINK-38092 (100012)` while the panel showed ELECTRON KICK STAGE R/B.

It was never user-visible — computed `display: none`, bounding width 0 — so the
chip was only correct by virtue of being invisible. Show it in any other state,
or read the DOM, and it names an object the panel is no longer about. On a
ticket whose whole subject is "the panel names the wrong conjunction", leaving
that in place was the wrong call.

Fixed by routing all three sites through one `setEncounterAlert(risk, label)`,
so the class and the text can no longer drift apart. Below the watch line, or on
a clear, the chip is hidden **and** emptied.

Verified twice:

**Directly, over the whole state surface.**

| call | class | display | text |
|---|---|---|---|
| `setEncounterAlert('RED', …30994)` | `encounter-alert red` | block | `RED ALERT — STARLINK-30994 (58483)` |
| `setEncounterAlert('AMBER', …38092)` | `encounter-alert amber` | block | `AMBER — STARLINK-38092 (100012)` |
| `setEncounterAlert('GREEN', …68293)` | `encounter-alert` | none | *(empty)* |
| `setEncounterAlert(null)` | `encounter-alert` | none | *(empty)* |

**Through the real path**, on the transition that exposed it. Selecting row 0
scored AMBER, giving `AMBER — STARLINK-30994 (58483)`, visible. Selecting row 12
scored GREEN:

    text ''            (was: 'AMBER — STARLINK-30994 (58483)')
    display none
    header  Encounter: STARLINK-38296 (100376) — Miss 24675m

`keptPreviousObjectText: false`.

## Limits of this verification

- **The tab reported `document.visibilityState: "hidden"`** throughout. Several
  attempts to foreground it, including AppleScript raising the window, did not
  change what the automation tab reported. Unlike the globe work this does not
  undermine the result: the fix is about `fetch`/`await` ordering and DOM writes,
  neither of which is throttled the way `requestAnimationFrame` is, and the
  ~13-22 s real round-trips completed normally throughout. A human should still
  click through it once on a visible tab.
- One asset (SWARM C) and one session's conjunction set.
- No automated coverage: there is no JS test harness in this repo, so nothing
  here will stay verified on its own.
