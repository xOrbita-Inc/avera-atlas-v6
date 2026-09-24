# SCRUM-448 live verification — covariance ellipsoids on the globe

Run 2026-09-24 on the local Docker stack with **both** containers rebuilt
(`docker compose up -d --build ui planner`), against the real LeoLabs API.
Asset: SWARM C (39453).

## Confirm the data first

Parsed the fixture CDM inside the planner container before writing any endpoint
code, per the kickoff:

    primary/SAT1   3x3 float64, symmetric (max|A-A^T| 4.7e-10), finite, PSD
                   eigenvalues [6.29, 501.8, 1.313e7] m^2
                   sigmas      [3623.8, 22.4, 2.5] m
    secondary/SAT2 3x3 float64, symmetric (max|A-A^T| 7.5e-9), finite, PSD
                   eigenvalues [251.0, 1060.0, 1.610e8] m^2
                   sigmas      [12688.2, 32.6, 15.8] m

So `cov_eci_pos_m2` is exactly what the plan said: a 3x3 position covariance in
m^2, symmetric-PSD, already in ECI. No new physics was needed.

## The finding that changed the renderer

The plan calls for the ellipsoids to be "exaggerated to a visible minimum" with a
persistent "uncertainty not to scale" note. Measured against the **live** feed
rather than the fixture, that premise does not hold:

| | one-sigma major axis |
|---|---|
| smallest drawn object | 5.8 km |
| median | 144.6 km |
| largest (ELECTRON KICK STAGE R/B) | 1,569.4 km |

At globe scale (1 scene unit = 6,371 km), a median 3-sigma major axis is already
**0.068 scene units** — roughly four times a RED marker — and the largest is
**0.74 Earth radii**. There is nothing to exaggerate. Any magnification above
1x would bury the globe under the uncertainty volumes.

So the ellipsoids are drawn at **true scale**, `GLOBE_COV_EXAGGERATION = 1.0`,
at the conventional 3-sigma. The constant is still named and the caption reads
the scale off it, so a future dataset that genuinely needs magnification changes
one number and the note stops claiming true scale by itself. The caption reads
`3σ CDM position uncertainty — true scale`, which is the honest version of the
plan's intent: the operator is told what the volume is and at what scale.

The covariances are also wildly anisotropic (ratios of 166:1 up to 39,000:1
across the drawn set), so an ellipsoid renders as a long thin sliver along-track.
That is what the measurement says, and it is deliberately not rounded off.

## Endpoint

    GET /v1/leolabs/orbits?primary_norad=39453   ->  200 in 17.5 s

    drawn 20, every one carrying cov
    asset.cov sigmas  [2300.9, 13.6, 4.4] m
    asset.cov_epoch_utc 2026-09-24T20:35:54Z   (a CDM epoch, not the state epoch)

Status contract unchanged: 200/404/422 re-checked, feed-off still 503, and the
503 body carries no asset or covariance data.

## Browser, on a FOREGROUND tab

The plan called this out specifically, and it mattered: the tab reported
`document.hidden = true` at first and had to be brought forward
(`osascript -e 'tell application "Google Chrome" to activate'`) before
requestAnimationFrame ran — 56 frames in 900 ms once foreground, versus 1 and
then nothing while hidden.

| check | result |
|---|---|
| COVARIANCE is the fifth control | `AUTO-ROTATE, LABELS, NOMINAL, COVARIANCE, FOCUS WORST` |
| default off | button not `.on`, `globeShowCovariance` false, 0 of 21 visible, note hidden |
| on draws every ellipsoid | 21 visible (20 objects + asset), note shows |
| scale is correct | mesh scale re-derived independently from the payload: exact match to 1e-12 |
| orientation is correct | mesh local X vs the dominant ECI eigenvector mapped to scene: **dot = 1.0** |
| NOMINAL takes its ellipsoids with it | haze off → 17 nominal ellipsoids stop rendering, 4 non-nominal keep rendering; the ellipsoids' own `visible` flags are untouched, so COVARIANCE stays authoritative |
| refetch honours both toggles | with COVARIANCE on and NOMINAL off: 21 flagged visible, 4 actually rendering |
| no accumulation on refetch | `globeCovParts` stays 21 and `assetMarker.children` stays 1 across a refetch |
| true sigma is reported | FOCUS WORST: `Focused STARLINK-30994 — RED — 1σ 5.8 km / 31 m / 4 m` |

The asset-marker check is there because the asset marker is persistent — built
once in `initGlobe` and never removed — so its ellipsoid would have survived a
refetch and a second one been added beside it, invisible to the toggle. That bug
existed in the first version and is fixed by detaching in `clearGlobeTracks`.

## Not verified

- **An object with no usable covariance, against real data.** All 20 live objects
  carried one, so the no-`cov` path is covered by the offline test only
  (`test_an_object_with_no_usable_covariance_is_drawn_without_one`) and has not
  fired on the real feed.
- **Any asset other than SWARM C.**
- **How this reads at other zoom levels.** Checked at the default camera and
  after FOCUS WORST; a 0.74-Earth-radius ellipsoid on the worst-case object was
  not inspected close up.

## Suites

    python3 -m pytest services/planner      1456 passed, 2 skipped   (+28)
