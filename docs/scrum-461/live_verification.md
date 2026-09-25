# SCRUM-461 live verification — a truncated view now announces itself

Run 2026-09-25 on the local stack against the real LeoLabs API, ui rebuilt from this
branch, driven in Chrome. No KVM deploy; that is John's manual step.

## What the backend sends, read from the SCRUM-459 code

Both the list (`server.py:1683`) and the globe (`server.py:2009`) responses carry
the same three fields, and `truncation` is `null` when complete:

    "complete": page.complete,
    "partial":  not page.complete,
    "truncation": {"cdms_pulled":    page.pulled_cdms,
                   "cdms_in_window": page.window_cdm_total,
                   "truncated_by":   page.truncation_reason}

The dashboard read none of them, so a truncated SWARM B listing rendered as an
ordinary list.

## Live responses, before touching the dashboard

    SWARM B list   partial=True   3,000 of 52,558 pulled, truncated_by=deadline
    SWARM B globe  partial=True   3,000 of 52,604 pulled, truncated_by=deadline
                                  counts: drawn=43, events_in_window=408
    SWARM C list   partial=False  truncation=null, 103 events

The globe line is the one worth pausing on: **43 rings drawn, from a window holding
52,604 conjunction messages of which 3,000 were read**. An uncluttered globe is the
most convincing "quiet sky" this dashboard can draw, and that is one.

(The window totals drift by a few dozen between calls because it is a live feed and
the window is relative to the wall clock. The banner always shows the numbers from
the response it is drawn for, never a remembered one.)

## The banner, driven through the real controls

LIVE ASSET, SWARM B selected from the asset dropdown, FETCH LIVE CONJUNCTIONS
clicked. Not by poking globals: an earlier attempt called `fetchLiveConjunctions()`
directly while the page was still in Scenario mode, which put a live banner over
scenario rows. That was my test being wrong, not the code, and it is exactly the
inconsistency the mode-switch hook prevents — but it was worth catching, because a
banner describing a different view than the one under it would be its own lie.

**The table** (`docs/scrum-461/partial-banner-table.jpg`):

> **⚠ PARTIAL VIEW — THE WORST CONJUNCTION MAY NOT BE SHOWN**
> Only **3,000** of **52,636** conjunction messages in the window were read before
> the fetch time limit was reached. Messages arrive in LeoLabs order, not risk
> order, so this is a prefix of the window and not its top. Narrow the window or the
> reporting volume for a complete view.

with, underneath it and unaffected:

    SOURCE: LEOLABS LIVE — real_cdm covariance — SWARM B (39451)
    1-100 of 408
    STARLINK-31075  11198 m  4.99e-04  T+84.4h   RED
    STARLINK-36467   4384 m  3.58e-04  T+102.6h  RED
    STARLINK-30531  18977 m  2.23e-04  T+106.3h  RED

100 rows still rendered. The banner sits with the rows, never in place of them: a
partial view is useful, it just must not claim to be whole.

**The globe** (`docs/scrum-461/partial-banner-globe.jpg`): the same banner across
the top of the canvas, opaque so it stays readable over the earth, and clear of both
`#globeMsg` (bottom-left, "43 objects from 408 events") and the control cluster
(bottom-right). Asserted identical, not just similar:

    globe innerHTML === table innerHTML   True

## A complete asset shows nothing new

SWARM C, selected and fetched the same way:

    source            SOURCE: LEOLABS LIVE — real_cdm covariance — SWARM C (39453)
    pager             1-100 of 103
    rows              100
    conj banner       shown=False, innerHTML=""
    globe banner      shown=False

Not merely hidden — emptied, so there is no markup left to be revealed by a stray
class.

## The banner never outlives the view it describes

Switching asset from SWARM B to SWARM C cleared both banners **before** the new
fetch returned:

    banners cleared on asset switch:  conj=False  globe=False

That matters because several paths in the list fetch return early on a 404, 422,
503 or an unparseable body. Without clearing up front, a banner from the previous
asset would sit over the next asset's table. Both fetches clear first, and both
clear paths (`clearAll`, `clearConjunctionsForSourceSwitch`) clear them too.

## The honesty rule, as implemented

`renderPartialBanner` shows the banner when the response says `partial === true`
**or** `complete === false`, so a response that flags either way is announced. A
response with neither field — the pre-SCRUM-459 shape — is treated as not partial,
which is correct for it and is the only silent case.

The denominator is the backend's or it is absent. When `cdms_in_window` is null the
sentence says "out of an unreported total" rather than reusing the pulled count,
which would quietly imply the window had been read whole. With no counts at all it
still warns, in words, without a dangling "of". Offline checks cover `null`,
`undefined`, `NaN`, `Infinity`, strings and objects reaching the formatter, and
assert no `NaN`/`undefined`/`null` can leak into the copy.

`truncated_by` is translated for the reader: `deadline` becomes "time limit",
`cap` becomes "size limit", and an unrecognised reason simply omits that clause
rather than printing a raw enum.

## Tests

    python3 -m pytest services/ui/tests services/planner    1856 passed, 2 skipped
    node services/ui/tests/partial_banner_harness.mjs       49/49 checks passed

`partial_banner_harness.mjs` extracts the real banner functions from `index.html`
and drives them against a stub DOM, the same approach as SCRUM-457's row harness, so
the test cannot drift from the template. It covers the copy, the counts, the
never-invented denominator, both truncation reasons, and six shapes of partial
response that must all announce themselves.

`test_partial_banner.py` runs that harness under pytest (skipped without node) and
adds template-wiring checks that need no node at all: both elements exist, they
share one style definition, the style is hidden by default, each fetch renders from
its own response and clears first, the rows are not gated behind the banner (the
table render precedes it with no `else` between), the markup sits outside the table
body, both clear paths hide it, and the copy reads only the three fields the backend
actually sends. The harness proves the functions behave; these prove they are
connected — a perfect render function nothing calls would pass the first and ship a
dashboard that still lies.

## Not changed

No planner change, no change to the responses, and no change to the fetch. This
reads and renders what SCRUM-459 already sends.
