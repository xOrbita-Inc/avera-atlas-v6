# SCRUM-462 live verification — one toast, one persistent chip, source-neutral

Run 2026-09-25 on the local stack against the real LeoLabs API, ui rebuilt from this
branch and driven in Chrome. No KVM deploy; that is John's manual step.

## Branch note, which a reviewer needs first

This ticket redesigns code that is **not on main yet**: SCRUM-461 is PR #125, still
open. Branching off main as the kickoff says would have meant writing the toast and
chip from scratch with no banners to remove, and two conflicting implementations of
the same warning once both merged. So this branches off `scrum-461-partial-banner`
instead. **PR #125 must merge first**; after it does, this PR is the 462 delta alone.
Flagged rather than done silently, because it is a deviation from the kickoff.

## What was replaced

SCRUM-461's two in-panel banners (`#conjPartialBanner` over the table,
`#globePartialBanner` over the globe), their `.partial-banner` styles,
`renderPartialBanner` and `hidePartialBanner`, are gone. Asserted gone by test, not
by eye: five parametrised checks that none of those names survives anywhere in the
template, because a leftover banner would show alongside the toast and there would
be three signals instead of one.

Kept, because they were right: the partial-ness test
(`data.partial === true || data.complete === false`, absent fields meaning
complete), the honest denominator, the number formatter, and the discipline of
clearing on every fetch, switch and clear.

## Source-neutral copy

Before: *"Messages arrive in **LeoLabs** order, not risk order…"*
Now: *"Messages are not in risk order, so this is a prefix of the window and not its
top."*

The warning will outlive the provider, so it no longer names one. Asserted two ways:
the harness joins the toast text, the chip tooltip, both sentence forms and the
heading and checks that `LeoLabs`, `leolabs`, `LEOLABS` and `Pulse` appear in none
of them; and a template test greps the copy-building functions directly.

Live, the toast read:

> **⚠ PARTIAL VIEW — THE WORST CONJUNCTION MAY NOT BE SHOWN**
> Only **3,000** of **52,212** conjunction messages in the window were read before
> the fetch time limit was reached. Messages are not in risk order, so this is a
> prefix of the window and not its top. Narrow the window or the reporting volume
> for a complete view.

`mentions_leolabs: false`, checked on the live DOM.

## The toast — `docs/scrum-462/partial-toast.jpg`

SWARM B, LIVE ASSET, fetched through the real dropdown and FETCH button. The toast
slides in at the top of the viewport, spans the app rather than a panel, carries the
real counts and an **x**. Underneath it, unaffected: 100 real STARLINK rows and the
pager reading `1-100 of 412`. The rows are never gated behind the warning.

One toast for the app, not one per panel: asserted in the DOM,
`document.querySelectorAll('.partial-toast').length === 1`, and the template test
pins that it is `position:fixed` and declared between `<body>` and the conjunction
toolbar.

## The chip, and why it is the load-bearing half — `partial-chip-after-toast.jpg`

The toast auto-dismisses after 8 s. A truncated view stays partial for as long as it
is on screen, so something has to outlive it or the condition is silently forgotten
the moment it fades — which is the whole reason SCRUM-461 exists. The chip is that
something, and the screenshot is taken deliberately **after** the toast has gone:
the amber `⚠ PARTIAL VIEW` chip sits beside `1-100 of 412`, with the full sentence
and the counts in its tooltip.

Measured live across four states:

    immediately after the fetch     toast gone (8 s elapsed)   chip TRUE
    after the auto-dismiss          toast false                chip TRUE
    after clicking the chip         toast TRUE (brought back)  chip TRUE
    after closing with the x        toast false                chip TRUE

    chip tooltip still carries "3,000 of 52,212": true

Dismissing the notification is not the same as the view ceasing to be partial, and
only one of those is the operator's to decide. The code is shaped so that stays
true: `dismissPartialToast` does not mention the chip or the state,
`renderPartialChip` reads `partialViewState` and never the toast, and the
auto-dismiss timer body cannot reach the chip. Three template tests pin each, so a
later edit that looks harmless cannot quietly couple them.

## Both views — `partial-chip-globe.jpg`

The chip rides with the pager, which is on screen in the 2D table and the globe
alike, so the condition follows the operator across. In the globe view the chip is
still there beside `1-100 of 412` — and note there is no longer a banner over the
canvas, which is the point of moving to one app-level signal.

    globe view:  toast false, chip TRUE, "42 objects from 412 events"

42 rings drawn from a 412-event view whose window was read 3,000 of 52,212. Without
the chip that globe is simply a quiet sky.

## A complete asset shows neither

SWARM C, selected and fetched the same way:

    pager 1-100 of 102, 100 rows
    toast false, chip false, chip tooltip "", toast innerHTML ""

Not merely hidden — emptied, so there is no markup or stale tooltip left for a stray
class to reveal.

## Clearing

    asset switch (SWARM C -> SWARM B)   toast false, chip false, before the new fetch returned
    CLEAR button                        toast false, chip false, tooltip "", toast html ""

Both clear before every fetch as well, which matters because the list and globe
fetches have several paths that return early on a 404, 422, 503 or an unparseable
body. A stale "partial" over a fresh complete view is the failure this guards.

## One behaviour worth a reviewer's attention

Re-fetching the *same* partial view after the operator has dismissed the toast does
not push the toast back at them; a *different* truncation does. That is a deliberate
politeness and it is safe only because the chip is unaffected either way — the
condition is never hidden, only the notification is quieted. The harness pins both
halves.

## Toast timing

8 s, as the plan proposed. Long enough to read three lines, short enough not to sit
over the dashboard. The harness asserts it stays within 5–15 s, so the number can be
tuned without silently becoming either useless or permanent. It is a comfort
setting, not a safety one: the chip carries the condition afterwards.

## Tests

    python3 -m pytest services/ui/tests services/planner    1877 passed, 2 skipped
    node services/ui/tests/partial_warning_harness.mjs      64/64 checks passed

`partial_warning_harness.mjs` replaces the SCRUM-461 harness and extracts the real
functions from `index.html` with *controllable timers*, so it can fire the
auto-dismiss and assert the chip survives it — the property the whole redesign turns
on. It also covers the source-neutral copy, both truncation reasons, the honest
denominator, the six shapes of partial response that must all announce themselves,
and the pre-459 shape that must not.

`test_partial_warning.py` runs that harness under pytest (skipped without node) and
adds template-wiring checks that need no node: the old banners are gone, the copy
names no source and reads only the three backend fields, one toast and one chip
exist, both are hidden until shown, each fetch drives the warning from its own
response and clears before it fetches, the rows are not gated, and the chip/toast
independence described above. The harness proves the functions behave; these prove
they are connected and stay decoupled — the point SCRUM-461 made, that a render
function nothing calls still passes a pure harness.

## Not changed

No planner change, no change to the responses, no change to the fetch. The
SCRUM-459 fields and the honesty rule are exactly as they were; this is presentation.
