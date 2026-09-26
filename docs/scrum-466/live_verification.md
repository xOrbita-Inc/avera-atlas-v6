# SCRUM-466 live verification: LeoLabs attribution banner

Branch `scrum-466-leolabs-attribution`, off main at `ddbc31f`.
Verified 2026-09-26 against the local Docker stack, ui rebuilt from this branch.
Not deployed to the KVM; that stays John's manual step.

## What shipped

`services/ui/app/templates/index.html` only. No planner change, no response change,
no new flag.

The pill is the first child of `.header-right`, before `#sysStatus`:

    <span class="attrib-pill" id="leolabsAttrib"><span class="attrib-dot"></span>UTILIZING LEOLABS PULSE SPACE SAFETY SERVICE</span>

Styling is mockup A: `.attrib-pill` with the `.sys-status` metrics so it sits level
with the status pill and the clock, in the ARBITER cyan rather than the status green.

Two deliberate choices worth flagging for review:

- The plan's markup put the dot span and the words on separate source lines. They
  are on **one** line here. The requirement is that the element's text content is the
  exact phrase; a line break between them would put a newline inside that text and
  would also stop the phrase being a contiguous run in the served bytes. On one line
  the empty dot span contributes nothing and the text content is exactly the phrase.
- The mockup styles the dot as `.attrib-pill .dot`. `class="dot"` is already used six
  times in this template for the pipeline nodes, so the dedicated `.attrib-dot` class
  from the plan is used instead, avoiding the collision.

## The phrase is stated verbatim

Served bytes from the running container:

    $ curl -s http://localhost:8080/ | grep -c "UTILIZING LEOLABS PULSE SPACE SAFETY SERVICE"
    1

Exactly one occurrence. The explanatory HTML comment above the pill deliberately does
**not** repeat the phrase, so the shipped page holds a single copy and a compliance
grep cannot return a false positive from a comment that has drifted from the markup.

In the browser, measured on the live page:

| property | value |
|---|---|
| `textContent` | `"UTILIZING LEOLABS PULSE SPACE SAFETY SERVICE"` |
| matches required phrase (case-insensitive) | true |
| visible (display/visibility/opacity/box) | true |
| parent is `.header-right` | true |
| precedes `#sysStatus` in document order | true |
| truncated (`scrollWidth > clientWidth`) | false |
| overlaps title / status / clock | false / false / false |
| colour | `rgb(42,158,148)` on `rgba(79,248,232,0.15)`, border `rgba(79,248,232,0.28)` |
| dot | `rgb(79,248,232)` |

Vertical centres of the pill, `SYSTEM ONLINE` and the clock are all `23.5px` in a
48px header, so the three sit on one line. (An earlier check compared `top` rather
than centre and reported them as misaligned; that was the measurement being wrong,
not the layout, since the clock is 18px tall against the pills' 22px.)

Header capture for the LeoLabs submission, 1600px wide at 2x:
`docs/scrum-466/leolabs-attribution-header.png`.

## Always on, in both modes

Not gated to data, mode, fetch or flag. Verified by driving the real page:

- **Scenario mode** (default): present, visible, verbatim, 339px wide.
- **Live Asset mode**: clicked the `Live Asset` source button, confirmed it became
  active and the asset picker appeared, then selected `SWARM A — 39452`. Source note
  read `SOURCE: LEOLABS LIVE — real_cdm covariance — SWARM A (39452)`. Pill unchanged:
  present, visible, verbatim, 339px.
- **Live LeoLabs data on screen**: clicked `FETCH LIVE CONJUNCTIONS`, which returned
  **100 rows of real LeoLabs conjunctions** (first row `STARLINK-37550 15596
  5.77e-08 T+10.9h NOMINAL`, 1-100 of 101). Pill still present, visible and verbatim.
  This is the moment the credit exists for, and it is there.

A first attempt to switch modes clicked an element matching the uppercase rendered
text and silently did nothing; the mode was still Scenario, so that reading was
discarded and the switch redone against the real `.src-btn` control and confirmed
active before re-measuring.

## Responsive: it moves, it never abbreviates

Two breakpoints, both of which only relocate the credit. Neither hides nor shortens it.

- **`max-width:1000px`** — the header is allowed to grow past 48px and wrap, and the
  pill takes its own row beneath the status and clock, right-aligned. The wrap is
  forced by a zero-height `.header-right::after` flex item rather than by giving the
  pill `flex-basis:100%`, so the pill keeps its own width instead of stretching into
  a full-width bar. (The first version did stretch; it looked like a banner rather
  than a pill and was changed.)
- **`max-width:700px`** — `.header-right` takes the full header width. Below roughly
  this point the status and clock alone no longer size the cluster wide enough to
  hold a ~340px nowrap credit, and without this the phrase ran past the right edge.

Measured by capturing the running container headlessly at each width:

| viewport | credit | note |
|---|---|---|
| 1600 | inline, right of the title | mockup A |
| 1100 | inline, right of the title | |
| 1000 | own row under status/clock | full phrase |
| 900 | own row | full phrase |
| 760 | own row | full phrase |
| 700 | own row, cluster full width | full phrase |
| 600 | own row, cluster full width | full phrase |

Captures: `docs/scrum-466/leolabs-attribution-responsive.png`.

**Known limit, pre-existing and not introduced here.** Below about 600px the *whole
dashboard* overflows horizontally, because `.main` holds fixed-width columns
(285 + 469 + 340px). At 420px the pipeline bar, the left sidebar and the right
decision panel are all clipped along with the right of the header. The credit is
clipped there exactly as every other element is, and horizontal scrolling reveals it.
Fixing the dashboard's minimum width is out of scope for this ticket. The demo runs
on a wide screen.

## Test

`services/ui/tests/test_leolabs_attribution.py`, 15 tests. It asserts against the
page **rendered** through the app's own Jinja environment with the index endpoint's
real context, not against the template source, because the rendered bytes are what a
LeoLabs reviewer sees.

It pins three distinct failure modes:

1. the words being edited, abbreviated or reworded;
2. the phrase being **split across elements** — the required string is reconstructed
   from the parsed DOM and compared whole, and separately the raw served bytes must
   contain it as one contiguous run;
3. the pill being **gated** — the header block must carry no Jinja statement tag, no
   script may reference the element, and no rule may hide it at any breakpoint.

Mutation-checked; every one of these is caught:

| mutation | result |
|---|---|
| pill removed | 11 failed |
| phrase split with `<b>LEOLABS PULSE</b>` | 2 failed |
| phrase abbreviated | 4 failed |
| moved after `#sysStatus` | 1 failed |
| wrapped in `{% if leolabs_assets %}` | 1 failed |
| `display:none` in the narrow media query | 1 failed |
| hidden by script at runtime | 1 failed |
| restyled in the status green | 1 failed |

The split case is the one that matters most: wrapping part of the phrase in a `<b>`
leaves `textContent` **identical**, so only the contiguous-bytes assertion catches it.

## Suites

    services/ui/tests        126 passed
    services/planner/tests  1829 passed, 2 skipped

## Note on the local environment, not a defect in the image

`GET /` raises in this checkout under pytest: local starlette is 1.0.0, which dropped
the legacy positional `TemplateResponse("index.html", {...})` call that
`services/ui/app/main.py:81` uses. The image pins `fastapi==0.115.6`, where that call
is supported, and the running container serves `/` with HTTP 200 (293,671 bytes). So
this is a local version drift, not a shipping bug, and modernising the call is out of
scope for a template-only ticket. The test therefore renders through the app's own
Jinja environment, which is version-stable. Worth a separate ticket before anyone
else trips over it.
