# SCRUM-465 live verification — the globe bottom dock

Run 2026-09-26 on the local stack against the real LeoLabs API, ui rebuilt from this
branch. No KVM deploy.

## The dock, against the concept

`docs/scrum-465/globe-dock.jpg`, bottom-right of the globe, three panels stacked in
the concept's order:

    [ SWARM C · 97 ] [ CRYOSAT 2 ] [ GRACE-FO 1 ] [ GRA… ]     <- asset switch

    STARLINK-38092                                  [ RED ]
    Worst of 97 conjunctions in the reporting volume
    Asset                                            SWARM C
    Miss                                            2.484 km
    Pc (LeoLabs)                                    4.96e-04
    TCA                                  2026-09-26 18:05:13Z
    Covariance                                   [● REAL CDM]

    [AUTO-ROTATE] [LABELS] [NOMINAL] [COVARIANCE] [FOCUS WORST]

Every field is from the live response. The panels are translucent with a blur, the
values are tabular-nums, and the current asset chip is the cyan pressed state.

## The design point the plan left open

The concept hard-codes two asset tabs; there are **ten** subscribed assets here. As
proposed, it is a **single-row strip that scrolls horizontally**, current asset
first, rather than tabs wrapping into several rows — it reads like the concept and
keeps the dock one panel tall. Confirmed live: 10 chips, SWARM C first and pressed.

## One selection state, verified both directions

The dock adds a second control over state the LIVE ASSET dropdown already owns, so
this was the part worth proving rather than asserting.

**Dropdown to dock**, before any fetch:

    dropdown set to 39453  ->  liveAssetNorad "39453", liveAssetName "SWARM C"
                               pressed chip: SWARM C

**Dock to dropdown**, clicking the CRYOSAT 2 chip:

    before  norad 39453  name "SWARM C"    dropdown 39453
    after   norad 36508  name "CRYOSAT 2"  dropdown 36508
            dropdown_matches_state: true
            pressed chip: CRYOSAT 2

There is one code path, not two: `selectGlobeAsset` writes the dropdown and then
calls `onLiveAssetChange`, the dropdown's own handler. It never assigns
`liveAssetNorad` itself, and a test pins that **no code outside `onLiveAssetChange`
assigns it at all**, so a future edit cannot quietly add a second path.

## A bug this found, live

After switching from SWARM C to CRYOSAT 2 the card still read:

    STARLINK-38092   RED   Worst conjunction in the reporting volume

That is SWARM C's conjunction, relabelled with CRYOSAT 2 in the Asset row — the
previous asset's encounter presented as this one's. The card reads `globeWorstMeta`,
which survived the switch because only a completed fetch clears it.

It is the same failure the 2D table clears itself to avoid, and the fix is in the
same place: the worst-conjunction state is now dropped in `onLiveAssetChange`, on
the one selection path, so the dropdown and the dock both get it. Re-verified:

    immediately after the switch:
      card "No data yet — fetch live conjunctions"
      globeWorstMeta cleared: true
      asset "CRYOSAT 2", dropdown 36508

    after the new fetch landed:
      worst      GRAVITAS
      sub        "Worst of 14 conjunctions in the reporting volume"
      Asset row  CRYOSAT 2

A regression test pins it.

## Counts are never invented

The concept prints a count on each tab. It could, with two assets and no live data;
ten assets cannot be counted without ten window pulls, and a guessed count on a
safety dashboard is the same class of lie as a fabricated total.

So a chip shows a count only for an asset actually fetched this session, written in
exactly one place — from `data.counts.events_in_window` of a real response. After
visiting three assets:

    SWARM B · 412    CRYOSAT 2 · 14    SWARM C · 97

and the remaining seven chips carry no count at all.

The card's empty state distinguishes the two facts that matter: with no asset
selected it says "Select a subscribed asset"; with an asset chosen but not yet
fetched, **"No data yet — fetch live conjunctions"**; only after a fetch that
returned nothing does it say "No conjunction in the reporting volume". Saying the
sky is empty when we have not looked is the dangerous direction.

## Partial-view honesty survives

On SWARM B, whose window is truncated (SCRUM-459):

    partial chip shown:  true   ("⚠ PARTIAL VIEW")
    card count line:     "Worst of 412 conjunctions in the reporting volume"
    chip:                "SWARM B · 412"

412 is the honest count of events this view actually holds, which on a truncated
window is the short count and not the whole 52k-CDM window. Both the card and the
chip use that number rather than inventing a total.

The SCRUM-461/462 chip lives with the pager in the Active Conjunctions panel, which
is outside the globe pane, so the dock cannot cover it — asserted by test on the
element ordering as well as observed live.

## One thing the dock forced

The SCRUM-448 covariance caption floated at a fixed `bottom:46px`, which cleared a
single-row control cluster. The dock is several panels tall, so that offset would now
land on top of it — the same collision SCRUM-463 hit and for the same reason. The
caption is now a row of the dock's control panel, above the buttons at any height.
Three tests pin the new placement.

## Tests

    python3 -m pytest services/ui/tests services/planner    1939 passed, 2 skipped

`test_globe_dock.py`, 33 cases: the dock holds the three panels in the concept's
order and is what the view shows and hides; the control row keeps its id and all five
handlers; **one selection path**, with the global assertion that nothing outside
`onLiveAssetChange` assigns the selection; reselecting the current asset is a no-op;
counts written in exactly one place and only from a response; the strip scrolls
rather than wraps; the card binds name, risk, miss, Pc, TCA and an honest covariance
badge; not-yet-fetched is not reported as an empty sky; the worst state is dropped on
switch; and the partial chip and toast still exist and are still driven by the globe
fetch.

## Not changed

No planner change, no change to the orbits response. This is layout and wiring over
data the page already had.
