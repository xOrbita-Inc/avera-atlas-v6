# SCRUM-462 implementation plan: partial-view warning as a top toast plus a persistent chip, source-neutral

Ticket: https://xorbita.atlassian.net/browse/SCRUM-462
Branch off main. UI only. Presentation change, safety-relevant. Part of SCRUM-433.
Relates to SCRUM-461 and SCRUM-459.

## What this is

SCRUM-461 shipped the partial-view warning as two in-panel banners, one over the
table and one over the globe, and its copy names LeoLabs. This ticket revises the
presentation: one top toast plus one persistent chip, and source-neutral copy. It
changes nothing about the SCRUM-459 fields it reads or the honesty rule; it only
changes how the warning looks and where it lives.

## The two problems being fixed

1. The copy names the source. It says "Messages arrive in LeoLabs order". The feed
   may come from another source later, so drop the name.
2. The placement is two in-panel banners. Replace them with one unobtrusive top
   toast and one small persistent chip.

## Copy, source-neutral

Build the sentence from the SCRUM-459 truncation fields, with no source name:

  PARTIAL VIEW — THE WORST CONJUNCTION MAY NOT BE SHOWN.
  Only {pulled} of {in_window} conjunction messages in the window were read before
  the fetch {time|size} limit was reached. Messages are not in risk order, so this
  is a prefix of the window and not its top. Narrow the window or the reporting
  volume for a complete view.

Keep the honest-denominator handling from SCRUM-461: when cdms_in_window is absent,
say "out of an unreported total" rather than reusing the pulled count; when
truncated_by is "cap", say "size limit" rather than "time limit". Reuse the existing
number formatting so 52,684 is punctuated. No source name anywhere in the copy.

## Placement and behaviour

Replace #conjPartialBanner and #globePartialBanner with:

1. One top toast. When a fetch returns partial, a toast slides in at the top of the
   view carrying the copy above. It auto-dismisses after a few seconds (propose 8 s,
   confirm at review) and has an x to dismiss early. One toast for the app, not one
   per panel, so the table and the globe never show two.

2. One persistent chip. A small "PARTIAL VIEW" marker sits next to the Active
   Conjunctions pager (the "1-100 of N" line, which is present in both 2D and globe).
   It stays for as long as the current view is truncated, independent of the toast.
   Give it a title/tooltip with the same counts, and optionally let a click re-show
   the toast (nice-to-have, not required).

## Why both, and which is load-bearing

The toast is the attention-getter and is allowed to disappear. The chip is the safety
element. A truncated view stays partial the entire time it is on screen, and the chip
is what keeps that from being silently forgotten once the toast fades — which is the
whole reason SCRUM-461 exists. So the auto-dismissing toast is only safe because the
chip persists. Build the chip as the durable state and the toast as a transient view
of it, not the other way around.

## Invariants to keep from SCRUM-461

- Partial is read as an explicit `data.partial === true || data.complete === false`.
  A response without the fields is complete (the pre-459 shape).
- The toast and the chip both clear before every fetch, on asset or source switch,
  and on clear-all, including the error paths that return early. Neither may describe
  a view it no longer applies to. This is the same discipline SCRUM-461 put on its
  banners; keep it, since a stale "partial" over a fresh complete view is the failure.
- The rows are never gated behind the warning.
- A complete view shows neither toast nor chip.

## Do not

- Do not change the planner, the responses, or the fetch. This reads what SCRUM-459
  already sends.
- Do not lose the honesty: no auto-dismiss without the persistent chip, no invented
  denominator, no source name.
- Do not leave the old two banners in place alongside the new toast; remove them so
  there is one signal, not three.

## Tests

- Update the SCRUM-461 template-wiring and harness tests to the new elements: the
  copy carries the backend counts and no source name (assert "LeoLabs" does not
  appear in the partial copy), the honest-denominator and time-versus-size wording
  hold, partial shows the toast and the chip while complete shows neither, and both
  clear on fetch start, asset switch and clear-all. Keep a node-harness check on the
  copy builder and template-wiring checks that the toast and chip are actually driven
  from the fetch responses, the point SCRUM-461 made that a render function nothing
  calls still passes a pure harness.
- Run the existing ui tests and confirm green.

## Live acceptance

Rebuild the ui. On SWARM B: the top toast appears with the source-neutral copy and
the real counts, auto-dismisses (and closes on the x), and the "PARTIAL VIEW" chip
stays by the pager in both the 2D and the globe view. On a sparse complete asset,
neither appears. Switching asset and clearing remove both. Capture a screenshot of the
toast and of the persistent chip. Record in docs/scrum-462/live_verification.md. Do
not deploy to the KVM.

## Commit

Branch off main. Include docs/scrum-462/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms the
copy is source-neutral with the honest denominator, that the chip persists while a
view is partial so the auto-dismissing toast never leaves the condition hidden, that
both clear on every fetch, switch and clear, and that a complete view shows neither,
then reviews the live toast and chip.
