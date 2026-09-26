# SCRUM-466 implementation plan: LeoLabs attribution banner in the dashboard header

Ticket: https://xorbita.atlassian.net/browse/SCRUM-466
Branch off main. UI only. Part of SCRUM-444. Compliance-driven, time-sensitive.

## Why

LeoLabs requires that any public material featuring results clearly state, verbatim,
"utilizing LeoLabs Pulse space safety service", and that anything naming LeoLabs be
submitted to them for review with at least two weeks' notice (Lois Reid, 2026-09-14;
confirmed by John 2026-09-18). The demo is Oct 13-15, so this must ship and be sent to
LeoLabs for review this week. This adds the persistent attribution banner to the
ARBITER header. It is a contractual string, not decorative copy.

## The change, services/ui/app/templates/index.html only

The header today (around line 470) is:

    <div class="header">
      <div class="logo"><h1>ARBITER</h1><span class="tag">APS v{{ aps_version }} Conjunction Assessment</span></div>
      <div class="header-right">
        <span class="sys-status" id="sysStatus">SYSTEM ONLINE</span>
        <span class="clock" id="clock">--:--:-- UTC</span>
      </div>
    </div>

Add the attribution pill as the FIRST child of `.header-right`, before `#sysStatus`, so
it sits to the right of the title and to the left of the status pill:

    <span class="attrib-pill" id="leolabsAttrib">
      <span class="attrib-dot"></span>UTILIZING LEOLABS PULSE SPACE SAFETY SERVICE
    </span>

CSS (mockup A, the accent-pill treatment, matching the .sys-status metrics so it sits
level with the status pill and the clock):

    .attrib-pill{display:inline-flex;align-items:center;gap:7px;font-size:9px;
      letter-spacing:1.5px;font-weight:500;padding:4px 10px;border-radius:3px;
      color:var(--accent-dim);background:var(--accent-glow);
      border:1px solid rgba(79,248,232,0.28);white-space:nowrap}
    .attrib-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);
      box-shadow:0 0 6px rgba(79,248,232,0.7);flex:0 0 auto}

It is in the ARBITER cyan rather than the green of SYSTEM ONLINE on purpose: it must
read as a data-source credit, not a second status light.

The approved look is in docs/scrum-466/leolabs-attribution-banner-mockup.html, variant A.

## Rules, non-negotiable

- Always on. It renders in both scenario and live modes, so it can never be missing
  during a live moment. Do not gate it to live data or to a LeoLabs fetch.
- Exact phrase. The words "utilizing LeoLabs Pulse space safety service" must appear
  verbatim. Uppercase display is fine; do not abbreviate, reword, reorder, or split the
  phrase across elements such that the text content is not the exact string.
- Responsive. On a narrow window let the pill wrap to its own line or move to the
  pipeline row rather than crowd or truncate the clock. It never abbreviates the phrase.
  (A simple approach: allow `.header` to wrap, or drop the pill below the title under a
  width breakpoint. Confirm at review; the demo runs on a wide screen.)

## Do not

- Do not change the planner or any response. This is a static header credit.
- Do not make the text depend on data, mode, or a feature flag.
- Do not use a LeoLabs logo or wordmark image; the text credit is the requirement and a
  logo would need their brand assets and separate approval.

## Test

A template/unit test in the existing ui pattern asserting:
- the rendered header contains the exact phrase, case-insensitive substring
  "utilizing leolabs pulse space safety service", as a single contiguous text run;
- the `#leolabsAttrib` pill element exists and sits inside `.header-right` before
  `#sysStatus`.
This pins the required attribution so it cannot be silently removed or altered later.
Run the existing ui tests and confirm green.

## Live acceptance

Rebuild the ui. The ARBITER header shows the cyan attribution pill to the right of the
title and before SYSTEM ONLINE, always visible, phrase verbatim, matching mockup A, in
both scenario and live modes. Capture a clean screenshot of the header for the LeoLabs
submission. Record in docs/scrum-466/live_verification.md. Do not deploy to the KVM.

## Commit

Branch off main. Include docs/scrum-466/ (the plan, the kickoff, and the mockup). No
attribution trailers, author John Avera <javera@xorbita.com>. PR against main; report
the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms the
phrase is present verbatim as a contiguous string, that the pill is always on and not
gated to data or mode, that a test pins the phrase, and that the placement matches
mockup A, then reviews the live header. The rendered banner is itself material naming
LeoLabs, so its screenshot goes to Lois for review before public use (SCRUM-444).
