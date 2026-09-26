# SCRUM-466 kickoff for Claude Code

Add the required LeoLabs attribution banner to the ARBITER dashboard header: a compact
cyan pill reading "UTILIZING LEOLABS PULSE SPACE SAFETY SERVICE", to the right of the
title, always on. This is a contractual string LeoLabs requires stated verbatim on any
public material. Part of SCRUM-444. Ticket:
https://xorbita.atlassian.net/browse/SCRUM-466

Branch off main. UI only. No planner change.

Read docs/scrum-466/implementation_plan.md first, and open
docs/scrum-466/leolabs-attribution-banner-mockup.html (variant A) for the exact look.

Run in order:

1. Read docs/scrum-466/implementation_plan.md.

2. In services/ui/app/templates/index.html, find the header (the `.header` block with
   `.logo` and `.header-right`, around line 470).

3. Add the attribution pill as the FIRST child of `.header-right`, before
   `#sysStatus`: a span `#leolabsAttrib` class `attrib-pill` containing a `.attrib-dot`
   span and the text `UTILIZING LEOLABS PULSE SPACE SAFETY SERVICE` as one contiguous
   text run.

4. Add the CSS from the plan: `.attrib-pill` and `.attrib-dot`, matching the
   `.sys-status` metrics, in the ARBITER cyan (var(--accent-dim) text, var(--accent-glow)
   background, rgba(79,248,232,0.28) border) so it reads as a data-source credit, not a
   status light.

5. Keep the rules: always on in scenario and live modes, never gated to data or a flag;
   the phrase "utilizing LeoLabs Pulse space safety service" verbatim (uppercase display
   is fine, no abbreviation, rewording, or reordering, and the text content is the exact
   string); responsive so on a narrow window it wraps or drops below the title rather
   than crowding or truncating the clock, never abbreviating.

6. Do not change the planner or any response, do not make the text depend on data or
   mode, do not use a LeoLabs logo image.

7. Test: a ui template/unit test asserting the header contains the exact phrase
   (case-insensitive substring "utilizing leolabs pulse space safety service" as a
   single contiguous run) and that `#leolabsAttrib` exists inside `.header-right` before
   `#sysStatus`. Run the existing ui tests and confirm green.

8. Rebuild the ui. The header shows the cyan pill to the right of the title and before
   SYSTEM ONLINE, always visible, phrase verbatim, matching mockup A, in both scenario
   and live modes. Capture a clean screenshot of the header. Record in
   docs/scrum-466/live_verification.md. Do not deploy to the KVM.

9. Commit off main, include docs/scrum-466/, push, open a PR against main. No
   attribution trailers, author John Avera <javera@xorbita.com>. Report the PR number
   and head SHA.

Then Cowork runs the verification-first review before John merges, and the header
screenshot goes to LeoLabs for review before public use.
