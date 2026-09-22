# SCRUM-438 (reopened) — cursor param fix, kickoff for Claude Code

SCRUM-438 was reopened. PR #104 (merged a94abf0) set `paginate=true` and added
the `total` cross-check, but the pagination loop echoed the cursor back under
the wrong request-parameter name. LeoLabs returns the cursor in the response
field `nextToken` and expects it back on the next request as the parameter
`token`. The merged code sent it as `nextToken`, which LeoLabs ignores, so the
loop re-fetched page one and never advanced past the first 1,000 items — the
>1,000 case the ticket exists to fix was not actually fixed. The ticket's own
acceptance says it plainly: "Read 'nextToken' from each response and pass it as
'token' on the next call." LeoLabs support (Lois Reid, 2026-08-31) says the same.

Ticket: https://xorbita.atlassian.net/browse/SCRUM-438

## The fix is already in the working tree

Cowork made the surgical edits in the working tree — do NOT re-derive them.
`git diff` against main a94abf0 shows exactly two files:

- `services/planner/common/leolabs_client.py` — in `_paginate`, the cursor is
  now sent as `page_params["token"] = next_token` (was `nextToken`). The
  response read stays `data.get("nextToken")`. A four-line comment records why
  the two names differ.
- `services/planner/tests/test_leolabs_client.py` — the two existing tests that
  asserted `second_params["nextToken"] == "tok"` now assert
  `second_params["token"] == "tok"` and `"nextToken" not in second_params`, and
  a new test `test_paginate_sends_cursor_under_token_not_next_token` locks the
  wire name so the two names cannot drift back together.

Cowork ran the planner suite on this tree: 1380 passed, 2 skipped.

## Run in order

1. Review the `git diff` for the two files above and confirm it matches the
   description here — nothing else should be modified.
2. Run the planner suite from the repo root and confirm green:
   `python3 -m pytest services/planner`
   The change touches no UI and no propagator, so those suites are unaffected.
3. Commit on a branch named `scrum-438-cursor-token`, including this
   `docs/scrum-438/pagination_cursor_fix_kickoff.md` in the commit. No AI
   attribution in the commit message or PR body.
4. Push and open the PR against main, then report the PR number and head SHA.

Suggested commit subject:
`SCRUM-438: send LeoLabs pagination cursor as token, not nextToken`

Cowork will do the verification-first review (read the committed diff, run the
planner suite on the branch, confirm the wire-name test discriminates) before
John merges. Acceptance against a live LeoLabs call is tracked separately on the
SCRUM-439 spike, so this PR is merged on the unit evidence, and the live
paginate/token behavior is proven there before we rely on it in the demo.
