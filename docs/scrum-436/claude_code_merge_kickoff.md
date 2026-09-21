# SCRUM-436 — merge kickoff for Claude Code

PR #101, branch `scrum-436-onload-burn-window`, expected head `87c4dbc`
("SCRUM-436: derive the scenario burn from the TCA instead of the clock").
Single commit, ahead 1 / behind 0 on `main` (`5bd7057`). UI-only: the diff is
`services/ui/app/templates/index.html` (+69/-6) plus the two `docs/scrum-436/`
planning files. No `.py` files change, so the suite outcome is main's by
construction. Review is done and clean (John): the real `scenarioBurnWindow`
keeps the burn strictly inside `(now, TCA)` for every valid input and returns
`null` for missing/<=0/NaN, which both callers turn into the "No upcoming TCA"
panel instead of a 422.

Run these in order. Nothing here needs undoing.

## 1. Confirm the PR head is what was reviewed
```
gh pr view 101 --json headRefOid,headRefName,state,mergeStateStatus
```
Head must be `87c4dbc...`. If it is anything else, stop and report — the branch
moved since review.

## 2. Squash-merge and delete the branch
```
# --match-head-commit needs the FULL 40-char head OID from step 1, not the
# abbreviated SHA (GitHub rejects the short form: "Could not coerce value to
# GitObjectID"). Substitute the full headRefOid you just verified.
gh pr merge 101 --squash --match-head-commit <full-40-char-headRefOid> --delete-branch
```
`--match-head-commit` makes the merge refuse if the head drifted between step 1
and here.

## 3. Update local main
```
git checkout main
git pull origin main
git log --oneline -1
```

## 4. Close the ticket
Transition SCRUM-436 to Done and post this closing comment verbatim:

> Fixed. The scenario evaluate now derives the burn from the row's TCA instead
> of a fixed now+2h, so near-TCA rows (under 2h out) stop coming back 422 and
> reading as EVALUATE FAILED. `scenarioBurnWindow` puts the burn 6h before TCA,
> or halfway between now and TCA when that is sooner, and returns nothing for a
> row with no usable TCA — those rows show "No upcoming TCA" rather than firing
> a request that can only fail. Both the on-load auto-evaluate and the payload
> preview go through the same helper, so the previewed request is the one that
> is sent. UI-only; suite unchanged. Shipped in PR #101.

## 5. Report back
Reply with the new `main` SHA from step 3 so the sprint anchor can be refreshed.
