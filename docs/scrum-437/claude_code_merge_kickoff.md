# SCRUM-437 — merge kickoff for Claude Code

PR #102, branch `scrum-437-index-html-utf8-repair`, expected head
`48bd080b1648ea540e12c96c3508d0534f54a1d4` (48bd080), "SCRUM-437 repair
double-encoded UTF-8 in the dashboard template". Single commit, base main
(9539d37), MERGEABLE/CLEAN. UI-only: `services/ui/app/templates/index.html`
+9/-9, text only, no `.py`. Review is done and clean (John and Cowork both
verified): the committed blob decodes as valid UTF-8, carries zero residual
mojibake, holds the intended characters, and the extracted page script passes
`node --check`.

Run these in order. Nothing here needs undoing.

## 1. Confirm the PR head is what was reviewed
```
gh pr view 102 --json headRefOid,headRefName,state,mergeStateStatus
```
headRefOid must be `48bd080b1648ea540e12c96c3508d0534f54a1d4`. If it is anything
else, stop and report.

## 2. Squash-merge and delete the branch
```
gh pr merge 102 --squash --match-head-commit 48bd080b1648ea540e12c96c3508d0534f54a1d4 --delete-branch
```
Pass the FULL 40-character head OID, not the abbreviated SHA. GitHub rejects the
short form with "Could not coerce value to GitObjectID".

## 3. Update local main
```
git checkout main
git pull origin main
git log --oneline -1
```

## 4. Close the ticket
Transition SCRUM-437 to Done and post this closing comment verbatim:

> Fixed. The dashboard template held double-encoded UTF-8, so the Live Asset
> dropdown and a few other strings rendered as mojibake ("CRYOSAT 2 â€" 36508").
> Repaired the bytes back to the intended characters in
> services/ui/app/templates/index.html, text only, no logic and no Jinja tags.
> Ten runs across nine lines: the dropdown separator, the two SOURCE notes, the
> scenario note, the live-conjunctions message, a CSS box rule, the LeoLabs
> apostrophe, m² in a comment, and the 1σ COV canvas label. Verified the file
> decodes as clean UTF-8 with zero residual mojibake and the extracted page
> script passes node --check. UI-only. Shipped in PR #102, reaches prod on a
> deploy.sh ui rebuild.

## 5. Report back
Reply with the new `main` SHA from step 3.

Note: this reaches prod on a `deploy.sh ui` rebuild, which John runs on the KVM
after the merge. That is not a step for you.
