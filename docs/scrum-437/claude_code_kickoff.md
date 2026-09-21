# SCRUM-437 — kickoff for Claude Code (UTF-8 repair in the dashboard template)

The fix is already in the working tree. Cowork repaired the double-encoded
UTF-8 in `services/ui/app/templates/index.html` (+9/-9, text only, inside CSS
comments and JS string literals — no logic, no Jinja tags touched). The em dash
separator —, the box-drawing rule ──, the apostrophe in "LeoLabs'", m², and
"1σ COV" now hold their intended characters instead of mojibake. Verified: a
full byte-signature sweep of the file is clean, and the intended characters are
present. Only this one file is modified; `main.py`'s π/³/μ were already correct
and are untouched.

Your job is the git side only. Run in order, nothing here needs undoing.

## 1. Confirm the tree holds exactly this one change
```
git status --porcelain
```
The only tracked modification should be `services/ui/app/templates/index.html`.
If anything else tracked is modified, stop and report.

## 2. Branch, commit, push, open the PR
```
git checkout -b scrum-437-index-html-utf8-repair
git add services/ui/app/templates/index.html
git commit -m "SCRUM-437 repair double-encoded UTF-8 in the dashboard template"
git push -u origin scrum-437-index-html-utf8-repair
gh pr create --fill --base main
```
No AI attribution in the commit or the PR body.

## 3. Report back
Reply with the PR number and its head SHA so the review can confirm the head
before merge.

Notes: UI-only, no `.py` changed, so the backend suite is unaffected. This
reaches prod on a `deploy.sh ui` rebuild after John merges (a second, cheap UI
deploy on top of today's).
