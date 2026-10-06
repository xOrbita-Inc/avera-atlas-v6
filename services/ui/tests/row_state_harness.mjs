/* services/ui/tests/row_state_harness.mjs
 *
 * SCRUM-457: drive the real Secondary conflict row code from index.html under
 * node, with a stub DOM, and assert the honesty rule in every state.
 *
 * The plan expected the JS to be verified only live, because the dashboard has no
 * JS unit harness. The Chrome extension was not available on this run, so rather
 * than leave the row states unverified this harness extracts the actual functions
 * from the template by name and exercises them. It is not a substitute for seeing
 * the row repaint in a browser -- it cannot catch a CSS or layout problem -- but it
 * does check the logic that decides CLEAR versus NOT CLEAR versus SCREENING, which
 * is the part that matters for safety.
 *
 * Run:  node services/ui/tests/row_state_harness.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const template = readFileSync(join(here, '..', 'app', 'templates', 'index.html'), 'utf8');

// SCRUM-492: pollSecondaryScreen now also propagates the resolved verdict to the
// decision badge and the MAF mode, so the real propagation is extracted too
// rather than stubbed away. That keeps this harness exercising the actual
// poll-to-headline wiring, which is the integration the ticket creates.
const NAMES = ['_ar', 'secondaryScreenIsClear', 'cancelSecondaryScreenPoll',
  'armSecondaryScreenPoll', 'refreshPostManeuverPanel', 'pollSecondaryScreen',
  'propagateSecondaryScreenResolution', 'refreshVerificationPanel',
  'renderVerificationResult', 'renderPostManeuverProjection'];

const sources = NAMES.map(name => {
  const re = new RegExp('^(?:async )?function ' + name + '\\(.*?^\\}', 'ms');
  const m = template.match(re);
  if (!m) throw new Error('could not extract ' + name + ' from index.html');
  return m[0];
});

// The mode constant lives at module scope in the template, so it is lifted out
// by value rather than redeclared here; a copy would be free to drift.
const HOLD_MODE_SRC = template.match(/^const SECONDARY_HOLD_MAF_MODE = '[^']*';/m);
if (!HOLD_MODE_SRC) throw new Error('could not extract SECONDARY_HOLD_MAF_MODE');

// --- stub DOM, just enough for the row and the headline fields it now writes ---
const els = {
  postManeuverBody: { innerHTML: '' },
  'portlet-post-maneuver-projection': { style: {} },
  decisionBadge: {
    textContent: 'GO', className: 'decision-badge badge-go',
    classList: { contains(c) { return els.decisionBadge.className.split(' ').includes(c); } },
  },
  dMafMode: { textContent: 'M1' },
  verificationBody: { innerHTML: '' },
  verifyOverallBadge: { innerHTML: '', textContent: '', style: {} },
};
globalThis.document = { getElementById: id => els[id] || null };

// --- module state the extracted code closes over ---
let evalSeq = 0;
let secondaryScreen = null;
let secondaryScreenAbort = null;
let secondaryScreenTimer = null;
let currentPlannerResult = null;
const SECONDARY_POLL_INTERVAL_MS = 5000;
const SECONDARY_POLL_CAP_MS = 360000;

const ctx = {};
const body = sources.join('\n\n') + `
ctx.render = renderPostManeuverProjection;
ctx.isClear = secondaryScreenIsClear;
ctx.arm = armSecondaryScreenPoll;
ctx.cancel = cancelSecondaryScreenPoll;
ctx.poll = pollSecondaryScreen;
ctx.propagate = propagateSecondaryScreenResolution;
ctx.renderVerification = renderVerificationResult;
ctx.setSeq = v => { evalSeq = v; };
ctx.getSeq = () => evalSeq;
ctx.setState = v => { secondaryScreen = v; };
ctx.getState = () => secondaryScreen;
ctx.setResult = v => { currentPlannerResult = v; };
`;
// eslint-disable-next-line no-new-func
new Function('ctx', 'document', 'AbortController', 'fetch', 'setTimeout',
  'clearTimeout',
  `let evalSeq=0, secondaryScreen=null, secondaryScreenAbort=null,
       secondaryScreenTimer=null, currentPlannerResult=null;
   const SECONDARY_POLL_INTERVAL_MS=${SECONDARY_POLL_INTERVAL_MS};
   const SECONDARY_POLL_CAP_MS=${SECONDARY_POLL_CAP_MS};
   ${HOLD_MODE_SRC[0]}
   ${body}`
)(ctx, globalThis.document, globalThis.AbortController,
  (...a) => globalThis.__fetch(...a), globalThis.setTimeout, globalThis.clearTimeout);

// --- helpers ---
let failures = 0, checks = 0;
function ok(cond, label) {
  checks += 1;
  if (!cond) { failures += 1; console.log('  FAIL  ' + label); }
  else { console.log('  ok    ' + label); }
}
const PM_PENDING = {
  mahalanobis_post: 12.7, risk_surrogate_post: 0.0, slot_recovery_required: false,
  secondary_conflict: {
    screen_pending: true, screen_job_id: 'job-1', screen_deferred: false,
    secondary_check_performed: false, secondary_conjunction_clear: false,
    operator_note: 'running in the background',
  },
};
function renderWith(state, seq = 1, pm = PM_PENDING) {
  ctx.setSeq(seq);
  ctx.setState(state);
  ctx.render(pm);
  return els.postManeuverBody.innerHTML;
}
// Match to the end of the row div, not the first </span>: the screening label
// contains a nested spinner span, and stopping at the first close would truncate it.
const rowOf = html => {
  const m = html.match(/Secondary conflict<\/span><span class="artifact-val ([a-z]*)">(.*?)<\/span><\/div>/);
  return m ? { cls: m[1], label: m[2] } : null;
};

console.log('\n1. the honesty allow-list');
ok(ctx.isClear({ status: 'clear', clear: true }) === true, 'a real clear is clear');
for (const p of [
  { status: 'pending', clear: false }, { status: 'not_clear', clear: false },
  { status: 'error', clear: false }, { status: 'clear', clear: false },
  { status: 'CLEAR', clear: true }, { status: 'cleared', clear: true },
  { clear: true }, { status: 'clear' }, {}, null, undefined,
]) {
  ok(ctx.isClear(p) === false, 'not clear: ' + JSON.stringify(p));
}

console.log('\n2. the row in each state');
let row = rowOf(renderWith({ seq: 1, phase: 'screening', jobId: 'job-1' }));
ok(/SCREENING IN PROGRESS/.test(row.label), 'screening shows SCREENING IN PROGRESS');
ok(/spinner/.test(row.label), 'screening shows a spinner');
ok(row.cls === 'amber', 'screening is amber, not green');
ok(!/CLEAR<\/span>/.test(row.label), 'screening never says CLEAR alone');

let clearHtml = renderWith({ seq: 1, phase: 'resolved', payload: {
  status: 'clear', clear: true, screening_id: '603659',
  conjunctions_total: 1, conjunctions_truncated: false,
  conjunctions: [{
    object_id: 'SAFE-SAT', secondary_norad: 12345,
    tca_utc: '2026-09-26T22:00:00Z',
    miss_distance_km: 18.25, pc: 2.5e-8,
    covariance_repaired: false, covariance_untrusted: false,
  }],
  verdict: { clear: true, evaluated: 1, breaches: [] },
} });
row = rowOf(clearHtml);
ok(row.label === 'CLEAR' && row.cls === 'green', 'a real clear shows CLEAR green');
ok(/Full LeoLabs catalog/.test(clearHtml), 'clear names the catalog screened');
ok(/SAFE-SAT/.test(clearHtml), 'clear shows a non-breaching residual conjunction');
ok(/2\.500e-8/.test(clearHtml), 'clear shows the residual covariance-backed Pc');
ok(/accepted/.test(clearHtml), 'clear shows the covariance state');

let html = renderWith({ seq: 1, phase: 'resolved', payload: {
  status: 'not_clear', clear: false, screening_id: '603659',
  flagged_objects: ['STARLINK-1661', 'R5-S4'],
  closest_object_id: 'STARLINK-1661', closest_approach_km: 4.396523,
  conjunctions_total: 1223, conjunctions_truncated: true,
  conjunctions: [{
    object_id: 'STARLINK-1661', secondary_norad: 44713,
    tca_utc: '2026-09-26T23:00:00Z',
    miss_distance_km: 4.396523, pc: 1.23e-5,
    covariance_repaired: true, covariance_untrusted: false,
  }],
  verdict: { clear: false, evaluated: 1223, breaches: [
    { object_id: 'STARLINK-1661', limbs: ['mahalanobis'] },
    { object_id: 'R5-S4', limbs: ['mahalanobis'] }] },
} });
row = rowOf(html);
ok(row.label === 'NOT CLEAR' && row.cls === 'red', 'not_clear shows NOT CLEAR red');
ok(/STARLINK-1661, R5-S4/.test(html), 'not_clear names the breaching objects');
ok(/mahalanobis/.test(html), 'not_clear names the limb breached');
ok(/1223/.test(html), 'not_clear reports how many were screened');
ok(/603659/.test(html), 'not_clear reports the screening id');
ok(/4\.397 km/.test(html), 'not_clear reports the closest approach');
ok(/Full LeoLabs catalog/.test(html), 'not_clear names the catalog screened');
ok(/Residual conjunctions/.test(html) && /1223/.test(html),
  'not_clear reports the full residual count');
ok(/1 of 1223/.test(html), 'not_clear says when residual details are truncated');
ok(/1\.230e-5/.test(html), 'not_clear shows the residual covariance-backed Pc');
ok(/repaired/.test(html), 'not_clear shows repaired covariance provenance');

row = rowOf(renderWith({ seq: 1, phase: 'resolved', payload: {
  status: 'error', clear: false, error: 'poll timed out',
  operator_note: 'could not be retrieved' } }));
ok(row.label === 'NOT CLEAR' && row.cls === 'red', 'error shows NOT CLEAR red');

html = renderWith({ seq: 1, phase: 'failed', note: 'did not resolve within 360 s' });
row = rowOf(html);
ok(row.label === 'NOT CLEAR' && row.cls === 'red', 'the cap shows NOT CLEAR red');
ok(/360 s/.test(html), 'the cap note says what the bound was');
ok(!/spinner/.test(html), 'the cap leaves no spinner behind');

console.log('\n3. a pending artifact with nothing tracking it');
html = renderWith(null, 1);
row = rowOf(html);
ok(row.label === 'NOT CLEAR' && row.cls === 'red',
  'pending with no live poll reads NOT CLEAR, not a spinner and not clear');
ok(!/spinner/.test(html), 'no endless spinner');

console.log('\n4. latest selection wins (the SCRUM-449 rule)');
html = renderWith({ seq: 1, phase: 'resolved',
  payload: { status: 'clear', clear: true } }, 2);   // state is for a stale evaluate
row = rowOf(html);
ok(row.label !== 'CLEAR',
  'a stale CLEAR for a superseded evaluate does not paint the row');
ok(row.label === 'NOT CLEAR',
  'the superseded row falls back to the fail-closed reading');

console.log('\n5. the non-async shapes are untouched');
for (const [sc, want, cls] of [
  [{ screen_deferred: true, secondary_check_performed: false,
     secondary_conjunction_clear: false }, 'DEFERRED', ''],
  [{ screen_deferred: false, secondary_check_performed: false,
     secondary_conjunction_clear: false }, 'NOT CHECKED', ''],
  [{ screen_deferred: false, secondary_check_performed: true,
     secondary_conjunction_clear: true }, 'CLEAR', 'green'],
  [{ screen_deferred: false, secondary_check_performed: true,
     secondary_conjunction_clear: false }, 'CONFLICT DETECTED', 'red'],
]) {
  row = rowOf(renderWith(null, 1, { secondary_conflict: sc }));
  ok(row.label === want && row.cls === cls, `legacy shape renders ${want}`);
}

console.log('\n6. the poll loop');
ctx.setSeq(7);
ctx.setResult({ atlas_artifact: { post_maneuver: PM_PENDING } });
globalThis.__fetch = async () => ({ json: async () => (
  { status: 'not_clear', clear: false, screening_id: '603659',
    verdict: { clear: false, evaluated: 1223, breaches: [] } }) });
ctx.setState({ seq: 7, jobId: 'job-1', phase: 'screening', payload: null,
  note: null, startedAt: Date.now(), polls: 0 });
await ctx.poll();
ok(ctx.getState().phase === 'resolved', 'a resolved answer stops the poll');
ok(rowOf(els.postManeuverBody.innerHTML).label === 'NOT CLEAR',
  'the row repainted NOT CLEAR from the poll');

ctx.setState({ seq: 7, jobId: 'job-1', phase: 'screening', payload: null,
  note: null, startedAt: Date.now() - (SECONDARY_POLL_CAP_MS + 1000), polls: 0 });
let fetched = 0;
globalThis.__fetch = async () => { fetched += 1; return { json: async () => ({}) }; };
await ctx.poll();
ok(fetched === 0, 'past the cap it stops polling rather than issuing another request');
ok(ctx.getState().phase === 'failed', 'past the cap the state is failed');

ctx.setState({ seq: 3, jobId: 'job-old', phase: 'screening', payload: null,
  note: null, startedAt: Date.now(), polls: 0 });
ctx.setSeq(9);                                   // superseded
fetched = 0;
await ctx.poll();
ok(fetched === 0, 'a poll for a superseded evaluate issues no request at all');

globalThis.__fetch = async () => { throw Object.assign(new Error('x'), { name: 'AbortError' }); };
ctx.setSeq(11);
ctx.setState({ seq: 11, jobId: 'j', phase: 'screening', payload: null, note: null,
  startedAt: Date.now(), polls: 0 });
await ctx.poll();
ok(ctx.getState().phase === 'screening',
  'an aborted poll is not a failure and does not repaint');

globalThis.__fetch = async () => ({ json: async () => { throw new Error('bad json'); } });
ctx.setSeq(12);
ctx.setState({ seq: 12, jobId: 'j', phase: 'screening', payload: null, note: null,
  startedAt: Date.now(), polls: 0 });
await ctx.poll();
ok(ctx.getState().phase === 'failed', 'an unreadable answer fails closed');
ok(rowOf(els.postManeuverBody.innerHTML).label === 'NOT CLEAR',
  'and the row shows NOT CLEAR');

console.log('\n7. cancel and arm');
ctx.setState({ seq: 1, phase: 'screening' });
ctx.cancel();
ok(ctx.getState() === null, 'cancel clears the state');
ctx.setSeq(20);
ctx.arm({ post_maneuver: PM_PENDING }, 20);
ok(ctx.getState() && ctx.getState().phase === 'screening', 'arm sets screening');
ok(ctx.getState().jobId === 'job-1', 'arm picks up screen_job_id');
ctx.cancel();
ctx.arm({ post_maneuver: { secondary_conflict: {
  screen_pending: false, secondary_check_performed: true,
  secondary_conjunction_clear: true } } }, 20);
ok(ctx.getState() === null, 'arm does nothing when the screen is not pending');
ctx.arm({ post_maneuver: { secondary_conflict: {
  screen_pending: true, screen_job_id: null } } }, 20);
ok(ctx.getState() === null, 'arm does nothing without a job id');
ctx.cancel();

// SCRUM-492: the resolved verdict reaches the headline, not just the A4 row.
//
// The defect was that only A4 updated, so the card could read GO with the MAF
// row at its evaluate-time mode while A4 directly below said the post-burn
// screen was NOT CLEAR. These check the three states the operator can see.
console.log('\n8. the resolution reaches the decision headline');
function headline(badgeText, badgeCls, mode){
  els.decisionBadge.textContent = badgeText;
  els.decisionBadge.className = 'decision-badge ' + badgeCls;
  els.dMafMode.textContent = mode;
}
const HOLD_MODE = HOLD_MODE_SRC[0].match(/'([^']*)'/)[1];

// HOLD_MODE is read out of the template, so every check below that compares
// dMafMode against it is circular: it passes whatever the constant says, even if
// the constant is wrong. This is the one assertion that pins the VALUE, against
// the planner rule it is supposed to mirror -- safety_monitor section 4.2
// escalates a performed-and-not-clear secondary to FlightMode.M4_SAFE_HOLD, and
// a planner test pins that transition. Changing the template constant to
// anything else has to fail here.
ok(HOLD_MODE === 'M4',
  'hold mode is M4, FlightMode.M4_SAFE_HOLD per safety_monitor 4.2');

ctx.setSeq(30);
ctx.setResult({ decision_state_machine: { to_mode: 'M1' } });

headline('GO', 'badge-go', 'M1');
ctx.setState({ seq: 30, phase: 'resolved', payload: { status: 'not_clear', clear: false } });
ctx.propagate();
ok(els.decisionBadge.textContent === 'SECONDARY HOLD',
  'a resolved NOT CLEAR moves the badge off GO');
ok(!els.decisionBadge.className.includes('badge-go'),
  'and it no longer carries the GO style');
ok(els.dMafMode.textContent === HOLD_MODE,
  'and the MAF mode escalates to ' + HOLD_MODE);

headline('GO', 'badge-go', 'M1');
ctx.setState({ seq: 30, phase: 'resolved', payload: { status: 'clear', clear: true } });
ctx.propagate();
ok(els.decisionBadge.textContent === 'GO', 'a resolved CLEAR keeps GO');
ok(els.dMafMode.textContent === 'M1', 'and restores the evaluate-time MAF mode');

headline('GO', 'badge-go', 'M1');
ctx.setState({ seq: 30, phase: 'screening', payload: null });
ctx.propagate();
ok(els.decisionBadge.textContent === 'GO' && els.dMafMode.textContent === 'M1',
  'a screen still running leaves the headline alone');

headline('GO', 'badge-go', 'M1');
ctx.setState({ seq: 30, phase: 'failed', payload: null });
ctx.propagate();
ok(els.decisionBadge.textContent === 'SECONDARY HOLD' && els.dMafMode.textContent === HOLD_MODE,
  'a screen that could not be read holds, the same as NOT CLEAR');

// The SCRUM-449 rule applies here too: a screen for a superseded decision must
// never paint the headline of the one now on screen.
headline('GO', 'badge-go', 'M1');
ctx.setState({ seq: 29, phase: 'resolved', payload: { status: 'not_clear', clear: false } });
ctx.propagate();
ok(els.decisionBadge.textContent === 'GO' && els.dMafMode.textContent === 'M1',
  'a stale screen does not touch the current headline');

headline('WATCH', 'badge-nogo', 'M1');
ctx.setState({ seq: 30, phase: 'resolved', payload: { status: 'not_clear', clear: false } });
ctx.propagate();
ok(els.decisionBadge.textContent === 'WATCH',
  'a badge that was never GO is left as it is');

// Through the REAL poll, not by calling propagate directly. Everything above
// would still pass if pollSecondaryScreen simply never called it, which would
// leave the whole feature dead and the suite green.
ctx.setSeq(31);
ctx.setResult({ atlas_artifact: { post_maneuver: PM_PENDING },
                decision_state_machine: { to_mode: 'M1' } });
globalThis.__fetch = async () => ({ json: async () => (
  { status: 'not_clear', clear: false, screening_id: '603659',
    verdict: { clear: false, evaluated: 1223, breaches: [] } }) });
headline('GO', 'badge-go', 'M1');
ctx.setState({ seq: 31, jobId: 'job-1', phase: 'screening', payload: null,
  note: null, startedAt: Date.now(), polls: 0 });
await ctx.poll();
ok(els.decisionBadge.textContent === 'SECONDARY HOLD',
  'the poll itself carries a NOT CLEAR through to the badge');
ok(els.dMafMode.textContent === HOLD_MODE,
  'and through to the MAF mode');

globalThis.__fetch = async () => ({ json: async () => (
  { status: 'clear', clear: true, screening_id: '603660',
    verdict: { clear: true, evaluated: 1223, breaches: [] } }) });
headline('GO', 'badge-go', 'M1');
ctx.setState({ seq: 31, jobId: 'job-1', phase: 'screening', payload: null,
  note: null, startedAt: Date.now(), polls: 0 });
await ctx.poll();
ok(els.decisionBadge.textContent === 'GO',
  'and a CLEAR through the poll leaves GO standing');

// SCRUM-492: the DEMO path reaches the same propagation as the live poll.
//
// OBJ-DEMO-FLIP resolves its screen synchronously when the scenario renders,
// never going through pollSecondaryScreen, so before this it was the one path
// that could still show a GO badge above a verification row reading FAIL. It is
// also the path most likely to be demonstrated, which is why it is driven here
// end to end rather than trusted to the shared function.
console.log('\n9. the demo not-clear path reaches all three panels');

// The payload the demo fixture actually emits for OBJ-DEMO-FLIP.
const DEMO_NOT_CLEAR = {
  status: 'not_clear', clear: false, pending: false,
  catalog_screened: 'Controlled demo catalog', screening_id: 'DEMO-HELD',
  conjunctions_total: 1, conjunctions_truncated: false,
  conjunctions: [{ object_id: 'DEMO-HELD-SECONDARY', secondary_norad: 90002,
    tca_utc: new Date(Date.now() + 3600e3).toISOString(),
    miss_distance_km: 0.4, pc: 1.2e-4,
    covariance_repaired: false, covariance_untrusted: false }],
  verdict: { clear: false, evaluated: 1, breaches: ['miss_distance'] },
  operator_note: 'Controlled demo fixture: post-burn secondary screen NOT CLEAR.',
};
const VERIFICATION = {
  passed: true, risk_reduced: true, utility_positive: true,
  secondary_clear: true, budget_within_limits: true,
  recovery_within_limits: true, verification_note: null,
};

ctx.setSeq(40);
ctx.setResult({
  atlas_artifact: {
    post_maneuver: PM_PENDING, recommendation: { direction: 'prograde' },
    verification: VERIFICATION,
  },
  decision_state_machine: { to_mode: 'M1' },
});
headline('GO', 'badge-go', 'M1');
els.verificationBody.innerHTML = '';
// Exactly what the scenario path does: set the resolved payload, then propagate.
ctx.setState({ seq: 40, jobId: null, phase: 'resolved', payload: DEMO_NOT_CLEAR,
  note: null, startedAt: Date.now(), polls: 0 });
ctx.propagate();

ok(els.decisionBadge.textContent === 'SECONDARY HOLD',
  'demo NOT CLEAR moves the badge off GO');
ok(!els.decisionBadge.className.includes('badge-go'),
  'demo NOT CLEAR drops the GO style');
ok(els.dMafMode.textContent === HOLD_MODE,
  'demo NOT CLEAR escalates the MAF mode to ' + HOLD_MODE);

// The verification row, read out of the rendered markup rather than inferred.
const secondaryRow = els.verificationBody.innerHTML
  .split('verify-check')
  .find(chunk => chunk.includes('Secondary conflict clear'));
ok(!!secondaryRow, 'the verification panel rendered a secondary-conflict row');
ok(!!secondaryRow && secondaryRow.includes('FAIL'),
  'and that row reads the resolved FAIL');
ok(!!secondaryRow && !secondaryRow.includes('PASS'),
  'and does not still read PASS from the evaluate-time artifact');

// The same, with a LIVE-shaped payload: no catalog_screened, so it does not take
// the SCRUM-443 demo branch. renderVerificationResult used to read the resolved
// screen only for the demo catalog, which is what left the live path showing a
// stale PASS; re-gating it to demo-only has to fail here.
//
// The artifact deliberately says secondary_clear TRUE, so the row can only read
// FAIL if the resolved screen actually overrode it. If the override is removed
// the row falls back to that TRUE and renders PASS.
ctx.setResult({
  atlas_artifact: {
    post_maneuver: PM_PENDING, recommendation: { direction: 'prograde' },
    verification: VERIFICATION,
  },
  decision_state_machine: { to_mode: 'M1' },
});
headline('GO', 'badge-go', 'M1');
els.verificationBody.innerHTML = '';
ctx.setState({ seq: 40, jobId: 'job-live', phase: 'resolved',
  payload: {
    status: 'not_clear', clear: false, pending: false,
    screening_id: '603659',
    conjunctions_total: 1223, conjunctions_truncated: true, conjunctions: [],
    verdict: { clear: false, evaluated: 1223, breaches: ['miss_distance'] },
    operator_note: 'Live LeoLabs on-demand screen: NOT CLEAR.',
  },
  note: null, startedAt: Date.now(), polls: 0 });
ctx.propagate();

const liveRow = els.verificationBody.innerHTML
  .split('verify-check')
  .find(chunk => chunk.includes('Secondary conflict clear'));
ok(!!liveRow, 'a live-shaped resolved screen renders a secondary-conflict row');
ok(!!liveRow && liveRow.includes('FAIL'),
  'and a LIVE resolved NOT CLEAR reads FAIL, not just the demo-catalog one');
ok(!!liveRow && !liveRow.includes('PASS'),
  'and does not fall back to the artifact\'s stale secondary_clear');

// The clear demo fixture must not be dragged along with it.
ctx.setResult({
  atlas_artifact: {
    post_maneuver: PM_PENDING, recommendation: { direction: 'prograde' },
    verification: VERIFICATION,
  },
  decision_state_machine: { to_mode: 'M1' },
});
headline('GO', 'badge-go', 'M1');
ctx.setState({ seq: 40, jobId: null, phase: 'resolved',
  payload: { status: 'clear', clear: true, catalog_screened: 'Controlled demo catalog' },
  note: null, startedAt: Date.now(), polls: 0 });
ctx.propagate();
ok(els.decisionBadge.textContent === 'GO', 'a clear demo screen keeps GO');
ok(els.dMafMode.textContent === 'M1', 'and leaves the MAF mode at its evaluate value');

// SCRUM-492: both call sites exist.
//
// Everything above drives propagateSecondaryScreenResolution directly, so it all
// still passes if a call site is deleted and the function is simply never
// reached -- which is the exact way this feature could die silently. The poll
// path is covered behaviourally in section 8; the demo call site sits inside
// evaluateConjunction, which pulls in the whole evaluate pipeline and cannot be
// driven here, so it is pinned at the wiring level instead.
//
// A source assertion is weaker than a behavioural one and is used only because
// the alternative is no coverage at all for a line whose deletion is invisible.
console.log('\n10. both call sites are wired');
const pollSrc = template.match(/^async function pollSecondaryScreen\(.*?^\}/ms);
ok(!!pollSrc, 'pollSecondaryScreen is in the template');
ok(!!pollSrc && (pollSrc[0].match(/propagateSecondaryScreenResolution\(\)/g) || []).length >= 2,
  'the poll propagates from every terminal branch, not just one');
ok(/if\(demoScreen\)\s*propagateSecondaryScreenResolution\(\);/.test(template),
  'the demo scenario path propagates too');
ok((template.match(/function propagateSecondaryScreenResolution\(/g) || []).length === 1,
  'and there is one implementation of the rule, not a demo fork of it');

console.log(`\n${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
