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

const NAMES = ['_ar', 'secondaryScreenIsClear', 'cancelSecondaryScreenPoll',
  'armSecondaryScreenPoll', 'refreshPostManeuverPanel', 'pollSecondaryScreen',
  'renderPostManeuverProjection'];

const sources = NAMES.map(name => {
  const re = new RegExp('^(?:async )?function ' + name + '\\(.*?^\\}', 'ms');
  const m = template.match(re);
  if (!m) throw new Error('could not extract ' + name + ' from index.html');
  return m[0];
});

// --- stub DOM, just enough for the row ---
const els = {
  postManeuverBody: { innerHTML: '' },
  'portlet-post-maneuver-projection': { style: {} },
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

row = rowOf(renderWith({ seq: 1, phase: 'resolved',
  payload: { status: 'clear', clear: true, screening_id: '603659' } }));
ok(row.label === 'CLEAR' && row.cls === 'green', 'a real clear shows CLEAR green');

let html = renderWith({ seq: 1, phase: 'resolved', payload: {
  status: 'not_clear', clear: false, screening_id: '603659',
  flagged_objects: ['STARLINK-1661', 'R5-S4'],
  closest_object_id: 'STARLINK-1661', closest_approach_km: 4.396523,
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

console.log(`\n${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
