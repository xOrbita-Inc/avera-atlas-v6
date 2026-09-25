/* services/ui/tests/partial_warning_harness.mjs
 *
 * SCRUM-462: drive the real partial-view warning out of index.html under node.
 *
 * The warning is now one app-level toast plus one persistent chip, replacing the two
 * in-panel banners SCRUM-461 shipped, and the copy no longer names the feed.
 *
 * The property that matters most here is the relationship between the two. The toast
 * auto-dismisses; the chip does not. A truncated view stays partial for as long as it
 * is on screen, so if the chip did not outlive the toast the condition would be
 * silently forgotten the moment it faded -- which is the whole reason SCRUM-461
 * existed. So the harness fires the auto-dismiss timer and the x, and checks the chip
 * is still there afterwards.
 *
 * Same extraction approach as before: pull the real functions out of the template by
 * name so the test cannot drift from what ships.
 *
 * Run:  node services/ui/tests/partial_warning_harness.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const template = readFileSync(join(here, '..', 'app', 'templates', 'index.html'), 'utf8');

const NAMES = ['pbNum', 'partialViewSentence', 'partialViewCopy', 'renderPartialChip',
  'showPartialToast', 'dismissPartialToast', 'setPartialView', 'clearPartialView'];
const sources = NAMES.map(name => {
  const re = new RegExp('^function ' + name + '\\(.*?^\\}', 'ms');
  const m = template.match(re);
  if (!m) throw new Error('could not extract ' + name + ' from index.html');
  return m[0];
});
// The constants the extracted code closes over.
const headM = template.match(/^const PARTIAL_HEAD = .*$/m);
const msM = template.match(/^const PARTIAL_TOAST_MS = .*$/m);
if (!headM || !msM) throw new Error('could not extract the partial-view constants');

// --- stub DOM ---
function makeEl() {
  const classes = new Set();
  return {
    innerHTML: '', title: '',
    classList: {
      add: c => classes.add(c), remove: c => classes.delete(c),
      contains: c => classes.has(c),
    },
  };
}
const els = { partialToast: makeEl(), partialChip: makeEl() };
globalThis.document = { getElementById: id => els[id] || null };

// --- controllable timers, so the auto-dismiss can be fired on demand ---
const timers = new Map();
let nextTimer = 1;
const fakeSetTimeout = (fn, ms) => { const id = nextTimer++; timers.set(id, { fn, ms }); return id; };
const fakeClearTimeout = id => timers.delete(id);
function fireTimers() { const all = [...timers.values()]; timers.clear(); all.forEach(t => t.fn()); }

const ctx = {};
new Function('ctx', 'document', 'setTimeout', 'clearTimeout',
  `${headM[0]}
   ${msM[0]}
   let partialViewState = null, partialToastTimer = null, partialToastDismissed = false;
   ${sources.join('\n\n')}
   ctx.pbNum = pbNum;
   ctx.sentence = partialViewSentence;
   ctx.copy = partialViewCopy;
   ctx.set = setPartialView;
   ctx.clear = clearPartialView;
   ctx.show = showPartialToast;
   ctx.dismiss = dismissPartialToast;
   ctx.state = () => partialViewState;
   ctx.TOAST_MS = PARTIAL_TOAST_MS;
   ctx.HEAD = PARTIAL_HEAD;
`)(ctx, globalThis.document, fakeSetTimeout, fakeClearTimeout);

let checks = 0, failures = 0;
function ok(cond, label) {
  checks += 1;
  if (!cond) { failures += 1; console.log('  FAIL  ' + label); }
  else { console.log('  ok    ' + label); }
}
const toastShown = () => els.partialToast.classList.contains('show');
const chipShown = () => els.partialChip.classList.contains('show');
const toastText = () => els.partialToast.innerHTML.replace(/<[^>]*>/g, '');
const chipTip = () => els.partialChip.title;

const TRUNC = { cdms_pulled: 3000, cdms_in_window: 52684, truncated_by: 'deadline' };
const PARTIAL = { complete: false, partial: true, truncation: TRUNC };
const COMPLETE = { complete: true, partial: false, truncation: null };

console.log('\n1. the copy is source-neutral');
ctx.set(PARTIAL);
const everywhere = [toastText(), chipTip(), ctx.copy(TRUNC), ctx.sentence(TRUNC, true),
  ctx.sentence(TRUNC, false), ctx.HEAD].join(' | ');
for (const name of ['LeoLabs', 'leolabs', 'LEOLABS', 'Pulse']) {
  ok(!everywhere.includes(name), `"${name}" appears nowhere in the partial copy`);
}
ok(/not in risk order/.test(toastText()), 'it says messages are not in risk order');
ok(!/arrive in .* order/.test(toastText()), 'and does not attribute an order to a feed');

console.log('\n2. the copy carries the backend counts and the right wording');
ok(/PARTIAL VIEW/.test(toastText()), 'it says PARTIAL VIEW');
ok(/WORST CONJUNCTION MAY NOT BE SHOWN/.test(toastText()), 'and names the risk');
ok(/3,000/.test(toastText()) && /52,684/.test(toastText()), 'both counts, formatted');
ok(/time limit/.test(toastText()), 'deadline reads as a time limit');
ctx.set({ partial: true, truncation: { cdms_pulled: 10000, cdms_in_window: 75257, truncated_by: 'cap' } });
ok(/size limit/.test(toastText()) && !/time limit/.test(toastText()),
  'cap reads as a size limit, not a time limit');
ctx.set({ partial: true, truncation: { cdms_pulled: 500, cdms_in_window: 900, truncated_by: 'mystery' } });
ok(!/limit was reached/.test(toastText()), 'an unknown reason omits the clause');
ok(!/mystery/.test(toastText()), 'and never prints the raw enum');

console.log('\n3. the denominator is never invented');
ctx.set({ partial: true, truncation: { cdms_pulled: 3000, cdms_in_window: null, truncated_by: 'deadline' } });
ok(/unreported total/.test(toastText()), 'an unknown window size says so');
ok(!/of 3,000/.test(toastText()), 'the pulled count is not reused as the denominator');
ctx.set({ partial: true, truncation: {} });
ok(/could not be read in full/.test(toastText()), 'with no counts it still warns');
ok(!/NaN|undefined|null/.test(toastText()), 'no NaN, undefined or null leaks in');
ok(!/NaN|undefined|null/.test(chipTip()), 'nor into the chip tooltip');

console.log('\n4. partial shows both; complete shows neither');
ctx.set(PARTIAL);
ok(toastShown(), 'partial shows the toast');
ok(chipShown(), 'partial shows the chip');
ctx.set(COMPLETE);
ok(!toastShown(), 'complete hides the toast');
ok(!chipShown(), 'complete hides the chip');
ok(els.partialToast.innerHTML === '', 'and leaves no toast markup behind');
ok(chipTip() === '', 'and no stale tooltip on the chip');

console.log('\n5. THE SAFETY PROPERTY: the chip outlives the toast');
ctx.set(PARTIAL);
ok(toastShown() && chipShown(), 'both up to begin with');
fireTimers();                                  // the auto-dismiss fires
ok(!toastShown(), 'the toast auto-dismisses');
ok(chipShown(), 'the CHIP REMAINS after the toast has gone');
ok(/3,000/.test(chipTip()) && /52,684/.test(chipTip()),
  'and it still carries the counts in its tooltip');
ctx.set(PARTIAL);
ctx.dismiss();                                 // the x
ok(!toastShown(), 'the x dismisses the toast');
ok(chipShown(), 'the CHIP REMAINS after the x too');
ok(ctx.state() !== null, 'and the view is still known to be partial');

console.log('\n6. dismissing is not the same as the view changing');
ctx.clear(); ctx.set(PARTIAL); ctx.dismiss();
ok(!toastShown(), 'dismissed');
ctx.set(PARTIAL);                              // the same partial view, refetched
ok(!toastShown(), 'the same warning does not push itself back after a dismissal');
ok(chipShown(), 'but the chip is still there, so nothing is hidden');
ctx.set({ partial: true, truncation: { cdms_pulled: 9, cdms_in_window: 99, truncated_by: 'cap' } });
ok(toastShown(), 'a DIFFERENT truncation does show the toast again');
ctx.show(true);
ok(toastShown(), 'and the chip click can force it back');

console.log('\n7. no partial response can render as complete');
for (const [label, data] of [
  ['partial:true', { partial: true, truncation: TRUNC }],
  ['complete:false', { complete: false, truncation: TRUNC }],
  ['both flags', PARTIAL],
  ['partial with no truncation block', { partial: true }],
  ['partial with an empty truncation', { partial: true, truncation: {} }],
  ['partial with null counts', { partial: true,
    truncation: { cdms_pulled: null, cdms_in_window: null, truncated_by: null } }],
]) {
  ctx.clear();
  ctx.set(data);
  ok(chipShown(), `the chip is shown for ${label}`);
  ok(/PARTIAL VIEW/.test(chipTip()), `and the tooltip warns for ${label}`);
}

console.log('\n8. absent flags are complete (the pre-SCRUM-459 shape)');
for (const [label, data] of [
  ['an empty object', {}], ['null', null], ['undefined', undefined],
  ['rows but no flags', { conjunctions: [{}], total: 5 }],
]) {
  ctx.clear(); ctx.set(data);
  ok(!chipShown() && !toastShown(), `neither shown for ${label}`);
}

console.log('\n9. clearing drops both');
ctx.set(PARTIAL);
ok(toastShown() && chipShown(), 'shown first');
ctx.clear();
ok(!toastShown(), 'clear hides the toast');
ok(!chipShown(), 'clear hides the chip');
ok(ctx.state() === null, 'and forgets the state');

console.log('\n10. the number formatter and the timing');
ok(ctx.pbNum(52684) === '52,684', '52684 formats with separators');
for (const bad of [null, undefined, NaN, Infinity, 'abc', {}]) {
  ok(ctx.pbNum(bad) === null, `pbNum rejects ${String(bad)}`);
}
ok(ctx.TOAST_MS >= 5000 && ctx.TOAST_MS <= 15000,
  `the toast lingers ${ctx.TOAST_MS}ms -- long enough to read, short enough to pass`);

console.log(`\n${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
