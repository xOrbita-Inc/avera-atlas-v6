/* services/ui/tests/partial_banner_harness.mjs
 *
 * SCRUM-461: drive the real partial-view banner code out of index.html under node,
 * with a stub DOM, and assert the honesty rule.
 *
 * The rule: a truncated window must announce itself and say the worst conjunction
 * may not be shown. SCRUM-459 made the planner send complete / partial /
 * truncation{cdms_pulled, cdms_in_window, truncated_by}; the dashboard ignored all
 * three, so a truncated SWARM B listing rendered as an ordinary list. The truncated
 * set is a prefix in LeoLabs order, not Pc order, so the worst conjunction can be
 * past the cut.
 *
 * Same approach as SCRUM-457's row harness: extract the real functions by name so
 * the test cannot drift from the template. It checks the logic and the copy, not the
 * appearance -- the visual is covered live.
 *
 * Run:  node services/ui/tests/partial_banner_harness.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const template = readFileSync(join(here, '..', 'app', 'templates', 'index.html'), 'utf8');

const NAMES = ['pbNum', 'partialViewCopy', 'renderPartialBanner', 'hidePartialBanner'];
const sources = NAMES.map(name => {
  const re = new RegExp('^function ' + name + '\\(.*?^\\}', 'ms');
  const m = template.match(re);
  if (!m) throw new Error('could not extract ' + name + ' from index.html');
  return m[0];
});

// --- stub DOM ---
const els = {
  conjPartialBanner: makeEl(),
  globePartialBanner: makeEl(),
};
function makeEl() {
  const classes = new Set();
  return {
    innerHTML: '',
    classList: {
      add: c => classes.add(c),
      remove: c => classes.delete(c),
      contains: c => classes.has(c),
    },
    _classes: classes,
  };
}
globalThis.document = { getElementById: id => els[id] || null };

const ctx = {};
new Function('ctx', 'document', sources.join('\n\n') + `
ctx.pbNum = pbNum;
ctx.copy = partialViewCopy;
ctx.render = renderPartialBanner;
ctx.hide = hidePartialBanner;
`)(ctx, globalThis.document);

let checks = 0, failures = 0;
function ok(cond, label) {
  checks += 1;
  if (!cond) { failures += 1; console.log('  FAIL  ' + label); }
  else { console.log('  ok    ' + label); }
}
const shown = id => els[id].classList.contains('show');
const html = id => els[id].innerHTML;
const text = id => html(id).replace(/<[^>]*>/g, '');

const PARTIAL = {
  complete: false, partial: true,
  truncation: { cdms_pulled: 3000, cdms_in_window: 53034, truncated_by: 'deadline' },
  conjunctions: [{}, {}], total: 297,
};
const COMPLETE = {
  complete: true, partial: false, truncation: null,
  conjunctions: [{}], total: 97,
};

console.log('\n1. a partial list shows the banner with the backend numbers');
ctx.render('conjPartialBanner', PARTIAL);
ok(shown('conjPartialBanner'), 'the banner is shown');
ok(/PARTIAL VIEW/.test(text('conjPartialBanner')), 'it says PARTIAL VIEW');
ok(/WORST CONJUNCTION MAY NOT BE SHOWN/.test(text('conjPartialBanner')),
  'it says the worst conjunction may not be shown');
ok(/3,000/.test(text('conjPartialBanner')), 'it carries cdms_pulled, formatted');
ok(/53,034/.test(text('conjPartialBanner')), 'it carries cdms_in_window, formatted');
ok(/time limit/.test(text('conjPartialBanner')), 'it says which bound was hit');
ok(/LeoLabs order, not risk order/.test(text('conjPartialBanner')),
  'it explains why a prefix is not the top');

console.log('\n2. a complete list shows nothing');
ctx.render('conjPartialBanner', COMPLETE);
ok(!shown('conjPartialBanner'), 'the banner is hidden');
ok(html('conjPartialBanner') === '', 'and it leaves no markup behind');

console.log('\n3. the globe uses the same words');
ctx.render('globePartialBanner', PARTIAL);
ok(shown('globePartialBanner'), 'the globe banner is shown');
ok(text('globePartialBanner') === text('conjPartialBanner') ||
   text('globePartialBanner').length > 0, 'the globe banner renders');
ctx.render('conjPartialBanner', PARTIAL);
ok(html('globePartialBanner') === html('conjPartialBanner'),
  'the globe and the table say exactly the same thing');
ctx.render('globePartialBanner', COMPLETE);
ok(!shown('globePartialBanner'), 'a complete globe shows nothing');

console.log('\n4. no partial response can render as complete');
for (const [label, data] of [
  ['partial:true', { partial: true, truncation: PARTIAL.truncation }],
  ['complete:false', { complete: false, truncation: PARTIAL.truncation }],
  ['both flags', PARTIAL],
  ['partial with no truncation block', { partial: true }],
  ['partial with an empty truncation', { partial: true, truncation: {} }],
  ['partial with null counts', { partial: true,
    truncation: { cdms_pulled: null, cdms_in_window: null, truncated_by: null } }],
]) {
  ctx.render('conjPartialBanner', data);
  ok(shown('conjPartialBanner'), `shown for ${label}`);
  ok(/PARTIAL VIEW/.test(text('conjPartialBanner')),
    `and it still says PARTIAL VIEW for ${label}`);
}

console.log('\n5. the denominator is never invented');
ctx.render('conjPartialBanner', { partial: true,
  truncation: { cdms_pulled: 3000, cdms_in_window: null, truncated_by: 'deadline' } });
let t = text('conjPartialBanner');
ok(/3,000/.test(t), 'the pulled count is shown when known');
ok(/unreported total/.test(t), 'and an unknown window size says so');
ok(!/of 3,000/.test(t), 'the pulled count is never reused as the denominator');

ctx.render('conjPartialBanner', { partial: true, truncation: {} });
t = text('conjPartialBanner');
ok(/could not be read in full/.test(t), 'with no counts at all it still warns');
ok(!/\bof\b\s*\./.test(t), 'and carries no dangling "of"');
ok(!/NaN|undefined|null/.test(t), 'no NaN, undefined or null leaks into the copy');

console.log('\n6. absent flags are not partial (the pre-SCRUM-459 shape)');
for (const [label, data] of [
  ['an empty object', {}],
  ['null', null],
  ['undefined', undefined],
  ['rows but no flags', { conjunctions: [{}], total: 5 }],
]) {
  ctx.render('conjPartialBanner', data);
  ok(!shown('conjPartialBanner'), `not shown for ${label}`);
}

console.log('\n7. hide clears whatever was there');
ctx.render('conjPartialBanner', PARTIAL);
ok(shown('conjPartialBanner'), 'shown first');
ctx.hide('conjPartialBanner');
ok(!shown('conjPartialBanner'), 'hidden after hidePartialBanner');
ok(html('conjPartialBanner') === '', 'and emptied');

console.log('\n8. the number formatter');
ok(ctx.pbNum(53034) === '53,034', '53034 formats with separators');
ok(ctx.pbNum(0) === '0', 'zero formats');
for (const bad of [null, undefined, NaN, Infinity, 'abc', {}]) {
  ok(ctx.pbNum(bad) === null, `pbNum rejects ${String(bad)}`);
}

console.log('\n9. the cap bound reads differently from the deadline');
ctx.render('conjPartialBanner', { partial: true,
  truncation: { cdms_pulled: 10000, cdms_in_window: 75257, truncated_by: 'cap' } });
t = text('conjPartialBanner');
ok(/size limit/.test(t), 'a cap says size limit');
ok(!/time limit/.test(t), 'and not time limit');
ok(/10,000/.test(t) && /75,257/.test(t), 'with both counts');

console.log(`\n${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
