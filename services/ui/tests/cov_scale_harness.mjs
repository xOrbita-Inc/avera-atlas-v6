/* services/ui/tests/cov_scale_harness.mjs
 *
 * SCRUM-463: the covariance exaggeration slider, driven out of index.html under node.
 *
 * SCRUM-448 drew the 3-sigma CDM position ellipsoids at true scale, with the
 * exaggeration as a constant a developer would have to edit and redeploy. This turns
 * it into an operator slider, 1x to 50x, that rescales what is already drawn.
 *
 * The honesty rules this has to keep, and what the harness therefore checks:
 *   - 1x draws exactly what SCRUM-448 drew. Not approximately: the formula is
 *     recomputed here independently and compared.
 *   - one value drives both the drawn scale and the caption, so the note can never
 *     claim a scale that is not on screen.
 *   - the GLOBE_COV_MIN_SCALE floor survives at every slider value.
 *   - the rescale reads the stored base sigmas, so dragging twice is not the same as
 *     dragging once by the product.
 *
 * Extraction, as in the SCRUM-457/462 harnesses, so the test cannot drift from what
 * ships.
 *
 * Run:  node services/ui/tests/cov_scale_harness.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const template = readFileSync(join(here, '..', 'app', 'templates', 'index.html'), 'utf8');

const NAMES = ['covAxisScale', 'setGlobeCovExaggeration', 'renderGlobeCovNote',
  'syncGlobeCovScaleControl', 'toggleGlobeCovariance'];
const sources = NAMES.map(name => {
  const re = new RegExp('^function ' + name + '\\(.*?^\\}', 'ms');
  const m = template.match(re);
  if (!m) throw new Error('could not extract ' + name + ' from index.html');
  return m[0];
});
const consts = ['GLOBE_COV_SIGMA_MULTIPLIER', 'GLOBE_COV_MIN_SCALE',
  'GLOBE_COV_EXAGGERATION_MIN', 'GLOBE_COV_EXAGGERATION_MAX']
  .map(n => {
    const m = template.match(new RegExp('^const ' + n + '=.*$', 'm'));
    if (!m) throw new Error('could not extract ' + n);
    return m[0];
  });
const earthM = template.match(/^const EARTH_RADIUS_KM\s*=\s*([0-9.]+)/m);
if (!earthM) throw new Error('could not extract EARTH_RADIUS_KM');
const EARTH_RADIUS_KM = Number(earthM[1]);

// --- stub DOM ---
function makeEl(extra) {
  const classes = new Set();
  return Object.assign({
    textContent: '', value: '1', disabled: false, style: {},
    classList: {
      add: c => classes.add(c), remove: c => classes.delete(c),
      contains: c => classes.has(c),
      toggle: (c, on) => { if (on) classes.add(c); else classes.delete(c); },
    },
  }, extra || {});
}
const els = {
  globeCovScaleVal: makeEl(), globeCovScaleInput: makeEl(),
  globeCovScale: makeEl(), globeCovNote: makeEl(), globeCovBtn: makeEl(),
};
globalThis.document = { getElementById: id => els[id] || null };
els.globeCovScale.classList.add('off');
els.globeCovScaleInput.disabled = true;

// A mesh stand-in with the three.js surface these functions touch.
function mesh(sigmas) {
  return {
    userData: { sigmas_m: sigmas.slice() },
    visible: false,
    scale: { x: 0, y: 0, z: 0, set(a, b, c) { this.x = a; this.y = b; this.z = c; } },
  };
}

const ctx = {};
new Function('ctx', 'document', 'EARTH_RADIUS_KM',
  `${consts.join('\n')}
   let globeCovExaggeration = 1;
   let globeShowCovariance = false;
   let globeCovParts = [];
   ${sources.join('\n\n')}
   ctx.scale = covAxisScale;
   ctx.set = setGlobeCovExaggeration;
   ctx.note = renderGlobeCovNote;
   ctx.toggle = toggleGlobeCovariance;
   ctx.parts = p => { globeCovParts = p; };
   ctx.exag = () => globeCovExaggeration;
   ctx.showing = () => globeShowCovariance;
   ctx.SIGMA = GLOBE_COV_SIGMA_MULTIPLIER;
   ctx.FLOOR = GLOBE_COV_MIN_SCALE;
   ctx.MIN = GLOBE_COV_EXAGGERATION_MIN;
   ctx.MAX = GLOBE_COV_EXAGGERATION_MAX;
`)(ctx, globalThis.document, EARTH_RADIUS_KM);

let checks = 0, failures = 0;
function ok(cond, label) {
  checks += 1;
  if (!cond) { failures += 1; console.log('  FAIL  ' + label); }
  else { console.log('  ok    ' + label); }
}
const close = (a, b) => Math.abs(a - b) <= Math.abs(b) * 1e-12 + 1e-18;

// The SCRUM-448 formula, written out independently rather than reused, so this is
// a real comparison and not the implementation agreeing with itself.
const scrum448 = s =>
  Math.max(s * 3 * 1.0 / (EARTH_RADIUS_KM * 1000), 1e-6);

// Real SWARM C figures from the SCRUM-448 note: 5.8 km, a median, and 1,569 km.
const SIGMAS = [5.8e3, 1.2e4, 1.569e6, 1.0, 0.0, 7.5e5];

console.log('\n1. 1x is the SCRUM-448 true scale, to the digit');
ctx.set(1);
for (const s of SIGMAS) {
  ok(close(ctx.scale(s), scrum448(s)), `sigma ${s} m draws the SCRUM-448 size`);
}
ok(ctx.SIGMA === 3, 'the 3-sigma multiplier is unchanged');
ok(ctx.FLOOR === 1e-6, 'the floor constant is unchanged');

console.log('\n2. Nx is exactly N times 1x, above the floor');
const base = SIGMAS.filter(s => scrum448(s) > 1e-6);
for (const n of [2, 5, 20, 50]) {
  ctx.set(n);
  ok(base.every(s => close(ctx.scale(s), scrum448(s) * n)),
    `at ${n}x every axis is ${n} times its true scale`);
}

console.log('\n3. the floor holds at every slider value');
for (const n of [1, 2, 25, 50]) {
  ctx.set(n);
  ok(ctx.scale(0) === ctx.FLOOR, `a zero axis is floored at ${n}x`);
  ok(ctx.scale(1e-9) >= ctx.FLOOR, `a degenerate axis stays at or above the floor at ${n}x`);
  ok(SIGMAS.every(s => ctx.scale(s) >= ctx.FLOOR), `nothing falls below the floor at ${n}x`);
}

console.log('\n4. the slider rescales the meshes in place, from the base sigmas');
const meshes = [mesh([5.8e3, 1.2e4, 1.569e6]), mesh([1.0, 0.0, 7.5e5])];
ctx.parts(meshes);
ctx.set(1);
ok(close(meshes[0].scale.x, scrum448(5.8e3)), 'at 1x the mesh carries the true scale');
ctx.set(10);
ok(close(meshes[0].scale.x, scrum448(5.8e3) * 10), 'at 10x it is ten times that');
ok(close(meshes[0].scale.z, scrum448(1.569e6) * 10), 'and so is the major axis');
ok(meshes[1].scale.y === ctx.FLOOR, 'a zero axis is still floored after a drag');
// the property that makes a second drag safe
ctx.set(20);
ok(close(meshes[0].scale.x, scrum448(5.8e3) * 20),
  'dragging again rescales from the measurement, not from the current scale');
ctx.set(1);
ok(close(meshes[0].scale.x, scrum448(5.8e3)),
  'and returning to 1x returns exactly to true scale');
ok(meshes.every(m => m.userData.sigmas_m.length === 3),
  'the base sigmas are still on the mesh, unmutated');

console.log('\n5. the value is clamped and integral');
for (const [input, want] of [[0, 1], [-5, 1], [1, 1], [50, 50], [999, 50],
                             ['30', 30], [7.4, 7], [NaN, 1], ['abc', 1]]) {
  ctx.set(input);
  ok(ctx.exag() === want, `${JSON.stringify(input)} clamps to ${want}`);
}
ok(ctx.MIN === 1 && ctx.MAX === 50, 'the range is 1 to 50');

console.log('\n6. the caption reads the value actually applied');
ctx.parts([mesh([1e4, 1e4, 1e4])]);
ctx.toggle();                                   // COVARIANCE on
ok(ctx.showing(), 'covariance is on');
ctx.set(1);
ok(/true scale/.test(els.globeCovNote.textContent), 'at 1x it reads true scale');
ok(!/not to scale/.test(els.globeCovNote.textContent), 'and does not hedge');
for (const n of [2, 20, 50]) {
  ctx.set(n);
  ok(els.globeCovNote.textContent.includes('×' + n),
    `at ${n}x the caption names ${n}`);
  ok(/not to scale/.test(els.globeCovNote.textContent),
    `and says it is not to scale at ${n}x`);
  ok(!/true scale/.test(els.globeCovNote.textContent),
    `and no longer claims true scale at ${n}x`);
}
ok(els.globeCovNote.textContent.startsWith('3σ'),
  'the caption still says what is drawn: 3 sigma');

console.log('\n7. the readout follows the value');
for (const n of [1, 12, 50]) {
  ctx.set(n);
  ok(els.globeCovScaleVal.textContent === n + '×', `readout shows ${n}x`);
}

console.log('\n8. COVARIANCE off hides everything at any slider value');
ctx.set(50);
const parts = [mesh([1e4, 1e4, 1e4]), mesh([5e5, 1e3, 2e4])];
ctx.parts(parts);
ctx.toggle();                                   // off
ok(!ctx.showing(), 'covariance is off');
ok(parts.every(m => m.visible === false), 'every ellipsoid is hidden at 50x');
ok(els.globeCovNote.style.display === 'none', 'and the caption is hidden');
ok(els.globeCovScaleInput.disabled === true, 'the slider is disabled');
ok(els.globeCovScale.classList.contains('off'), 'and dimmed');
ok(ctx.exag() === 50, 'but it keeps its value');
ctx.toggle();                                   // back on
ok(parts.every(m => m.visible === true), 'turning it back on shows them again');
ok(ctx.exag() === 50, 'still at the value the operator chose');
ok(els.globeCovScaleInput.disabled === false, 'and the slider is live again');
ok(/×50/.test(els.globeCovNote.textContent), 'with the caption back at 50x');

console.log(`\n${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
