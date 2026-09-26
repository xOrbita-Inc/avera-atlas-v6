/* services/ui/tests/cov_dome_harness.mjs
 *
 * SCRUM-464: the covariance dome's size normalizer, driven out of index.html.
 *
 * The rendering splits the drawing into a part that is a measurement and a part that
 * is a choice, and the whole honesty of the feature is that the split is clean:
 *
 *   ORIENTATION is real -- the dome is tilted by the covariance's own principal axes.
 *   SIZE is normalized -- the major axis is pinned, so magnitude never affects it.
 *   SHAPE is compressed -- monotonically, so the axis ORDER and the relative
 *   roundness between assets survive, but the raw ratio does not.
 *
 * That last one is a deliberate departure from "real shape", forced by the data: a
 * tracked LEO asset's position covariance runs 299:1 to 15,764:1 along-track, so the
 * true ratios at a readable major axis are a sub-pixel needle on every asset. The
 * compression is what makes it a dome instead of a line, and the caption says the
 * shape is not to scale because of it.
 *
 * So the harness checks: major pinned, ordering preserved, rounder draws rounder,
 * isotropic draws a sphere, floor kept, degenerate input handled without a divide by
 * zero -- and explicitly that the raw ratio is NOT preserved, so nobody restores the
 * needle by "fixing" this back. It replaces the SCRUM-463 exaggeration harness.
 *
 * Run:  node services/ui/tests/cov_dome_harness.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const template = readFileSync(join(here, '..', 'app', 'templates', 'index.html'), 'utf8');

const fn = template.match(/^function covDomeAxisScales\(.*?^\}/ms);
if (!fn) throw new Error('could not extract covDomeAxisScales from index.html');
const consts = ['GLOBE_COV_READABLE_MAJOR', 'GLOBE_COV_MIN_SCALE',
  'GLOBE_COV_ASPECT_GAMMA', 'GLOBE_COV_MIN_AXIS_RATIO'].map(n => {
  const m = template.match(new RegExp('^const ' + n + '=.*$', 'm'));
  if (!m) throw new Error('could not extract ' + n);
  return m[0];
});

const ctx = {};
new Function('ctx', `${consts.join('\n')}
  ${fn[0]}
  ctx.scales = covDomeAxisScales;
  ctx.MAJOR = GLOBE_COV_READABLE_MAJOR;
  ctx.FLOOR = GLOBE_COV_MIN_SCALE;
  ctx.GAMMA = GLOBE_COV_ASPECT_GAMMA;
  ctx.MIN_RATIO = GLOBE_COV_MIN_AXIS_RATIO;
`)(ctx);

let checks = 0, failures = 0;
function ok(cond, label) {
  checks += 1;
  if (!cond) { failures += 1; console.log('  FAIL  ' + label); }
  else { console.log('  ok    ' + label); }
}
const close = (a, b) => Math.abs(a - b) <= Math.abs(b) * 1e-12 + 1e-18;

// Real SWARM covariance shapes: strongly along-track dominated, which is why
// exaggerating them (SCRUM-463) produced a streak rather than a volume.
const REAL = [
  [27829.5, 6.8, 1.77],          // measured live on SWARM B
  [5800, 400, 250],
  [1569000, 12000, 8000],        // the worst SWARM C case
  [100, 100, 100],               // isotropic
];

console.log('\n1. the major axis is pinned to the readable size');
for (const sig of REAL) {
  const sc = ctx.scales(sig);
  ok(close(Math.max(...sc), ctx.MAJOR),
    `major axis is READABLE_MAJOR for sigmas ${sig[0]}/${sig[1]}/${sig[2]}`);
}
ok(close(Math.max(...ctx.scales([1, 1, 1])), ctx.MAJOR),
  'and for a tiny isotropic covariance too');
ok(close(Math.max(...ctx.scales([1e9, 1e9, 1e9])), ctx.MAJOR),
  'and for an enormous one -- size never depends on magnitude');

console.log('\n2. the ORDER of the axes is preserved');
for (const sig of REAL) {
  const sc = ctx.scales(sig);
  const byS = sig.map((s, i) => i).sort((a, b) => sig[b] - sig[a]);
  const byC = sc.map((v, i) => i).sort((a, b) => sc[b] - sc[a]);
  ok(byS.join() === byC.join(),
    `the longest sigma draws longest for ${sig[0]}/${sig[1]}/${sig[2]}`);
}

console.log('\n3. the compression is monotonic: rounder draws rounder');
const roundness = s => { const sc = ctx.scales(s); return Math.min(...sc) / Math.max(...sc); };
// SWARM C is 299:1, SWARM B is 15,764:1 -- C must read as the rounder of the two.
ok(roundness([1800.7, 8.7, 6.0]) > roundness([27829.5, 6.8, 1.77]),
  'a 299:1 covariance draws rounder than a 15,764:1 one');
ok(roundness([1000, 900, 800]) > roundness([1000, 100, 80]),
  'and a near-isotropic one rounder than an elongated one');
ok(close(roundness([100, 100, 100]), 1),
  'an isotropic covariance draws as a sphere');

console.log('\n4. magnitude never affects the drawing');
const small = ctx.scales([100, 50, 25]);
const large = ctx.scales([100e6, 50e6, 25e6]);
ok(small.every((v, i) => close(v, large[i])),
  'the same shape at 1,000,000x the magnitude draws identically');
const flat = ctx.scales([1000, 10, 10]);
const round = ctx.scales([1000, 1000, 1000]);
ok(!close(flat[1], round[1]), 'a different shape draws differently');

console.log('\n5. the raw ratio is deliberately NOT preserved');
// Stated as a test so this cannot be "fixed" back into an invisible needle without
// the test saying so. The real ratios are in REAL[0]: 1 : 0.00024 : 0.00006.
const swarmB = ctx.scales([27829.5, 6.8, 1.77]);
const rawMinor = 6.8 / 27829.5;
const drawnMinor = swarmB[1] / swarmB[0];
ok(drawnMinor > rawMinor * 100,
  'the minor axis is compressed far above its raw ratio');
ok(drawnMinor >= 0.05,
  `the minor axis draws at ${drawnMinor.toFixed(3)} of the major -- visible, not a needle`);
ok(drawnMinor < 1, 'but still shorter than the major, so the elongation reads');

console.log('\n6. the floor is kept');
const extreme = ctx.scales([1e9, 1, 0]);
ok(extreme.every(v => v >= ctx.FLOOR), 'no axis falls below the floor');
ok(extreme[2] === ctx.FLOOR, 'a zero minor axis sits exactly on the floor');
ok(close(extreme[0], ctx.MAJOR), 'while the major axis is still readable');

console.log('\n7. a degenerate covariance does not divide by zero');
const zero = ctx.scales([0, 0, 0]);
ok(Array.isArray(zero) && zero.length === 3, 'all-zero returns three scales');
ok(zero.every(v => v === ctx.FLOOR), 'all three sit on the floor');
ok(zero.every(v => isFinite(v)), 'and none is NaN or Infinity');

console.log('\n8. unreadable input draws nothing rather than guessing');
for (const [label, bad] of [
  ['null', null], ['undefined', undefined], ['a number', 5],
  ['two axes', [1, 2]], ['four axes', [1, 2, 3, 4]],
  ['a NaN', [1, NaN, 3]], ['an Infinity', [1, Infinity, 3]],
  ['a negative sigma', [1, -2, 3]], ['a string', [1, '2', 3]],
  ['an empty array', []],
]) {
  ok(ctx.scales(bad) === null, `rejects ${label}`);
}

console.log('\n9. the readable size is sane against the asset marker');
// The asset marker is a sphere of radius 0.018 scene units.
ok(ctx.MAJOR > 0.018 * 3, 'the dome is bigger than the marker it surrounds');
ok(ctx.MAJOR < 1.0, 'and far smaller than the Earth it orbits');

console.log(`\n${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
