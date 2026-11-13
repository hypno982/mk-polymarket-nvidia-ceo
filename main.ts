// make.fun polymarket-nvidia-ceo-forecast-01 — a quantile forecast entry (Node 22, standard library only).
//
// The grader runs this once per case, with only that case's file in the input directory:
//
//     /input/<id>.in    one makefun/polymarket-case/v1 file
//     /output/<id>.out  the forecast it reads back
//
// A case is a checkpoint `at` and, for every market of the event, that market's Yes price on a 1-minute
// UTC grid from its `start` up to `at`. The forecast the contest asks for is, for each market and each
// minute h = 1..horizon_minutes after `at`, five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the price:
//
//     q = clamp(x + drift * h / 60 + z * sigma * sqrt(h), 0, 1)
//
// x the market's last price in the case, z the standard normal's quantile at each level, sigma a spread in
// price units per square-root minute and drift an expected move in price units per hour. The case is scored
// by the pinball loss of those five numbers against the price the market actually printed at each minute.
//
// So the whole job is two numbers a market: how wide its price will wander over four hours (sigma), and
// which way it is being pushed (drift). Both are read off the case's own grid:
//
//   sigma  a realised variance of the increments at several spans (1, 5, 15, 30 and 60 minutes), each
//          normalised to one minute, weighted toward the recent tape and pooled toward the other markets of
//          the event for a market whose grid is too short or too stale to say much on its own. Reading one
//          span alone is wrong on the tape this contest is about: a forward-filled grid looks dead at one
//          minute and jumpy at sixty, and a price that steps once every twenty minutes has the same variance
//          at twenty minutes as at one, not a twentieth of it.
//   drift  a small, heavily shrunk slope of the recent tape, plus a pull toward the middle for a price
//          pinned near 0 or 1, where the clamp already does the work and a loose drift would only drag the
//          quantiles inward.
//
// Deterministic by construction: no clock, no randomness, no network, no iteration over anything unordered.
//
// Outside the grader's container MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The levels each row holds, in order, and the standard normal's quantile at each (the contract's own). */
const QUANTILES: readonly number[] = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z: readonly number[] = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

/** The contract's caps. */
const SIGMA_MAX = 0.05;
const DRIFT_MAX = 0.2;
const MAX_HORIZON = 1440;

/** The most recent minutes the variance reads, so a market that changed regime two hours ago is not read at its old one. */
const LOOKBACK = 360;
/** The spans a variance is read at, in minutes, and how much each counts. */
const SPANS: readonly (readonly [number, number])[] = [
  [1, 0.15],
  [5, 0.2],
  [15, 0.25],
  [30, 0.2],
  [60, 0.2],
];
/** A floor on sigma: a market that has not printed a move in the sample still carries overnight news risk. */
const SIGMA_FLOOR = 0.0006;
/** How many minutes a market's own tape must hold before the event's pooled sigma stops standing in for it. */
const OWN_MINUTES = 45;
/** A price pinned within this of a bound is pulled gently back toward it rather than allowed to drift past. */
const EDGE = 0.05;
/** The most an hour of edge pull may add, in price units: a market at 0 gets the full pull, one at EDGE none. */
const EDGE_PULL = 0.004;
/** Drift's slope lookback, minutes, and how hard it is shrunk (the tape's slope is mostly noise over four hours). */
const DRIFT_LOOKBACK = 90;
const DRIFT_SHRINK = 0.15;
/** A slope below this many price units a minute is read as no trend at all. */
const DRIFT_DEADBAND = 0.004;

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

/** A price clamped to [0,1] and rounded to 6 decimals (0, never -0): the baseline's own rounding. */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** One market's rows: h = 1..horizon, five quantiles, non-decreasing by construction (sigma >= 0, z ascending). */
function rows(x: number, sigma: number, drift: number, horizon: number): number[][] {
  const out: number[][] = [];
  for (let h = 1; h <= horizon; h++) out.push(Z.map((z) => price6(x + (drift * h) / 60 + z * sigma * Math.sqrt(h))));
  return out;
}

/** One market's mean squared increment per minute at `span`, over the last LOOKBACK minutes. */
function spanVariance(prices: readonly number[], span: number): { variance: number; count: number } {
  const n = prices.length;
  if (n - span < 1) return { variance: 0, count: 0 };
  const oldest = Math.max(0, n - 1 - LOOKBACK);
  let sum = 0;
  let count = 0;
  for (let i = n - 1; i - span >= oldest; i--) {
    const d = (prices[i] as number) - (prices[i - span] as number);
    sum += d * d;
    count += 1;
  }
  return count === 0 ? { variance: 0, count: 0 } : { variance: sum / count / span, count };
}

/** A market's spread over one minute, from its own grid (0 when the grid is too short to say). */
function marketVariance(prices: readonly number[]): number {
  let acc = 0;
  let weight = 0;
  for (const [span, w] of SPANS) {
    const { variance, count } = spanVariance(prices, span);
    // A span with fewer than a handful of increments says more about the grid's edge than the market.
    if (count < 4) continue;
    acc += w * variance;
    weight += w;
  }
  return weight === 0 ? 0 : acc / weight;
}

/** A market's drift: a shrunk recent slope, plus a small pull for a price pinned at a bound.
 *  A price at 0 or 1 is held there by the market's structure, not pushed to 0.5: the pull is away from the
 *  bound only in proportion to how deep in the edge band the price sits, and it is small, so a market at 3c
 *  barely moves while one at 0.2c gets the most. */
function marketDrift(prices: readonly number[]): number {
  const n = prices.length;
  const x = prices[n - 1] as number;
  let drift = 0;
  const look = Math.min(DRIFT_LOOKBACK, n - 1);
  if (look >= 5) {
    const slope = (x - (prices[n - 1 - look] as number)) / look;
    drift = Math.abs(slope) < DRIFT_DEADBAND ? 0 : slope * 60 * DRIFT_SHRINK;
  }
  if (x < EDGE) drift += (1 - x / EDGE) * EDGE_PULL;
  else if (x > 1 - EDGE) drift -= (1 - (1 - x) / EDGE) * EDGE_PULL;
  return Math.max(-DRIFT_MAX, Math.min(DRIFT_MAX, drift));
}

/** Case `file`'s output text, or why it has none. */
function caseOutput(file: string): string {
  const c: unknown = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8'));
  if (!isObject(c)) throw new Error('it is not a JSON object');
  const horizon = c.horizon_minutes;
  if (typeof horizon !== 'number' || !Number.isInteger(horizon) || horizon < 1 || horizon > MAX_HORIZON) {
    throw new Error(`its horizon_minutes ${JSON.stringify(horizon)} is not a whole number from 1 to ${MAX_HORIZON}`);
  }
  const levels = c.quantiles;
  if (!Array.isArray(levels) || levels.length !== QUANTILES.length || levels.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`it asks for quantiles ${JSON.stringify(levels)}, not ${JSON.stringify(QUANTILES)}`);
  }
  if (!Array.isArray(c.markets) || c.markets.length === 0) throw new Error('it holds no markets');
  const markets = c.markets.map((m: unknown, k: number) => {
    if (!isObject(m) || typeof m.id !== 'string' || !Array.isArray(m.prices) || m.prices.length === 0) {
      throw new Error(`markets[${k}] is not {id, prices}`);
    }
    for (const p of m.prices) if (typeof p !== 'number' || !Number.isFinite(p)) throw new Error(`market ${m.id} holds a price that is not a number`);
    return { id: m.id, prices: m.prices as number[] };
  });
  // The event's pooled variance: the markets' own, each weighted by how much tape it has to prove it.
  let pooledAcc = 0;
  let pooledWeight = 0;
  for (const m of markets) {
    const v = marketVariance(m.prices);
    const w = Math.min(m.prices.length - 1, LOOKBACK);
    if (v <= 0 || w <= 0) continue;
    pooledAcc += v * w;
    pooledWeight += w;
  }
  const pooled = pooledWeight === 0 ? SIGMA_FLOOR * SIGMA_FLOOR : pooledAcc / pooledWeight;
  const forecasts = markets.map((m) => {
    const n = m.prices.length;
    const own = marketVariance(m.prices);
    // Partial pooling: a short or flat grid borrows the event's variance, a long one keeps its own.
    const k = Math.max(0, Math.min(1, (n - 1) / OWN_MINUTES));
    const variance = n >= 2 ? k * (own > 0 ? own : pooled) + (1 - k) * (own > 0 ? Math.max(own, pooled) : pooled) : pooled;
    const sigma = Math.max(SIGMA_FLOOR, Math.min(SIGMA_MAX, Math.sqrt(variance)));
    return { id: m.id, q: rows(m.prices[n - 1] as number, sigma, marketDrift(m.prices), horizon) };
  });
  return `${JSON.stringify({ forecasts })}\n`;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
for (const file of readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort()) {
  let out: string;
  try {
    out = caseOutput(file);
  } catch (err) {
    process.stderr.write(`forecast: ${file}: ${err instanceof Error ? err.message : String(err)}, so it gets no output\n`);
    process.exitCode = 1;
    continue;
  }
  writeFileSync(join(OUTPUT_DIR, `${file.slice(0, -'.in'.length)}.out`), out);
}
