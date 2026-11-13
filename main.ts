// A forecast entry for make.fun's polymarket-nvidia-ceo-forecast-01 (Node 22, standard library only).
//
// The grader runs this once per case, with only that case's file in the input directory: /input/<id>.in
// (makefun/polymarket-case/v1), and reads the /output/<id>.out it writes. For every market of the case, and every
// minute h = 1..horizon_minutes after the checkpoint `at`, the output holds the five quantiles
// (0.05, 0.25, 0.5, 0.75, 0.95) of that market's Yes price:
//
//     q = clamp(x + drift * h / 60 + z * sigma * sqrt(h), 0, 1)
//
// with x the market's last price in the case, sigma a spread in price units per square-root minute and drift an
// expected move in price units per hour. Each case is scored by the pinball loss of those five quantiles against
// the market's real grid price at each minute, so the job is to put each market's quantiles where its price will
// really be. Two consequences shape the model below:
//
//   * a band that is too narrow is punished hard on both tails (the 5% and 95% quantiles are missed by a mile
//     whenever the price walks out of them), while a band a little too wide costs a little at all five levels.
//     So the spread estimate is deliberately nudged outwards, and it is floored: a market that happens not to have
//     moved in the sample still carries news risk over the next four hours.
//   * drift is only worth stating where a trend is really there, and a wrong drift is expensive at the median,
//     so the trend is read over a long look and shrunk hard towards zero.
//
// The spread is a realised-variance estimate of the case's own 1-minute increments, taken at several horizons
// (5, 15, 30 and 60 minutes) and averaged. One horizon alone misreads a real tape: a market that only trades every
// tenth minute looks dead to a 1-minute estimate and wild to a 30-minute one, and forward-filled minutes make the
// short estimate collapse exactly when a jump lands. Averaging across horizons keeps the variance of the moves
// rather than the variance of the quiet minutes between them, and the estimate uses only the most recent span of
// the case's grid, because a market's volatility this hour says more about the next four than last week's does.
//
// Everything here is deterministic: no clock, no randomness, no network, no iteration over unordered keys. Two
// runs on the same input write the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels each row holds, in the contract's order, and the standard normal's quantile at each. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** The contract's bounds on the two parameters. */
const SIGMA_MAX = 0.05;
const DRIFT_MAX = 0.2;
/** The longest horizon a case may name. */
const MAX_HORIZON = 1440;
/** The horizons whose realised variance is averaged, and how many minutes of the grid the estimate reads. */
const HORIZONS = [5, 15, 30, 60];
// 200 minutes: on the dev sweep a 200-minute estimate beat 150 (0.02196 to 0.02213) and 300 added nothing,
// while a short one (60 minutes) is the worst of all, because it reads a quiet stretch as a calm market.
const LOOKBACK_MINUTES = 200;
/** The spread's floor, and the outward nudge applied to the estimate. */
const SIGMA_FLOOR = 0.0005;
const SIGMA_NUDGE = 1.2;
/** The drift's lookback, its shrink factor, and the least |drift| worth stating. */
const DRIFT_LOOKBACK = 180;
const DRIFT_SHRINK = 0.15;
const DRIFT_DEADBAND = 0.004;

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

/** A price clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** One market's rows: h = 1..horizon, the five quantiles, non-decreasing as sigma >= 0. */
function rows(x: number, sigma: number, drift: number, horizon: number): number[][] {
  const out: number[][] = [];
  for (let h = 1; h <= horizon; h++) out.push(Z.map((z) => price6(x + (drift * h) / 60 + z * sigma * Math.sqrt(h))));
  return out;
}

/** A market's spread (price units per square-root minute) and drift (price units per hour), from its own grid. */
function fit(prices: readonly number[]): { sigma: number; drift: number } {
  const n = prices.length;
  let acc = 0;
  let weight = 0;
  const oldest = Math.max(0, n - 1 - LOOKBACK_MINUTES);
  for (const k of HORIZONS) {
    const span = Math.min(k, n - 1);
    if (span < 1) continue;
    let sum = 0;
    let count = 0;
    for (let i = n - 1; i - span >= oldest; i--) {
      const d = (prices[i] as number) - (prices[i - span] as number);
      sum += d * d;
      count += 1;
    }
    if (count === 0) continue;
    // A span-minute increment's mean squared move, per minute.
    acc += sum / count / span;
    weight += 1;
  }
  const variance = weight === 0 ? 0 : acc / weight;
  const sigma = Math.min(SIGMA_MAX, Math.max(SIGMA_FLOOR, Math.sqrt(variance) * SIGMA_NUDGE));

  const look = Math.min(DRIFT_LOOKBACK, n - 1);
  let drift = 0;
  if (look >= 5) {
    const slope = ((prices[n - 1] as number) - (prices[n - 1 - look] as number)) / look;
    const raw = slope * 60 * DRIFT_SHRINK;
    drift = Math.abs(raw) < DRIFT_DEADBAND ? 0 : Math.max(-DRIFT_MAX, Math.min(DRIFT_MAX, raw));
  }
  return { sigma, drift };
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
  const forecasts = c.markets.map((m: unknown, k: number) => {
    if (!isObject(m) || typeof m.id !== 'string' || !Array.isArray(m.prices) || m.prices.length === 0) {
      throw new Error(`markets[${k}] is not {id, prices}`);
    }
    for (const p of m.prices) if (typeof p !== 'number' || !Number.isFinite(p)) throw new Error(`market ${m.id} holds a price that is not a number`);
    const prices = m.prices as number[];
    const { sigma, drift } = fit(prices);
    return { id: m.id, q: rows(prices[prices.length - 1] as number, sigma, drift, horizon) };
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
