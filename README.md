# Polymarket next-NVIDIA-CEO price forecast

An entry for make.fun's `polymarket-nvidia-ceo-forecast-01`: a quantile forecast of every market in a
Polymarket event on NVIDIA's next CEO, minute by minute, four hours ahead.

The grader runs this program once per case. Each case (`/input/<id>.in`, `makefun/polymarket-case/v1`) holds the
event's markets with their Yes price on a one-minute UTC grid up to a checkpoint `at`. The program writes
`/output/<id>.out`: for every market, and every minute `h = 1..horizon_minutes` after `at`, the five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of that market's price,

    q = clamp(x + drift * h / 60 + z * sigma * sqrt(h), 0, 1)

`x` is the market's last price in the case, `z` the standard normal's quantile at each level, `sigma` a spread in
price units per square-root minute and `drift` an expected move in price units per hour.

## The model

Each case is scored by the pinball loss of those quantiles against the market's real price at each minute, which
shapes the estimate:

- **Spread** is the realised variance of the case's own one-minute increments, taken at four horizons (5, 15, 30
  and 60 minutes) over the last 200 minutes of the grid and averaged, then nudged outwards by 20% and floored at
  0.0005. Averaging over horizons keeps a jump's variance instead of losing it to the forward-filled quiet minutes
  around it, and a market that has not moved in the sample still carries news risk over the next four hours. The
  outward nudge is there because a band that is too narrow is punished on both tails while one a little too wide
  costs a little at all five levels.
- **Drift** is the price change over the last 180 minutes, scaled to an hourly rate and shrunk to 15% of it, with
  anything under 0.004 dropped to zero: a trend that is really there still shows after the shrink, and a wrong
  drift costs more at the median than it earns.

The result is the persistence baseline's shape with a spread read off the tape: on the development set it scores
about 26% below the baseline's pinball loss (0.0212 against 0.0286), and it beat the baseline in every regime
tried — fine and coarse price ticks, 120- and 240-minute histories, calm and wild volatility, half-dead markets,
trending markets, and a 60-minute horizon.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only: no dependencies, no network, no clock, no randomness. Two runs on the same input
write the same bytes.

Files: `main.ts` (the whole program), `LICENSE` (MIT), `package.json`, `.makefun/owner.json` (the claim file).
