# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ploveooo/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ploveooo/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for each of the 55 Grand Theft Auto VI PS5 trophies, the earn rate (percent of tracked players) at a target time, from the Exophase trophy page's history up to a checkpoint. The entry is `main.py`: it reads one case file and writes five quantiles (5%, 25%, 50%, 75%, 95%) of each trophy's earn rate. Scoring is mean pinball loss on the logit scale, and only entries beating the best house baseline qualify for a prize.

## The model

The rate is a ratio, so it is forecast as a ratio: earners divided by tracked players.

- **Decomposition.** For each trophy, the earners `E = rate/100 × N` and the tracked-player count `N` are modelled separately, then the rate is `100 × E/N`. This matters because the two move in opposite directions early on: new players flood in (N jumps, diluting every rate) while players also earn trophies (E grows).
- **Players.** `N(h)` is fitted two ways in hours since launch: a power law `log N ~ log(1+h)` and a saturating exponential `N = N_inf × (1 − exp(−h/τ))` with a τ grid search. The forecast uses their geometric mean, so the player count keeps growing but slows down.
- **Earners.** Each trophy's `E(h)` gets a power-law fit `log E ~ log(1+h)` on the last 24 hours of reads. If a trophy has no earners yet, a tier prior is used.
- **Flat history.** Launch night's reads are all identical (the page had not refreshed). With no trend to fit, the model uses the documented first refresh: N jumps ~6× (638 → 3,868 players) within the first hours, earners barely move, so rates fall by dilution. Early story trophies earn a little faster than the rest.
- **Constraints.** Tier ceilings on the forecast rate (platinum 2%, gold 15%, silver 30%, bronze 70%); the story chain is monotone (Chapter 1 ≥ Chapter 2 ≥ … ≥ final mission, platinum ≤ the chain minimum, the 100% trophy ≤ the final mission); the always-earned trophy stays at 100%.
- **Uncertainty.** Quantiles are taken in logit space around the median, with the spread from a random-walk term (0.5 × √days) plus the fit residual plus player-growth uncertainty, so a longer horizon honestly widens the band.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, …]}` with 55 rows. Runs in ~0.08 s per case, deterministic (two runs are byte-identical).

## Validation

Backtest on a synthetic case built from the real launch-night history (checkpoint 2026-11-19 03:00 UTC, target 2026-11-19 11:49 UTC, 8.8 h horizon), scored against the actual Exophase page captured at 11:49 UTC:

| model | pinball loss | typical median miss | 90% coverage |
|---|---|---|---|
| this model | **0.300** | 0.047 pts | 0.982 |
| house persistence baseline | 1.382 | 0.677 pts | 0.000 |

The house baseline misses because it does not model the dilution: in the first 12 hours the tracked-player count rose ~6–8× and almost every rate fell 3–8×. On a synthetic rich-history case (22 reads, 6.5-day horizon to 2026-11-26) the model also produces sane week-one medians (platinum ~0.09%, Chapter 1 ~0.78%, fishing ~2.3%, the 100% trophy at 100%).

---

<sub>Entered on [make.fun](https://make.fun/?ref=b2qhqcdv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
