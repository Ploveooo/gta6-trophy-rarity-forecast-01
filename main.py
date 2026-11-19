"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

A counts-based dilution + progression forecaster. Exophase earn rates are
r = 100 * E / N, where N is the players tracked and E the earners of a trophy.
As launch week runs, N explodes (dilution pulls every rate down) while E grows
as players progress through the game (progression pushes rates up). The model
fits both counts separately and divides, instead of extrapolating the rate
itself, so a slowing player-base ramp shows up as rates recovering.

Per trophy, from the case history (deduplicated to distinct reads):
  - N(h): player base, hours h since launch. Fit a power law log N ~ log(1+h)
    and a saturating exponential N = N_inf * (1 - exp(-h/tau)) (tau by grid
    search, N_inf by least squares in log space); the forecast N is the
    geometric mean of the two, so a slowing ramp is neither ignored nor
    extrapolated forever.
  - E_i(h): earners of trophy i. Fit a power law log E ~ log(1+h) on the last
    24h of reads (the current regime); fall back to earners growing like the
    square root of player growth when the fit is degenerate.
  - median rate = 100 * E_i(target) / N(target), capped by a tier ceiling
    (platinum/gold/silver/bronze cannot plausibly exceed it in week one) and
    by the story chain: chapter trophies are non-increasing in story order,
    the platinum is at most the lowest rate, 100% completion at most the
    final-mission rate.
  - quantiles in logit space around the median, spread = random-walk term
    (0.5 * sqrt(days)) combined with the fit residual and the player-growth
    uncertainty, so a long or noisy extrapolation is honestly wide.
Flat history (the practice cases: every read identical) has no trend to fit,
so it falls back to a dilution prior: a refresh jump in N scaled by the
horizon, earners nearly flat, and a wide spread.

Standard library only, deterministic, offline. Reads every *.in in the input
directory (MAKEFUN_INPUT_DIR, else /input) and writes one *.out per case to
the output directory (MAKEFUN_OUTPUT_DIR, else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Tier ceilings: the highest week-one earn rate (as a fraction) a trophy of
# this tier can plausibly reach. A safety cap on the extrapolation, not a
# prediction: story trophies are diluted by new players, grinds barely move.
TIER_CAP = {"platinum": 0.02, "gold": 0.15, "silver": 0.30, "bronze": 0.70}
# Week-one rate prior (percent) for a trophy nobody had earned by the
# checkpoint: essentially "someone will get it, but almost nobody".
TIER_PRIOR_PCT = {"platinum": 0.02, "gold": 0.10, "silver": 0.30, "bronze": 0.80}
# Random-walk spread of a rate's move, in logit units per square-root day.
SIGMA = 0.5


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (so 1.00005 -> 1.0001, not 1.0)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation), for the levels."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("level out of (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


Z = [norm_ppf(p) for p in LEVELS]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fit_power_law(xs, ys):
    """Least squares log y = a + b * log(1 + x). Returns (a, b, resid_std) or None."""
    pts = [(math.log(1 + x), math.log(y)) for x, y in zip(xs, ys) if y > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 1e-12:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = my - b * mx
    resid = [p[1] - (a + b * p[0]) for p in pts]
    var = sum(r * r for r in resid) / max(1, n - 2)
    return (a, b, math.sqrt(var))


def fit_saturating_n(hours, counts):
    """N(h) = N_inf * (1 - exp(-h/tau)); tau by grid search, N_inf by log-space
    least squares for each tau. Returns N_inf, tau, or None."""
    pts = [(h, c) for h, c in zip(hours, counts) if c > 0 and h > 0]
    if len(pts) < 3:
        return None
    best = None
    tau = 1.0
    while tau <= 2000.0:
        logs = []
        ok = True
        for h, c in pts:
            f = 1.0 - math.exp(-h / tau)
            if f <= 1e-9:
                ok = False
                break
            logs.append(math.log(c) - math.log(f))
        if ok:
            log_ninf = sum(logs) / len(logs)
            sse = sum((math.log(c) - (log_ninf + math.log(1.0 - math.exp(-h / tau)))) ** 2
                      for h, c in pts)
            if best is None or sse < best[0]:
                best = (sse, math.exp(log_ninf), tau)
        tau *= 1.25
    if best is None:
        return None
    return best[1], best[2]


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    # Launch reference: midnight UTC on the release date.
    rel = case.get("game", {}).get("release_date", "2026-11-19")
    try:
        launch = datetime.fromisoformat(rel + "T00:00:00+00:00")
    except ValueError:
        launch = parse_time(history[0]["slot"]).replace(hour=0, minute=0, second=0, microsecond=0)
    target = parse_time(case["target"])
    h_target = max(0.0, (target - launch).total_seconds() / 3600.0)

    # Deduplicate to distinct reads (identical consecutive states carry no
    # information and would overweight the flat launch-night page).
    reads = []
    for rec in history:
        key = (rec["players_tracked"], tuple(rec["rates"]))
        if not reads or reads[-1][0] != key:
            reads.append((key, rec))
    states = [rec for _, rec in reads]
    hours = [(parse_time(r["at"]) - launch).total_seconds() / 3600.0 for r in states]
    counts = [float(r["players_tracked"]) for r in states]
    n_last = counts[-1]
    flat = len(states) < 3 or (len(set(counts)) == 1 and
                               all(tuple(r["rates"]) == tuple(states[0]["rates"]) for r in states))

    # ---- player base N(target) ----
    if flat:
        # No trend to fit: assume a refresh jump in N. The rules document the
        # launch-night refresh taking N 638 -> 3868, about 6x; a horizon of
        # 1.5h or more is taken to include such a refresh in full.
        n_target = n_last * (1.0 + 5.0 * min(1.0, horizon_hours / 1.5))
        n_growth = n_target / n_last
    else:
        # Fit on the last 24h when there is enough of it, else the whole history.
        cut = hours[-1] - 24.0
        idx = [i for i, h in enumerate(hours) if h >= cut]
        if len(idx) < 3:
            idx = list(range(len(hours)))
        fh = [hours[i] for i in idx]
        fn = [counts[i] for i in idx]
        pl = fit_power_law(fh, fn)
        n_power = math.exp(pl[0] + pl[1] * math.log(1 + h_target)) if pl else None
        sat = fit_saturating_n(fh, fn)
        n_sat = sat[0] * (1.0 - math.exp(-h_target / sat[1])) if sat else None
        cands = [v for v in (n_power, n_sat) if v is not None and v > 0]
        if not cands:
            n_target = n_last
        elif len(cands) == 1:
            n_target = cands[0]
        else:
            n_target = math.sqrt(cands[0] * cands[1])
        n_target = max(n_target, n_last)
        n_growth = n_target / n_last

    # ---- per-trophy earner counts and median rates ----
    medians = {}
    spreads = {}
    for ti, trophy in enumerate(trophies):
        series = []
        for j, r in enumerate(states):
            rate = r["rates"][ti]
            if rate is None:
                continue
            series.append((hours[j], rate / 100.0 * counts[j]))
        last_rate = states[-1]["rates"][ti]
        last_rate = 0.0 if last_rate is None else float(last_rate)
        # Tier ceiling on the forecast rate (fraction).
        cap = TIER_CAP.get(trophy.get("tier", "bronze"), 0.70)
        if last_rate >= 99.99:
            cap = 1.0
        else:
            cap = max(cap, min(0.99, last_rate / 100.0 * 2.0))

        if flat:
            # Earners barely move over a refresh window; the rate falls by
            # the dilution factor (N jumps ~6x). Immediate trophies
            # (already ~100%) stay. Early story trophies earn a little
            # faster than the rest, but not enough to offset dilution.
            if last_rate >= 99.99:
                med = 100.0
            else:
                e_last = last_rate / 100.0 * n_last
                prog = 1.0 if trophy.get("tier") in ("bronze",) and ti < 14 else 0.3
                e_t = e_last * (1.0 + 0.2 * (horizon_hours / 24.0) * prog)
                med = 100.0 * e_t / n_target
            u = 1.2
        else:
            cut = hours[-1] - 24.0
            recent = [(h, e) for h, e in series if h >= cut]
            if len(recent) < 3:
                recent = series
            e_last = series[-1][1] if series else 0.0
            if e_last <= 0:
                # Nobody had earned it by the checkpoint: tier prior.
                med = TIER_PRIOR_PCT.get(trophy.get("tier", "bronze"), 0.8)
                u = 1.0
            else:
                pl = fit_power_law([h for h, _ in recent], [e for _, e in recent])
                if pl is not None:
                    a, b, resid = pl
                    e_t = math.exp(a + b * math.log(1 + h_target))
                    u = max(0.5, resid)
                else:
                    # Degenerate fit: earners grow like sqrt of player growth.
                    e_t = e_last * math.sqrt(n_growth)
                    u = 0.7
                e_t = min(e_t, n_target * cap)
                med = 100.0 * e_t / n_target
        med = clamp_pct(min(med, cap * 100.0))
        medians[trophy["id"]] = med
        # Spread: random-walk term + fit uncertainty + player-growth uncertainty.
        u_n = 0.25 * math.log(max(1.0, n_growth))
        spreads[trophy["id"]] = math.sqrt((SIGMA * math.sqrt(horizon_days)) ** 2 + u ** 2 + u_n ** 2)

    # ---- structural constraints (story order) ----
    # Chapter + story-mission trophies (ids 3..13) are earned in order: a later
    # one cannot exceed an earlier one. Platinum (1) needs them all; 100%
    # completion (14) needs the final mission (7).
    chain = [t["id"] for t in trophies if 3 <= t["id"] <= 13]
    run = None
    for tid in chain:
        if run is None or medians[tid] < run:
            run = medians[tid]
        else:
            medians[tid] = run
    if 1 in medians and chain:
        medians[1] = min(medians[1], min(medians[t] for t in chain))
    if 14 in medians and 7 in medians:
        medians[14] = min(medians[14], medians[7])

    rows = []
    for trophy in trophies:
        tid = trophy["id"]
        lg = logit(medians[tid])
        s = spreads[tid]
        qs = [round4(clamp_pct(inv_logit(lg + z * s))) for z in Z]
        # Rounding can only tie, never invert; enforce non-decreasing anyway.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
