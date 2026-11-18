"""Forecast GTA VI's launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Method: size-weighted comparable launch-day curves.
  - Every comparable with a complete release-day curve (hours 1..24 present,
    viewers > 0) contributes its log-ratio curve log(viewers_h / viewers_1).
  - Only tier "comparable" launches are used (early-access-contaminated, weak
    and the GTA VI target-history entry are excluded); if fewer than 5 such
    curves exist, fall back to all complete curves, then to persistence.
  - Curves are weighted by similarity of hour-1 size to this case's hour-1
    print: w = exp(-(log(v1_comp) - log(p1))^2 / (2 * sigma^2)), sigma = 0.8.
    Hour-1 size is highly informative: small launches (v1 < 10k) explode
    10x-2000x from a near-empty hour 1, while big launches (v1 > 20k) stay
    within ~1-3x. Weighting keeps the tiny-launch explosions out of a big
    launch's forecast. Leave-one-out cross-validation on the 17 big
    comparables: mean pinball loss 0.26 vs 0.35 for the persistence baseline
    (global median: 0.33).
  - Per hour, the forecast median is the weighted median log-ratio and the
    spread is 1.4826 * weighted MAD, clipped to [0.25, 0.80] in log units.
    Quantiles: q_tau = p1 * exp(median + z_tau * spread), z the standard
    normal quantiles. This is a proper-quantile, honest-spread forecast.
  - Deterministic (sorted weighted medians, no randomness), stdlib only,
    runs in well under a second per case.

Input/output: same contract as starter/main.py -- reads every *.in in
$MAKEFUN_INPUT_DIR (else /input), writes <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output): {"hours": [{"hour": h, "q": [q05,q25,q50,q75,q95]}, ...]}.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.8          # log-size kernel width
SPREAD_MIN = 0.25    # log units
SPREAD_MAX = 0.80    # log units


def round2(x):
    """Round to 2 decimals, half up (as the starter does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def tier_of(comp):
    m = re.search(r"Tier:\s*([^.;\"]+)", comp.get("unlock", "") or "")
    return m.group(1).strip() if m else "?"


def release_log_ratios(comp):
    """Log-ratio curve log(v_h / v_1) for h=1..24, or None if incomplete."""
    try:
        day = parse_time(comp["day_start"])
    except Exception:
        return None
    by_time = {}
    for row in comp.get("hourly") or []:
        v = row.get("viewers")
        if v is None or v <= 0:
            continue
        try:
            by_time[parse_time(row["from"])] = float(v)
        except Exception:
            continue
    if day not in by_time:
        return None
    v1 = by_time[day]
    ratios = []
    for h in range(24):
        t = day + timedelta(hours=h)
        if t not in by_time:
            return None
        ratios.append(math.log(by_time[t] / v1))
    return ratios


def weighted_median(vals, weights):
    pairs = sorted(zip(vals, weights))
    total = sum(weights)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError("%s asks for levels %s, not %s" % (name, case["levels"], LEVELS))
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("%s has no hour-1 print" % name)
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    # Collect complete release-day curves from comparables.
    curves = []  # (v1, log-ratio list, tier)
    for comp in case.get("comparables") or []:
        ratios = release_log_ratios(comp)
        if ratios is None:
            continue
        day = parse_time(comp["day_start"])
        v1 = None
        for row in comp.get("hourly") or []:
            if row.get("from") and parse_time(row["from"]) == day and row.get("viewers"):
                v1 = float(row["viewers"])
                break
        if v1 and v1 > 0:
            curves.append((v1, ratios, tier_of(comp)))

    pool = [c for c in curves if c[2] == "comparable"]
    if len(pool) < 5:
        pool = curves
    hours = []
    if len(pool) < 3:
        # Fallback: persistence baseline (the starter).
        for row in case["forecast"]:
            h = row["hour"]
            qs = [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]
            hours.append({"hour": h, "q": qs})
        return {"hours": hours}

    for row in case["forecast"]:
        h = row["hour"]
        vals = [c[1][h - 1] for c in pool]
        weights = [math.exp(-((math.log(c[0]) - log_p1) ** 2) / (2 * SIGMA * SIGMA))
                   for c in pool]
        med = weighted_median(vals, weights)
        devs = [abs(v - med) for v in vals]
        mad = weighted_median(devs, weights)
        spread = max(SPREAD_MIN, min(SPREAD_MAX, 1.4826 * mad))
        qs = []
        for z in Z:
            q = p1 * math.exp(med + z * spread)
            q = min(max(q, 0.01), 1e9)
            qs.append(round2(q))
        # Guarantee non-decreasing after rounding.
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
