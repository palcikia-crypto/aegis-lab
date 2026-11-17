"""Feature engineering on 1m bars (pure python, fast enough for 200 bars)."""
import statistics


def ema(values, period):
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    e = statistics.fmean(values[:period])
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e


def atr(bars, period=14):
    """ATR on the given bars (use 15m-resampled for stability)."""
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["h"], bars[i]["l"], bars[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return statistics.fmean(trs[-period:])


def resample(bars, minutes=15):
    """Resample ascending 1m bars to N-minute bars."""
    out = []
    bucket_ms = minutes * 60_000
    cur = None
    for b in bars:
        bt = b["ts"] - (b["ts"] % bucket_ms)
        if cur is None or bt != cur["ts"]:
            if cur:
                out.append(cur)
            cur = {"ts": bt, "o": b["o"], "h": b["h"], "l": b["l"],
                   "c": b["c"], "v": b["v"]}
        else:
            cur["h"] = max(cur["h"], b["h"])
            cur["l"] = min(cur["l"], b["l"])
            cur["c"] = b["c"]
            cur["v"] += b["v"]
    if cur:
        out.append(cur)
    return out


def features(bars, cfg):
    """Compute signal features from ascending 1m bars. Returns dict or None."""
    if len(bars) < 60:
        return None
    closes = [b["c"] for b in bars]
    last = bars[-1]
    f = closes[-cfg["ema_fast"] - 1:]
    s = closes[-cfg["ema_slow"] - 1:]
    ef = ema(f, cfg["ema_fast"])
    es = ema(s, cfg["ema_slow"])
    if ef is None or es is None:
        return None
    mb = cfg["momentum_bars"]
    mom = closes[-1] - closes[-1 - mb]
    vols = [b["v"] for b in bars[-21:-1]]
    vavg = statistics.fmean(vols) if vols else 0.0
    vol_spike = (last["v"] / vavg) if vavg > 0 else 0.0
    a15 = atr(resample(bars, 15), cfg["atr_period"])
    return {
        "close": last["c"],
        "ema_fast": ef,
        "ema_slow": es,
        "momentum": mom,
        "vol_spike": round(vol_spike, 3),
        "atr15": a15,
        "ts": last["ts"],
    }


def signal(feat, cfg):
    """Deterministic rule: volume spike + EMA alignment + momentum.
    Returns 'long' | 'short' | None."""
    if feat is None or feat["atr15"] is None:
        return None
    if feat["vol_spike"] < cfg["vol_spike_min"]:
        return None
    d = cfg["atr_k"] * feat["atr15"]
    if d < feat["close"] * cfg["min_atr_frac"]:
        return None  # distance too small vs fees/spread
    up = feat["ema_fast"] > feat["ema_slow"] and feat["momentum"] > 0
    dn = feat["ema_fast"] < feat["ema_slow"] and feat["momentum"] < 0
    if up:
        return "long"
    if dn:
        return "short"
    return None


def ml_vector(feat, side):
    """Feature vector for the ML gate (train.py uses the same layout)."""
    c = max(feat["close"], 1e-9)
    return [
        (feat["ema_fast"] - feat["ema_slow"]) / c,
        feat["momentum"] / c,
        feat["vol_spike"],
        feat["atr15"] / c,
        1.0 if side == "long" else -1.0,
    ]


def ml_prob(model, feat, side):
    """Probability that this signal wins, from a trained model dict."""
    import math
    x = ml_vector(feat, side)
    z = model["bias"]
    for w, v, m, s in zip(model["weights"], x, model["mean"], model["std"]):
        z += w * ((v - m) / s if s > 1e-12 else 0.0)
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
