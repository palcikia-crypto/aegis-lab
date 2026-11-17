"""AEGIS-Lab worker: fetch -> resolve -> signal -> open -> save -> dashboard."""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src import market, features, engine, dashboard  # noqa: E402

with open(os.path.join(HERE, "config", "settings.json")) as f:
    CFG = json.load(f)

MODEL_PATH = os.path.join(HERE, "results", "model.json")


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def gate(model, feat, side):
    """ML filter (Этап 3): None model -> pass-through."""
    if model is None:
        return True, None
    p = features.ml_prob(model, feat, side)
    return p >= model["threshold"], round(p, 4)


def log_crash(exc):
    os.makedirs(engine.RESULTS, exist_ok=True)
    with open(engine.CRASH_PATH, "a") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===\n")
        f.write(traceback.format_exc())


def run_cycle():
    now_ms = int(time.time() * 1000)
    st = engine.load_state(CFG)
    trades = engine.load_trades()

    symbol_bars = {}
    sources = {}
    for sym in CFG["symbols"]:
        bars, src = market.fetch_klines(sym, limit=300)
        symbol_bars[sym] = bars
        sources[sym] = src
        time.sleep(0.3)

    # 1) resolve open positions against fresh bars
    engine.resolve_positions(st, trades, symbol_bars, CFG, now_ms)

    # 2) day rollover + breakers
    engine.roll_day(st, now_ms)
    ok, reason = engine.breakers_ok(st, CFG, now_ms)

    opened = []
    gated = 0
    model = load_model()
    if ok:
        for sym in CFG["symbols"]:
            if len(st["positions"]) >= CFG["max_concurrent"]:
                break
            if any(p["symbol"] == sym for p in st["positions"]):
                continue
            bars = symbol_bars[sym]
            if not bars or now_ms - bars[-1]["ts"] > 5 * 60_000:
                continue  # stale data, skip
            feat = features.features(bars, CFG)
            side = features.signal(feat, CFG)
            if side:
                allow, p = gate(model, feat, side)
                if not allow:
                    gated += 1
                    continue
                pos = engine.open_positions(st, CFG, sym, feat, side, now_ms,
                                            sources[sym])
                if pos:
                    opened.append(pos["id"])

    # 3) persist
    engine.save_state(st)
    engine.save_trades(trades)
    engine.append_equity(now_ms, st["equity"])
    dashboard.generate(st, trades, CFG)

    result = {
        "now": now_ms,
        "sources": sources,
        "opened": opened,
        "ml_gated": gated,
        "ml_active": model is not None,
        "open_positions": len(st["positions"]),
        "equity": round(st["equity"], 2),
        "breaker": reason,
        "stats": st["stats"],
    }
    print(json.dumps(result))
    return result


def main():
    try:
        run_cycle()
    except Exception:
        log_crash(sys.exc_info()[1])
        raise  # never swallow crashes (lesson 2026-11-10..16)


if __name__ == "__main__":
    main()
