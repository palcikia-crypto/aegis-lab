"""AEGIS-Lab worker: fetch -> resolve -> decide (multi-strategy + regime)
-> ML gate -> open -> adapt -> save -> dashboard."""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src import market, features, engine, dashboard  # noqa: E402
from src import strategies, universe  # noqa: E402

OPT_PATH = os.path.join(HERE, "results", "opt.json")
MODEL_PATH = os.path.join(HERE, "results", "model.json")

with open(os.path.join(HERE, "config", "settings.json")) as f:
    CFG = json.load(f)

if os.path.exists(OPT_PATH):
    try:
        with open(OPT_PATH) as f:
            CFG.update(json.load(f).get("cfg_overrides", {}))
    except Exception:
        pass


def load_opt():
    if not os.path.exists(OPT_PATH):
        return None
    try:
        with open(OPT_PATH) as f:
            return json.load(f)
    except Exception:
        return None


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

    opt = load_opt()
    if opt and not st.get("adapt"):
        engine.seed_adapt(st, opt.get("combo_stats", {}))

    symbols = universe.load_universe(n=CFG.get("universe_size", 8)) \
        if CFG.get("auto_universe", True) else CFG["symbols"]
    # positions may live on coins that left the universe — resolve them too
    for p in st["positions"]:
        if p["symbol"] not in symbols:
            symbols.append(p["symbol"])
    universe.apply_specs(CFG, symbols)

    symbol_bars = {}
    sources = {}
    for sym in symbols:
        bars, src = market.fetch_klines(sym, limit=800)
        symbol_bars[sym] = bars
        sources[sym] = src
        time.sleep(0.2)

    # 1) resolve open positions against fresh bars (updates adapt brain)
    engine.resolve_positions(st, trades, symbol_bars, CFG, now_ms)

    # 2) day rollover + breakers
    engine.roll_day(st, now_ms)
    ok, reason = engine.breakers_ok(st, CFG, now_ms)

    opened = []
    gated = 0
    model = load_model()
    regimes = {}
    if ok:
        for sym in symbols:
            if len(st["positions"]) >= CFG["max_concurrent"]:
                break
            if any(p["symbol"] == sym for p in st["positions"]):
                continue
            bars = symbol_bars[sym]
            if not bars or now_ms - bars[-1]["ts"] > 5 * 60_000:
                continue  # stale data, skip
            feat = features.features(bars, CFG)
            decision = strategies.decide(feat, CFG, st.get("adapt"))
            if decision:
                side, strat, reg = decision
                regimes[sym] = reg
                allow, p = gate(model, feat, side)
                if not allow:
                    gated += 1
                    continue
                pos = engine.open_positions(st, CFG, sym, feat, side, now_ms,
                                            sources[sym], strategy=strat,
                                            regime=reg)
                if pos:
                    opened.append(f"{pos['id']}:{strat}")

    # 3) persist
    engine.save_state(st)
    engine.save_trades(trades)
    engine.append_equity(now_ms, st["equity"])
    dashboard.generate(st, trades, CFG)

    result = {
        "now": now_ms,
        "sources": sources,
        "regimes": regimes,
        "opened": opened,
        "ml_gated": gated,
        "ml_active": model is not None,
        "adapt_combos": len(st.get("adapt", {})),
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
