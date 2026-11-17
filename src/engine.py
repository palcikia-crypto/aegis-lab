"""AEGIS engine: 1:1 risk/reward paper trades with expiration timers and breakers."""
import json
import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
STATE_PATH = os.path.join(RESULTS, "state.json")
TRADES_PATH = os.path.join(RESULTS, "trades.jsonl")
EQUITY_PATH = os.path.join(RESULTS, "equity.csv")
CRASH_PATH = os.path.join(RESULTS, "crash.log")


def load_state(cfg):
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "equity": cfg["start_equity"],
        "day_key": None,
        "day_start_equity": None,
        "consec_losses": 0,
        "paused_until": 0,
        "halted_until": 0,
        "positions": [],
        "stats": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        "updated": 0,
    }


def save_state(st):
    os.makedirs(RESULTS, exist_ok=True)
    st["updated"] = int(__import__("time").time() * 1000)
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=1)


def load_trades():
    if not os.path.exists(TRADES_PATH):
        return []
    out = []
    with open(TRADES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_trades(trades):
    """Keep last 5000 in jsonl, archive older into trades_archive.jsonl."""
    os.makedirs(RESULTS, exist_ok=True)
    if len(trades) > 5000:
        arch = trades[:-5000]
        apath = os.path.join(RESULTS, "trades_archive.jsonl")
        with open(apath, "a") as f:
            for t in arch:
                f.write(json.dumps(t) + "\n")
        trades = trades[-5000:]
    with open(TRADES_PATH, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def append_equity(ts_ms, equity):
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    if os.path.exists(EQUITY_PATH):
        with open(EQUITY_PATH) as f:
            rows = [l for l in f.read().splitlines() if l.strip()]
    rows.append(f"{ts_ms},{equity:.4f}")
    with open(EQUITY_PATH, "w") as f:
        f.write("\n".join(rows[-5000:]) + "\n")


def round_qty(qty, step):
    return math.floor(qty / step + 1e-12) * step


def risk_qty(cfg, symbol, entry, dist):
    """Qty so that a stop-out loses ~risk_per_trade including fees."""
    per_unit = dist + entry * cfg["fee_rate"]
    if per_unit <= 0:
        return 0.0
    qty = round_qty(cfg["risk_per_trade"] / per_unit, cfg["qty_step"][symbol])
    if qty < cfg["min_qty"][symbol]:
        return 0.0
    return qty


def resolve_positions(st, trades, symbol_bars, cfg, now_ms):
    """Resolve open positions against fresh 1m bars. Mutates state."""
    still_open = []
    for pos in st["positions"]:
        bars = symbol_bars.get(pos["symbol"], [])
        rel = [b for b in bars if pos["opened_ts"] < b["ts"] <= now_ms]
        expiry = pos["opened_ts"] + pos["ttl"] * 1000
        closed = None
        for b in rel:
            hit_tp = b["h"] >= pos["tp"] if pos["side"] == "long" else b["l"] <= pos["tp"]
            hit_sl = b["l"] <= pos["sl"] if pos["side"] == "long" else b["h"] >= pos["sl"]
            if hit_tp and hit_sl:
                closed = ("loss", pos["sl"], min(b["ts"], expiry))  # conservative
                break
            if hit_tp:
                closed = ("win", pos["tp"], min(b["ts"], expiry))
                break
            if hit_sl:
                closed = ("loss", pos["sl"], min(b["ts"], expiry))
                break
            if b["ts"] >= expiry:
                closed = ("timeout", b["c"], expiry)
                break
        if closed is None:
            if now_ms >= expiry and rel:
                closed = ("timeout", rel[-1]["c"], expiry)
            else:
                still_open.append(pos)
                continue
        kind, price, closed_ts = closed
        direction = 1 if pos["side"] == "long" else -1
        gross = (price - pos["entry"]) * direction * pos["qty"]
        fees = (pos["entry"] + price) * pos["qty"] * cfg["fee_rate"] / 2
        pnl = gross - fees
        pos.update({"status": kind, "exit": price, "closed_ts": closed_ts,
                    "pnl": round(pnl, 4)})
        st["equity"] += pnl
        st["stats"]["trades"] += 1
        st["stats"]["pnl"] += pnl
        if pnl > 0:
            st["stats"]["wins"] += 1
            st["consec_losses"] = 0
        else:
            st["stats"]["losses"] += 1
            st["consec_losses"] += 1
        trades.append(pos)
    st["positions"] = still_open


def breakers_ok(st, cfg, now_ms):
    if now_ms < st.get("paused_until", 0):
        return False, "paused"
    if now_ms < st.get("halted_until", 0):
        return False, "daily_halt"
    if st["consec_losses"] >= cfg["loss_streak_pause"]:
        st["paused_until"] = now_ms + cfg["pause_minutes"] * 60_000
        st["consec_losses"] = 0
        return False, "loss_streak"
    if st.get("day_start_equity") is not None:
        if st["equity"] <= st["day_start_equity"] * (1 - cfg["daily_drawdown_stop"]):
            st["halted_until"] = ((now_ms // 86_400_000) + 1) * 86_400_000
            return False, "daily_drawdown"
    return True, "ok"


def roll_day(st, now_ms):
    day = now_ms // 86_400_000
    if st.get("day_key") != day:
        st["day_key"] = day
        st["day_start_equity"] = st["equity"]


def open_positions(st, cfg, symbol, feat, side, now_ms, source):
    sym = symbol
    dist = cfg["atr_k"] * feat["atr15"]
    entry = feat["close"]
    qty = risk_qty(cfg, sym, entry, dist)
    if qty <= 0:
        return None
    direction = 1 if side == "long" else -1
    ttl = cfg["expiry_seconds"][len(st["positions"]) % len(cfg["expiry_seconds"])]
    pos = {
        "id": f"{sym}-{now_ms}",
        "symbol": sym,
        "side": side,
        "entry": entry,
        "dist": round(dist, 6),
        "tp": round(entry + direction * dist, 6),
        "sl": round(entry - direction * dist, 6),
        "qty": qty,
        "risk": cfg["risk_per_trade"],
        "ttl": ttl,
        "opened_ts": now_ms,
        "expires_ts": now_ms + ttl * 1000,
        "status": "open",
        "source": source,
        "features": {k: (round(v, 6) if isinstance(v, float) else v)
                     for k, v in feat.items()},
    }
    st["positions"].append(pos)
    return pos
