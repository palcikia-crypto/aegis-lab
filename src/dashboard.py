"""Generate the dashboard: docs/index.html (static, dark, no external deps)."""
import json
import os
import statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
DOCS = os.path.join(HERE, "docs")
SIM_STATE_PATH = os.path.join(RESULTS, "sim_state.json")
SIM_TRADES_PATH = os.path.join(RESULTS, "sim_trades.jsonl")
SIM_EQUITY_PATH = os.path.join(RESULTS, "sim_equity.csv")
MODEL_PATH = os.path.join(RESULTS, "model.json")
OPT_PATH = os.path.join(RESULTS, "opt.json")
UNIVERSE_PATH = os.path.join(RESULTS, "universe.json")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_sim():
    st = None
    if os.path.exists(SIM_STATE_PATH):
        with open(SIM_STATE_PATH) as f:
            st = json.load(f)
    return st, load_jsonl(SIM_TRADES_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH) as f:
        return json.load(f)


def load_opt():
    if not os.path.exists(OPT_PATH):
        return None
    with open(OPT_PATH) as f:
        return json.load(f)


def load_universe():
    if not os.path.exists(UNIVERSE_PATH):
        return None
    with open(UNIVERSE_PATH) as f:
        return json.load(f)


def leaderboard_rows(opt):
    if not opt:
        return ""
    rows = ""
    for r in opt.get("leaderboard", [])[:8]:
        p = r.get("params", {})
        pnl = r.get("pnl", 0)
        cls = "up" if pnl >= 0 else "down"
        rows += (f"<tr><td>ATR k={p.get('atr_k')} · vol≥{p.get('vol_spike_min')}</td>"
                 f"<td>{r.get('trades')}</td><td>{r.get('win_rate')}%</td>"
                 f"<td>PF {r.get('profit_factor')}</td>"
                 f"<td class='{cls}'>{pnl:+.2f}</td>"
                 f"<td class='{'up' if r.get('chunk_worst_pnl', 0) > 0 else 'muted'}'>{r.get('chunk_worst_pnl')}</td></tr>")
    return rows


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def equity_svg(rows, w=760, h=180):
    if len(rows) < 2:
        return '<div class="muted">Недостаточно данных для графика</div>'
    vals = [float(r["eq"]) for r in rows]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1
    pad = 10
    pts = []
    for i, v in enumerate(vals):
        x = pad + i * (w - 2 * pad) / (len(vals) - 1)
        y = h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    color = "#2dd4a7" if vals[-1] >= vals[0] else "#f26d7d"
    area = f"M{pts[0]} L" + " L".join(pts) + f" L{pts[-1].split(',')[0]},{h-pad} L{pad},{h-pad} Z"
    return (f'<svg viewBox="0 0 {w} {h}" class="chart">'
            f'<path d="{area}" fill="{color}" opacity="0.12"/>'
            f'<path d="M{" L".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/></svg>')


def trade_row(t):
    cls = {"win": "up", "loss": "down"}.get(t.get("status"), "")
    pnl = t.get("pnl")
    pnl_s = f"{pnl:+.2f}" if pnl is not None else "—"
    return (f"<tr><td>{esc(t.get('symbol'))}</td>"
            f"<td class='{'up' if t.get('side')=='long' else 'down'}'>{esc(t.get('side','').upper())}</td>"
            f"<td>{t.get('entry')}</td><td>{t.get('ttl')}s</td>"
            f"<td class='{cls}'>{esc(t.get('status'))}</td>"
            f"<td class='{cls}'>{pnl_s}</td></tr>")


def sim_trade_row(t):
    cls = {"win": "up", "loss": "down"}.get(t.get("status"), "")
    pnl = t.get("pnl")
    pnl_s = f"{pnl:+.2f}" if pnl is not None else "—"
    return (f"<tr><td>{esc(t.get('symbol'))}</td>"
            f"<td class='{'up' if t.get('side')=='long' else 'down'}'>{esc(t.get('side','').upper())}</td>"
            f"<td>{t.get('entry')}</td><td>{t.get('ttl')}s</td>"
            f"<td class='{cls}'>{esc(t.get('status'))}</td>"
            f"<td class='{cls}'>{pnl_s}</td></tr>")


def generate(st, trades, cfg):
    os.makedirs(DOCS, exist_ok=True)
    eq_rows = []
    epath = os.path.join(RESULTS, "equity.csv")
    if os.path.exists(epath):
        with open(epath) as f:
            eq_rows = [{"ts": p[0], "eq": p[1]}
                       for p in (l.split(",") for l in f.read().splitlines() if l.strip())]

    s = st["stats"]
    wr = (s["wins"] / s["trades"] * 100) if s["trades"] else 0.0
    gross_w = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_l = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))
    pf = (gross_w / gross_l) if gross_l > 0 else float("inf")
    pnls = [t["pnl"] for t in trades if "pnl" in t]
    sd = statistics.pstdev(pnls) if len(pnls) > 1 else 0.0
    sharpe = (statistics.fmean(pnls) / sd * (len(pnls) ** 0.5)) if sd > 0 and pnls else 0.0

    pos_rows = "".join(
        f"<tr><td>{esc(p['symbol'])}</td>"
        f"<td class='{'up' if p['side']=='long' else 'down'}'>{p['side'].upper()}</td>"
        f"<td>{p['entry']}</td><td>{p['tp']}</td><td>{p['sl']}</td>"
        f"<td>{p['ttl']}s</td><td>{p['qty']}</td></tr>"
        for p in reversed(st["positions"][-20:]))

    tr_rows = "".join(trade_row(t) for t in reversed(trades[-30:]))
    breaker = "OK" if (st["equity"] and not st.get("paused_until", 0)) else "check state"
    status_badge = "🟢 trading" if not st["positions"] else f"🟢 trading · {len(st['positions'])} open"

    sim_st, sim_trades = load_sim()
    model = load_model()
    opt = load_opt()
    sim_cards = ""
    sim_panel = ""
    if sim_st:
        ss = sim_st["stats"]
        swr = (ss["wins"] / ss["trades"] * 100) if ss["trades"] else 0.0
        sim_eq_rows = []
        if os.path.exists(SIM_EQUITY_PATH):
            with open(SIM_EQUITY_PATH) as f:
                sim_eq_rows = [{"ts": p[0], "eq": p[1]}
                               for p in (l.split(",") for l in f.read().splitlines() if l.strip())]
        sim_rows = "".join(sim_trade_row(t) for t in reversed(sim_trades[-15:]))
        sim_cards = f"""
<div class="card"><div class="k">Sim-сделок (песочница)</div><div class="v">{ss['trades']}</div></div>
<div class="card"><div class="k">Sim Win Rate</div><div class="v">{swr:.1f}%</div></div>
<div class="card"><div class="k">Sim PnL</div><div class="v {'up' if ss['pnl'] >= 0 else 'down'}">{ss['pnl']:+.2f}</div></div>"""
        sim_panel = f"""
<div class="panel"><h2>Песочница · sim-эквити</h2>{equity_svg(sim_eq_rows)}</div>
<div class="panel"><h2>Песочница · последние sim-сделки</h2>
<table><tr><th>Символ</th><th>Сторона</th><th>Вход</th><th>TTL</th><th>Итог</th><th>PnL $</th></tr>
{sim_rows or '<tr><td colspan="6" class="muted">Sim-сделок пока нет</td></tr>'}</table></div>"""
    uni = load_universe()
    uni_panel = ""
    if uni:
        syms = ", ".join(uni.get("symbols", []))
        uni_panel = f"""
<div class="panel"><h2>Авто-вселенная монет (PRO)</h2>
<div class="sub">система сама скачивает топ-ликвидные инструменты с Binance (включая золото XAUUSDT, когда оно в топе) и их торговые спецификации</div>
<p style="font-size:14px;margin-top:8px">{esc(syms)}</p></div>"""
    strat_panel = ""
    if opt:
        best_p = opt.get("best", {}).get("params", {})
        rows = leaderboard_rows(opt)
        n_combos = len(opt.get("combo_stats", {}))
        strat_panel = f"""
<div class="panel"><h2>Оптимизатор WFO · лидерборд параметров</h2>
<div class="sub">лучшие: ATR k={best_p.get('atr_k')} · vol≥{best_p.get('vol_spike_min')} · адаптивных комбинаций стратегия|режим: {n_combos}</div>
<table><tr><th>Параметры</th><th>Сделок</th><th>WR</th><th>PF</th><th>PnL $</th><th>Худший чанк</th></tr>
{rows}</table></div>"""
    ml_card = ""
    if model:
        wfo = model.get("wfo", {})
        ml_card = f"""
<div class="card"><div class="k">ML-гейт (Этап 3)</div><div class="v up">ON</div></div>
<div class="card"><div class="k">ML порог / точность</div><div class="v">{model['threshold']} / {wfo.get('val_accuracy', '—')}</div></div>"""
    else:
        ml_card = """
<div class="card"><div class="k">ML-гейт (Этап 3)</div><div class="v muted">OFF</div></div>"""

    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AEGIS-Lab · Paper 1:1</title>
<style>
:root{{--bg:#0b0f17;--card:#131a26;--line:#22304a;--txt:#e6ecf5;--muted:#8296b3;
--up:#2dd4a7;--down:#f26d7d;--acc:#5b8cff}}
*{{box-sizing:border-box;margin:0}}
body{{background:var(--bg);color:var(--txt);
font:15px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif;padding:24px}}
.wrap{{max-width:860px;margin:0 auto}}
h1{{font-size:22px;letter-spacing:.5px}}
h1 span{{color:var(--acc)}}
.sub{{color:var(--muted);font-size:13px;margin:4px 0 20px}}
.badge{{display:inline-block;background:var(--card);border:1px solid var(--line);
border-radius:99px;padding:2px 12px;font-size:12px;color:var(--up)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}}
.card .k{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px}}
.card .v{{font-size:22px;font-weight:600;margin-top:4px}}
.up{{color:var(--up)}}.down{{color:var(--down)}}.muted{{color:var(--muted)}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin:14px 0}}
.panel h2{{font-size:14px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:10px}}
.chart{{width:100%;height:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{color:var(--muted);text-align:left;font-weight:500;padding:6px 8px;border-bottom:1px solid var(--line)}}
td{{padding:6px 8px;border-bottom:1px solid rgba(34,48,74,.5)}}
footer{{color:var(--muted);font-size:12px;margin-top:24px;text-align:center}}
a{{color:var(--acc)}}
</style></head><body><div class="wrap">
<h1>AEGIS<span>-Lab</span> · Paper Engine 1:1</h1>
<div class="sub">Bybit/OKX публичные данные · риск {cfg['risk_per_trade']} USDT на сделку · экспирация {cfg['expiry_seconds'][0]}/{cfg['expiry_seconds'][1]} сек · TP/SL = {cfg['atr_k']}×ATR15m · обновляется воркером GitHub Actions каждые 5 мин</div>
<span class="badge">{status_badge}</span>
<div class="grid">
<div class="card"><div class="k">Equity</div><div class="v">${st['equity']:.2f}</div></div>
<div class="card"><div class="k">Win Rate</div><div class="v">{wr:.1f}%</div></div>
<div class="card"><div class="k">Profit Factor</div><div class="v">{'∞' if pf==float('inf') else f'{pf:.2f}'}</div></div>
<div class="card"><div class="k">Сделок</div><div class="v">{s['trades']}</div></div>
<div class="card"><div class="k">Sharpe (paper)</div><div class="v">{sharpe:.2f}</div></div>
<div class="card"><div class="k">Серия убытков</div><div class="v">{st['consec_losses']} / {cfg['loss_streak_pause']}</div></div>
{ml_card}{sim_cards}
</div>
<div class="panel"><h2>Кривая эквити</h2>{equity_svg(eq_rows)}</div>
{sim_panel}{strat_panel}{uni_panel}
<div class="panel"><h2>Открытые позиции ({len(st['positions'])}/{cfg['max_concurrent']})</h2>
<table><tr><th>Символ</th><th>Сторона</th><th>Вход</th><th>TP</th><th>SL</th><th>TTL</th><th>Qty</th></tr>
{pos_rows or '<tr><td colspan="7" class="muted">Позиций нет</td></tr>'}</table></div>
<div class="panel"><h2>Последние сделки</h2>
<table><tr><th>Символ</th><th>Сторона</th><th>Вход</th><th>TTL</th><th>Итог</th><th>PnL $</th></tr>
{tr_rows or '<tr><td colspan="6" class="muted">Сделок пока нет — воркер собирает сигналы</td></tr>'}</table></div>
<footer>Breaker: {esc(breaker)} · paper-only, не реальные деньги · <a href="https://github.com/palcikia-crypto/aegis-lab">репозиторий</a> · данные: results/state.json</footer>
</div></body></html>"""
    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write(html)
