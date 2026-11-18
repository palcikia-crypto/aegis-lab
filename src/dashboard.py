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


def _read(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def equity_svg(rows, w=760, h=190):
    if len(rows) < 2:
        return '<div class="empty">Недостаточно данных для графика</div>'
    vals = [float(r["eq"]) for r in rows]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1
    pad = 8
    pts = []
    for i, v in enumerate(vals):
        x = pad + i * (w - 2 * pad) / (len(vals) - 1)
        y = h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    color = "#34d399" if vals[-1] >= vals[0] else "#fb7185"
    area = f"M{pts[0]} L" + " L".join(pts) + f" L{pts[-1].split(',')[0]},{h - pad} L{pad},{h - pad} Z"
    return (f'<svg viewBox="0 0 {w} {h}" class="chart">'
            f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".25"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<path d="{area}" fill="url(#g)"/>'
            f'<path d="M{" L".join(pts)}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/></svg>')


def trade_row(t):
    cls = {"win": "up", "loss": "down"}.get(t.get("status"), "")
    pnl = t.get("pnl")
    pnl_s = f"{pnl:+.2f}" if pnl is not None else "—"
    strat = t.get("strategy") or ""
    return (f"<tr><td class='mono'>{esc(t.get('symbol'))}</td>"
            f"<td><span class='pill {'p-up' if t.get('side')=='long' else 'p-down'}'>{esc(t.get('side','')).upper()}</span></td>"
            f"<td class='mono'>{t.get('entry')}</td><td class='mono'>{t.get('ttl')}s</td>"
            f"<td><span class='pill {'p-up' if t.get('status')=='win' else 'p-down' if t.get('status')=='loss' else ''}'>{esc(t.get('status'))}</span></td>"
            f"<td class='mono {cls}'>{pnl_s}</td>"
            f"<td class='muted'>{esc(strat)}</td></tr>")


def sim_trade_row(t):
    cls = {"win": "up", "loss": "down"}.get(t.get("status"), "")
    pnl = t.get("pnl")
    pnl_s = f"{pnl:+.2f}" if pnl is not None else "—"
    return (f"<tr><td class='mono'>{esc(t.get('symbol'))}</td>"
            f"<td><span class='pill {'p-up' if t.get('side')=='long' else 'p-down'}'>{esc(t.get('side','')).upper()}</span></td>"
            f"<td class='mono'>{t.get('entry')}</td><td class='mono'>{t.get('ttl')}s</td>"
            f"<td><span class='pill {'p-up' if t.get('status')=='win' else 'p-down' if t.get('status')=='loss' else ''}'>{esc(t.get('status'))}</span></td>"
            f"<td class='mono {cls}'>{pnl_s}</td></tr>")


def leaderboard_rows(opt):
    if not opt:
        return ""
    rows = ""
    for r in opt.get("leaderboard", [])[:8]:
        p = r.get("params", {})
        pnl = r.get("pnl", 0)
        cls = "up" if pnl >= 0 else "down"
        rows += (f"<tr><td class='mono'>ATR {p.get('atr_k')} · vol {p.get('vol_spike_min')}</td>"
                 f"<td class='mono'>{r.get('trades')}</td><td class='mono'>{r.get('win_rate')}%</td>"
                 f"<td class='mono'>PF {r.get('profit_factor')}</td>"
                 f"<td class='mono {cls}'>{pnl:+.2f}</td>"
                 f"<td class='mono {'up' if r.get('chunk_worst_pnl', 0) > 0 else 'muted'}'>{r.get('chunk_worst_pnl')}</td></tr>")
    return rows


def generate(st, trades, cfg):
    os.makedirs(DOCS, exist_ok=True)
    import time as _t

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
        f"<tr><td class='mono'>{esc(p['symbol'])}</td>"
        f"<td><span class='pill {'p-up' if p['side']=='long' else 'p-down'}'>{p['side'].upper()}</span></td>"
        f"<td class='mono'>{p['entry']}</td><td class='mono'>{p['tp']}</td><td class='mono'>{p['sl']}</td>"
        f"<td class='mono'>{p['ttl']}s</td><td class='mono'>{p['qty']}</td>"
        f"<td class='muted'>{esc(p.get('strategy') or '')} · {esc(p.get('regime') or '')}</td></tr>"
        for p in reversed(st["positions"][-20:]))

    tr_rows = "".join(trade_row(t) for t in reversed(trades[-30:]))
    breaker = "OK" if (st["equity"] and not st.get("paused_until", 0)) else "check state"
    trading = "🟢 Торговля" if not st["positions"] else f"🟢 Торговля · {len(st['positions'])} поз."

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
<div class="card b"><div class="k">Песочница · сделок</div><div class="v">{ss['trades']}</div><div class="d">обучение на истории</div></div>
<div class="card b"><div class="k">Песочница · WR</div><div class="v">{swr:.1f}%</div><div class="d">win rate</div></div>
<div class="card b"><div class="k">Песочница · PnL</div><div class="v {'up' if ss['pnl'] >= 0 else 'down'}">{ss['pnl']:+.2f}</div><div class="d">{esc('equity $' + format(ss.get('equity', 0.0), '.2f'))}</div></div>"""
        sim_panel = f"""
<div class="panel"><div class="ph"><span class="ico">🧪</span>Песочница · sim-эквити</div>{equity_svg(sim_eq_rows)}</div>
<div class="panel"><div class="ph"><span class="ico">🧪</span>Песочница · последние sim-сделки</div>
<table><tr><th>Символ</th><th>Сторона</th><th>Вход</th><th>TTL</th><th>Итог</th><th>PnL $</th></tr>
{sim_rows or '<tr><td colspan="6" class="muted">Sim-сделок пока нет</td></tr>'}</table></div>"""

    upd_ago = max(0, int((_t.time() * 1000 - st.get("updated", 0)) / 60000))
    upd_s = f"обновлено {upd_ago} мин назад" if st.get("updated") else ""

    train_panel = ""
    if model:
        wfo = model.get("wfo", {})
        mc = model.get("monte_carlo") or {}
        age_h = max(0, int((_t.time() * 1000 - model.get("trained_ts", 0)) / 3_600_000))
        train_panel = f"""
<div class="panel"><div class="ph"><span class="ico">🎓</span>ML-гейт · статус обучения</div>
<table><tr><th>Сделок в датасете</th><th>Порог</th><th>Точность (val)</th><th>AUC</th><th>MC p5 PnL</th><th>Обучено</th></tr>
<tr><td class="mono">{model.get('n_trades')}</td><td class="mono">{model.get('threshold')}</td>
<td class="mono">{wfo.get('val_accuracy', '—')}</td><td class="mono">{wfo.get('val_auc', '—')}</td>
<td class="mono">{mc.get('p5_final_pnl', '—')}</td><td class="mono">{age_h} ч назад</td></tr></table>
<div class="sub" style="margin-top:10px">переобучение: ежедневно 02:25 UTC (песочница 30 дней → walk-forward → модель)</div></div>"""

    adapt = st.get("adapt", {})
    adapt_panel = ""
    if adapt:
        arows = ""
        for k in sorted(adapt, key=lambda x: -adapt[x].get("ewma", 0)):
            b = adapt[k]
            on = b.get("n", 0) < 10 or b.get("ewma", 0) > -0.35
            arows += (f"<tr><td class='mono'>{esc(k)}</td><td class='mono'>{b.get('n', 0)}</td>"
                      f"<td class='mono {'up' if b.get('ewma', 0) > 0 else 'down'}'>{b.get('ewma', 0):+.3f}</td>"
                      f"<td><span class='pill {'p-up' if on else 'p-down'}'>{'ON' if on else 'OFF'}</span></td></tr>")
        adapt_panel = f"""
<div class="panel"><div class="ph"><span class="ico">🧠</span>Адаптивный мозг · связки стратегия|режим</div>
<table><tr><th>Связка</th><th>Сделок</th><th>Ewma R</th><th>Торговля</th></tr>
{arows}</table>
<div class="sub" style="margin-top:10px">убыточные связки отключаются сами, пробы 1/40 проверяют их восстановление</div></div>"""

    uni = load_universe()
    uni_panel = ""
    if uni:
        syms = ", ".join(uni.get("symbols", []))
        uni_panel = f"""
<div class="panel"><div class="ph"><span class="ico">🪐</span>Авто-вселенная монет</div>
<div class="sub">система сама скачивает топ-ликвидные инструменты с Binance (включая золото XAUUSDT, когда оно в топе) и их торговые спецификации</div>
<p style="font-size:14px;margin-top:8px" class="mono">{esc(syms)}</p></div>"""

    strat_panel = ""
    if opt:
        best_p = opt.get("best", {}).get("params", {})
        rows = leaderboard_rows(opt)
        n_combos = len(opt.get("combo_stats", {}))
        strat_panel = f"""
<div class="panel"><div class="ph"><span class="ico">⚙️</span>Оптимизатор WFO · лидерборд</div>
<div class="sub">лучшие: ATR {best_p.get('atr_k')} · vol {best_p.get('vol_spike_min')} · адаптивных связок: {n_combos}</div>
<table><tr><th>Параметры</th><th>Сделок</th><th>WR</th><th>PF</th><th>PnL $</th><th>Худший чанк</th></tr>
{rows}</table></div>"""

    ml_card = ""
    if model:
        wfo = model.get("wfo", {})
        ml_card = f"""
<div class="card"><div class="k">ML-гейт</div><div class="v up">ON</div><div class="d">AUC {wfo.get('val_auc', '—')}</div></div>
<div class="card"><div class="k">ML порог</div><div class="v">{model['threshold']}</div><div class="d">точн. {wfo.get('val_accuracy', '—')}</div></div>"""
    else:
        ml_card = """
<div class="card"><div class="k">ML-гейт</div><div class="v muted">OFF</div><div class="d">ждёт данных</div></div>"""

    ai_panel = ""
    rep = _read(os.path.join(RESULTS, "ai_report.json"))
    if rep and rep.get("ok"):
        acts = ", ".join(f"{a['param']} = {a['value']}" for a in rep.get("actions_applied", [])) or "нет правок"
        chip = {"ok": "chip-ok", "warn": "chip-warn", "bad": "chip-bad"}.get(rep.get("verdict"), "chip-warn")
        vtxt = {"ok": "✓ Норма", "warn": "! Внимание", "bad": "✖ Опасность"}.get(rep.get("verdict"), rep.get("verdict", "?"))
        risks = "".join(
            f"<div class='risk'>⚠ {esc(str(r))}</div>"
            for r in rep.get("risks", [])[:4])
        ai_panel = f"""
<div class="panel ai"><div class="ph"><span class="ico">🤖</span>AI-оператор <span class="muted small">· {esc(rep.get("ts", ""))}</span></div>
<div class="chiprow"><span class="chip {chip}">{esc(vtxt)}</span><span class="chip">{esc(rep.get("model", ""))}</span></div>
<p class="lead">{esc(rep.get("summary", ""))}</p>
{risks}
<div class="small muted" style="margin-top:10px">правки параметров: {esc(acts)} · разборов: {sum(1 for l in open(os.path.join(RESULTS, "ai_journal.jsonl"))) if os.path.exists(os.path.join(RESULTS, 'ai_journal.jsonl')) else 0}</div>
</div>"""
    else:
        ai_panel = f"""
<div class="panel ai"><div class="ph"><span class="ico">🤖</span>AI-оператор</div>
<div class="small muted">отчёта нет ({esc(str((rep or {}).get("error", "?")))}) — обновится в следующем цикле</div></div>"""

    dev = _read(os.path.join(RESULTS, "ai_dev_report.json"))
    reg = _read(os.path.join(RESULTS, "ai_modules.json")) or []
    dev_panel = ""
    if isinstance(dev, dict) and dev.get("ok"):
        exp_rows = ""
        for r in (dev.get("experiments") or [])[:4]:
            score = r.get("score")
            score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
            failed = isinstance(score, (int, float)) and score <= -1e8
            params = ", ".join(f"{k}={v}" for k, v in (r.get("params") or {}).items()) or "—"
            exp_rows += (f"<tr><td class='mono'>{esc(r.get('name',''))}</td><td class='mono small'>{esc(params)}</td>"
                         f"<td class='mono'>{r.get('trades')}</td><td class='mono'>{r.get('wr')}%</td>"
                         f"<td class='mono {'up' if (r.get('pnl') or 0) >= 0 else 'down'}'>{r.get('pnl'):+.2f}</td>"
                         f"<td class='mono {'up' if not failed else 'down'}'>{score_s}</td></tr>")
        exp_block = ""
        if exp_rows:
            exp_block = f"""
<div class="sub" style="margin:10px 0 4px">эксперименты в песочнице (бэктест 7 дней, шаг 15 мин):</div>
<table><tr><th>Вариант</th><th>Параметры</th><th>Сделок</th><th>WR</th><th>PnL</th><th>Score</th></tr>
{exp_rows}</table>"""
        mod_rows = ""
        for m in reg:
            status = m.get("status", "?")
            stx = {"active": "p-up", "retired": "muted", "failed": "p-down"}.get(status, "")
            mod_rows += (f"<tr><td class='mono'>{esc(m.get('name',''))}</td>"
                         f"<td class='small'>{esc(m.get('purpose','') or m.get('fail_reason',''))}</td>"
                         f"<td><span class='pill {stx}'>{esc(status)}</span></td>"
                         f"<td class='mono'>{m.get('vetoes', 0)}/{m.get('decisions', 0)} вето</td></tr>")
        mods_block = ""
        if mod_rows:
            mods_block = f"""
<div class="sub" style="margin:10px 0 4px">свои модули-фильтры (авто-отключение при сбое или глушении сигналов):</div>
<table><tr><th>Модуль</th><th>Назначение</th><th>Статус</th><th>Действие</th></tr>
{mod_rows}</table>"""
        params_ai = ""
        if dev.get("params_applied"):
            pa = ", ".join(f"{a['param']}={a['value']}" for a in dev["params_applied"])
            params_ai = f"<div class='good-p'>🛠 внедрено: {esc(pa)}</div>"
        exp_applied = ""
        if dev.get("experiment_applied"):
            b = dev["experiment_applied"][0]
            exp_applied = f"<div class='good-p'>🏆 победитель экспериментов внедрён: {esc(b.get('name',''))} (score {b.get('score')})</div>"
        dev_panel = f"""
<div class="panel ai dev"><div class="ph"><span class="ico">🏗</span>AI-архитектор <span class="muted small">· {esc(dev.get("ts",""))} · {esc(dev.get("model",""))}</span></div>
<p class="lead">{esc(dev.get("thoughts",""))}</p>
{params_ai}{exp_applied}{exp_block}{mods_block}</div>"""

    vcheck = _read(os.path.join(RESULTS, "real_check.json"))
    vcheck_panel = ""
    if isinstance(vcheck, dict) and vcheck.get("trades"):
        vrows = ""
        for o in reversed(vcheck["trades"][-15:]):
            amb = "⚠ двусмысленно" if o.get("ambiguous") else ""
            match = o.get("match")
            mtxt = "✓" if match else "✗ расходится"
            mcls = "up" if match else "down"
            pr = o.get("pnl_real") or 0
            vrows += (f"<tr><td class='mono'>{esc(o.get('symbol',''))}</td>"
                      f"<td class='muted'>{esc(o.get('opened',''))}</td>"
                      f"<td><span class='pill {'p-up' if o.get('kind_real')=='win' else 'p-down' if o.get('kind_real')=='loss' else ''}'>{esc(o.get('kind_real') or o.get('status_recorded',''))}</span></td>"
                      f"<td class='mono {'up' if (o.get('pnl_recorded') or 0) >= 0 else 'down'}'>{(o.get('pnl_recorded') or 0):+.3f}</td>"
                      f"<td class='mono {'up' if pr >= 0 else 'down'}'>{pr:+.3f}</td>"
                      f"<td class='mono {mcls}'>{mtxt}</td><td class='muted small'>{amb}</td></tr>")
        sm = vcheck.get("summary") or {}
        n_total = s.get("trades", 0)
        n_checked = sm.get("trades", 0)
        vrows_tot = (f"<tr><td><b>TOTAL</b></td><td></td><td class='mono'><b>{n_checked}/{n_total}</b> проверено</td>"
                     f"<td class='mono {'up' if sm.get('recorded_total', 0) >= 0 else 'down'}'><b>{sm.get('recorded_total', 0):+.2f}</b></td>"
                     f"<td class='mono {'up' if sm.get('real_total', 0) >= 0 else 'down'}'><b>{sm.get('real_total', 0):+.2f}</b></td>"
                     f"<td class='mono muted'>{sm.get('realism_bias', 0):+.2f} (смещение)</td><td></td></tr>")
        vcheck_panel = f"""
<div class="panel"><div class="ph"><span class="ico">🔍</span>Авто-проверка на реальных биржевых данных</div>
<div class="sub">каждая сделка автоматически пересчитывается по реальным 1m-свечам Binance/Bybit: вход по следующему бару с проскальзыванием, полная тейкер-комиссия обе стороны, стоп глубже на проскальзывание. этими данными питаются ИИ-мозги · обновлено: {esc(sm.get('updated',''))}</div>
<table style="margin-top:8px"><tr><th>Символ</th><th>Открыта</th><th>Реальный итог</th><th>Записано</th><th>Реально</th><th>Сходится</th><th></th></tr>
{vrows}{vrows_tot}</table></div>"""

    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#070a12">
<title>AEGIS-Lab PRO · Adaptive Fee-Edge</title>
<style>
:root{{--bg:#070a12;--card:rgba(255,255,255,.035);--card2:rgba(255,255,255,.055);
--line:rgba(255,255,255,.08);--line2:rgba(255,255,255,.14);--txt:#eef2fa;--muted:#8ba0c0;
--up:#34d399;--down:#fb7185;--acc:#6d7cff;--acc2:#22d3ee;
--grad:linear-gradient(135deg,#6d7cff 0%,#22d3ee 100%)}}
*{{box-sizing:border-box;margin:0;padding:0}}
html{{-webkit-text-size-adjust:100%}}
body{{background:var(--bg);color:var(--txt);
font:15px/1.55 "Inter","SF Pro Display",-apple-system,"Segoe UI",Roboto,sans-serif;
background-image:radial-gradient(900px 500px at 85% -10%,rgba(109,124,255,.16),transparent 60%),
radial-gradient(700px 400px at -10% 0%,rgba(34,211,238,.10),transparent 55%),
radial-gradient(900px 600px at 50% 120%,rgba(52,211,153,.07),transparent 60%);
background-attachment:fixed;padding:0 16px 32px}}
.wrap{{max-width:920px;margin:0 auto}}
header.bar{{position:sticky;top:0;z-index:50;display:flex;align-items:center;gap:12px;
padding:14px 4px 12px;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
background:rgba(7,10,18,.72);border-bottom:1px solid var(--line)}}
.brand{{font-weight:800;font-size:19px;letter-spacing:.4px;
background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}}
.brand small{{display:block;font-size:10px;font-weight:600;letter-spacing:2.2px;color:var(--muted);
background:none;-webkit-background-clip:unset;-webkit-text-fill-color:var(--muted)}}
.live{{display:flex;align-items:center;gap:7px;background:var(--card2);border:1px solid var(--line2);
border-radius:99px;padding:5px 13px;font-size:12px;font-weight:600}}
.dot{{width:8px;height:8px;border-radius:50%;background:var(--up);box-shadow:0 0 0 0 rgba(52,211,153,.6);
animation:pulse 2s infinite}}
@keyframes pulse{{0%{{box-shadow:0 0 0 0 rgba(52,211,153,.55)}}70%{{box-shadow:0 0 0 9px rgba(52,211,153,0)}}100%{{box-shadow:0 0 0 0 rgba(52,211,153,0)}}}}
.timer{{color:var(--muted);font-variant-numeric:tabular-nums}}
.hero{{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:14px;padding:22px 4px 6px}}
.hero .eq{{font-size:15px;color:var(--muted)}}
.hero .eq b{{display:block;font-size:40px;font-weight:800;letter-spacing:-1px;color:var(--txt);
font-variant-numeric:tabular-nums}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px 18px;
transition:transform .18s,border-color .18s}}
.card:hover{{transform:translateY(-2px);border-color:var(--line2)}}
.card .k{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:1.1px;font-weight:600}}
.card .v{{font-size:24px;font-weight:700;margin-top:6px;font-variant-numeric:tabular-nums;letter-spacing:-.3px}}
.card .d{{color:var(--muted);font-size:12px;margin-top:2px}}
.card.hl{{background:linear-gradient(135deg,rgba(109,124,255,.14),rgba(34,211,238,.08));
border-color:rgba(109,124,255,.35)}}
.up{{color:var(--up)}}.down{{color:var(--down)}}.muted{{color:var(--muted)}}
.mono{{font-variant-numeric:tabular-nums}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px;margin:14px 0}}
.panel.ai{{border-color:rgba(109,124,255,.28);background:linear-gradient(160deg,rgba(109,124,255,.08),var(--card) 55%)}}
.panel.dev{{border-color:rgba(34,211,238,.25);background:linear-gradient(160deg,rgba(34,211,238,.07),var(--card) 55%)}}
.ph{{font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;display:flex;align-items:center;gap:9px}}
.ph .ico{{font-size:17px}}
.ph .small{{text-transform:none;letter-spacing:0;font-weight:500}}
.lead{{font-size:14.5px;line-height:1.6}}
.chiprow{{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 10px}}
.chip{{border:1px solid var(--line2);border-radius:99px;padding:3px 12px;font-size:12px;font-weight:600;background:var(--card2)}}
.chip-ok{{color:var(--up);border-color:rgba(52,211,153,.4)}}
.chip-warn{{color:#fbbf24;border-color:rgba(251,191,36,.4)}}
.chip-bad{{color:var(--down);border-color:rgba(251,113,133,.4)}}
.good-p{{background:rgba(52,211,153,.09);border:1px solid rgba(52,211,153,.3);color:var(--up);
border-radius:10px;padding:7px 11px;font-size:12.5px;font-weight:600;margin:8px 0}}
.risk{{color:var(--down);font-size:12.5px;margin-top:4px}}
.chart{{width:100%;height:auto;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{color:var(--muted);text-align:left;font-weight:600;font-size:11.5px;text-transform:uppercase;
letter-spacing:.6px;padding:7px 8px;border-bottom:1px solid var(--line2)}}
td{{padding:8px;border-bottom:1px solid rgba(255,255,255,.05)}}
tr:last-child td{{border-bottom:none}}
.pill{{display:inline-block;border-radius:99px;padding:1px 9px;font-size:11px;font-weight:700;
background:var(--card2);border:1px solid var(--line2)}}
.p-up{{color:var(--up);border-color:rgba(52,211,153,.4)}}
.p-down{{color:var(--down);border-color:rgba(251,113,133,.4)}}
.empty{{color:var(--muted);font-size:13px;padding:8px 0}}
.small{{font-size:12.5px}}.sub{{color:var(--muted);font-size:12.5px}}
footer{{color:var(--muted);font-size:12px;margin-top:26px;text-align:center;line-height:1.8}}
a{{color:var(--acc2);text-decoration:none}}a:hover{{text-decoration:underline}}
@media (max-width:640px){{
body{{padding:0 10px 24px;font-size:14px}}
.hero .eq b{{font-size:32px}}
.grid{{grid-template-columns:repeat(2,1fr);gap:9px}}
.card{{padding:12px 13px;border-radius:15px}}.card .v{{font-size:19px}}
.panel{{padding:13px;border-radius:15px}}
.ph{{font-size:12.5px}}
table{{font-size:11.5px;display:block;overflow-x:auto;white-space:nowrap;-webkit-overflow-scrolling:touch}}
th,td{{padding:5px 6px}}
}}
</style></head><body><div class="wrap">
<header class="bar">
<div class="brand">AEGIS<small>PAPER TRADING LAB</small></div>
<div class="live"><span class="dot"></span><span class="timer" id="tm">120s</span></div>
<div class="live muted">{trading}</div>
</header>
<div class="hero">
<div class="eq">Equity (paper)<b>${st['equity']:.2f}</b></div>
<div class="eq" style="text-align:right">Binance → Bybit → OKX<br>
<span class="muted small">риск {cfg['risk_per_trade']} USDT · RR 1:{cfg.get('rr', 1.0)} · Fee-Edge ×{cfg.get('fee_edge_mult', 3.0)} · TTL {cfg.get('ttl_min', 300)}–{cfg.get('ttl_max', 900)}с · {upd_s}</span></div>
</div>
<div class="grid">
<div class="card hl"><div class="k">Win Rate</div><div class="v {'up' if wr >= 50 else ''}">{wr:.1f}%</div><div class="d">{s['wins']}W / {s['losses']}L</div></div>
<div class="card"><div class="k">Profit Factor</div><div class="v">{'∞' if pf == float('inf') else f'{pf:.2f}'}</div><div class="d">прибыль/убыток</div></div>
<div class="card"><div class="k">Сделок</div><div class="v">{s['trades']}</div><div class="d">paper-циклы 5 мин</div></div>
<div class="card"><div class="k">Sharpe</div><div class="v">{sharpe:.2f}</div><div class="d">paper</div></div>
<div class="card"><div class="k">Серия убытков</div><div class="v {'down' if st['consec_losses'] > 0 else ''}">{st['consec_losses']} / {cfg['loss_streak_pause']}</div><div class="d">стоп-лимит</div></div>
{ml_card}{sim_cards}
</div>
{ai_panel}{dev_panel}{vcheck_panel}
<div class="panel"><div class="ph"><span class="ico">📈</span>Кривая эквити</div>{equity_svg(eq_rows)}</div>
{sim_panel}
<div class="panel"><div class="ph"><span class="ico">🎯</span>Открытые позиции ({len(st['positions'])}/{cfg['max_concurrent']})</div>
<div class="sub">движок проверяет рынок каждые 5 минут; сигнал приходит в среднем раз в 30–90 минут — когда условия совпадут, сделка появится здесь автоматически</div>
<table style="margin-top:8px"><tr><th>Символ</th><th>Сторона</th><th>Вход</th><th>TP</th><th>SL</th><th>TTL</th><th>Qty</th><th>Стратегия · режим</th></tr>
{pos_rows or '<tr><td colspan="8" class="muted">Позиций нет — ждём сигнал (проверка каждые 5 мин)</td></tr>'}</table></div>
<div class="panel"><div class="ph"><span class="ico">💰</span>Последние сделки</div>
<table><tr><th>Символ</th><th>Сторона</th><th>Вход</th><th>TTL</th><th>Итог</th><th>PnL $</th><th>Стратегия</th></tr>
{tr_rows or '<tr><td colspan="7" class="muted">Сделок пока нет — воркер собирает сигналы</td></tr>'}</table></div>
{strat_panel}{uni_panel}{train_panel}{adapt_panel}
<footer>Breaker: {esc(breaker)} · paper-only, не реальные деньги ·
<a href="https://github.com/palcikia-crypto/aegis-lab">репозиторий</a> ·
зеркало: <a href="https://aegis-lab.pages.dev">aegis-lab.pages.dev</a> · AI-мозги: Cloudflare Workers AI</footer>
<script>var __t=120;setInterval(function(){{__t=Math.max(0,__t-1);var e=document.getElementById('tm');if(e)e.textContent=__t+'s';}},1000);</script>
</div></body></html>"""
    for name in ("index.html", "pro.html"):
        with open(os.path.join(DOCS, name), "w") as f:
            f.write(html)
