"""AEGIS AI-оператор: Workers AI разбирает состояние системы каждый цикл,
пишет отчёт в панель и вносит только разрешённые правки параметров."""
import json
import os
import re
import time
import urllib.request

from src import engine

RESULTS = engine.RESULTS
AI_CFG_PATH = os.path.join(RESULTS, "ai_cfg.json")
AI_REPORT_PATH = os.path.join(RESULTS, "ai_report.json")
AI_JOURNAL_PATH = os.path.join(RESULTS, "ai_journal.jsonl")

MODELS = ["@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/meta/llama-3.1-8b-instruct-fast"]
API = "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}"

WHITELIST = {
    "fee_edge_mult": (2.0, 4.0, float),
    "ttl_min": (180, 600, int),
    "ttl_max": (600, 1800, int),
    "max_concurrent": (1, 3, int),
    "risk_per_trade": (0.5, 1.5, float),
    "loss_streak_pause": (4, 8, int),
}

SYSTEM = (
    "Ты AI-оператор торговой системы AEGIS-Lab: бумажная торговля (paper-only) крипты "
    "на Binance/Bybit/OKX, стратегии bin-trend/baseline с Fee-Edge фильтром (сделка открывается "
    "только когда расстояние до тайка в fee_edge_mult раз больше комиссии), ML-гейт по AUC, "
    "адаптивный мозг (EWMA R по связке стратегия|режим), предохранители: loss_streak_pause "
    "сделок убытка подряд -> пауза, дневной стоп. Шаг цикла 5 минут, TTL позиции 300-900с, "
    "риск на сделку ~1 USDT. Реальные деньги ЗАБЛОКИРОВАНЫ до подтверждённого эджа. "
    "Твоя работа: 1) вердикт как идёт торговля; 2) риски; 3) при необходимости предложить "
    "ИЗМЕНЕНИЯ ТОЛЬКО из белого списка параметров (укажи точные имена и значения). "
    "Отвечай СТРОГО валидным JSON без markdown, по схеме: "
    '{"verdict": "ok|warn|bad", "summary": "краткий разбор по-русски до 120 слов", '
    '"risks": ["..."], "actions": [{"param": "имя", "value": число, "reason": "кратко"}]}. '
    "Имена параметров только из списка. Если менять ничего не нужно - actions: []."
)


def _ai_run(acc, tok, model, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        API.format(acc, model), data=data,
        headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _ask_llm(digest):
    tok = os.environ.get("WAI_TOKEN")
    acc = os.environ.get("WAI_ACCOUNT")
    if not tok or not acc:
        return None, "no WAI_TOKEN/WAI_ACCOUNT"
    body = {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Состояние системы (JSON):\n" + json.dumps(digest, ensure_ascii=False)},
    ], "max_tokens": 700, "temperature": 0.3}
    last_err = None
    for m in MODELS:
        try:
            r = _ai_run(acc, tok, m, body)
            if not r.get("success"):
                last_err = r.get("errors")
                continue
            res = r.get("result") or {}
            text = res.get("response") if isinstance(res.get("response"), str) else None
            if not text and isinstance(res, dict) and res.get("choices"):
                text = ((res["choices"][0].get("message") or {}).get("content")) or ""
            return {"model": m, "text": text or ""}, None
        except Exception as e:
            last_err = str(e)
    return None, str(last_err)


def load_ai_overrides():
    if not os.path.exists(AI_CFG_PATH):
        return {}
    try:
        with open(AI_CFG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _digest(st, trades, cfg, extra):
    s = st["stats"]
    last = [
        {"sym": t.get("symbol"), "side": t.get("side"), "pnl": round(t.get("pnl", 0), 3),
         "closed": t.get("closed_reason", t.get("reason", ""))}
        for t in trades[-12:]
    ]
    return {
        "equity": round(st["equity"], 2),
        "day_start_equity": st.get("day_start_equity"),
        "day_pnl": round(st["equity"] - st.get("day_start_equity", st["equity"]), 2),
        "consec_losses": st.get("consec_losses", 0),
        "paused_until": st.get("paused_until", 0),
        "stats": {"trades": s.get("trades", 0), "wins": s.get("wins", 0), "losses": s.get("losses", 0)},
        "open_positions": [
            {"sym": p["symbol"], "side": p["side"], "entry": p["entry"],
             "ttl": p.get("ttl"), "strat": p.get("strategy"), "regime": p.get("regime")}
            for p in st["positions"]
        ],
        "last_trades": last,
        "params_now": {k: cfg.get(k) for k in WHITELIST},
        "adapt_combos": len(st.get("adapt", {})),
        "top_adapt": sorted(
            ({"combo": k, "r_ewma": round(v.get("r_ewma", 0), 3), "n": v.get("n", 0)}
             for k, v in (st.get("adapt") or {}).items()),
            key=lambda x: -x["r_ewma"])[:5],
        "ml": extra.get("ml"),
        "sim": extra.get("sim"),
        "sources": extra.get("sources", {}),
        "ml_gated_last": extra.get("ml_gated", 0),
    }


def _whitelist_validation(actions):
    applied, rejected = [], []
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        name = a.get("param")
        val = a.get("value")
        if name not in WHITELIST or val is None:
            rejected.append({**a, "why": "param вне белого списка"})
            continue
        lo, hi, cast = WHITELIST[name]
        try:
            v = cast(float(val))
        except (TypeError, ValueError):
            rejected.append({**a, "why": "значение не число"})
            continue
        clamped = max(lo, min(hi, v))
        rec = {"param": name, "value": clamped}
        if clamped != v:
            rec["reason"] = f"{a.get('reason', '')} [обрезано до {lo}..{hi}]"
        else:
            rec["reason"] = a.get("reason", "")
        applied.append(rec)
    return applied, rejected


def _parse_json(text):
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def maybe_run(st, trades, cfg, extra):
    digest = _digest(st, trades, cfg, extra)
    t0 = time.time()
    answer, err = _ask_llm(digest)
    report = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "latency_s": round(time.time() - t0, 1),
        "ok": False,
    }
    if err:
        report["error"] = err
    else:
        parsed = _parse_json(answer["text"])
        if not parsed:
            report["error"] = "не удалось разобрать ответ модели"
            report["raw"] = (answer["text"] or "")[:400]
        else:
            applied, rejected = _whitelist_validation(parsed.get("actions"))
            if applied:
                merged = load_ai_overrides()
                merged.update({a["param"]: a["value"] for a in applied})
                os.makedirs(RESULTS, exist_ok=True)
                with open(AI_CFG_PATH, "w") as f:
                    json.dump(merged, f, indent=1)
            report.update({
                "ok": True,
                "model": answer["model"],
                "verdict": parsed.get("verdict", "?"),
                "summary": parsed.get("summary", ""),
                "risks": parsed.get("risks", []),
                "actions_applied": applied,
                "actions_rejected": rejected,
            })
            with open(AI_JOURNAL_PATH, "a") as f:
                f.write(json.dumps(
                    {k: report.get(k) for k in ("ts", "model", "verdict", "summary",
                                                "risks", "actions_applied", "actions_rejected")},
                    ensure_ascii=False) + "\n")
    with open(AI_REPORT_PATH, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return report
