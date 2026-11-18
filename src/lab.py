"""AEGIS Лаборатория ("вторая реальность"): автономный ИИ-конструктор.

Полная копия системы живёт в results/lab/: у неё свои параметры, свои
модули-фильтры, свои бэктесты на реальной истории и свой журнал.
Цель — построить вариант системы, который стабильно ОБГОНЯЕТ оригинал
по WFO-скору на paper-торговле. Что сработало — портится в оригинал
(results/lab/recommend.json + запись модуля в главный реестр),
что не сработало — остаётся в журнале как опыт (авто-откат к лучшему).

Бюджетная дисциплина: один шаг за вызов, LLM-запросы ограничены в день,
между планами пауза; очередь экспериментов выполняется без LLM.
Ограничения: код модулей проходит ту же песочницу, что и
AI-архитектор (ai_dev.validate_module); данные скачиваются только с
биржевых API (market.py); портация только при устойчивом превосходстве.
"""
import argparse
import json
import os
import re
import time

from src import engine
from src import sim as sim_mod
from src import optimize as opt_mod
from src import ai_advisor
from src import ai_dev

RESULTS = engine.RESULTS
LAB_DIR = os.path.join(RESULTS, "lab")
STATE_PATH = os.path.join(LAB_DIR, "state.json")
JOURNAL_PATH = os.path.join(LAB_DIR, "journal.jsonl")
REPORT_PATH = os.path.join(LAB_DIR, "report.json")
RECOMMEND_PATH = os.path.join(LAB_DIR, "recommend.json")
CKPT_DIR = os.path.join(LAB_DIR, "checkpoints")
MODULES_REGISTRY = os.path.join(LAB_DIR, "modules.json")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_DIR = os.path.join(HERE, "ai_modules")

LLM_DAY_LIMIT = int(os.environ.get("AEGIS_LAB_LLM_DAY", "6"))
PLAN_INTERVAL_S = int(os.environ.get("AEGIS_LAB_PLAN_INTERVAL_S", "2700"))
STEP_TIME_S = int(os.environ.get("AEGIS_LAB_STEP_TIME_S", "150"))
EVAL_DAYS = int(os.environ.get("AEGIS_LAB_DAYS", "5"))
MAX_SYMBOLS = 2
CACHE_DIR = os.path.join(LAB_DIR, "cache")
CACHE_TTL_S = 12 * 3600
MAX_VARIANTS_PER_STEP = 1
MAX_CHECKPOINTS = 12
MIN_PROMOTE_TRADES = 25
BASELINE_REFRESH_S = 4 * 3600

def merged_bounds():
    from src import evo as evo_mod
    b = dict(ai_dev.DEV_BOUNDS)
    for k, v in evo_mod.BOUNDS.items():
        b[k] = v
    return b


SYSTEM = (
    "Ты — ЛАБОРАТОРИЯ 'Вторая реальность' торговой системы AEGIS (paper-only, "
    "реальные деньги заблокированы). У тебя есть точная копия системы: свои "
    "параметры, свои модули-фильтры, бэктесты на реальной истории бирж. "
    "Твоя единственная цель — построить вариант, который СТАБИЛЬНО обгоняет "
    "оригинал по WFO-скору (прибыль во всех временных чанках, PF>1, depth "
    "трейдов) на paper-торговле. Ты разработчик: можешь предлагать любые "
    "параметры из списка (границы дам), писать свои модули-фильтры сигналов "
    "(python: import math/statistics, экспорт NAME + "
    "filter_signal(feat, cfg) -> (allow: bool, reason); ВЕТО-философия: "
    "блокируй только явно рискованные входы, не глуши всё подряд), "
    "запрашивать данные (символ/дни) — система скачает их сама с бирж."
    "Отвечай СТРОГО валидным JSON без markdown:\n"
    '{"thoughts": "что понял из досье, где искать улучшение (по-русски, до 150 слов)", '
    '"queue": [{"name": "короткое_имя", "params": {"param": число}, '
    '"module": null | {"name": "snake_case", "code": "полный код"}, '
    '"note": "гипотеза"}], "data": [{"symbol": "BTCUSDT", "days": 10}]}\n'
    "Правила: queue длиной 1-3 варианта; каждый вариант меняет 1-4 параметра "
    "и/или добавляет один модуль; имена параметров только из списка границ; "
    "избегай повторов того, что уже провалилось (см. журнал).")


def _now_ms():
    return int(time.time() * 1000)


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def journal(event, **kw):
    row = {"ts": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "event": event}
    row.update(kw)
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _day_key():
    return time.strftime("%Y%m%d", time.gmtime())


def fresh_state():
    return {"phase": "boot", "steps": 0, "llm_used": 0, "llm_day": _day_key(),
            "last_plan_ts": 0, "baseline_ts": 0, "baseline": None,
            "queue": [], "q_idx": 0, "best": None, "promoted": [],
            "ckpts": 0}


def load_state():
    st = _load_json(STATE_PATH, None) or fresh_state()
    for k, v in fresh_state().items():
        st.setdefault(k, v)
    return st


def save_state(st):
    _save_json(STATE_PATH, st)


def _clamp_params(params):
    bounds = merged_bounds()
    out, rejected = {}, []
    for k, v in (params or {}).items():
        if k not in bounds:
            rejected.append(f"{k} (не в списке)")
            continue
        lo, hi, cast = bounds[k]
        try:
            out[k] = max(lo, min(hi, cast(float(v))))
        except Exception:
            rejected.append(f"{k} (нечисл.")
    return out, rejected


# ------------------------------------------------------------ baseline cache
def _cached_windows(symbols, days):
    """История с бирж кэшируется на диск (вне git) — бэктесты в рамках
    сессии не бьют повторно по API."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"hist_{days}d.json")
    cache = _load_json(path, {})
    if (cache.get("symbols") == symbols and cache.get("days") == days
            and _now_ms() - cache.get("ts", 0) < CACHE_TTL_S * 1000
            and cache.get("windows")):
        return {s: sim_mod.Window([
                    dict(zip(("ts", "o", "h", "l", "c", "v"), row))
                    for row in rows])
                for s, rows in cache["windows"].items()}, cache.get("srcs", {})
    wins, srcs = sim_mod.fetch_windows(days, symbols)
    _save_json(path, {
        "ts": _now_ms(), "symbols": symbols, "days": days, "srcs": srcs,
        "windows": {s: [[b["ts"], b["o"], b["h"], b["l"], b["c"], b["v"]]
                        for b in w.bars] for s, w in wins.items()},
    })
    return wins, srcs


def ensure_baseline(deadline=None):
    st_lab = load_state()
    fresh = (_now_ms() - st_lab.get("baseline_ts", 0)) < BASELINE_REFRESH_S * 1000
    if fresh and st_lab.get("baseline"):
        return st_lab["baseline"]
    symbols = _eval_symbols()
    cfg = _main_cfg({})
    wins, _srcs = _cached_windows(symbols, EVAL_DAYS)
    st_sim, trades, _eq = sim_mod.replay(
        wins, cfg, step_ms=opt_mod.OPT_STEP_MS, symbols=symbols)
    base = _pack(st_sim, trades, cfg, "baseline-оригинал")
    st_lab["baseline"] = base
    st_lab["baseline_ts"] = _now_ms()
    save_state(st_lab)
    return base


def _eval_symbols():
    try:
        syms = json.load(open(os.path.join(RESULTS, "universe.json")))
        if isinstance(syms, dict):
            syms = syms.get("symbols") or []
        if isinstance(syms, list) and syms and isinstance(syms[0], str):
            return syms[:MAX_SYMBOLS]
    except Exception:
        pass
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"][:MAX_SYMBOLS]


def _main_cfg(override):
    with open(os.path.join(HERE, "config", "settings.json")) as f:
        cfg = json.load(f)
    for name, key in (("opt.json", "cfg_overrides"),
                      ("lab/recommend.json", "params")):
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            try:
                data = json.load(open(p))
                cfg.update(data.get(key) or {})
            except Exception:
                pass
    cfg.update(ai_advisor.load_ai_overrides())
    cfg.update(override)
    return cfg


def _pack(st_sim, trades, cfg, name):
    s = st_sim["stats"]
    return {
        "name": name,
        "score": opt_mod.combo_score(trades, st_sim),
        "trades": s.get("trades", 0),
        "wins": s.get("wins", 0),
        "losses": s.get("losses", 0),
        "pnl": round(s.get("pnl", 0.0), 2),
        "params": {k: cfg.get(k) for k in merged_bounds()},
    }


# ------------------------------------------------------------ LLM план
def _llm_call(prompt):
    tok = os.environ.get("WAI_TOKEN")
    acc = os.environ.get("WAI_ACCOUNT")
    if not tok or not acc:
        return None, "нет WAI_TOKEN/WAI_ACCOUNT"
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
            "max_tokens": 2400, "temperature": 0.5}
    errs = []
    for m in ai_advisor.MODELS:
        try:
            r = ai_advisor._ai_run(acc, tok, m, body)
            if not r.get("success"):
                errs.append(str(r.get("errors"))[:80])
                continue
            res = r.get("result") or {}
            text = res.get("response") if isinstance(res.get("response"), str) else None
            if not text and isinstance(res, dict) and res.get("choices"):
                text = ((res["choices"][0].get("message") or {}).get("content")) or ""
            if text:
                return text, None
            errs.append("empty")
        except Exception as e:
            errs.append(str(e)[:80])
    return None, "; ".join(errs)


def _dossier(st_lab, baseline, extra):
    tail = []
    if os.path.exists(JOURNAL_PATH):
        with open(JOURNAL_PATH, encoding="utf-8") as f:
            rows = f.readlines()[-14:]
        for r in rows:
            try:
                d = json.loads(r)
                tail.append(f"{d.get('event')}: {d.get('detail', '')[:120]}")
            except Exception:
                tail.append(r[:120])
    data_cfg = _main_cfg({})
    bounds = {k: [b[0], b[1]] for k, b in merged_bounds().items()}
    return json.dumps({
        "цель": "обогнать оригинал по score при PF>1 и глубине трейдов",
        "оригинал_baseline": {k: baseline.get(k) for k in
                              ("name", "score", "trades", "pnl") if k in baseline},
        "лаборатория_лучшая": st_lab.get("best"),
        "внедрено_в_оригинал": st_lab.get("promoted", [])[-4:],
        "журнал_недавнее": tail,
        "параметры_сейчас": {k: data_cfg.get(k) for k in list(bounds)[:12]},
        "границы": bounds,
        "ml": extra.get("ml"),
        "real_check": extra.get("real_check"),
        "universe": _eval_symbols(),
    }, ensure_ascii=False)


def _plan(st_lab, baseline, extra, deadline):
    st_lab["llm_used"] = (st_lab.get("llm_used", 0) + 1
                          if st_lab.get("llm_day") == _day_key() else 1)
    st_lab["llm_day"] = _day_key()
    st_lab["last_plan_ts"] = _now_ms()
    text, err = _llm_call(_dossier(st_lab, baseline, extra))
    if text is None:
        journal("plan_failed", detail=err or "llm недоступен")
        return False
    data = ai_advisor._parse_json(text)
    if not isinstance(data, dict):
        journal("plan_failed", detail="невалидный JSON от LLM")
        return False
    queue = []
    for i, v in enumerate((data.get("queue") or [])[:3]):
        if not isinstance(v, dict):
            continue
        params, rej = _clamp_params(v.get("params"))
        mod_entry = None
        mod = v.get("module")
        if isinstance(mod, dict) and isinstance(mod.get("code"), str):
            ns, merr = ai_dev.validate_module(mod["code"])
            mod_entry = {"name": ns["name"], "code": ns["code"]} if ns else {
                "name": str(mod.get("name", f"lab_mod_{i}")), "error": merr}
        queue.append({"name": str(v.get("name", f"exp_{i}"))[:40],
                      "params": params, "module": mod_entry,
                      "note": str(v.get("note", ""))[:200],
                      "rejected": rej})
    thoughts = str(data.get("thoughts", ""))[:200]
    for d in (data.get("data") or [])[:2]:
        if isinstance(d, dict) and str(d.get("symbol", ""))[:16]:
            journal("data_request", symbol=str(d.get("symbol"))[:16],
                    days=int(d.get("days", 10) or 10),
                    detail="скачано при следующем бэктесте")
    st_lab["queue"] = queue
    st_lab["q_idx"] = 0
    st_lab["phase"] = "evaluate"
    save_state(st_lab)
    journal("plan", detail=thoughts, queued=len(queue))
    return True


# ------------------------------------------------------------ шаг лаборатории
def maybe_step(st, trades, cfg_main, extra=None, time_budget_s=STEP_TIME_S):
    deadline = time.time() + time_budget_s
    st_lab = load_state()
    summary = {"ok": False, "phase": st_lab.get("phase")}
    try:
        return _step(st_lab, deadline, summary)
    except Exception as e:
        journal("step_crash", detail=f"{type(e).__name__}: {str(e)[:160]}")
        summary["error"] = str(e)[:120]
        _write_report(st_lab, summary)
        return summary


def _step(st_lab, deadline, summary):
    if st_lab.get("llm_day") != _day_key():
        st_lab["llm_used"] = 0
        st_lab["llm_day"] = _day_key()

    if not st_lab.get("baseline") and time.time() < deadline - 40:
        base = ensure_baseline(deadline)
        if base:
            journal("baseline", score=base.get("score"),
                    pnl=base.get("pnl"), trades=base.get("trades"))
            st_lab["baseline"] = base
            st_lab["baseline_ts"] = _now_ms()
            st_lab["phase"] = "plan"
            summary["baseline"] = {k: base.get(k) for k in ("score", "trades", "pnl")}

    queue = st_lab.get("queue") or []
    q_idx = int(st_lab.get("q_idx", 0))
    llm_ok = (st_lab.get("llm_used", 0) < LLM_DAY_LIMIT
              and _now_ms() - st_lab.get("last_plan_ts", 0) > PLAN_INTERVAL_S * 1000)

    pend = st_lab.get("pending")
    if pend and time.time() < deadline - 60:
        ck = _load_json(os.path.join(CKPT_DIR, f"{int(pend):03d}.json"), None)
        if ck:
            if _confirm(ck, deadline):
                _promote(st_lab, ck, {"module": ck.get("module")})
                journal("promoted", name=ck.get("name"),
                        params=ck.get("params"), score=ck.get("score"),
                        detail="подтверждено вторым окном предыдущего шага")
            else:
                journal("confirm_failed", name=ck.get("name"),
                        detail="второе окно не подтвердило, вариант отброшен")
        st_lab["pending"] = None

    if queue[q_idx:] and time.time() < deadline - 50:
        summary.update(_evaluate(st_lab, deadline))
    elif llm_ok and time.time() < deadline - 60:
        st_lab["phase"] = "plan"
        save_state(st_lab)
        done = _plan(st_lab, st_lab.get("baseline") or {}, {}, deadline)
        summary["planned"] = done
    else:
        st_lab["phase"] = "cooldown" if not llm_ok else st_lab["phase"]
        summary["idle"] = True
        summary["llm_used"] = st_lab.get("llm_used", 0)
    st_lab["steps"] = st_lab.get("steps", 0) + 1
    save_state(st_lab)
    _write_report(st_lab, summary)
    summary["ok"] = True
    summary["phase"] = st_lab["phase"]
    return summary


# --------------------------------------------------------------- evaluation
def _evaluate(st_lab, deadline):
    out = {"evaluated": 0}
    queue = st_lab.get("queue") or []
    q_idx = int(st_lab.get("q_idx", 0))
    base = st_lab.get("baseline")
    if not base:
        return out

    while q_idx < len(queue) and out["evaluated"] < MAX_VARIANTS_PER_STEP \
            and time.time() < deadline - 45:
        variant = queue[q_idx]
        if variant.get("module") and variant["module"].get("error"):
            journal("module_rejected", name=variant["module"].get("name"),
                    detail=variant["module"]["error"][:160])
            q_idx += 1
            continue
        symbols = _eval_symbols()
        cfg = _main_cfg(variant.get("params") or {})
        if variant.get("module"):
            reg = _lab_registry()
            exists = {m["name"] for m in reg}
            if variant["module"]["name"] not in exists:
                path = os.path.join(DEV_DIR, f"lab_{variant['module']['name']}.py")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(variant["module"]["code"])
                reg.append({"name": variant["module"]["name"], "file": path,
                            "status": "active", "decisions": 0, "vetoes": 0,
                            "fails": 0, "origin": "lab"})
                _save_json(MODULES_REGISTRY, reg)
            _with_lab_modules()
        wins, _srcs = _cached_windows(symbols, EVAL_DAYS)
        st_sim, tr, _eq = sim_mod.replay(wins, cfg,
                                         step_ms=opt_mod.OPT_STEP_MS,
                                         symbols=symbols)
        _restore_main_modules()
        res = _pack(st_sim, tr, cfg, variant["name"])
        res["note"] = variant.get("note", "")
        res["module"] = (variant.get("module") or {}).get("name") \
            if variant.get("module") else None
        ok, why = _robust(res, base)
        if ok:
            st_lab["best"] = res
            st_lab["ckpts"] = st_lab.get("ckpts", 0) + 1
            _save_json(os.path.join(CKPT_DIR, f"{st_lab['ckpts']:03d}.json"),
                       {"ts": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                        **res})
            _prune_checkpoints()
            if time.time() < deadline - 60 and _confirm(res, deadline):
                _promote(st_lab, res, variant)
                journal("promoted", name=variant["name"],
                        params=res.get("params"), score=res.get("score"),
                        detail="второе окно подтвердило превосходство")
            elif time.time() < deadline - 60:
                journal("confirm_failed", name=variant["name"],
                        detail="второе окно не подтвердило, вариант отброшен")
            else:
                st_lab["pending"] = st_lab["ckpts"]
                journal("checkpoint", name=variant["name"], score=res.get("score"),
                        detail="лучше baseline, подтверждение перенесено на следующий шаг")
        else:
            journal("failed", name=variant["name"], detail=why[:160],
                    score=res.get("score"), pnl=res.get("pnl"))
        out["evaluated"] += 1
        q_idx += 1
    if q_idx >= len(queue):
        queue.clear()
        q_idx = 0
        st_lab["queue"], st_lab["q_idx"] = [], 0
    else:
        st_lab["q_idx"] = q_idx
    return out


_saved_modules_path = None
_saved_cache = None


def _with_lab_modules():
    """На время бэктеста лабы активны ЕЁ модули (lab/modules.json) —
    оригинальный реестр не трогаем."""
    global _saved_modules_path, _saved_cache
    _saved_modules_path = ai_dev.MODULES_PATH
    _saved_cache = dict(ai_dev._module_cache)
    ai_dev.MODULES_PATH = MODULES_REGISTRY
    ai_dev._module_cache = {"key": None, "mods": []}


def _restore_main_modules():
    global _saved_modules_path, _saved_cache
    if _saved_modules_path is not None:
        ai_dev.MODULES_PATH = _saved_modules_path
        ai_dev._module_cache = _saved_cache
        _saved_modules_path = None


def _robust(res, base):
    if res["score"] <= -1e8:
        return False, "недостаточно трейдов или противоречивые чанки"
    if res["pnl"] <= base["pnl"]:
        return False, f"pnl {res['pnl']} <= baseline {base['pnl']}"
    if base["score"] > -1e8 and res["score"] <= base["score"] * 1.02:
        return False, (f"score {res['score']} не выше baseline "
                       f"{base['score']} +2%")
    return True, ""


def _confirm(res, deadline):
    """Второе окно: более старый кусок истории. Только если есть время."""
    if time.time() > deadline - 60:
        return False
    symbols = _eval_symbols()
    cfg = _main_cfg({})
    for k, v in (res.get("params") or {}).items():
        if _is_bounded(k):
            cfg[k] = v
    wins, _srcs = _cached_windows(symbols, EVAL_DAYS + 6)
    ts_list = sorted(w.ts for w in wins.values()
                     if len(w.ts) > sim_mod.WARMUP_BARS)
    if len(ts_list) < 2:
        return False
    sym0 = symbols[0]
    w = wins[sym0]
    i = sim_mod.WARMUP_BARS + 6 * 24 * 60  # 6 дней назад от окна eval
    if i >= len(w.ts):
        return False
    end_old = w.ts[i]
    st_sim, tr, _eq = sim_mod.replay(wins, cfg, start_ts=end_old,
                                     step_ms=opt_mod.OPT_STEP_MS,
                                     symbols=symbols)
    res2 = _pack(st_sim, tr, cfg, "confirm")
    return res2["score"] > -1e8 and res2["pnl"] > 0


def _is_bounded(k):
    return k in merged_bounds()


# --------------------------------------------------------------- promotion
def _promote(st_lab, res, variant):
    params = {k: v for k, v in (res.get("params") or {}).items() if _is_bounded(k)}
    prev = _load_json(RECOMMEND_PATH, {})
    merged = dict(prev.get("params") or {})
    merged.update(params)
    _save_json(RECOMMEND_PATH, {
        "ts": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "params": merged,
        "robust": True,
        "source": "lab seconda realidad",
        "last_score": res.get("score"),
    })
    promoted = st_lab.setdefault("promoted", [])
    mod = variant.get("module")
    if mod and not mod.get("error"):
        reg_main = []
        mreg = os.path.join(RESULTS, "ai_modules.json")
        if os.path.exists(mreg):
            try:
                reg_main = json.load(open(mreg))
            except Exception:
                reg_main = []
        if not any(m.get("name") == f"lab_{mod['name']}" for m in reg_main):
            src_path = os.path.join(DEV_DIR, f"lab_{mod['name']}.py")
            if os.path.exists(src_path):
                reg_main.append({"name": f"lab_{mod['name']}",
                                 "file": src_path, "status": "active",
                                 "decisions": 0, "vetoes": 0, "fails": 0,
                                 "origin": "lab"})
                with open(mreg, "w", encoding="utf-8") as f:
                    json.dump(reg_main, f, ensure_ascii=False, indent=1)
                promoted.append({"module": f"lab_{mod['name']}"})
    promoted.append({"params": params, "score": res.get("score"),
                     "ts": time.strftime("%Y-%m-%d %H:%M", time.gmtime())})
    st_lab["promoted"] = promoted[-10:]


def load_recommend_overrides():
    data = _load_json(RECOMMEND_PATH, {})
    if data.get("robust") and data.get("params"):
        return data["params"]
    return {}


def _prune_checkpoints():
    if not os.path.isdir(CKPT_DIR):
        return
    files = sorted(os.listdir(CKPT_DIR))
    while len(files) > MAX_CHECKPOINTS:
        os.remove(os.path.join(CKPT_DIR, files.pop(0)))


def _lab_registry():
    return _load_json(MODULES_REGISTRY, [])


# ----------------------------------------------------------------- report
def _write_report(st_lab, summary):
    rep = {
        "ts": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "phase": st_lab.get("phase"),
        "steps": st_lab.get("steps", 0),
        "llm_used_today": st_lab.get("llm_used", 0),
        "llm_day_limit": LLM_DAY_LIMIT,
        "queue_left": max(len(st_lab.get("queue") or []) - int(st_lab.get("q_idx", 0)), 0),
        "checkpoints": st_lab.get("ckpts", 0),
        "baseline": st_lab.get("baseline") and {
            k: st_lab["baseline"].get(k) for k in ("score", "trades", "wins", "pnl")},
        "best": st_lab.get("best") and {
            k: st_lab["best"].get(k) for k in ("name", "score", "trades", "wins", "pnl", "note")},
        "promoted": (st_lab.get("promoted") or [])[-3:],
        "last": {k: summary.get(k) for k in ("evaluated", "planned", "idle", "error")
                 if k in summary},
    }
    _save_json(REPORT_PATH, rep)
    return rep


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description="AEGIS лаборатория (вторая реальность)")
    ap.add_argument("--run", action="store_true", help="один бюджетный шаг")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--rollback", type=int, metavar="N",
                    help="откатить лучшее к чекпоинту N")
    ap.add_argument("--budget-s", type=int, default=STEP_TIME_S)
    args = ap.parse_args()
    if args.status:
        print(json.dumps(_load_json(REPORT_PATH, {}), ensure_ascii=False, indent=1))
        return
    if args.rollback is not None:
        path = os.path.join(CKPT_DIR, f"{args.rollback:03d}.json")
        ck = _load_json(path, None)
        st_lab = load_state()
        if ck:
            st_lab["best"] = ck
            save_state(st_lab)
            journal("rollback", detail=f"лучшая точка = чекпоинт {args.rollback}")
            print("rollback ok")
        else:
            print("checkpoint not found")
        return
    st = engine.load_state({})
    trs = engine.load_trades()
    _ = trs
    summary = maybe_step(st, trs, json.load(open(os.path.join(HERE, "config", "settings.json"))), {})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
