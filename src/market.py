"""AEGIS-Lab: market data ingestion with source fallback.

Primary: Bybit V5 public REST (api.bytick.com -> api.bybit.com).
Fallback: OKX public candles (same instruments, linear swaps).
No API keys needed for public market data.
"""
import json
import os
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(HERE, "config", "settings.json")) as f:
    CFG = json.load(f)

UA = {"User-Agent": "aegis-lab/1.0 (paper research)"}
TIMEOUT = 15


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def fetch_klines_bybit(symbol, interval="1", limit=200):
    """Returns ascending list of bars: {ts,o,h,l,c,v}. ts in ms."""
    last_err = None
    for host in ("api.bytick.com", "api.bybit.com"):
        url = (f"https://{host}/v5/market/kline?category=linear"
               f"&symbol={symbol}&interval={interval}&limit={limit}")
        try:
            data = _get_json(url)
            lst = data.get("result", {}).get("list") or []
            if not lst:
                last_err = f"{host}: empty list, retCode={data.get('retCode')}"
                continue
            bars = [{"ts": int(b[0]), "o": float(b[1]), "h": float(b[2]),
                     "l": float(b[3]), "c": float(b[4]), "v": float(b[5])}
                    for b in lst]
            bars.sort(key=lambda x: x["ts"])
            return bars, host
        except Exception as e:  # network/geo/parse
            last_err = f"{host}: {e}"
    raise RuntimeError(f"bybit klines failed for {symbol}: {last_err}")


def fetch_klines_okx(symbol, limit=200):
    inst = symbol.replace("USDT", "-USDT-SWAP")
    url = (f"https://www.okx.com/api/v5/market/candles"
           f"?instId={inst}&bar=1m&limit={min(limit, 300)}")
    data = _get_json(url)
    rows = data.get("data") or []
    if not rows:
        raise RuntimeError(f"okx klines empty for {symbol}: {data}")
    bars = [{"ts": int(b[0]), "o": float(b[1]), "h": float(b[2]),
             "l": float(b[3]), "c": float(b[4]), "v": float(b[5])}
            for b in rows]
    bars.sort(key=lambda x: x["ts"])
    return bars, "okx"


def fetch_klines(symbol, limit=200):
    """Try configured sources in order; return (bars, source_name)."""
    errors = []
    for src in CFG["data_sources"]:
        try:
            if src in ("bytick", "bybit"):
                return fetch_klines_bybit(symbol, limit=limit)
            if src == "okx":
                return fetch_klines_okx(symbol, limit=limit)
        except Exception as e:
            errors.append(str(e))
            time.sleep(0.5)
    raise RuntimeError(f"all sources failed for {symbol}: " + " | ".join(errors))
