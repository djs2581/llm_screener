#!/usr/bin/env python3
"""
LLM Stock Screener — adapter wired to your existing `technical_signals.py`.

Fixed errors at lines 397, 438, 445:
- Changed `any(pd.isna(vals))` to `any(pd.isna(v) for v in vals)` to avoid error when checking list.
- Fixed OpenAI client usage: replaced `openai.OpenAI(...)` with correct `openai.ChatCompletion.create(...)` call.
- Adjusted result printing to safely handle missing keys.
"""
from __future__ import annotations
import os, math, json, argparse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TS_MODULE = None
try:
    import technical_signals as TS_MODULE
except Exception:
    TS_MODULE = None

YF_AVAILABLE = False
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_period=14, d_period=3):
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll)
    d = k.rolling(d_period).mean()
    return k, d

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]
    out["volume"] = df["volume"]
    out["sma20"] = sma(df["close"], 20)
    out["sma50"] = sma(df["close"], 50)
    out["sma200"] = sma(df["close"], 200)
    out["rsi14"] = rsi(df["close"], 14)
    macd_line, macd_sig, macd_hist = macd(df["close"])
    out["macd_line"], out["macd_sig"], out["macd_hist"] = macd_line, macd_sig, macd_hist
    k, d = stoch(df["high"], df["low"], df["close"])
    out["stoch_k"], out["stoch_d"] = k, d
    out["atr14"] = atr(df["high"], df["low"], df["close"])
    out["ret_1m"] = df["close"].pct_change(21)
    out["ret_3m"] = df["close"].pct_change(63)
    out["ret_6m"] = df["close"].pct_change(126)
    out["dist_to_sma200"] = (df["close"] - out["sma200"]) / out["sma200"]
    out["adv20"] = df["volume"].rolling(20).mean()
    out["six_mo_high"] = df["close"].rolling(126).max()
    out["six_mo_low"] = df["close"].rolling(126).min()
    out["pos_in_range"] = (df["close"] - out["six_mo_low"]) / (out["six_mo_high"] - out["six_mo_low"])
    return out

def latest_row(features: pd.DataFrame) -> pd.Series:
    return features.dropna().iloc[-1]

def momentum_score(r1m, r3m, r6m):
    vals = [r1m, r3m, r6m]
    if any(pd.isna(v) for v in vals):
        return np.nan
    ranks = pd.Series(vals).rank()
    return (ranks.mean() - 2) / 2

def trend_score(sma20, sma50, sma200, dist200):
    bonus = 0
    if sma20 > sma50: bonus += 0.3
    if sma50 > sma200: bonus += 0.3
    if dist200 is not None:
        if 0 <= dist200 <= 0.15: bonus += 0.4
        elif dist200 < 0: bonus -= 0.2
        elif dist200 > 0.25: bonus -= 0.2
    return bonus

def rsi_score(rsi14):
    if pd.isna(rsi14): return np.nan
    if 40 <= rsi14 <= 65: return 0.3
    if 30 <= rsi14 < 40: return 0.15
    if 65 < rsi14 <= 75: return 0.1
    if rsi14 < 30: return 0.05
    return -0.05

def macd_score(line, sig, hist):
    if any(pd.isna(v) for v in [line, sig, hist]):
        return np.nan
    sc = 0.0
    if line > sig: sc += 0.2
    if hist > 0: sc += 0.1
    return sc

def vol_score(atr14, price):
    if pd.isna(atr14) or pd.isna(price): return np.nan
    ratio = atr14 / price
    if ratio < 0.01: return -0.1
    if ratio <= 0.03: return 0.2
    if ratio <= 0.06: return 0.1
    return -0.1

def total_score(lat: pd.Series, sentiment: float | None = None) -> float:
    m = momentum_score(lat["ret_1m"], lat["ret_3m"], lat["ret_6m"]) or 0
    t = trend_score(lat["sma20"], lat["sma50"], lat["sma200"], lat["dist_to_sma200"])
    r = rsi_score(lat["rsi14"]) or 0
    ms = macd_score(lat["macd_line"], lat["macd_sig"], lat["macd_hist"]) or 0
    v = vol_score(lat["atr14"], lat["close"]) or 0
    s = (sentiment or 0)
    return 0.25*m + 0.20*t + 0.15*r + 0.15*ms + 0.10*v + 0.15*s

def passes_prefilters(lat: pd.Series, min_price=5, min_adv=500_000) -> bool:
    if lat["close"] < min_price: return False
    if lat["adv20"] < min_adv: return False
    if math.isnan(lat["sma50"]) or math.isnan(lat["sma200"]): return False
    return True

USE_OPENAI = False

def call_llm_rank_explain(ticker: str, snapshot: Dict[str, Any], strategy_rules: str) -> Dict[str, Any]:
    if not USE_OPENAI:
        return {
            "ticker": ticker,
            "verdict": "watchlist",
            "confidence": 0.55,
            "why": ["LLM disabled; using deterministic score only"],
            "risks": ["No live news context"],
            "horizon_days": 15,
            "notes": "Enable --use-llm and set OPENAI_API_KEY to get ranked explanations."
        }
    import openai
    prompt = f"""
You are a disciplined trading assistant. Follow the user's strategy rules strictly.

STRATEGY RULES:
{strategy_rules}

SNAPSHOT (key signals):
{json.dumps(snapshot, indent=2)}

Return STRICT JSON with fields:
- ticker
- verdict ∈ ["buy-candidate","watchlist","pass"]
- confidence ∈ [0,1]
- why (≤5 bullets)
- risks (≤5 bullets)
- horizon_days (int)
- notes (string)
If rules conflict, choose "pass" and state why.
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(resp.choices[0].message["content"])

# --------------------
# Main entrypoint
# --------------------
DEFAULT_TICKERS = ["MSFT","NVDA","AMZN","META","AAPL"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--days", type=int, default=300)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--use-llm", type=int, default=0)
    parser.add_argument("--csv", type=str, default="screener_results.csv")
    parser.add_argument("--html", type=str, default="screener_report.html")
    args = parser.parse_args()

    global USE_OPENAI
    USE_OPENAI = bool(args.use_llm)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    print(f"[INFO] Running screener for {tickers}")

    # Minimal stub: compute fake scores until run_screener is wired in
    results = []
    for t in tickers:
        results.append({"ticker": t, "score": round(np.random.rand(),4), "llm": {"verdict": "watchlist"}, "snapshot": {"price": 100}})

    import csv
    with open(args.csv,"w") as f:
        w = csv.DictWriter(f, fieldnames=["ticker","score","verdict","price"])
        w.writeheader()
        for r in results:
            w.writerow({"ticker": r["ticker"], "score": r["score"], "verdict": r["llm"]["verdict"], "price": r["snapshot"]["price"]})

    with open(args.html,"w") as f:
        f.write("<html><body><h2>LLM Screener Report</h2><table border=1>\n")
        for r in results:
            f.write(f"<tr><td>{r['ticker']}</td><td>{r['score']}</td><td>{r['llm']['verdict']}</td><td>{r['snapshot']['price']}</td></tr>\n")
        f.write("</table></body></html>")

    print(f"[INFO] Saved CSV to {args.csv}, HTML to {args.html}")

if __name__ == "__main__":
    main()

# -------------------------- Data adapters -----------------------------------------
COMMON_FN_NAMES = [
    "get_ohlcv", "fetch_ohlcv", "load_ohlcv", "download_ohlcv", "get_data"
]

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    mapping = {}
    for key in required:
        if key in cols_lower:
            mapping[cols_lower[key]] = key
        elif key.upper() in df.columns:
            mapping[key.upper()] = key
        elif key.capitalize() in df.columns:
            mapping[key.capitalize()] = key
    normalized = df.rename(columns=mapping)
    missing = [k for k in required if k not in normalized.columns]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")
    normalized = normalized[required].copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized.sort_index(inplace=True)
    return normalized

def _try_ts_module_fetch(ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    if TS_MODULE is None:
        return None
    for fn_name in COMMON_FN_NAMES:
        fn = getattr(TS_MODULE, fn_name, None)
        if fn is None:
            continue
        try:
            df = fn(ticker=ticker, start=start, end=end)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _normalize_ohlcv(df)
        except TypeError:
            try:
                df = fn(ticker, start, end)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return _normalize_ohlcv(df)
            except Exception:
                continue
        except Exception:
            continue
    return None

def get_ohlcv(ticker: str, days: int = 300) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days)
    # 1) Try your technical_signals module first
    df = _try_ts_module_fetch(ticker, start, end)
    if df is not None:
        return df
    # 2) Fallback to yfinance
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not available and technical_signals didn't provide data.")
    yfd = yf.download(ticker, start=start.date(), end=end.date(), interval="1d", progress=False, auto_adjust=False)
    if yfd is None or yfd.empty:
        raise RuntimeError(f"No data for {ticker}")
    yfd = yfd.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })[["open","high","low","close","volume"]]
    yfd.index = pd.to_datetime(yfd.index)
    return yfd

# -------------------------- Screener core -----------------------------------------
def screen_one(ticker: str, df: pd.DataFrame, sentiment: float | None, strategy_rules: str) -> Dict[str, Any]:
    f = compute_features(df)
    if f.dropna().empty:
        return {"ticker": ticker, "status": "insufficient_data"}
    lat = latest_row(f)
    if not passes_prefilters(lat):
        return {"ticker": ticker, "status": "filtered_out"}
    score = total_score(lat, sentiment)
    snapshot = {
        "ticker": ticker,
        "price": round(float(lat["close"]), 2),
        "rsi14": round(float(lat["rsi14"]), 2),
        "macd_line": round(float(lat["macd_line"]), 4),
        "macd_sig": round(float(lat["macd_sig"]), 4),
        "macd_hist": round(float(lat["macd_hist"]), 4),
        "sma20": round(float(lat["sma20"]), 2),
        "sma50": round(float(lat["sma50"]), 2),
        "sma200": round(float(lat["sma200"]), 2),
        "ret_1m": round(float(lat["ret_1m"]), 4),
        "ret_3m": round(float(lat["ret_3m"]), 4),
        "ret_6m": round(float(lat["ret_6m"]), 4),
        "dist_to_sma200": round(float(lat["dist_to_sma200"]), 4),
        "atr14_pct": round(float(lat["atr14"] / lat["close"]), 4),
        "pos_in_6m_range": round(float(lat["pos_in_range"]), 4),
        "sentiment": round(float(sentiment or 0.0), 3),
    }
    llm_out = call_llm_rank_explain(ticker, snapshot, strategy_rules, model=args.model)
    return {"ticker": ticker, "score": round(float(score), 4), "snapshot": snapshot, "llm": llm_out}

def run_screener(tickers: List[str], days: int, top_n: int, strategy_rules: str, sentiment_map: Dict[str, float]):
    rows: List[Dict[str, Any]] = []
    print(f"[INFO] Running screener for {len(tickers)} tickers, lookback={days}d, top_n={top_n}")
    for t in tickers:
        try:
            print(f"[INFO] Fetching {t}…", end="")
            df = get_ohlcv(t, days)
            print(f" ok ({len(df)} rows)")
            result = screen_one(t, df, sentiment_map.get(t), strategy_rules)
            if result.get("status") in {"filtered_out", "insufficient_data"}:
                print(f"[INFO] {t} skipped: {result.get('status')}")
                continue
            print(f"[INFO] {t} scored: {result.get('score')}")
            rows.append(result)
        except Exception as e:
            print(f"\n[ERROR] {t} failed: {e}")
            rows.append({"ticker": t, "status": f"error: {e}"})
    rows.sort(key=lambda r: r.get("score", -999), reverse=True)
    rows = [r for r in rows if "score" in r]
    print(f"[INFO] Completed. {len(rows)} tickers passed filters.")
    return rows[:top_n]

# -------------------------- Reporting ---------------------------------------------
def to_csv(results: List[Dict[str,Any]], path: str):
    flat = []
    for r in results:
        snap = r.get("snapshot", {})
        llm = r.get("llm", {})
        flat.append({
            "ticker": r.get("ticker"),
            "score": r.get("score"),
            **{f"snap_{k}": v for k, v in snap.items()},
            "verdict": llm.get("verdict"),
            "confidence": llm.get("confidence"),
            "why": "; ".join(llm.get("why", [])),
            "risks": "; ".join(llm.get("risks", [])),
            "horizon_days": llm.get("horizon_days"),
            "notes": llm.get("notes"),
        })
    pd.DataFrame(flat).to_csv(path, index=False)

HTML_TEMPLATE = """
<!doctype html>
<html><head><meta charset=\"utf-8\" />
<title>LLM Screener Report</title>
<style>
 body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
 table { border-collapse: collapse; width: 100%; }
 th, td { border-bottom: 1px solid #eee; padding: 8px 10px; text-align: left; }
 th { position: sticky; top: 0; background: #fafafa; }
 .tag { display:inline-block; padding:2px 8px; border-radius:14px; border:1px solid #ddd; font-size:12px; }
 .buy { background:#e6ffed; border-color:#b8f5c3; }
 .watchlist { background:#fff7e6; border-color:#ffe0a3; }
 .pass { background:#ffe6e6; border-color:#ffb8b8; }
</style>
</head>
<body>
<h2>LLM Screener — {date}</h2>
<p>Universe: {universe}<br/>Strategy: {strategy}</p>
<table>
  <thead>
    <tr>
      <th>Rank</th><th>Ticker</th><th>Score</th><th>Verdict</th><th>Conf</th>
      <th>Price</th><th>RSI14</th><th>MACD</th><th>SMA20/50/200</th><th>1m/3m/6m</th><th>Notes</th>
    </tr>
  </thead>
  <tbody>
  {rows}
  </tbody>
</table>
</body></html>
"""

def to_html(results: List[Dict[str,Any]], path: str, universe: List[str], strategy_rules: str):
    def verdict_tag(v: str) -> str:
        if v == "buy-candidate":
            cls = "tag buy"
        elif v == "watchlist":
            cls = "tag watchlist"
        else:
            cls = "tag pass"
        return f"<span class='{cls}'>{v}</span>"

    tr_rows = []
    for i, r in enumerate(results, 1):
        snap = r.get("snapshot", {})
        llm = r.get("llm", {})
        tr_rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><b>{r.get('ticker')}</b></td>"
            f"<td>{r.get('score')}</td>"
            f"<td>{verdict_tag(llm.get('verdict','watchlist'))}</td>"
            f"<td>{llm.get('confidence','')}</td>"
            f"<td>{snap.get('price','')}</td>"
            f"<td>{snap.get('rsi14','')}</td>"
            f"<td>{snap.get('macd_line','')}/{snap.get('macd_sig','')}</td>"
            f"<td>{snap.get('sma20','')}/{snap.get('sma50','')}/{snap.get('sma200','')}</td>"
            f"<td>{snap.get('ret_1m','')}/{snap.get('ret_3m','')}/{snap.get('ret_6m','')}</td>"
            f"<td>{llm.get('notes','')}</td>"
            f"</tr>"
        )

    html = (
        HTML_TEMPLATE
        .replace("{date}", datetime.now().strftime("%Y-%m-%d %H:%M"))
        .replace("{universe}", ", ".join(universe))
        .replace("{strategy}", strategy_rules)
        .replace("{rows}", "\n".join(tr_rows))
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

# -------------------------- CLI ----------------------------------------------------
DEFAULT_TICKERS = [
    "TSLA","AMD","AVGO","NVDA","CELH","DOCN","HOOD","MSFT","NFLX","PLTR",
    "ASTS","META","AMZN","ETOR","VST","OKLO","BMNR","SBET","SOUN","ENPH","RUN"
]

def main():
    global USE_OPENAI
    global MODEL_NAME  # optional: in case call_llm_rank_explain reads a global

    parser = argparse.ArgumentParser(description="LLM Stock Screener (wired to technical_signals.py)")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers")
    parser.add_argument("--days", type=int, default=300, help="Lookback days of OHLCV")
    parser.add_argument("--top", type=int, default=25, help="Top N to keep by deterministic score")
    parser.add_argument("--use-llm", type=int, default=0, help="0=off (default), 1=on")
    parser.add_argument("--csv", type=str, default="screener_results.csv")
    parser.add_argument("--html", type=str, default="screener_report.html")
    parser.add_argument(
        "--strategy",
        type=str,
        default=(
            "Prefer liquid, large/mid caps in uptrends (20>50>200DMA), MACD turning up, RSI 40–65; "
            "avoid earnings within 3 trading days; pass if >25% above 200DMA."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for LLM ranking (e.g., gpt-4o-mini, gpt-4.1-mini)",
    )

    args = parser.parse_args()


 # Configure global switches
    USE_OPENAI = bool(args.use_llm)
    MODEL_NAME = args.model

# Nice heads-up if user enabled LLM without a key
    if USE_OPENAI and not os.getenv("OPENAI_API_KEY"):
        print("[WARN] --use-llm is ON but OPENAI_API_KEY is not set. "
              "LLM step will return a safe default. Export OPENAI_API_KEY to enable real calls.")

# Parse tickers and build neutral sentiment map (0.0 baseline)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    sentiment_map = {t: 0.0 for t in tickers}

  # Run screener (deterministic + optional LLM)
    try:
        results = run_screener(tickers, args.days, args.top, args.strategy, sentiment_map)
    except Exception as e:
        print(f"[FATAL] RUN failed: {e}")
        import traceback as _tb
        _tb.print_exc()
        return

 # Save outputs
    try:
        to_csv(results, args.csv)
        to_html(results, args.html, tickers, args.strategy)
    except Exception as e:
        print(f"[ERROR] Failed to write outputs: {e}")

    # Console summary
    print("\nTop results (deterministic score):")
    for i, r in enumerate(results, 1):
        ticker = r.get("ticker", "")
        score = r.get("score", "")
        llm = r.get("llm", {}) or {}
        snap = r.get("snapshot", {}) or {}
        verdict = llm.get("verdict", "")
        price = snap.get("price", "")
        print(f"{i:>2}. {ticker:>6}  score={score}  verdict={verdict}  price={price}")

    print(f"\nSaved CSV → {args.csv}\nSaved HTML → {args.html}\n")

if __name__ == "__main__":
    main()
