"""
NSE Stock Analysis API — for n8n integration
================================================================================
A single FastAPI service that does ALL the real work (data pull + every
indicator built in our Jupyter session) and exposes it as one clean HTTP
endpoint. n8n calls this URL and gets back a complete JSON analysis for
ANY NSE-listed company — no coding required inside n8n itself.

REQUIREMENTS
------------
    pip install fastapi uvicorn yfinance pandas numpy

RUN LOCALLY (for testing in Jupyter/terminal)
-----------------------------------------------
    uvicorn nse_analysis_api:app --reload --port 8000

Then test in your browser or Jupyter:
    http://127.0.0.1:8000/analyze/RELIANCE

EXPOSING THIS TO n8n
---------------------
- If n8n runs on the SAME machine: use http://127.0.0.1:8000 directly (self-hosted n8n)
- If n8n is in the cloud (n8n.cloud) or on a different machine: you need a
  public URL. Easiest free options:
    1. ngrok (quick, good for testing):  ngrok http 8000
       -> gives you a temporary public URL like https://abc123.ngrok.io
    2. Deploy this file properly on Render.com or Railway.app (free tier)
       for a permanent URL — recommended once you move past testing

USAGE FROM n8n
---------------
In n8n, add an "HTTP Request" node:
    Method: GET
    URL: http://YOUR_URL/analyze/{{ $json.symbol }}
No code needed on the n8n side at all — this endpoint does everything.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="NSE Stock Analysis API")

# Allow any frontend (v0.dev, Lovable, or anywhere else) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── INDICATOR FUNCTIONS (same ones built in our Jupyter session today) ────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_moving_averages(prices, short_window=50, long_window=200):
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()
    return ma_short, ma_long

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100

def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth — the corrected version, fixing the bug
    we caught earlier today where fixed-position matching broke on
    stocks with an irregularly reported quarter."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def safe_round(val, decimals=2):
    if val is None or pd.isna(val):
        return None
    return round(float(val), decimals)


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    """Same date-matched logic as earnings growth, applied to revenue."""
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Extended fundamentals: earnings growth, revenue growth, margins, ROE, debt-to-equity.

    IMPORTANT: this deliberately avoids stock.info — Yahoo Finance blocks/rate-limits
    that specific endpoint for many cloud hosting providers (Render, Railway, Heroku),
    returning an almost-empty dict even though the price-history endpoint works fine.
    Instead, everything here is calculated directly from the income statement and
    balance sheet, which use the endpoint that's actually working for us."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        # margins calculated directly — same math a real analyst would do by hand
        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)

    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None

            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    equity_series = balance_sheet.loc[equity_label].dropna()
                    if len(equity_series) > 0:
                        equity = equity_series.iloc[0]
                        break

            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    debt_series = balance_sheet.loc[debt_label].dropna()
                    if len(debt_series) > 0:
                        total_debt = debt_series.iloc[0]
                        break

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                # ROE using latest quarter's net income annualized (x4) against latest equity —
                # a simplified TTM-style approximation, standard when only one quarter is available
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)

    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    # market cap: try the lighter fast_info endpoint, which is less likely to be blocked
    # than the full .info dict — separate try block so a failure here doesn't affect anything above
    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass  # market cap is a nice-to-have; not worth failing the request over

    return fundamentals


def get_technical_snapshot(hist):
    """The core technical calculation, factored out so both /analyze and /watchlist can reuse it."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


@app.get("/")
def health_check():
    return {"status": "ok", "message": "NSE Stock Analysis API is running"}


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        technical, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = get_fundamentals(stock)

        return {
            "symbol": ticker,
            "company_name": ticker,
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years — "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag else None
            ),
            "technical": technical,
            "fundamentals": fundamentals,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}")
def chart_data(symbol: str, months: int = 6):
    """
    OHLC candlestick data + overlay indicator series for charting.
    Returns one row per trading day: date, OHLCV, RSI, MACD line/signal, MA50, MA200.
    `months` controls how much history to return in the chart (default 6, max 24).
    """
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    months = min(max(months, 1), 24)

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)  # need 2y so MA200 has enough lookback, even though we only return the recent slice
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, _ = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)

        trading_days = months * 21  # approx trading days per month
        recent = hist.tail(trading_days)

        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]),
                "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]),
                "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]) if date in rsi.index else None,
                "macd_line": safe_round(macd_line.loc[date]) if date in macd_line.index else None,
                "macd_signal": safe_round(signal_line.loc[date]) if date in signal_line.index else None,
                "ma_50": safe_round(ma50.loc[date]) if date in ma50.index else None,
                "ma_200": safe_round(ma200.loc[date]) if date in ma200.index else None,
            })

        return {"symbol": ticker, "months": months, "data": rows}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist")
def watchlist(symbols: str):
    """
    Lightweight multi-stock snapshot for a watchlist view.
    Pass symbols comma-separated: /watchlist?symbols=RELIANCE,TCS,INFY
    Capped at 10 symbols per request to keep response time reasonable on free hosting.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y", auto_adjust=True)
            if hist.empty:
                results.append({"symbol": ticker, "error": "No data found"})
                continue

            technical, corporate_action_flag = get_technical_snapshot(hist)
            results.append({
                "symbol": ticker,
                "company_name": ticker,
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": corporate_action_flag,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    return {"count": len(results), "stocks": results}


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURES (v2):  /market  ·  /news/{symbol}  ·  /backtest/{symbol}
#  Uses only the packages already in requirements.txt (no new installs).
# ══════════════════════════════════════════════════════════════════════════
import time

# Small in-memory cache so repeated clicks don't hammer Yahoo Finance
# (keeps the free Render tier fast and avoids rate-limits).
_CACHE = {}

def cached(key, ttl_seconds, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl_seconds:
        return hit[1]
    value = fn()
    _CACHE[key] = (now, value)
    return value


# ── 1. MARKET OVERVIEW (NIFTY 50 / SENSEX / BANK NIFTY) ───────────────────

INDICES = [("NIFTY 50", "^NSEI"), ("SENSEX", "^BSESN"), ("BANK NIFTY", "^NSEBANK")]

def _fetch_market():
    out = []
    for name, tkr in INDICES:
        try:
            h = yf.Ticker(tkr).history(period="10d").dropna(subset=["Close"])
            if len(h) < 2:
                out.append({"name": name, "symbol": tkr, "error": "No data"})
                continue
            last = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2])
            out.append({
                "name": name,
                "symbol": tkr,
                "level": round(last, 2),
                "change": round(last - prev, 2),
                "change_pct": round((last / prev - 1) * 100, 2),
                "as_of": h.index[-1].strftime("%Y-%m-%d"),
            })
        except Exception as e:
            out.append({"name": name, "symbol": tkr, "error": str(e)})
    return {"indices": out, "timestamp": datetime.now().isoformat()}


@app.get("/market")
def market_overview():
    """Index level + daily change for NIFTY 50, SENSEX and BANK NIFTY (cached 2 min)."""
    return cached("market", 120, _fetch_market)


# ── 2. NEWS HEADLINES ──────────────────────────────────────────────────────

def _parse_news_item(item):
    """yfinance changed its news format; this handles both the new and old layouts."""
    content = item.get("content") if isinstance(item, dict) else None
    if content:
        url = ((content.get("canonicalUrl") or {}).get("url")
               or (content.get("clickThroughUrl") or {}).get("url"))
        return {
            "title": content.get("title"),
            "publisher": (content.get("provider") or {}).get("displayName"),
            "published": content.get("pubDate"),
            "url": url,
            "summary": content.get("summary"),
        }
    ts = item.get("providerPublishTime")
    return {
        "title": item.get("title"),
        "publisher": item.get("publisher"),
        "published": datetime.fromtimestamp(ts).isoformat() if ts else None,
        "url": item.get("link"),
        "summary": None,
    }


@app.get("/news/{symbol}")
def stock_news(symbol: str):
    """Latest headlines for one stock (max 10, cached 5 min)."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    def fetch():
        try:
            raw = yf.Ticker(ticker).news or []
        except Exception as e:
            return {"symbol": ticker, "count": 0, "news": [],
                    "note": f"News temporarily unavailable: {e}"}
        items = [_parse_news_item(i) for i in raw]
        items = [i for i in items if i.get("title")][:10]
        return {"symbol": ticker, "count": len(items), "news": items,
                "note": None if items else "No recent headlines found for this stock."}

    return cached(f"news:{ticker}", 300, fetch)


# ── 3. BACKTESTING ENGINE ──────────────────────────────────────────────────

BACKTEST_STRATEGIES = {
    "ma_crossover": "Hold while the short MA is above the long MA (golden cross in, death cross out)",
    "rsi": "Buy when RSI falls below the low level, sell when it rises above the high level",
    "macd": "Hold while the MACD line is above its signal line",
    "bollinger": "Buy when price closes below the lower Bollinger Band, sell when it returns to the middle band",
    "buy_and_hold": "Buy on day one and never sell (the benchmark)",
}


def _stateful_signal(entry_cond, exit_cond):
    """Turns 'enter when X, exit when Y' rules into a 0/1 holding series."""
    state = 0
    out = []
    for enter, leave in zip(entry_cond, exit_cond):
        if enter is None:                 # indicator not ready yet
            out.append(np.nan)
            continue
        if state == 0 and enter:
            state = 1
        elif state == 1 and leave:
            state = 0
        out.append(state)
    return out


def build_signal(close, strategy, p):
    """Returns a Series of 1 (in the market) / 0 (out) / NaN (indicator warming up)."""
    if strategy == "ma_crossover":
        s, l = calculate_moving_averages(close, p["short_window"], p["long_window"])
        sig = (s > l).astype(float)
        sig[s.isna() | l.isna()] = np.nan
        return sig

    if strategy == "macd":
        m, sg, _ = calculate_macd(close)
        sig = (m > sg).astype(float)
        sig.iloc[:35] = np.nan            # let the EMAs settle first
        return sig

    if strategy == "rsi":
        rsi = calculate_rsi(close)
        ready = ~rsi.isna()
        entry = [bool(v < p["rsi_low"]) if r else None for v, r in zip(rsi, ready)]
        exit_ = [bool(v > p["rsi_high"]) if r else None for v, r in zip(rsi, ready)]
        return pd.Series(_stateful_signal(entry, exit_), index=close.index)

    if strategy == "bollinger":
        pb, _ = calculate_bollinger_bands(close)
        ready = ~pb.isna()
        entry = [bool(v < 0) if r else None for v, r in zip(pb, ready)]
        exit_ = [bool(v > 0.5) if r else None for v, r in zip(pb, ready)]
        return pd.Series(_stateful_signal(entry, exit_), index=close.index)

    # buy_and_hold
    return pd.Series(1.0, index=close.index)


def run_backtest(close, strategy, p):
    sig = build_signal(close, strategy, p)
    start = sig.first_valid_index()
    if start is None:
        raise ValueError("Not enough history to compute this strategy's indicators.")
    sig = sig.loc[start:].fillna(0)
    close = close.loc[start:]
    if len(close) < 30:
        raise ValueError("Not enough history after indicator warm-up. Try more years.")

    # Signal is known at today's close -> position is held from the NEXT day's return
    position = sig.shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    turnover = position.diff().abs().fillna(0)
    strat_ret = position * ret - turnover * (p["cost_pct"] / 100)

    equity = (1 + strat_ret).cumprod() * 100
    bench = (1 + ret).cumprod() * 100

    # ── trades ──
    changes = sig.diff().fillna(sig.iloc[0])
    trades, entry = [], None
    for date, ch in changes.items():
        if ch > 0:
            entry = (date, float(close.loc[date]))
        elif ch < 0 and entry is not None:
            px = float(close.loc[date])
            trades.append({
                "entry_date": entry[0].strftime("%Y-%m-%d"), "entry_price": round(entry[1], 2),
                "exit_date": date.strftime("%Y-%m-%d"), "exit_price": round(px, 2),
                "return_pct": round((px / entry[1] - 1) * 100, 2), "open": False,
            })
            entry = None
    if entry is not None:
        px = float(close.iloc[-1])
        trades.append({
            "entry_date": entry[0].strftime("%Y-%m-%d"), "entry_price": round(entry[1], 2),
            "exit_date": close.index[-1].strftime("%Y-%m-%d"), "exit_price": round(px, 2),
            "return_pct": round((px / entry[1] - 1) * 100, 2), "open": True,
        })

    closed = [t for t in trades if not t["open"]]
    wins = [t for t in closed if t["return_pct"] > 0]

    def summarize(eq, rets):
        years = max(len(eq) / 252, 1 / 252)
        total = eq.iloc[-1] / 100 - 1
        cagr = (eq.iloc[-1] / 100) ** (1 / years) - 1
        dd = (eq / eq.cummax() - 1).min()
        vol = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() * 252) / vol if vol and not pd.isna(vol) and vol > 0 else None
        return {
            "total_return_pct": safe_round(total * 100),
            "cagr_pct": safe_round(cagr * 100),
            "max_drawdown_pct": safe_round(dd * 100),
            "annual_volatility_pct": safe_round(vol * 100),
            "sharpe_ratio": safe_round(sharpe),
        }

    strategy_stats = summarize(equity, strat_ret)
    strategy_stats.update({
        "num_trades": len(trades),
        "win_rate_pct": safe_round(len(wins) / len(closed) * 100) if closed else None,
        "time_in_market_pct": safe_round(float(position.mean()) * 100),
    })

    rows = [{
        "date": d.strftime("%Y-%m-%d"),
        "close": safe_round(close.loc[d]),
        "strategy_equity": safe_round(equity.loc[d]),
        "buy_hold_equity": safe_round(bench.loc[d]),
        "in_market": int(position.loc[d]),
    } for d in close.index]

    return {
        "strategy": strategy,
        "description": BACKTEST_STRATEGIES[strategy],
        "period": {"start": close.index[0].strftime("%Y-%m-%d"),
                   "end": close.index[-1].strftime("%Y-%m-%d"), "trading_days": len(close)},
        "currently_in_market": bool(sig.iloc[-1] == 1),
        "strategy_stats": strategy_stats,
        "buy_hold_stats": summarize(bench, ret),
        "trades": trades,
        "data": rows,
        "assumptions": (f"Long-only, one share class, signals act at the close and earn returns from the "
                        f"next day, {p['cost_pct']}% cost per trade, no tax/slippage. Educational only — "
                        f"past performance does not predict future returns."),
    }


@app.get("/backtest/strategies")
def list_strategies():
    """Which strategies the backtester supports (for building a dropdown)."""
    return {"strategies": [{"id": k, "description": v} for k, v in BACKTEST_STRATEGIES.items()]}


@app.get("/backtest/{symbol}")
def backtest(symbol: str,
             strategy: str = "ma_crossover",
             years: int = 3,
             short_window: int = 50,
             long_window: int = 200,
             rsi_low: int = 30,
             rsi_high: int = 70,
             cost_pct: float = 0.1):
    """
    Backtest one strategy on one NSE stock.
    Example: /backtest/RELIANCE?strategy=rsi&years=3&rsi_low=30&rsi_high=70
    strategy = ma_crossover | rsi | macd | bollinger | buy_and_hold
    """
    if strategy not in BACKTEST_STRATEGIES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown strategy. Choose one of: {', '.join(BACKTEST_STRATEGIES)}")
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    years = min(max(years, 1), 5)
    params = {
        "short_window": min(max(short_window, 2), 100),
        "long_window": min(max(long_window, 5), 250),
        "rsi_low": min(max(rsi_low, 5), 49),
        "rsi_high": min(max(rsi_high, 51), 95),
        "cost_pct": min(max(cost_pct, 0.0), 2.0),
    }
    if params["short_window"] >= params["long_window"]:
        raise HTTPException(status_code=400, detail="short_window must be smaller than long_window.")

    try:
        hist = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")
        result = run_backtest(hist["Close"].dropna(), strategy, params)
        result["symbol"] = ticker
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backtesting {ticker}: {str(e)}")
