"""
NSE Darvas Box Scanner - Technical Indicators
Stateless functions operating on pandas Series/DataFrames.
All calculations are vectorised with pandas/numpy – no external TA libraries required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Moving Averages ──────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ─── RSI ──────────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── ATR ──────────────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ─── ADX ──────────────────────────────────────────────────────────────────────

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Returns DataFrame with columns: ADX, +DI, -DI."""
    high_diff  = high.diff()
    low_diff   = (-low).diff()

    plus_dm  = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=high.index)
    minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=high.index)

    _atr = atr(high, low, close, period)

    plus_di  = 100 * (plus_dm.ewm(com=period - 1, adjust=False).mean()  / _atr)
    minus_di = 100 * (minus_dm.ewm(com=period - 1, adjust=False).mean() / _atr)

    dx  = (((plus_di - minus_di).abs()) / (plus_di + minus_di).replace(0, np.nan)) * 100
    _adx = dx.ewm(com=period - 1, adjust=False).mean()

    return pd.DataFrame({"ADX": _adx, "+DI": plus_di, "-DI": minus_di})


# ─── Volume helpers ───────────────────────────────────────────────────────────

def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume / n-period average."""
    return volume / volume.rolling(period).mean()


def is_accumulation_day(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Up day on above-average volume."""
    up   = close.diff() > 0
    hvol = volume > volume.rolling(20).mean()
    return up & hvol


# ─── Trend Structure ──────────────────────────────────────────────────────────

def higher_highs_higher_lows(high: pd.Series, low: pd.Series, lookback: int = 20) -> bool:
    """True if the last *lookback* bars form an uptrend (HH-HL structure)."""
    if len(high) < lookback:
        return False
    h = high.iloc[-lookback:]
    l = low.iloc[-lookback:]
    # Simple linear slope test
    x = np.arange(lookback)
    h_slope = np.polyfit(x, h.values, 1)[0]
    l_slope = np.polyfit(x, l.values, 1)[0]
    return h_slope > 0 and l_slope > 0


def trend_label(close: pd.Series, fast: int = 50, slow: int = 200) -> str:
    """Return 'bullish', 'bearish', or 'neutral'."""
    if len(close) < slow:
        return "neutral"
    f = ema(close, fast).iloc[-1]
    s = ema(close, slow).iloc[-1]
    last = close.iloc[-1]
    if last > f > s:
        return "bullish"
    if last < f < s:
        return "bearish"
    return "neutral"


# ─── Relative Strength Rating (O'Neil style) ─────────────────────────────────

def rs_rating(
    close: pd.Series,
    benchmark: pd.Series,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Returns a 1-99 RS Rating comparing *close* to *benchmark*.
    Uses weighted performance across 3, 6, 9, 12-month periods.
    """
    if weights is None:
        weights = {"3m": 0.40, "6m": 0.20, "9m": 0.20, "12m": 0.20}

    periods = {"3m": 63, "6m": 126, "9m": 189, "12m": 252}
    stock_score = 0.0
    bench_score = 0.0
    total_w     = 0.0

    for key, w in weights.items():
        n = periods[key]
        if len(close) > n and len(benchmark) > n:
            s_ret = (close.iloc[-1] / close.iloc[-n] - 1)
            b_ret = (benchmark.iloc[-1] / benchmark.iloc[-n] - 1)
            stock_score += s_ret * w
            bench_score += b_ret * w
            total_w += w

    if total_w == 0 or bench_score == 0:
        return 50.0

    ratio = (stock_score / total_w) / (bench_score / total_w + 1e-9)
    # Normalise to 1-99
    raw = (ratio - 0.5) * 99
    return float(np.clip(raw + 50, 1, 99))


# ─── SEPA (Minervini) Checks ──────────────────────────────────────────────────

def sepa_score(close: pd.Series) -> tuple[float, dict]:
    """
    Returns (score_0_to_10, details_dict).
    Checks the 8 core SEPA template criteria.
    """
    checks: dict[str, bool] = {}
    if len(close) < 252:
        return 0.0, checks

    last    = close.iloc[-1]
    ema_50  = ema(close, 50).iloc[-1]
    ema_150 = ema(close, 150).iloc[-1]
    ema_200 = ema(close, 200).iloc[-1]
    high_52 = close.iloc[-252:].max()
    low_52  = close.iloc[-252:].min()

    checks["price_above_50"]    = last > ema_50
    checks["price_above_150"]   = last > ema_150
    checks["price_above_200"]   = last > ema_200
    checks["ema150_above_200"]  = ema_150 > ema_200
    checks["ema50_above_150"]   = ema_50 > ema_150
    checks["ema50_above_200"]   = ema_50 > ema_200
    checks["within_25pct_high"] = last >= high_52 * 0.75
    checks["low_30pct_below"]   = low_52 <= last * 0.70

    score = sum(checks.values()) / len(checks) * 10
    return round(score, 2), checks
