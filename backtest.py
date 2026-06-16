"""
NSE Darvas Box Scanner - Backtesting & Forward Validation
==========================================================
Two testing modes:

1. Traditional Backtest
   Simulates the strategy on historical data for each symbol,
   then computes: CAGR, Win Rate, Profit Factor, Max Drawdown,
   Sharpe, Sortino, Average Hold Period.

2. Forward Validation
   Uses actual generated signals (from the database) and their
   tracked outcomes to measure real performance – no look-ahead bias.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from database import get_all_signals_df
from darvas import detect_darvas_boxes
from downloader import load_daily
from indicators import atr as calc_atr, rsi as calc_rsi, adx as calc_adx, ema
from logger_utils import get_logger

log = get_logger("performance")

RISK_FREE_RATE = 0.065  # RBI repo rate proxy


# ─── Forward Validation (uses actual tracked signals) ─────────────────────────

def forward_validation_report() -> pd.DataFrame:
    """
    Aggregate performance of tracked signals from the database.
    Returns a DataFrame with yearly statistics.
    """
    df = get_all_signals_df()
    if df.empty:
        log.warning("No signals in database for forward validation.")
        return pd.DataFrame()

    triggered = df[df["entry_triggered"] == 1].copy()
    if triggered.empty:
        return pd.DataFrame()

    triggered["scan_date"] = pd.to_datetime(triggered["scan_date"])
    triggered["year"] = triggered["scan_date"].dt.year

    yearly_rows = []
    for year, grp in triggered.groupby("year"):
        row = _calc_metrics(grp, label=str(year))
        yearly_rows.append(row)

    overall = _calc_metrics(triggered, label="All Years")
    yearly_rows.append(overall)

    result = pd.DataFrame(yearly_rows)
    log.info("Forward validation: %d years of data, %d total triggered signals",
             len(yearly_rows) - 1, len(triggered))
    return result


def _calc_metrics(df: pd.DataFrame, label: str) -> dict:
    total   = len(df)
    wins    = int((df["t2_achieved"] == 1).sum() + (df["t3_achieved"] == 1).sum())
    losses  = int((df["stopped_out"] == 1).sum())
    win_r   = wins / total if total else 0

    rr_series  = df["realised_rr"].dropna()
    profits    = rr_series[rr_series > 0].sum()
    gross_loss = rr_series[rr_series < 0].abs().sum()
    pf         = profits / gross_loss if gross_loss else float("inf")

    avg_rr     = float(rr_series.mean()) if len(rr_series) else 0.0
    expect     = win_r * avg_rr - (1 - win_r) * abs(avg_rr)

    # Drawdown on cumulative RR equity curve
    cum = rr_series.cumsum()
    roll_max = cum.cummax()
    dd_series = (cum - roll_max)
    max_dd = float(dd_series.min()) if len(dd_series) else 0.0

    # Sharpe / Sortino (RR-unit returns)
    avg_hold = float(df["days_to_target"].dropna().mean()) if "days_to_target" in df else 0
    std  = float(rr_series.std()) if len(rr_series) > 1 else 1e-9
    semi = float(rr_series[rr_series < 0].std()) if (rr_series < 0).any() else 1e-9
    sharpe  = (avg_rr - RISK_FREE_RATE / 252) / std  if std  else 0
    sortino = (avg_rr - RISK_FREE_RATE / 252) / semi if semi else 0

    return {
        "period":        label,
        "total_signals": total,
        "wins":          wins,
        "losses":        losses,
        "win_rate":      round(win_r * 100, 1),
        "avg_rr":        round(avg_rr, 3),
        "profit_factor": round(pf, 2),
        "expectancy":    round(expect, 3),
        "max_drawdown":  round(max_dd, 3),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "avg_hold_days": round(avg_hold, 1),
    }


# ─── Traditional Backtest ─────────────────────────────────────────────────────

def backtest_symbol(symbol: str) -> Optional[dict]:
    """
    Run a single-symbol Darvas backtest.
    Simulates entries at box bottom (close to box low + 5%)
    and exits at Target 2 or stop loss.
    Returns a metrics dict or None if no trades found.
    """
    daily = load_daily(symbol)
    if daily is None or len(daily) < 300:
        return None

    boxes = detect_darvas_boxes(symbol, daily)
    if not boxes:
        return None

    close = daily["Close"]
    high  = daily["High"]
    low   = daily["Low"]

    rr_list   = []
    hold_days = []

    for box in boxes:
        # Find bar when price was in entry zone (bottom 30% of box)
        entry_high = box.box_low + (box.box_high - box.box_low) * 0.30
        atr_s  = calc_atr(high, low, close, 14)
        atr_val = float(atr_s[atr_s.index.date <= box.start_date].iloc[-1]) if len(atr_s) > 0 else 0
        stop   = box.box_low - 1.5 * atr_val
        target = box.box_high + (box.box_high - box.box_low)

        # Find entry bar
        try:
            mask = (close.index.date >= box.start_date) & \
                   (close.values <= entry_high) & (close.values >= box.box_low)
            entry_bars = close[mask]
            if entry_bars.empty:
                continue
            entry_idx = entry_bars.index[0]
            entry_px  = float(entry_bars.iloc[0])
        except Exception:
            continue

        # Simulate forward
        future = daily[daily.index > entry_idx]
        for _, bar in future.iterrows():
            if bar["Low"] <= stop:
                rr = (stop - entry_px) / (entry_px - stop)
                rr_list.append(-1.0)
                hold_days.append(len(future.loc[:bar.name]))
                break
            if bar["High"] >= target:
                rr = (target - entry_px) / (entry_px - stop)
                rr_list.append(round(rr, 2))
                hold_days.append(len(future.loc[:bar.name]))
                break

    if not rr_list:
        return None

    wins = [r for r in rr_list if r > 0]
    losses = [r for r in rr_list if r <= 0]
    win_r = len(wins) / len(rr_list)
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss else float("inf")

    # Equity curve for CAGR & drawdown
    equity = pd.Series([1.0] + rr_list).add(1).cumprod()
    years  = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr   = (equity.iloc[-1] ** (1 / max(years, 0.1)) - 1) * 100
    roll_max = equity.cummax()
    max_dd   = float(((equity - roll_max) / roll_max).min() * 100)

    rr_s   = pd.Series(rr_list)
    std    = rr_s.std()
    sharpe = (rr_s.mean() - RISK_FREE_RATE / 252) / std if std else 0

    return {
        "symbol":       symbol,
        "trades":       len(rr_list),
        "win_rate":     round(win_r * 100, 1),
        "profit_factor":round(pf, 2),
        "cagr_pct":     round(cagr, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe":       round(sharpe, 3),
        "avg_hold":     round(sum(hold_days) / len(hold_days), 1) if hold_days else 0,
    }
