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
import uuid
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from database import (
    get_all_signals_df, save_backtest_run,
    save_backtest_symbol_summary, save_backtest_trade_log,
)
from darvas import detect_darvas_boxes
from downloader import load_daily
from indicators import (
    atr as calc_atr, rsi as calc_rsi, adx as calc_adx, ema,
    volume_ratio as calc_volume_ratio, higher_highs_higher_lows,
    rs_rating as calc_rs, sepa_score as calc_sepa, trend_label,
)
from scanner import _compute_score  # reuse the LIVE scoring formula exactly
from config import (
    RSI_PERIOD, ADX_MIN, ATR_PERIOD, ATR_STOP_MULTIPLIER, EMA_TREND,
    ENTRY_ZONE_PCT, VOLUME_MA_FAST, VOLUME_MA_PERIOD, RSI_MIN, RSI_MAX,
    SCORE_THRESHOLDS, RS_WEIGHTS,
)
from logger_utils import get_logger

log = get_logger("performance")

RISK_FREE_RATE = 0.065  # RBI repo rate proxy
MIN_BARS_FOR_SCORING = 260   # need ~1yr+ history before box start to score a trade


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
    # Cap at 999.0 instead of literal infinity — same reasoning as the
    # backward backtest: inf doesn't round-trip safely through Excel/SQLite
    # and a zero-loss sample is still "very good," not literally infinite.
    pf = min(profits / gross_loss, 999.0) if gross_loss else 999.0

    avg_rr     = float(rr_series.mean()) if len(rr_series) else 0.0
    expect     = win_r * avg_rr - (1 - win_r) * abs(avg_rr)

    # Drawdown on cumulative RR equity curve
    cum = rr_series.cumsum()
    roll_max = cum.cummax()
    dd_series = (cum - roll_max)
    max_dd = float(dd_series.min()) if len(dd_series) else 0.0

    # Sharpe / Sortino (RR-unit returns).
    # NOTE: with fewer than 2 trades, std() is undefined — we report 0.0
    # rather than dividing by a tiny sentinel (1e-9), which previously
    # produced nonsensical billion-scale "Sharpe ratios" for single-trade
    # samples. 0.0 honestly signals "not enough data to compute this yet."
    avg_hold = float(df["days_to_target"].dropna().mean()) if "days_to_target" in df else 0
    std_raw  = rr_series.std() if len(rr_series) > 1 else None
    neg_rets = rr_series[rr_series < 0]
    semi_raw = neg_rets.std() if len(neg_rets) > 1 else None

    excess_return = avg_rr - RISK_FREE_RATE / 252
    sharpe  = (excess_return / std_raw)  if std_raw  and std_raw  > 1e-6 else 0.0
    sortino = (excess_return / semi_raw) if semi_raw and semi_raw > 1e-6 else 0.0

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

def backtest_symbol(symbol: str) -> tuple[Optional[dict], list[dict]]:
    """
    Run a single-symbol Darvas backtest across its ENTIRE available
    history. Simulates entries when price is in the bottom
    ENTRY_ZONE_PCT of an active box, exits at Target 2 or stop loss.

    Returns (summary_dict, trade_log_list):
      summary_dict — aggregate stats for this symbol (None if no trades)
      trade_log_list — one dict per individual trade, including the
                        composite score / RS rating / RSI / ADX the
                        signal had AT THE MOMENT OF ENTRY. This is what
                        lets the report later answer "do high-score
                        trades actually win more" with real evidence.

    NOTE on look-ahead bias: RS Rating here is computed against the
    Nifty 50 benchmark using only price history available UP TO the
    entry date (no future data leaks in). This mirrors how the live
    scanner computes it, just evaluated at a historical point in time
    instead of "today."
    """
    daily = load_daily(symbol)
    if daily is None or len(daily) < 300:
        return None, []

    benchmark = load_daily("^NSEI")

    boxes = detect_darvas_boxes(symbol, daily)
    if not boxes:
        return None, []

    close  = daily["Close"]
    high   = daily["High"]
    low    = daily["Low"]
    volume = daily["Volume"]

    trade_log: list[dict] = []

    for box in boxes:
        entry_low  = box.box_low
        entry_high = box.box_low + (box.box_high - box.box_low) * ENTRY_ZONE_PCT

        # Find the first bar where price entered the bottom zone of the box
        try:
            mask = (close.index.date >= box.start_date) & \
                   (close.values <= entry_high) & (close.values >= entry_low)
            entry_bars = close[mask]
            if entry_bars.empty:
                continue
            entry_idx = entry_bars.index[0]
            entry_px  = float(entry_bars.iloc[0])
        except Exception:
            continue

        loc = close.index.get_loc(entry_idx)
        if loc < MIN_BARS_FOR_SCORING:
            # Not enough history before this point to compute RS/SEPA
            # reliably — skip rather than report a misleading score.
            continue

        # ── Point-in-time technicals, using ONLY data up to entry_idx ────────
        hist_close  = close.iloc[: loc + 1]
        hist_high   = high.iloc[: loc + 1]
        hist_low    = low.iloc[: loc + 1]
        hist_volume = volume.iloc[: loc + 1]

        try:
            rsi_val = float(calc_rsi(hist_close, RSI_PERIOD).iloc[-1])
            adx_val = float(calc_adx(hist_high, hist_low, hist_close, 14)["ADX"].iloc[-1])
            atr_val = float(calc_atr(hist_high, hist_low, hist_close, ATR_PERIOD).iloc[-1])
            vol_fast = float(hist_volume.iloc[-VOLUME_MA_FAST:].mean())
            vol_slow = float(hist_volume.rolling(VOLUME_MA_PERIOD).mean().iloc[-1])
            vol_rat  = vol_fast / vol_slow if vol_slow > 0 else 0.0

            if benchmark is not None:
                bench_hist = benchmark["Close"].reindex(hist_close.index, method="ffill").dropna()
                rs_val = calc_rs(hist_close, bench_hist, RS_WEIGHTS)
            else:
                rs_val = 50.0

            sepa_val, _ = calc_sepa(hist_close)

            # Weekly/monthly trend at time of entry (resample only the
            # history available up to this point — no future leakage)
            from downloader import resample_weekly, resample_monthly
            hist_df = daily.iloc[: loc + 1]
            w_trend = trend_label(resample_weekly(hist_df)["Close"]) if len(hist_df) > 250 else "neutral"
            m_trend = trend_label(resample_monthly(hist_df)["Close"]) if len(hist_df) > 250 else "neutral"

            score = _compute_score(
                rs=rs_val, w_trend=w_trend, m_trend=m_trend,
                vol_ratio=vol_rat, box=box, adx=adx_val,
                rsi=rsi_val, sepa=sepa_val,
            )
        except Exception as e:
            log.debug("Scoring error for %s at %s: %s", symbol, entry_idx, e)
            continue

        # ── Risk management — same formula as the live scanner ───────────────
        stop    = box.box_low - ATR_STOP_MULTIPLIER * atr_val
        height  = box.box_high - box.box_low
        target1 = box.box_high
        target2 = box.box_high + height

        if entry_px <= stop:
            continue  # degenerate case, skip

        risk_per_share = entry_px - stop

        # ── Simulate forward day-by-day until target/stop/end-of-data ────────
        future = daily[daily.index > entry_idx]
        outcome     = "open_at_end"
        exit_price  = float(future["Close"].iloc[-1]) if len(future) else entry_px
        exit_date   = future.index[-1] if len(future) else entry_idx
        hold_days   = len(future)

        for bar_date, bar in future.iterrows():
            hit_stop   = bar["Low"]  <= stop
            hit_t2     = bar["High"] >= target2
            hit_t1     = bar["High"] >= target1

            # If both stop and a target could have been hit on the SAME
            # bar, we conservatively assume the stop was hit first (you
            # cannot know intraday sequencing from daily OHLC data, and
            # assuming the worse outcome avoids overstating performance).
            if hit_stop:
                outcome    = "stopped_out"
                exit_price = stop
                exit_date  = bar_date
                hold_days  = len(future.loc[:bar_date])
                break
            if hit_t2:
                outcome    = "target2_hit"
                exit_price = target2
                exit_date  = bar_date
                hold_days  = len(future.loc[:bar_date])
                break
            if hit_t1:
                # Target 1 reached but we keep holding for Target 2 in
                # this simulation (matches "Target 1 Achieved" being a
                # checkpoint, not an automatic full exit, in the live
                # signal tracker). Continue scanning forward.
                outcome = "target1_hit_holding"

        rr_realised = round((exit_price - entry_px) / risk_per_share, 3)

        score_band = (
            "elite"       if score >= SCORE_THRESHOLDS["elite"] else
            "very_strong" if score >= SCORE_THRESHOLDS["very_strong"] else
            "strong"      if score >= SCORE_THRESHOLDS["strong"] else
            "watch"
        )

        trade_log.append({
            "symbol":          symbol,
            "entry_date":      entry_idx.date().isoformat(),
            "exit_date":       exit_date.date().isoformat() if hasattr(exit_date, "date") else str(exit_date),
            "entry_price":     round(entry_px, 2),
            "exit_price":      round(exit_price, 2),
            "stop_loss":       round(stop, 2),
            "target1":         round(target1, 2),
            "target2":         round(target2, 2),
            "outcome":         "target1_hit" if outcome == "target1_hit_holding" else outcome,
            "rr_realised":     rr_realised,
            "hold_days":       int(hold_days),
            "composite_score": round(score, 1),
            "rs_rating":       round(rs_val, 1),
            "sepa_score":      round(sepa_val, 1),
            "rsi_at_entry":    round(rsi_val, 1),
            "adx_at_entry":    round(adx_val, 1),
            "box_width_pct":   round(box.width_pct, 1),
            "box_age_bars":    box.age_bars,
            "score_band":      score_band,
        })

    if not trade_log:
        return None, []

    # ── Aggregate summary for this symbol ─────────────────────────────────────
    rr_list = [t["rr_realised"] for t in trade_log]
    wins    = [r for r in rr_list if r > 0]
    losses  = [r for r in rr_list if r <= 0]
    win_r   = len(wins) / len(rr_list)
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss else float("inf")
    pf_capped = min(pf, 999.0) if pf != float("inf") else 999.0

    equity   = pd.Series([1.0] + rr_list).add(1).cumprod()
    years    = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr     = (equity.iloc[-1] ** (1 / max(years, 0.1)) - 1) * 100
    roll_max = equity.cummax()
    max_dd   = float(((equity - roll_max) / roll_max).min() * 100)

    rr_s = pd.Series(rr_list)
    std  = rr_s.std() if len(rr_s) > 1 else None
    sharpe = float((rr_s.mean() - RISK_FREE_RATE / 252) / std) if std and std > 1e-6 else 0.0

    summary = {
        "symbol":       symbol,
        "trades":       len(rr_list),
        "win_rate":     round(win_r * 100, 1),
        "profit_factor":round(pf_capped, 2),
        "cagr_pct":     round(cagr, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe":       round(sharpe, 3),
        "avg_hold":     round(sum(t["hold_days"] for t in trade_log) / len(trade_log), 1),
    }
    return summary, trade_log


# ─── Universe-wide Backtest (Backward Test) ───────────────────────────────────

def backtest_universe(symbols: list[str], notes: str = "") -> dict:
    """
    Run backtest_symbol() across every symbol in *symbols* and persist
    a complete run summary + per-symbol summary + full trade log to
    the database.

    This is the "backward" backtest: it replays the strategy across
    each stock's ENTIRE available history to validate whether the
    Darvas bottom-of-box approach has historically worked, BEFORE
    waiting weeks for forward-validation data to accumulate.

    Returns a summary dict and also writes:
      - backtest_runs            table: one row per run (aggregate stats)
      - backtest_symbol_summary  table: one row per symbol within that run
      - backtest_trade_log       table: one row per INDIVIDUAL TRADE,
                                         including score-at-entry — this
                                         is the ground truth the Excel
                                         report's score-band analysis is
                                         built from.

    A unique run_id is generated each call, so you can re-run this
    after tuning config.py parameters and compare results over time
    without overwriting previous runs.
    """
    run_id = f"bt_{date.today().isoformat()}_{uuid.uuid4().hex[:8]}"
    log.info("Starting universe backtest run %s across %d symbols", run_id, len(symbols))

    per_symbol_results: list[dict] = []
    all_trades: list[dict] = []

    for i, sym in enumerate(symbols, 1):
        if i % 200 == 0:
            log.info("  Backtest progress: %d/%d symbols (%d with trades, %d total trades)",
                     i, len(symbols), len(per_symbol_results), len(all_trades))
        try:
            summary, trades = backtest_symbol(sym)
            if summary:
                per_symbol_results.append(summary)
                all_trades.extend(trades)
        except Exception as e:
            log.debug("Backtest error for %s: %s", sym, e)
            continue

    if not per_symbol_results:
        log.warning("Universe backtest produced no trades across %d symbols", len(symbols))
        summary = {
            "run_id": run_id, "run_date": date.today().isoformat(),
            "symbols_tested": len(symbols), "symbols_with_trades": 0,
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
            "avg_cagr": 0.0, "avg_drawdown": 0.0, "avg_sharpe": 0.0,
            "avg_hold_days": 0.0, "notes": notes,
        }
        save_backtest_run(summary)
        return summary

    df = pd.DataFrame(per_symbol_results)
    total_trades = int(df["trades"].sum())

    # ── Universe-level win rate, profit factor, and expectancy MUST be
    # computed from the POOLED individual trades, not by averaging each
    # symbol's own ratio. Averaging per-symbol profit factors badly
    # distorts the result whenever a symbol with only 1-2 trades happens
    # to have zero losses (it gets the 999.0 cap) — a handful of small
    # samples like that can drag a 5-symbol average up to "799", which
    # falsely implies a near-perfect strategy. Pooling avoids this.
    trades_df = pd.DataFrame(all_trades)
    rr = trades_df["rr_realised"]

    pooled_win_rate = float((rr > 0).mean() * 100) if len(rr) else 0.0
    gross_profit = float(rr[rr > 0].sum())
    gross_loss   = float(rr[rr < 0].abs().sum())
    pooled_profit_factor = min(gross_profit / gross_loss, 999.0) if gross_loss > 0 else 999.0
    pooled_expectancy = float(rr.mean()) if len(rr) else 0.0

    summary = {
        "run_id":              run_id,
        "run_date":            date.today().isoformat(),
        "symbols_tested":      len(symbols),
        "symbols_with_trades": len(per_symbol_results),
        "total_trades":        total_trades,
        "win_rate":            round(pooled_win_rate, 2),
        "profit_factor":       round(pooled_profit_factor, 2),
        "expectancy":          round(pooled_expectancy, 3),
        "avg_cagr":            round(float(df["cagr_pct"].mean()), 2),
        "avg_drawdown":        round(float(df["max_drawdown"].mean()), 2),
        "avg_sharpe":          round(float(df["sharpe"].dropna().mean()) if df["sharpe"].notna().any() else 0.0, 3),
        "avg_hold_days":       round(float(df["avg_hold"].mean()), 1),
        "notes":               notes,
    }

    save_backtest_run(summary)
    save_backtest_symbol_summary(run_id, per_symbol_results)
    save_backtest_trade_log(run_id, all_trades)

    log.info(
        "Backtest run %s complete: %d/%d symbols had trades, "
        "%d total trades, win_rate=%.1f%%, avg_cagr=%.1f%%",
        run_id, len(per_symbol_results), len(symbols),
        total_trades, pooled_win_rate, summary["avg_cagr"],
    )
    return summary
