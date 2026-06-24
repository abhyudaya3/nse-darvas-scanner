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

import gc
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
from downloader import load_daily, resample_weekly, resample_monthly
from indicators import (
    atr as calc_atr, rsi as calc_rsi, adx as calc_adx, ema,
    volume_ratio as calc_volume_ratio, higher_highs_higher_lows,
    rs_rating as calc_rs, sepa_score as calc_sepa, trend_label,
)
from scanner import _compute_score  # reuse the LIVE scoring formula exactly
from cup_handle import detect_cup_and_handle
from config import (
    RSI_PERIOD, ADX_MIN, ATR_PERIOD, ATR_STOP_MULTIPLIER, EMA_TREND,
    ENTRY_ZONE_PCT, VOLUME_MA_FAST, VOLUME_MA_PERIOD, RSI_MIN, RSI_MAX,
    SCORE_THRESHOLDS, RS_WEIGHTS, MIN_SIGNAL_SCORE, RISK_PER_TRADE_PCT,
)
from logger_utils import get_logger

log = get_logger("performance")

RISK_FREE_RATE = 0.065  # RBI repo rate proxy
MIN_BARS_FOR_SCORING = 260   # need ~1yr+ history before box start to score a trade

# How many trading days to advance between each walk-forward detection
# attempt in backtest_symbol(). MUST be smaller than DARVAS_LOOKBACK (252)
# so consecutive scan windows overlap and no box near a window boundary
# is silently skipped. 60 days (~3 months) gives good coverage without
# making the full-universe backtest prohibitively slow — each additional
# walk step is one more detect_darvas_boxes() call per symbol.
DARVAS_WALK_STEP_DAYS = 60


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

    # Drawdown: FIXED 2026-06-20 for consistency with backtest_symbol() and
    # backtest_cup_handle_symbol() — this previously summed raw R-multiples
    # (cumsum), which doesn't have the catastrophic "-100% on any single
    # stop-loss" bug those two had, but DID report drawdown in raw
    # R-units rather than the percentage scale used everywhere else in
    # the system, making it silently inconsistent and harder to compare
    # across reports. Now uses the same compounded-equity, risk-scaled
    # percentage calculation as the backward backtests.
    risk_fraction = RISK_PER_TRADE_PCT / 100.0
    equity = pd.Series([1.0] + [r * risk_fraction for r in rr_series]).add(1).cumprod()
    roll_max = equity.cummax()
    max_dd = float(((equity - roll_max) / roll_max).min() * 100)

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

def backtest_symbol(
    symbol: str,
    benchmark: Optional[pd.DataFrame] = None,
) -> tuple[Optional[dict], list[dict]]:
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

    PERFORMANCE NOTE: RSI/ADX/ATR are EWM-based (backward-looking only),
    so computing them ONCE over the full price history and indexing
    into the result at each box's entry point gives mathematically
    IDENTICAL values to recomputing them from scratch on a truncated
    slice per box — but is vastly cheaper. The previous version
    recomputed every indicator (plus weekly/monthly resampling) from
    scratch for EVERY box in EVERY symbol, which caused excessive CPU
    and memory churn across a 2,374-symbol universe and led to the
    process being silently killed (OOM) partway through a full run
    with no Python-level exception ever logged.

    *benchmark* — pass the already-loaded Nifty 50 DataFrame so it
    isn't re-read from disk for every single symbol (2,374 redundant
    Parquet reads otherwise). If None, this function loads it itself
    (kept for backward compatibility with direct calls/tests).
    """
    daily = load_daily(symbol)
    if daily is None or len(daily) < 300:
        return None, []

    if benchmark is None:
        benchmark = load_daily("^NSEI")

    # ── Walk forward through FULL history to find EVERY historical box ───────
    # FIXED 2026-06-20: detect_darvas_boxes() internally does
    # `scan_start = max(0, n - DARVAS_LOOKBACK)` — it ONLY EVER scans the
    # most recent DARVAS_LOOKBACK (252) bars of whatever DataFrame it's
    # given. The previous version of this function called it ONCE on the
    # entire multi-year `daily` series, which meant it only ever found
    # boxes within the last ~1 calendar year from "today" — regardless
    # of whether the stock had 5+ years of history available. A real
    # backtest run on the full NSE universe confirmed this: 5,170 of
    # 5,175 total trades had entry dates in 2025-2026, with a near-total
    # dead zone from Oct 2023 to May 2025, even though most symbols had
    # years of earlier price history that was simply never being looked
    # at by the box detector. This silently understated the statistical
    # power of the backtest and meant it never validated the strategy
    # against earlier market regimes (corrections, different volatility
    # environments) at all.
    #
    # The fix mirrors backtest_cup_handle_symbol()'s walk-forward
    # approach: repeatedly call detect_darvas_boxes() on a truncated,
    # progressively-advancing slice of `daily`, so every ~1-year window
    # across the stock's ENTIRE history gets scanned, not just the most
    # recent one. Consecutive windows are spaced by DARVAS_WALK_STEP_DAYS
    # (smaller than DARVAS_LOOKBACK) so they overlap and no box near a
    # window boundary gets missed; duplicate detections of the SAME
    # physical box (re-found in multiple overlapping windows, often with
    # a progressively LATER end_date as more consolidation bars
    # accumulate in later windows) are de-duplicated by start_date alone
    # before being turned into trades — keying on (start_date, end_date)
    # was tried first and found to be WRONG: the same box detected twice
    # at different points in its life (still-forming vs. fully-formed)
    # has the SAME start_date but a DIFFERENT end_date each time, so that
    # key let the same physical box slip through as 2 separate "boxes",
    # double-counting trades. start_date alone reliably identifies a
    # single physical box (darvas.py's detection loop explicitly skips
    # past each confirmed box before searching for the next one, so two
    # genuinely different boxes for the same symbol won't share a
    # start_date). When duplicates ARE found, the version with the
    # LATEST end_date is kept — that's the most mature/complete
    # observation of the box, found once more bars had accumulated.
    n_total = len(daily)
    boxes_by_key: dict = {}

    window_start = 300   # need at least this many bars before any scan attempt
    while window_start <= n_total:
        slice_daily = daily.iloc[:window_start]
        try:
            window_boxes = detect_darvas_boxes(symbol, slice_daily)
        except Exception as e:
            log.debug("Darvas walk-forward detection error %s at bar %d: %s",
                      symbol, window_start, e)
            window_boxes = []

        for b in window_boxes:
            key = b.start_date
            if key not in boxes_by_key or b.end_date > boxes_by_key[key].end_date:
                boxes_by_key[key] = b

        window_start += DARVAS_WALK_STEP_DAYS

    # Always also scan the full series once at the end, in case the loop's
    # step size didn't land exactly on the final bar.
    try:
        for b in detect_darvas_boxes(symbol, daily):
            key = b.start_date
            if key not in boxes_by_key or b.end_date > boxes_by_key[key].end_date:
                boxes_by_key[key] = b
    except Exception:
        pass

    boxes = sorted(boxes_by_key.values(), key=lambda b: b.start_date)
    if not boxes:
        return None, []

    close  = daily["Close"]
    high   = daily["High"]
    low    = daily["Low"]
    volume = daily["Volume"]

    # ── Compute every full-history indicator ONCE for this symbol ────────────
    # (not once per box — see PERFORMANCE NOTE above)
    full_rsi      = calc_rsi(close, RSI_PERIOD)
    full_adx      = calc_adx(high, low, close, 14)["ADX"]
    full_atr      = calc_atr(high, low, close, ATR_PERIOD)
    full_vol_fast = volume.rolling(VOLUME_MA_FAST).mean()
    full_vol_slow = volume.rolling(VOLUME_MA_PERIOD).mean()
    full_ema50    = ema(close, 50)
    full_ema150   = ema(close, 150)
    full_ema200   = ema(close, 200)

    # Benchmark aligned to this symbol's index, once (not once per box)
    if benchmark is not None:
        bench_aligned = benchmark["Close"].reindex(close.index, method="ffill")
    else:
        bench_aligned = None

    # Weekly/monthly OHLCV resampled ONCE over the full history. Since
    # resample() only groups existing bars into periods (it doesn't
    # peek forward), w_full.loc[:entry_idx] below is safe — it can only
    # contain weeks/months that ended on or before the entry date.
    from downloader import resample_weekly, resample_monthly
    w_full = resample_weekly(daily)
    m_full = resample_monthly(daily)

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

        try:
            rsi_val = float(full_rsi.iloc[loc])
            adx_val = float(full_adx.iloc[loc])
            atr_val = float(full_atr.iloc[loc])
            vf = float(full_vol_fast.iloc[loc])
            vs = float(full_vol_slow.iloc[loc])
            vol_rat = vf / vs if vs > 0 else 0.0

            hist_close = close.iloc[: loc + 1]   # still needed for rs_rating/sepa_score,
                                                  # which internally do .iloc[-1] / .iloc[-252:]
            if bench_aligned is not None:
                bench_hist = bench_aligned.iloc[: loc + 1].dropna()
                rs_val = calc_rs(hist_close, bench_hist, RS_WEIGHTS)
            else:
                rs_val = 50.0

            sepa_val, _ = calc_sepa(hist_close)

            # Slice the PRE-COMPUTED weekly/monthly series up to entry_idx
            # instead of re-resampling the daily data from scratch.
            w_slice = w_full.loc[: entry_idx]
            m_slice = m_full.loc[: entry_idx]
            w_trend = trend_label(w_slice["Close"]) if len(w_slice) > 30 else "neutral"
            m_trend = trend_label(m_slice["Close"]) if len(m_slice) > 10 else "neutral"

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
            "watch"       if score >= SCORE_THRESHOLDS["watch"] else
            "below_watch"   # would NEVER reach the live scanner (score < 60),
                             # kept here only so the backtest's full historical
                             # record stays complete for analysis — exclude
                             # this band when judging live-strategy performance
        )

        trade_log.append({
            "symbol":          symbol,
            "pattern_type":    "darvas_box",
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
            "box_start_date":  box.start_date.isoformat(),
            "box_end_date":    box.end_date.isoformat(),
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

    # ── Equity curve: FIXED 2026-06-20 ──────────────────────────────────────
    # Each trade's rr_realised is an R-MULTIPLE (e.g. -1.0 = lost exactly
    # what was risked, +2.0 = gained 2x what was risked) — it is NOT a
    # direct percentage change in total account equity. The previous
    # formula `(1 + rr_realised)` treated them as the same thing, so any
    # single stopped-out trade (rr_realised == -1.0 exactly, by
    # construction) produced an equity multiplier of EXACTLY ZERO,
    # wiping the entire compounded equity curve to 0 regardless of how
    # many winning trades came before or after it. This was corrupting
    # max_drawdown (showing -100% for ~74% of all symbols in one real
    # backtest run, including symbols with a 66%+ win rate and a profit
    # factor above 3.0 — a fundamentally impossible combination) and
    # inflating/deflating CAGR by the same broken mechanism.
    #
    # The fix: scale each trade's R-multiple by the FRACTION of account
    # equity actually put at risk per trade (RISK_PER_TRADE_PCT/100,
    # i.e. the same position-sizing assumption already used everywhere
    # else in this codebase — see config.py). A -1.0R loss at 1% risk
    # per trade now correctly produces roughly a 1% equity decline, not
    # a 100% wipeout.
    risk_fraction = RISK_PER_TRADE_PCT / 100.0
    equity   = pd.Series([1.0] + [r * risk_fraction for r in rr_list]).add(1).cumprod()
    years    = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr     = (equity.iloc[-1] ** (1 / max(years, 0.1)) - 1) * 100
    roll_max = equity.cummax()
    max_dd   = float(((equity - roll_max) / roll_max).min() * 100)

    rr_s = pd.Series(rr_list)
    std  = rr_s.std() if len(rr_s) > 1 else None
    sharpe = float((rr_s.mean() - RISK_FREE_RATE / 252) / std) if std and std > 1e-6 else 0.0

    summary = {
        "symbol":       symbol,
        "pattern_type": "darvas_box",
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

    # Load the benchmark ONCE for the whole run, not once per symbol.
    # Previously backtest_symbol() called load_daily("^NSEI") internally
    # on every invocation — across 2,374 symbols that's 2,374 redundant
    # Parquet reads of the same unchanging file, adding meaningful I/O
    # and memory churn for no benefit.
    benchmark = load_daily("^NSEI")
    if benchmark is None:
        log.warning("No ^NSEI benchmark data found — RS Rating will default to 50.0 for all trades")

    per_symbol_results: list[dict] = []
    all_trades: list[dict] = []
    pending_symbol_results: list[dict] = []   # not-yet-persisted, flushed periodically
    pending_trades: list[dict] = []

    FLUSH_EVERY = 300   # symbols between incremental DB writes + gc

    for i, sym in enumerate(symbols, 1):
        if i % 200 == 0:
            log.info("  Backtest progress: %d/%d symbols (%d with trades, %d total trades)",
                     i, len(symbols), len(per_symbol_results), len(all_trades))
        try:
            summary, trades = backtest_symbol(sym, benchmark=benchmark)
            if summary:
                per_symbol_results.append(summary)
                all_trades.extend(trades)
                pending_symbol_results.append(summary)
                pending_trades.extend(trades)
        except Exception as e:
            log.debug("Backtest error for %s: %s", sym, e)
            continue

        # ── Periodic incremental flush ────────────────────────────────────────
        # Persist what we have so far and release memory. This means if the
        # process IS killed later (OOM, timeout, etc.) the database already
        # has real partial results instead of losing the entire run — and it
        # keeps peak memory bounded regardless of universe size.
        if i % FLUSH_EVERY == 0 or i == len(symbols):
            if pending_symbol_results or pending_trades:
                try:
                    save_backtest_symbol_summary(run_id, pending_symbol_results)
                    save_backtest_trade_log(run_id, pending_trades)
                    log.debug("Flushed %d symbol summaries / %d trades to DB at symbol %d/%d",
                              len(pending_symbol_results), len(pending_trades), i, len(symbols))
                except Exception as e:
                    log.error("Incremental flush failed at symbol %d: %s", i, e)
                pending_symbol_results = []
                pending_trades = []
            gc.collect()

    if not per_symbol_results:
        log.warning("Universe backtest produced no trades across %d symbols", len(symbols))
        summary = {
            "run_id": run_id, "run_date": date.today().isoformat(),
            "pattern_type": "darvas_box",
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
        "pattern_type":        "darvas_box",
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

    # NOTE: per-symbol summaries and individual trades are already persisted
    # incrementally during the loop above (including the final batch, via
    # the `i == len(symbols)` flush condition) — saving them again here
    # would create duplicate rows in backtest_trade_log, which has no
    # unique constraint. Only the run-level summary needs saving now.
    save_backtest_run(summary)

    log.info(
        "Backtest run %s complete: %d/%d symbols had trades, "
        "%d total trades, win_rate=%.1f%%, avg_cagr=%.1f%%",
        run_id, len(per_symbol_results), len(symbols),
        total_trades, pooled_win_rate, summary["avg_cagr"],
    )
    return summary


# ─── Cup and Handle Backtest (Backward Test) ──────────────────────────────────

CH_WALK_STEP_WEEKS = 1   # advance ONE week per detection attempt.
# This MUST be 1, not a larger stride. detect_cup_and_handle()'s
# breakout/volume-surge check only looks at the LAST bar of whatever
# slice it's given — with a stride > 1, many real breakout weeks would
# fall between sampled checkpoints and be silently skipped, understating
# how many genuine O'Neil-style breakouts actually occurred. The extra
# computation cost is the right tradeoff for correctness here.
CH_MIN_BARS_FOR_SCORING = 260   # need ~1yr+ daily history before a cup can be scored


def backtest_cup_handle_symbol(
    symbol: str,
    benchmark: Optional[pd.DataFrame] = None,
) -> tuple[Optional[dict], list[dict]]:
    """
    Walk forward through *symbol*'s ENTIRE daily history looking for
    every historical Cup and Handle occurrence (not just the most
    recent one, which is all the live scanner checks).

    Because detect_cup_and_handle() always evaluates the pattern
    relative to the END of whatever daily/weekly/monthly slice it's
    given, we simulate "looking back from week W" by repeatedly slicing
    daily data up to progressively later points in time and re-running
    detection — the same point-in-time approach used by
    backtest_symbol() for the Darvas Box, adapted to weekly granularity
    since O'Neil's own cup analysis is inherently a weekly-bar pattern.

    Returns (summary_dict, trade_log_list) — same contract as
    backtest_symbol(), so both patterns plug into backtest_universe()
    identically. Every trade dict includes pattern_type='cup_handle'
    plus all cup/handle-specific fields (dates, depth, duration) so the
    report can fully separate Darvas vs Cup & Handle results.
    """
    daily = load_daily(symbol)
    if daily is None or len(daily) < CH_MIN_BARS_FOR_SCORING:
        return None, []

    weekly_full = resample_weekly(daily)
    if len(weekly_full) < 60:
        return None, []

    trade_log: list[dict] = []
    seen_pivots: set[tuple] = set()   # (cup_start_date, handle_end_date) dedup key

    n_weeks = len(weekly_full)
    # Start far enough in to have CH_MIN_BARS_FOR_SCORING of daily history,
    # walk forward in CH_WALK_STEP_WEEKS increments.
    start_week_idx = max(60, int(CH_MIN_BARS_FOR_SCORING / 5))

    for week_idx in range(start_week_idx, n_weeks, CH_WALK_STEP_WEEKS):
        as_of_date = weekly_full.index[week_idx]
        daily_slice = daily[daily.index <= as_of_date]
        if len(daily_slice) < CH_MIN_BARS_FOR_SCORING:
            continue

        weekly_slice = weekly_full.loc[:as_of_date]
        monthly_slice = resample_monthly(daily_slice)

        try:
            pattern = detect_cup_and_handle(symbol, daily_slice, weekly_slice, monthly_slice)
        except Exception as e:
            log.debug("C&H detection error %s at %s: %s", symbol, as_of_date, e)
            continue

        # IMPORTANT: O'Neil's rule is "buy the breakout", not "buy any
        # valid-looking base". pattern.is_valid only checks the cup/handle
        # SHAPE criteria (depth, duration, position) — it does NOT require
        # a genuine volume-confirmed breakout above the pivot. Without also
        # requiring pattern.is_breaking_out, this backtest would have been
        # simulating entries on patterns that never actually broke out with
        # real institutional volume, which is not how this strategy is
        # meant to be traded and would misrepresent its real performance.
        if pattern is None or not pattern.is_valid or not pattern.is_breaking_out:
            continue

        # Dedup: the same physical cup will be re-detected on every walk
        # step while its handle is still forming. Only record it once,
        # keyed by its (cup_start, handle_end) — the first time we see a
        # given handle_end means the handle has just completed enough to
        # validate; later walk steps re-finding the SAME handle_end are
        # the same occurrence, not a new one.
        dedup_key = (pattern.cup_start_date, pattern.handle_end_date)
        if dedup_key in seen_pivots:
            continue
        seen_pivots.add(dedup_key)

        # ── Point-in-time entry: simulate buying at the confirmed breakout
        #    bar's close (the actual market price the day the volume-backed
        #    breakout was confirmed), NOT the theoretical pivot price ────────
        entry_loc = daily.index.get_loc(daily.index[daily.index <= as_of_date][-1])
        if entry_loc + 1 >= len(daily):
            continue   # no bar after this to even attempt entry on

        entry_date = daily.index[entry_loc]
        entry_price = float(daily["Close"].iloc[entry_loc])  # actual breakout-day close, not pivot
        if entry_price < pattern.buy_zone_low or entry_price > pattern.buy_zone_high:
            entry_price = pattern.pivot_price  # fall back to exact pivot if outside buy zone

        # ── Point-in-time technicals for scoring (no look-ahead — only
        #    data up to as_of_date is used) ──────────────────────────────────
        close_hist = daily_slice["Close"]
        high_hist  = daily_slice["High"]
        low_hist   = daily_slice["Low"]

        try:
            atr_val = float(calc_atr(high_hist, low_hist, close_hist, ATR_PERIOD).iloc[-1])
            rsi_val = float(calc_rsi(close_hist, RSI_PERIOD).iloc[-1])
            adx_val = float(calc_adx(high_hist, low_hist, close_hist, 14)["ADX"].iloc[-1])

            if benchmark is not None:
                bench_hist = benchmark["Close"].reindex(close_hist.index, method="ffill").dropna()
                rs_val = calc_rs(close_hist, bench_hist, RS_WEIGHTS)
            else:
                rs_val = 50.0
        except Exception as e:
            log.debug("C&H scoring error %s at %s: %s", symbol, as_of_date, e)
            continue

        # Quality score already computed inside detect_cup_and_handle()
        # (pattern.quality_score) using the O'Neil-weighted formula —
        # reuse it directly rather than recomputing.
        quality_score = pattern.quality_score

        score_band = (
            "elite"       if quality_score >= SCORE_THRESHOLDS["elite"] else
            "very_strong" if quality_score >= SCORE_THRESHOLDS["very_strong"] else
            "strong"      if quality_score >= SCORE_THRESHOLDS["strong"] else
            "watch"       if quality_score >= SCORE_THRESHOLDS["watch"] else
            "below_watch"
        )

        # ── Risk management: stop below handle low, targets from cup height ──
        stop = pattern.handle_low - ATR_STOP_MULTIPLIER * atr_val
        cup_height = pattern.cup_high - pattern.cup_low
        target1 = pattern.cup_high
        target2 = pattern.cup_high + cup_height

        if entry_price <= stop:
            continue
        risk_per_share = entry_price - stop

        # ── Simulate forward day-by-day until target/stop/end-of-data ────────
        future = daily[daily.index > entry_date]
        outcome    = "open_at_end"
        exit_price = float(future["Close"].iloc[-1]) if len(future) else entry_price
        exit_date  = future.index[-1] if len(future) else entry_date
        hold_days  = len(future)

        for bar_date, bar in future.iterrows():
            hit_stop = bar["Low"]  <= stop
            hit_t2   = bar["High"] >= target2
            hit_t1   = bar["High"] >= target1

            if hit_stop:
                outcome, exit_price, exit_date = "stopped_out", stop, bar_date
                hold_days = len(future.loc[:bar_date])
                break
            if hit_t2:
                outcome, exit_price, exit_date = "target2_hit", target2, bar_date
                hold_days = len(future.loc[:bar_date])
                break
            if hit_t1:
                outcome = "target1_hit_holding"

        rr_realised = round((exit_price - entry_price) / risk_per_share, 3)

        trade_log.append({
            "symbol":               symbol,
            "pattern_type":         "cup_handle",
            "entry_date":           entry_date.date().isoformat(),
            "exit_date":            exit_date.date().isoformat() if hasattr(exit_date, "date") else str(exit_date),
            "entry_price":          round(entry_price, 2),
            "exit_price":           round(exit_price, 2),
            "stop_loss":            round(stop, 2),
            "target1":              round(target1, 2),
            "target2":              round(target2, 2),
            "outcome":              "target1_hit" if outcome == "target1_hit_holding" else outcome,
            "rr_realised":          rr_realised,
            "hold_days":            int(hold_days),
            "composite_score":      round(quality_score, 1),   # unified column name across both patterns
            "rs_rating":            round(rs_val, 1),
            "sepa_score":           0.0,   # SEPA not computed for C&H in this version; see notes
            "rsi_at_entry":         round(rsi_val, 1),
            "adx_at_entry":         round(adx_val, 1),
            "score_band":           score_band,
            "cup_depth_pct":        pattern.cup_depth_pct,
            "cup_duration_weeks":   pattern.cup_duration_weeks,
            "cup_start_date":       pattern.cup_start_date.isoformat(),
            "cup_bottom_date":      pattern.cup_bottom_date.isoformat(),
            "cup_end_date":         pattern.cup_end_date.isoformat(),
            "handle_depth_pct":     pattern.handle_depth_pct,
            "handle_duration_weeks": pattern.handle_duration_weeks,
            "handle_start_date":    pattern.handle_start_date.isoformat(),
            "handle_end_date":      pattern.handle_end_date.isoformat(),
            "prior_uptrend_pct":    pattern.prior_uptrend_pct,
            "breakout_volume_ratio": round(pattern.breakout_volume_ratio, 2),
            "pattern_quality":      quality_score,
            # ADDED 2026-06-21: these boolean quality checks were computed
            # during detection and used in scoring, but were never actually
            # persisted to the trade log — meaning the 2026-06-20 backtest
            # run had no way to validate whether they correlate with real
            # outcomes at all. Adding them now so the NEXT run can.
            "cup_shape_ok":         int(pattern.cup_shape_ok),
            "cup_volume_dryup":     int(pattern.cup_volume_dryup),
            "handle_in_upper_zone": int(pattern.handle_in_upper_zone),
            "handle_volume_dryup":  int(pattern.handle_volume_dryup),
        })

    if not trade_log:
        return None, []

    rr_list = [t["rr_realised"] for t in trade_log]
    wins    = [r for r in rr_list if r > 0]
    losses  = [r for r in rr_list if r <= 0]
    win_r   = len(wins) / len(rr_list)
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss else float("inf")
    pf_capped = min(pf, 999.0) if pf != float("inf") else 999.0

    # Equity curve fix applied here too — see backtest_symbol() above for
    # the full explanation of why R-multiples must be scaled by
    # RISK_PER_TRADE_PCT before compounding, not used as direct multipliers.
    risk_fraction = RISK_PER_TRADE_PCT / 100.0
    equity   = pd.Series([1.0] + [r * risk_fraction for r in rr_list]).add(1).cumprod()
    years    = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr     = (equity.iloc[-1] ** (1 / max(years, 0.1)) - 1) * 100
    roll_max = equity.cummax()
    max_dd   = float(((equity - roll_max) / roll_max).min() * 100)

    rr_s = pd.Series(rr_list)
    std  = rr_s.std() if len(rr_s) > 1 else None
    sharpe = float((rr_s.mean() - RISK_FREE_RATE / 252) / std) if std and std > 1e-6 else 0.0

    summary = {
        "symbol":        symbol,
        "pattern_type":  "cup_handle",
        "trades":        len(rr_list),
        "win_rate":      round(win_r * 100, 1),
        "profit_factor": round(pf_capped, 2),
        "cagr_pct":      round(cagr, 2),
        "max_drawdown":  round(max_dd, 2),
        "sharpe":        round(sharpe, 3),
        "avg_hold":      round(sum(t["hold_days"] for t in trade_log) / len(trade_log), 1),
    }
    return summary, trade_log


def backtest_universe_cup_handle(symbols: list[str], notes: str = "") -> dict:
    """
    Run backtest_cup_handle_symbol() across every symbol in *symbols*
    and persist a complete run summary + per-symbol summary + full
    trade log to the database — the Cup and Handle equivalent of
    backtest_universe() (which only ever covered the Darvas Box).

    Uses the SAME incremental-flush + gc.collect() pattern as the
    Darvas universe backtest to stay memory-safe across the full
    2,000+ symbol NSE universe.
    """
    run_id = f"bt_ch_{date.today().isoformat()}_{uuid.uuid4().hex[:8]}"
    log.info("Starting Cup & Handle universe backtest run %s across %d symbols",
             run_id, len(symbols))

    benchmark = load_daily("^NSEI")
    if benchmark is None:
        log.warning("No ^NSEI benchmark — RS Rating will default to 50.0 for all C&H trades")

    per_symbol_results: list[dict] = []
    all_trades: list[dict] = []
    pending_symbol_results: list[dict] = []
    pending_trades: list[dict] = []

    FLUSH_EVERY = 300

    for i, sym in enumerate(symbols, 1):
        if i % 200 == 0:
            log.info("  C&H Backtest progress: %d/%d symbols (%d with trades, %d total trades)",
                     i, len(symbols), len(per_symbol_results), len(all_trades))
        try:
            summary, trades = backtest_cup_handle_symbol(sym, benchmark=benchmark)
            if summary:
                per_symbol_results.append(summary)
                all_trades.extend(trades)
                pending_symbol_results.append(summary)
                pending_trades.extend(trades)
        except Exception as e:
            log.debug("C&H backtest error for %s: %s", sym, e)
            continue

        if i % FLUSH_EVERY == 0 or i == len(symbols):
            if pending_symbol_results or pending_trades:
                try:
                    save_backtest_symbol_summary(run_id, pending_symbol_results)
                    save_backtest_trade_log(run_id, pending_trades)
                    log.debug("Flushed %d C&H symbol summaries / %d trades at symbol %d/%d",
                              len(pending_symbol_results), len(pending_trades), i, len(symbols))
                except Exception as e:
                    log.error("C&H incremental flush failed at symbol %d: %s", i, e)
                pending_symbol_results = []
                pending_trades = []
            gc.collect()

    if not per_symbol_results:
        log.warning("Cup & Handle universe backtest produced no trades across %d symbols", len(symbols))
        summary = {
            "run_id": run_id, "run_date": date.today().isoformat(),
            "pattern_type": "cup_handle",
            "symbols_tested": len(symbols), "symbols_with_trades": 0,
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
            "avg_cagr": 0.0, "avg_drawdown": 0.0, "avg_sharpe": 0.0,
            "avg_hold_days": 0.0, "notes": notes,
        }
        save_backtest_run(summary)
        return summary

    df = pd.DataFrame(per_symbol_results)
    total_trades = int(df["trades"].sum())

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
        "pattern_type":        "cup_handle",
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

    log.info(
        "C&H Backtest run %s complete: %d/%d symbols had trades, "
        "%d total trades, win_rate=%.1f%%, avg_cagr=%.1f%%",
        run_id, len(per_symbol_results), len(symbols),
        total_trades, pooled_win_rate, summary["avg_cagr"],
    )
    return summary
