"""
NSE Darvas Box Scanner - Cup and Handle Signal Scanner
=======================================================
Runs the Cup and Handle detector across the full NSE universe and
generates ranked, actionable signals with:
  • RS Rating (O'Neil cross-sectional percentile rank)
  • SEPA score (Minervini)
  • ATR-based position sizing and stop loss
  • Multi-timeframe trend alignment (daily / weekly / monthly)
  • O'Neil-weighted quality scoring
  • Near-pivot entry zone confirmation

Output is a CupHandleSignal dataclass (parallel structure to Signal in
scanner.py) that feeds into the Excel report and Telegram notifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ACCOUNT_SIZE, ATR_PERIOD, ATR_STOP_MULTIPLIER, CUPHANDLE_BUY_ZONE_PCT,
    EMA_TREND, MIN_AVG_VOLUME, MIN_SIGNAL_SCORE, RISK_PER_TRADE_PCT,
    RS_WEIGHTS, SCORE_THRESHOLDS,
)
from cup_handle import CupHandlePattern, detect_cup_and_handle
from downloader import load_daily, resample_weekly, resample_monthly
from indicators import (
    atr as calc_atr, ema, rs_rating as calc_rs,
    sepa_score as calc_sepa, trend_label,
)
from logger_utils import get_logger

log = get_logger("scanner")

# Minimum quality score for the C&H pattern itself to be emitted
# (separate from the Darvas MIN_SIGNAL_SCORE — C&H has its own scale)
CH_MIN_QUALITY_SCORE = 60.0


@dataclass
class CupHandleSignal:
    # Identity
    symbol:             str
    sector:             str
    scan_date:          date

    # Pattern geometry (from CupHandlePattern)
    cup_high:           float
    cup_low:            float
    cup_depth_pct:      float
    cup_duration_weeks: int
    cup_shape_ok:       bool
    cup_volume_dryup:   bool
    cup_start_date:     date     # when the cup's left-side high formed
    cup_bottom_date:    date     # cup's lowest point
    cup_end_date:       date     # right side rounds back near old high; handle begins here
    handle_high:        float
    handle_low:         float
    handle_depth_pct:   float
    handle_duration_weeks: int
    handle_in_upper_zone: bool
    handle_volume_dryup: bool
    handle_start_date:  date
    handle_end_date:    date
    prior_uptrend_pct:  float

    # Entry / risk management
    current_price:      float
    pivot_price:        float
    buy_zone_low:       float
    buy_zone_high:      float
    stop_loss:          float          # ATR-based below handle low
    target1:            float          # Cup's prior high
    target2:            float          # Prior high + cup height
    target3:            float          # Prior high + 2× cup height
    atr:                float
    risk_per_share:     float
    position_size:      int
    capital_required:   float
    risk_amount:        float
    rr_ratio:           float

    # Technical context
    rs_rating:          float
    sepa_score:         float
    sepa_checks:        dict = field(default_factory=dict)
    weekly_trend:       str = "neutral"
    monthly_trend:      str = "neutral"
    breakout_vol_ratio: float = 0.0
    is_breaking_out:    bool = False

    # Scoring
    pattern_quality:    float = 0.0   # 0-100 O'Neil-weighted quality score
    signal_id:          str = ""
    status:             str = "Watching"   # Watching / Near Pivot / Breaking Out

    def __post_init__(self):
        self.signal_id = f"CH_{self.symbol}_{self.scan_date.isoformat()}"
        self.status = (
            "Breaking Out" if self.is_breaking_out else
            "Near Pivot"   if self.current_price >= self.pivot_price * 0.97 else
            "Watching"
        )


def scan_cup_handle(
    symbol:    str,
    daily:     pd.DataFrame,
    benchmark: pd.DataFrame,
    sector:    str = "Unknown",
    rs_override: Optional[float] = None,
) -> Optional[CupHandleSignal]:
    """
    Full Cup and Handle scan for one symbol.
    Returns a CupHandleSignal if a valid (or near-valid) pattern is
    found within the O'Neil quality threshold, else None.

    *rs_override* — if provided, uses this RS Rating instead of the
    single-stock-vs-benchmark sigmoid estimate computed internally.
    FIXED 2026-06-20: main.py's live scan loop computes a proper
    cross-sectional RS Rating (true O'Neil 1-99 percentile rank across
    the WHOLE scanned universe) once in "Pass 1", then passes it into
    scan_symbol() for Darvas Box signals via this exact same pattern —
    but was NOT passing it into scan_cup_handle() at all, meaning Darvas
    and Cup & Handle signals generated on the same day for the SAME
    stock could show two different, methodologically inconsistent RS
    Ratings. This parameter closes that gap so both patterns use the
    identical, properly cross-sectional RS Rating.
    """
    if daily is None or len(daily) < 200:
        return None

    close  = daily["Close"]
    high   = daily["High"]
    low    = daily["Low"]
    volume = daily["Volume"]

    avg_vol = volume.iloc[-20:].mean()
    if avg_vol < MIN_AVG_VOLUME:
        return None

    weekly  = resample_weekly(daily)
    monthly = resample_monthly(daily)

    pattern = detect_cup_and_handle(symbol, daily, weekly, monthly)
    if pattern is None:
        return None

    # Require a minimum quality score even for invalid patterns —
    # we still want to surface "near-valid" patterns that are very close
    # to forming a proper setup (e.g. handle just started), as long as
    # the cup structure itself is high quality.
    #
    # NOTE: this intentionally does NOT require pattern.is_breaking_out.
    # The live scanner surfaces "Watching" and "Near Pivot" signals too
    # (patterns still forming or approaching the buy zone), giving advance
    # notice before a breakout happens — that's the whole point of
    # watching for these setups. backtest_cup_handle_symbol(), by
    # contrast, DOES hard-require is_breaking_out, since a backtest must
    # only count REALIZED, tradeable entries to produce honest win-rate
    # numbers — a "Watching" signal isn't a trade. Don't change one
    # without considering whether the other needs to match.
    if pattern.quality_score < CH_MIN_QUALITY_SCORE:
        log.debug("%s: C&H quality %.1f below threshold %.1f",
                  symbol, pattern.quality_score, CH_MIN_QUALITY_SCORE)
        return None

    current_price = float(close.iloc[-1])

    # Only emit signals where price is within or just below the buy zone
    # (at most 15% below the pivot — watching setups still forming handles)
    if current_price < pattern.pivot_price * 0.85:
        log.debug("%s: price %.2f too far below pivot %.2f",
                  symbol, current_price, pattern.pivot_price)
        return None

    # Trend filter — 200 EMA
    ema200 = float(ema(close, EMA_TREND).iloc[-1])
    if current_price < ema200:
        log.debug("%s: price %.2f below 200 EMA %.2f", symbol, current_price, ema200)
        return None

    # RS Rating — use the cross-sectional override if the caller (main.py's
    # live scan loop) provided one, matching exactly how scan_symbol()
    # (the Darvas scanner) already does this.
    if rs_override is not None:
        rs = rs_override
    else:
        try:
            bench_close = benchmark["Close"].reindex(close.index, method="ffill").dropna()
            rs = calc_rs(close, bench_close, RS_WEIGHTS)
        except Exception:
            rs = 50.0

    # SEPA
    try:
        sepa, sepa_checks = calc_sepa(close)
    except Exception:
        sepa, sepa_checks = 0.0, {}

    # ATR-based stop loss: below handle low by ATR_STOP_MULTIPLIER × ATR
    atr_val = float(calc_atr(high, low, close, ATR_PERIOD).iloc[-1])
    stop_loss = round(pattern.handle_low - ATR_STOP_MULTIPLIER * atr_val, 2)

    # Targets: T1 = cup high, T2 = cup high + cup height, T3 = + 2× height
    cup_height = pattern.cup_high - pattern.cup_low
    target1 = round(pattern.cup_high, 2)
    target2 = round(pattern.cup_high + cup_height, 2)
    target3 = round(pattern.cup_high + 2 * cup_height, 2)

    entry_price = current_price if pattern.is_breaking_out else pattern.pivot_price
    risk_ps = entry_price - stop_loss
    if risk_ps <= 0:
        return None

    rr_ratio    = round((target2 - entry_price) / risk_ps, 2)
    risk_amount = ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100
    pos_size    = max(1, int(risk_amount / risk_ps))
    cap_req     = round(pos_size * entry_price, 2)

    # Weekly / monthly trend
    w_trend = trend_label(weekly["Close"])  if len(weekly)  > 30 else "neutral"
    m_trend = trend_label(monthly["Close"]) if len(monthly) > 10 else "neutral"

    sig = CupHandleSignal(
        symbol             = symbol,
        sector             = sector,
        scan_date          = date.today(),
        cup_high           = pattern.cup_high,
        cup_low            = pattern.cup_low,
        cup_depth_pct      = pattern.cup_depth_pct,
        cup_duration_weeks = pattern.cup_duration_weeks,
        cup_shape_ok       = pattern.cup_shape_ok,
        cup_volume_dryup   = pattern.cup_volume_dryup,
        cup_start_date     = pattern.cup_start_date,
        cup_bottom_date    = pattern.cup_bottom_date,
        cup_end_date       = pattern.cup_end_date,
        handle_high        = pattern.handle_high,
        handle_low         = pattern.handle_low,
        handle_depth_pct   = pattern.handle_depth_pct,
        handle_duration_weeks = pattern.handle_duration_weeks,
        handle_in_upper_zone  = pattern.handle_in_upper_zone,
        handle_volume_dryup   = pattern.handle_volume_dryup,
        handle_start_date  = pattern.handle_start_date,
        handle_end_date    = pattern.handle_end_date,
        prior_uptrend_pct  = pattern.prior_uptrend_pct,
        current_price      = round(current_price, 2),
        pivot_price        = pattern.pivot_price,
        buy_zone_low       = pattern.buy_zone_low,
        buy_zone_high      = pattern.buy_zone_high,
        stop_loss          = stop_loss,
        target1            = target1,
        target2            = target2,
        target3            = target3,
        atr                = round(atr_val, 2),
        risk_per_share     = round(risk_ps, 2),
        position_size      = pos_size,
        capital_required   = cap_req,
        risk_amount        = round(risk_amount, 2),
        rr_ratio           = rr_ratio,
        rs_rating          = round(rs, 1),
        sepa_score         = sepa,
        sepa_checks        = sepa_checks,
        weekly_trend       = w_trend,
        monthly_trend      = m_trend,
        breakout_vol_ratio = pattern.breakout_volume_ratio,
        is_breaking_out    = pattern.is_breaking_out,
        pattern_quality    = pattern.quality_score,
    )

    log.info(
        "C&H SIGNAL %-20s quality=%5.1f  rs=%4.1f  depth=%4.1f%%  pivot=%.2f  status=%s",
        symbol, pattern.quality_score, rs, pattern.cup_depth_pct,
        pattern.pivot_price, sig.status,
    )
    return sig
