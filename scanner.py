"""
NSE Darvas Box Scanner - Scanner Engine  (v2 - all bugs fixed)
===============================================================
FIXES:
  BUG4: Entry zone widened to 40% of box height
  BUG5: Volume check uses 5-day avg > 20-day avg (accumulation), not single day
  BUG6: ADX_MIN lowered to 15 (NSE mid/small caps)
  BUG7: Score output threshold lowered to 60 (watch band added)
  EXTRA: Added rejection reason logging for every filter
  EXTRA: HH-HL check relaxed to 20 bars (was 30)
  EXTRA: Benchmark alignment check — handles missing/flat benchmark gracefully
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ACCOUNT_SIZE, ADX_MIN, ADX_PREFER_MAX, ADX_PREFER_MIN,
    ATR_PERIOD, ATR_STOP_MULTIPLIER, EMA_LONG, EMA_MID, EMA_SHORT,
    EMA_TREND, ENTRY_ZONE_PCT, RISK_PER_TRADE_PCT, RS_MIN_PREFERRED,
    RSI_MAX, RSI_MIN, RSI_PERIOD, SCORE_THRESHOLDS, SCORE_WEIGHTS,
    VOLUME_MA_FAST, VOLUME_MA_PERIOD, VOLUME_RATIO_MIN, RS_WEIGHTS,
)
from darvas import DarvasBox, get_active_box
from downloader import resample_weekly, resample_monthly
from indicators import (
    adx as calc_adx,
    atr as calc_atr,
    ema,
    higher_highs_higher_lows,
    rs_rating as calc_rs,
    rsi as calc_rsi,
    sepa_score as calc_sepa,
    trend_label,
    volume_ratio,
)
from logger_utils import get_logger

log = get_logger("scanner")


# ─── Signal Dataclass ─────────────────────────────────────────────────────────

@dataclass
class Signal:
    # Identity
    symbol:          str
    sector:          str
    scan_date:       date

    # Price context
    current_price:   float
    box_high:        float
    box_low:         float
    entry_zone_low:  float
    entry_zone_high: float

    # Risk management
    stop_loss:       float
    target1:         float
    target2:         float
    target3:         float
    atr:             float
    risk_per_share:  float
    position_size:   int
    capital_required: float
    risk_amount:     float
    rr_ratio:        float

    # Technical readings
    rsi_val:         float
    adx_val:         float
    volume_ratio:    float
    weekly_trend:    str
    monthly_trend:   str

    # Ratings
    rs_rating:       float
    sepa_score:      float
    sepa_checks:     dict = field(default_factory=dict)
    composite_score: float = 0.0
    classification:  str   = ""

    # Box metadata
    box_age_bars:    int   = 0
    box_width_pct:   float = 0.0
    box_quality:     float = 0.0

    # Tracking
    status:          str = "Waiting"
    signal_id:       str = ""

    def __post_init__(self):
        self.signal_id     = f"{self.symbol}_{self.scan_date.isoformat()}"
        self.classification = self._classify()

    def _classify(self) -> str:
        s = self.composite_score
        if s >= SCORE_THRESHOLDS["elite"]:        return "Elite Setup"
        if s >= SCORE_THRESHOLDS["very_strong"]:  return "Very Strong"
        if s >= SCORE_THRESHOLDS["strong"]:       return "Strong"
        if s >= SCORE_THRESHOLDS["watch"]:        return "Watch"
        return "Weak"


# ─── Main Scanner ─────────────────────────────────────────────────────────────

def scan_symbol(
    symbol:    str,
    daily:     pd.DataFrame,
    benchmark: pd.DataFrame,
    sector:    str = "Unknown",
) -> Optional[Signal]:
    """
    Full scan pipeline for one symbol.
    Logs the rejection reason at DEBUG level for every filter.
    Returns Signal if setup found, else None.
    """
    if daily is None or len(daily) < 200:
        log.debug("%s: skip — insufficient history (%d bars)",
                  symbol, len(daily) if daily is not None else 0)
        return None

    close  = daily["Close"]
    high   = daily["High"]
    low    = daily["Low"]
    volume = daily["Volume"]

    # ── 1. Darvas box ─────────────────────────────────────────────────────────
    box = get_active_box(symbol, daily)
    if box is None:
        log.debug("%s: skip — no active Darvas box", symbol)
        return None

    current_price = float(close.iloc[-1])
    box_height    = box.box_high - box.box_low

    # ── 2. Entry zone (bottom ENTRY_ZONE_PCT of box) ──────────────────────────
    entry_low  = box.box_low
    entry_high = box.box_low + box_height * ENTRY_ZONE_PCT
    if not (entry_low <= current_price <= entry_high):
        log.debug("%s: skip — price %.2f outside entry zone [%.2f–%.2f]",
                  symbol, current_price, entry_low, entry_high)
        return None

    # ── 3. Compute indicators ─────────────────────────────────────────────────
    rsi_val = float(calc_rsi(close, RSI_PERIOD).iloc[-1])
    adx_df  = calc_adx(high, low, close, 14)
    adx_val = float(adx_df["ADX"].iloc[-1])
    atr_val = float(calc_atr(high, low, close, ATR_PERIOD).iloc[-1])

    # Volume: use 5-day avg vs 20-day avg (accumulation signal)
    vol_fast = float(volume.iloc[-VOLUME_MA_FAST:].mean())
    vol_slow = float(volume.rolling(VOLUME_MA_PERIOD).mean().iloc[-1])
    vol_rat  = vol_fast / vol_slow if vol_slow > 0 else 0.0

    ema200 = float(ema(close, EMA_TREND).iloc[-1])

    # ── 4. Hard filters (log reason for each rejection) ──────────────────────

    if current_price < ema200:
        log.debug("%s: skip — price %.2f below 200 EMA %.2f", symbol, current_price, ema200)
        return None

    if not (RSI_MIN <= rsi_val <= RSI_MAX):
        log.debug("%s: skip — RSI %.1f outside [%.0f–%.0f]",
                  symbol, rsi_val, RSI_MIN, RSI_MAX)
        return None

    if adx_val < ADX_MIN:
        log.debug("%s: skip — ADX %.1f < %.0f", symbol, adx_val, ADX_MIN)
        return None

    if vol_rat < VOLUME_RATIO_MIN:
        log.debug("%s: skip — vol ratio %.2f < %.2f", symbol, vol_rat, VOLUME_RATIO_MIN)
        return None

    # HH-HL structure: check bars BEFORE box started (exclude consolidation)
    # Using bars before the box prevents the sideways consolidation from
    # appearing as a downtrend and incorrectly failing the filter
    pre_box_start = max(0, len(daily) - box.age_bars - 80)
    pre_box_end   = max(20, len(daily) - box.age_bars)
    h_pre = high.iloc[pre_box_start:pre_box_end]
    l_pre = low.iloc[pre_box_start:pre_box_end]
    if not higher_highs_higher_lows(h_pre, l_pre, len(h_pre)):
        log.debug("%s: skip — no HH-HL in pre-box trend", symbol)
        return None

    # ── 5. Multi-timeframe trends ─────────────────────────────────────────────
    try:
        weekly  = resample_weekly(daily)
        monthly = resample_monthly(daily)
        w_trend = trend_label(weekly["Close"])  if len(weekly)  > 30 else "neutral"
        m_trend = trend_label(monthly["Close"]) if len(monthly) > 10 else "neutral"
    except Exception:
        w_trend = m_trend = "neutral"

    # ── 6. RS Rating ──────────────────────────────────────────────────────────
    try:
        bench_close = benchmark["Close"].reindex(close.index, method="ffill").dropna()
        rs = calc_rs(close, bench_close, RS_WEIGHTS)
    except Exception:
        rs = 50.0   # neutral if benchmark unavailable

    # ── 7. SEPA ───────────────────────────────────────────────────────────────
    try:
        sepa, sepa_checks = calc_sepa(close)
    except Exception:
        sepa, sepa_checks = 0.0, {}

    # ── 8. Stop loss & targets ────────────────────────────────────────────────
    stop_loss   = round(box.box_low - ATR_STOP_MULTIPLIER * atr_val, 2)
    target1     = round(box.box_high, 2)
    target2     = round(box.box_high + box_height, 2)
    target3     = round(box.box_high + 2 * box_height, 2)

    risk_ps = current_price - stop_loss
    if risk_ps <= 0:
        log.debug("%s: skip — negative risk (stop above entry)", symbol)
        return None

    rr_ratio     = round((target2 - current_price) / risk_ps, 2)
    risk_amount  = ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100
    pos_size     = max(1, int(risk_amount / risk_ps))
    cap_required = round(pos_size * current_price, 2)

    # ── 9. Composite Score ────────────────────────────────────────────────────
    score = _compute_score(
        rs=rs, w_trend=w_trend, m_trend=m_trend,
        vol_ratio=vol_rat, box=box, adx=adx_val,
        rsi=rsi_val, sepa=sepa,
    )

    # FIX: Output 60+ (was 70+) — let "Watch" setups through to report
    if score < SCORE_THRESHOLDS["watch"]:
        log.debug("%s: skip — score %.1f below watch threshold", symbol, score)
        return None

    # ── 10. Build Signal ──────────────────────────────────────────────────────
    sig = Signal(
        symbol           = symbol,
        sector           = sector,
        scan_date        = date.today(),
        current_price    = round(current_price, 2),
        box_high         = box.box_high,
        box_low          = box.box_low,
        entry_zone_low   = round(entry_low,  2),
        entry_zone_high  = round(entry_high, 2),
        stop_loss        = stop_loss,
        target1          = target1,
        target2          = target2,
        target3          = target3,
        atr              = round(atr_val, 2),
        risk_per_share   = round(risk_ps, 2),
        position_size    = pos_size,
        capital_required = cap_required,
        risk_amount      = round(risk_amount, 2),
        rr_ratio         = rr_ratio,
        rsi_val          = round(rsi_val, 2),
        adx_val          = round(adx_val, 2),
        volume_ratio     = round(vol_rat, 2),
        weekly_trend     = w_trend,
        monthly_trend    = m_trend,
        rs_rating        = round(rs, 1),
        sepa_score       = sepa,
        sepa_checks      = sepa_checks,
        composite_score  = round(score, 1),
        box_age_bars     = box.age_bars,
        box_width_pct    = box.width_pct,
        box_quality      = box.quality_score,
    )

    log.info(
        "SIGNAL %-20s score=%5.1f  rs=%4.1f  rsi=%4.1f  adx=%4.1f  class=%s",
        symbol, score, rs, rsi_val, adx_val, sig.classification,
    )
    return sig


# ─── Composite Scoring ────────────────────────────────────────────────────────

def _compute_score(
    rs: float, w_trend: str, m_trend: str,
    vol_ratio: float, box: DarvasBox,
    adx: float, rsi: float, sepa: float,
) -> float:
    w     = SCORE_WEIGHTS
    score = 0.0

    # 1. RS Rating (25 pts)
    score += w["rs_rating"] * min(rs / 99, 1.0)

    # 2. Weekly trend (15 pts)
    trend_map = {"bullish": 1.0, "neutral": 0.5, "bearish": 0.0}
    score += w["weekly_trend"]  * trend_map.get(w_trend, 0.5)

    # 3. Monthly trend (10 pts)
    score += w["monthly_trend"] * trend_map.get(m_trend, 0.5)

    # 4. Volume expansion (10 pts) — 5d avg vs 20d avg
    score += w["volume_expansion"] * min(vol_ratio / 2.0, 1.0)

    # 5. Box quality (15 pts) — quality_score is already 0–15
    score += box.quality_score

    # 6. ADX strength (10 pts)
    if ADX_PREFER_MIN <= adx <= ADX_PREFER_MAX:
        adx_pts = 1.0
    elif adx >= ADX_MIN:
        adx_pts = 0.5
    else:
        adx_pts = 0.0
    score += w["adx_strength"] * adx_pts

    # 7. RSI reversal quality (5 pts) — ideal 35-48
    if 35 <= rsi <= 48:
        rsi_pts = 1.0
    elif RSI_MIN <= rsi <= RSI_MAX:
        rsi_pts = 0.6
    else:
        rsi_pts = 0.0
    score += w["rsi_reversal"] * rsi_pts

    # 8. SEPA (10 pts) — sepa is 0–10
    score += w["sepa_score"] * (sepa / 10.0)

    return score
