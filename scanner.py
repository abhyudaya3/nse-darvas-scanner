"""
NSE Darvas Box Scanner - Scanner Engine
========================================
Integrates:
  • Darvas box detection
  • Bottom-of-box entry logic
  • RS Rating (O'Neil)
  • SEPA score (Minervini)
  • Multi-timeframe trend alignment
  • Composite 100-point scoring
  • ATR-based position sizing
  • Stop loss and 4-target calculation
  • Sector tagging (best-effort from yfinance)
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
    EMA_TREND, RISK_PER_TRADE_PCT, RS_MIN_PREFERRED, RSI_MAX,
    RSI_MIN, RSI_PERIOD, SCORE_THRESHOLDS, SCORE_WEIGHTS,
    VOLUME_MA_PERIOD, VOLUME_RATIO_MIN, RS_WEIGHTS,
)
from darvas import DarvasBox, detect_darvas_boxes, get_active_box
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
    entry_zone_low:  float        # bottom quarter of box
    entry_zone_high: float

    # Risk management
    stop_loss:       float
    target1:         float        # box high
    target2:         float        # 1× height above box
    target3:         float        # 2× height above box
    atr:             float
    risk_per_share:  float
    position_size:   int          # shares
    capital_required: float
    risk_amount:     float
    rr_ratio:        float        # reward-to-risk (target2 / risk)

    # Technical readings
    rsi_val:         float
    adx_val:         float
    volume_ratio:    float
    weekly_trend:    str          # bullish / bearish / neutral
    monthly_trend:   str

    # Ratings
    rs_rating:       float        # 1-99
    sepa_score:      float        # 0-10
    sepa_checks:     dict         = field(default_factory=dict)
    composite_score: float        = 0.0
    classification:  str          = ""

    # Box metadata
    box_age_bars:    int          = 0
    box_width_pct:   float        = 0.0
    box_quality:     float        = 0.0

    # Tracking
    status:          str          = "Waiting"    # Waiting / Active / Triggered / etc.
    signal_id:       str          = ""

    def __post_init__(self):
        self.signal_id = f"{self.symbol}_{self.scan_date.isoformat()}"
        self.classification = self._classify()

    def _classify(self) -> str:
        s = self.composite_score
        if s >= SCORE_THRESHOLDS["elite"]:        return "Elite Setup"
        if s >= SCORE_THRESHOLDS["very_strong"]:  return "Very Strong"
        if s >= SCORE_THRESHOLDS["strong"]:       return "Strong"
        return "Weak"


# ─── Main Scanner ─────────────────────────────────────────────────────────────

def scan_symbol(
    symbol:    str,
    daily:     pd.DataFrame,
    benchmark: pd.DataFrame,
    sector:    str = "Unknown",
) -> Optional[Signal]:
    """
    Perform full scan for *symbol*.  Returns a Signal if a valid
    bottom-of-box setup is found, else None.
    """
    if daily is None or len(daily) < 252:
        return None

    close  = daily["Close"]
    high   = daily["High"]
    low    = daily["Low"]
    volume = daily["Volume"]

    # ── Darvas box ────────────────────────────────────────────────────────────
    box = get_active_box(symbol, daily)
    if box is None:
        return None

    current_price = float(close.iloc[-1])

    # ── Entry zone: bottom 25% of box ────────────────────────────────────────
    box_height    = box.box_high - box.box_low
    entry_low     = box.box_low
    entry_high    = box.box_low + box_height * 0.30   # bottom 30% of box
    if not (entry_low <= current_price <= entry_high):
        return None   # price not in entry zone

    # ── Compute indicators ────────────────────────────────────────────────────
    rsi_series  = calc_rsi(close, RSI_PERIOD)
    rsi_val     = float(rsi_series.iloc[-1])

    adx_df      = calc_adx(high, low, close, 14)
    adx_val     = float(adx_df["ADX"].iloc[-1])

    atr_series  = calc_atr(high, low, close, ATR_PERIOD)
    atr_val     = float(atr_series.iloc[-1])

    vol_ratio   = float(volume_ratio(volume, VOLUME_MA_PERIOD).iloc[-1])

    ema200      = ema(close, EMA_TREND)
    ema50       = ema(close, EMA_MID)
    ema20       = ema(close, EMA_SHORT)

    # ── Hard filters ─────────────────────────────────────────────────────────
    # 1. Price must be above 200 EMA (long-term uptrend)
    if current_price < float(ema200.iloc[-1]):
        return None

    # 2. RSI in reversal zone
    if not (RSI_MIN <= rsi_val <= RSI_MAX):
        return None

    # 3. ADX minimum trend strength
    if adx_val < ADX_MIN:
        return None

    # 4. Volume filter
    if vol_ratio < VOLUME_RATIO_MIN:
        return None

    # 5. Higher highs / higher lows structure
    if not higher_highs_higher_lows(high, low, 30):
        return None

    # ── Multi-timeframe trends ────────────────────────────────────────────────
    weekly  = resample_weekly(daily)
    monthly = resample_monthly(daily)
    w_trend = trend_label(weekly["Close"])  if len(weekly)  > 50 else "neutral"
    m_trend = trend_label(monthly["Close"]) if len(monthly) > 12 else "neutral"

    # ── RS Rating ─────────────────────────────────────────────────────────────
    bench_close = benchmark["Close"].reindex(close.index, method="ffill").dropna()
    rs = calc_rs(close, bench_close, RS_WEIGHTS)

    # ── SEPA ─────────────────────────────────────────────────────────────────
    sepa, sepa_checks = calc_sepa(close)

    # ── Stop loss & targets ───────────────────────────────────────────────────
    stop_loss    = round(box.box_low - ATR_STOP_MULTIPLIER * atr_val, 2)
    target1      = round(box.box_high, 2)
    target2      = round(box.box_high + box_height, 2)
    target3      = round(box.box_high + 2 * box_height, 2)

    risk_ps      = current_price - stop_loss
    if risk_ps <= 0:
        return None

    rr_ratio     = round((target2 - current_price) / risk_ps, 2)
    risk_amount  = ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100
    pos_size     = max(1, int(risk_amount / risk_ps))
    cap_required = round(pos_size * current_price, 2)

    # ── Composite Score ───────────────────────────────────────────────────────
    score = _compute_score(
        rs=rs, w_trend=w_trend, m_trend=m_trend,
        vol_ratio=vol_ratio, box=box, adx=adx_val,
        rsi=rsi_val, sepa=sepa,
    )

    if score < SCORE_THRESHOLDS["strong"]:
        return None   # filter out weak setups

    # ── Build Signal ──────────────────────────────────────────────────────────
    sig = Signal(
        symbol          = symbol,
        sector          = sector,
        scan_date       = date.today(),
        current_price   = round(current_price, 2),
        box_high        = box.box_high,
        box_low         = box.box_low,
        entry_zone_low  = round(entry_low,  2),
        entry_zone_high = round(entry_high, 2),
        stop_loss       = stop_loss,
        target1         = target1,
        target2         = target2,
        target3         = target3,
        atr             = round(atr_val, 2),
        risk_per_share  = round(risk_ps, 2),
        position_size   = pos_size,
        capital_required= cap_required,
        risk_amount     = round(risk_amount, 2),
        rr_ratio        = rr_ratio,
        rsi_val         = round(rsi_val, 2),
        adx_val         = round(adx_val, 2),
        volume_ratio    = round(vol_ratio, 2),
        weekly_trend    = w_trend,
        monthly_trend   = m_trend,
        rs_rating       = round(rs, 1),
        sepa_score      = sepa,
        sepa_checks     = sepa_checks,
        composite_score = round(score, 1),
        box_age_bars    = box.age_bars,
        box_width_pct   = box.width_pct,
        box_quality     = box.quality_score,
    )

    log.info("SIGNAL %s score=%.1f rs=%.1f sepa=%.1f class=%s",
             symbol, score, rs, sepa, sig.classification)
    return sig


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _compute_score(
    rs: float, w_trend: str, m_trend: str,
    vol_ratio: float, box: DarvasBox,
    adx: float, rsi: float, sepa: float,
) -> float:
    w = SCORE_WEIGHTS
    score = 0.0

    # 1. RS Rating (25 pts)
    score += w["rs_rating"] * min(rs / 99, 1.0)

    # 2. Weekly trend (15 pts)
    score += w["weekly_trend"]  * {"bullish": 1.0, "neutral": 0.5, "bearish": 0.0}[w_trend]

    # 3. Monthly trend (10 pts)
    score += w["monthly_trend"] * {"bullish": 1.0, "neutral": 0.5, "bearish": 0.0}[m_trend]

    # 4. Volume expansion (10 pts)
    score += w["volume_expansion"] * min(vol_ratio / 3.0, 1.0)

    # 5. Box quality (15 pts) – already 0-15
    score += box.quality_score

    # 6. ADX strength (10 pts)
    if ADX_PREFER_MIN <= adx <= ADX_PREFER_MAX:
        adx_pts = 1.0
    elif adx >= ADX_MIN:
        adx_pts = 0.6
    else:
        adx_pts = 0.0
    score += w["adx_strength"] * adx_pts

    # 7. RSI reversal quality (5 pts) – ideal 38-46
    rsi_pts = 1.0 if 38 <= rsi <= 46 else (0.5 if RSI_MIN <= rsi <= RSI_MAX else 0.0)
    score += w["rsi_reversal"] * rsi_pts

    # 8. SEPA (10 pts – sepa is 0-10)
    score += w["sepa_score"] * (sepa / 10)

    return score
