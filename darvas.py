"""
NSE Darvas Box Scanner - Darvas Box Detection Engine
=====================================================
Implements the classic Darvas Box algorithm adapted for NSE equities:
  • Scans historical daily data for valid box formations
  • Identifies the current / most-recent active box
  • Calculates box quality metrics: age, touches, width, tightness
  • Stores active boxes for watchlist tracking

Algorithm Summary
-----------------
1.  Find a new 52-week high (the *pivot*).
2.  The box HIGH = the highest close in the window BEFORE the pivot where
    no subsequent close exceeded it for DARVAS_HIGH_LOOKBACK days.
3.  The box LOW  = the lowest intraday low during the consolidation period
    that held for DARVAS_HIGH_LOOKBACK days (no close below it).
4.  Price must stay inside [box_low, box_high] for at least
    DARVAS_MIN_CONSOLIDATION bars for the box to be valid.
5.  An *active* box is one where the most recent close is still inside the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DARVAS_HIGH_LOOKBACK,
    DARVAS_LOOKBACK,
    DARVAS_MAX_WIDTH_PCT,
    DARVAS_MIN_CONSOLIDATION,
    DARVAS_MIN_WIDTH_PCT,
    DARVAS_BOX_TOUCH_MIN,
    MIN_AVG_VOLUME,
    MIN_HISTORY_DAYS,
)
from logger_utils import get_logger

log = get_logger("scanner")


@dataclass
class DarvasBox:
    symbol:         str
    box_high:       float
    box_low:        float
    start_date:     date
    end_date:       date          # last bar inside the box
    detected_date:  date
    age_bars:       int           # bars the price has been inside the box
    touches_high:   int           # times price tested box high from below
    touches_low:    int           # times price tested box low from above
    width_pct:      float         # (box_high - box_low) / box_low * 100
    is_active:      bool = True
    tightness:      float = 0.0   # 1 - (width_pct / DARVAS_MAX_WIDTH_PCT), higher = tighter
    quality_score:  float = 0.0   # 0-15 composite quality for scoring engine

    # ── Post-init computed fields ──
    def __post_init__(self):
        self.tightness    = max(0.0, 1.0 - self.width_pct / DARVAS_MAX_WIDTH_PCT)
        self.quality_score = self._calc_quality()

    def height(self) -> float:
        return self.box_high - self.box_low

    def midpoint(self) -> float:
        return (self.box_high + self.box_low) / 2

    def _calc_quality(self) -> float:
        """0-15 quality score used in composite scoring."""
        score = 0.0
        # Age: older boxes up to 60 bars are better (more proven support)
        score += min(self.age_bars / 60, 1.0) * 5
        # Touches: more = better (2-6 ideal)
        touch_total = self.touches_high + self.touches_low
        score += min(touch_total / 6, 1.0) * 5
        # Tightness: tighter box = higher quality
        score += self.tightness * 5
        return round(score, 2)


def detect_darvas_boxes(
    symbol: str,
    daily: pd.DataFrame,
) -> list[DarvasBox]:
    """
    Scan *daily* OHLCV data for all valid Darvas Boxes.
    Returns a list of DarvasBox objects; the last element (if any)
    is the most-recently detected box.
    """
    boxes: list[DarvasBox] = []

    if daily is None or len(daily) < MIN_HISTORY_DAYS:
        log.debug("%s: insufficient history (%d bars)", symbol, len(daily) if daily is not None else 0)
        return boxes

    # Liquidity check
    avg_vol = daily["Volume"].iloc[-20:].mean()
    if avg_vol < MIN_AVG_VOLUME:
        log.debug("%s: illiquid (avg_vol=%.0f)", symbol, avg_vol)
        return boxes

    close  = daily["Close"]
    high   = daily["High"]
    low    = daily["Low"]
    n      = len(daily)
    scan_start = max(0, n - DARVAS_LOOKBACK)

    i = scan_start + 10   # need some history for lookback

    while i < n:
        # ── Step 1: Detect box high ─────────────────────────────────────────
        # A box high is confirmed when no new high is made for DARVAS_HIGH_LOOKBACK bars
        box_high_idx = _find_box_high(high, i, n)
        if box_high_idx is None:
            i += 1
            continue

        box_high_val = high.iloc[box_high_idx]

        # ── Step 2: Find consolidation end (box low) ────────────────────────
        box_low_val, box_low_idx, consol_end = _find_box_low(
            close, low, high, box_high_idx, box_high_val, n
        )
        if box_low_val is None:
            i = box_high_idx + 1
            continue

        # ── Step 3: Validate box dimensions ────────────────────────────────
        width_pct = (box_high_val - box_low_val) / box_low_val * 100
        if not (DARVAS_MIN_WIDTH_PCT <= width_pct <= DARVAS_MAX_WIDTH_PCT):
            i = box_high_idx + 1
            continue

        age_bars = consol_end - box_high_idx + 1
        if age_bars < DARVAS_MIN_CONSOLIDATION:
            i = box_high_idx + 1
            continue

        # ── Step 4: Count touches ───────────────────────────────────────────
        touches_high, touches_low = _count_touches(
            close, high, low, box_high_idx, consol_end,
            box_high_val, box_low_val
        )
        if touches_low < DARVAS_BOX_TOUCH_MIN:
            i = box_high_idx + 1
            continue

        # ── Step 5: Is box still active (price inside box)? ─────────────────
        last_close  = close.iloc[-1]
        is_active   = box_low_val <= last_close <= box_high_val

        box = DarvasBox(
            symbol        = symbol,
            box_high      = round(box_high_val, 2),
            box_low       = round(box_low_val,  2),
            start_date    = daily.index[box_high_idx].date(),
            end_date      = daily.index[consol_end].date(),
            detected_date = date.today(),
            age_bars      = age_bars,
            touches_high  = touches_high,
            touches_low   = touches_low,
            width_pct     = round(width_pct, 2),
            is_active     = is_active,
        )
        boxes.append(box)
        log.debug("%s box: high=%.2f low=%.2f age=%d active=%s",
                  symbol, box_high_val, box_low_val, age_bars, is_active)

        # Skip ahead past this box to avoid sub-boxes
        i = consol_end + 1

    return boxes


def get_active_box(symbol: str, daily: pd.DataFrame) -> Optional[DarvasBox]:
    """Return the most recent active Darvas Box, or None."""
    boxes = detect_darvas_boxes(symbol, daily)
    active = [b for b in boxes if b.is_active]
    return active[-1] if active else None


# ─── Private helpers ──────────────────────────────────────────────────────────

def _find_box_high(high: pd.Series, start: int, n: int) -> Optional[int]:
    """
    Find the index of the bar whose high is NOT exceeded for
    DARVAS_HIGH_LOOKBACK subsequent bars.
    """
    lb = DARVAS_HIGH_LOOKBACK
    # We need at least `lb` bars after `start`
    limit = n - lb
    for idx in range(start, limit):
        h = high.iloc[idx]
        # Check the next `lb` bars
        future = high.iloc[idx + 1 : idx + 1 + lb]
        if (future <= h).all():
            return idx
    return None


def _find_box_low(
    close: pd.Series,
    low: pd.Series,
    high: pd.Series,
    box_high_idx: int,
    box_high_val: float,
    n: int,
) -> tuple[Optional[float], Optional[int], Optional[int]]:
    """
    After confirming the box high, find the box low:
    The lowest intraday low that holds (no close below it for
    DARVAS_HIGH_LOOKBACK bars) within the consolidation window.
    Also returns the last index inside the box.
    """
    lb = DARVAS_HIGH_LOOKBACK
    # Consolidation starts after box high is confirmed
    consol_start = box_high_idx + lb

    if consol_start >= n - lb:
        return None, None, None

    # Find where price breaks above box high (box breakout) or end of data
    consol_end = n - 1
    for idx in range(consol_start, n):
        if close.iloc[idx] > box_high_val:
            consol_end = idx - 1
            break

    if consol_end <= consol_start + DARVAS_MIN_CONSOLIDATION:
        return None, None, None

    # Box low = lowest low in consolidation that is NOT breached on close
    window_lows  = low.iloc[consol_start : consol_end + 1]
    window_close = close.iloc[consol_start : consol_end + 1]

    candidate = window_lows.min()
    box_low_idx = window_lows.idxmin()

    # Ensure no close broke below candidate
    if (window_close < candidate).any():
        # Adjust: use the lowest low where subsequent closes stay above
        for attempt_low in sorted(window_lows.unique()):
            if not (window_close < attempt_low).any():
                candidate = attempt_low
                break
        else:
            return None, None, None

    return candidate, close.index.get_loc(box_low_idx), consol_end


def _count_touches(
    close: pd.Series,
    high:  pd.Series,
    low:   pd.Series,
    start: int,
    end:   int,
    box_high: float,
    box_low:  float,
    tolerance_pct: float = 1.0,
) -> tuple[int, int]:
    """Count how many times price tested box_high and box_low within tolerance."""
    tol_h = box_high * (1 - tolerance_pct / 100)
    tol_l = box_low  * (1 + tolerance_pct / 100)

    h_window = high.iloc[start : end + 1]
    l_window = low.iloc[start : end + 1]

    touches_high = int((h_window >= tol_h).sum())
    touches_low  = int((l_window <= tol_l).sum())
    return touches_high, touches_low
