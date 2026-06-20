"""
NSE Darvas Box Scanner - Cup and Handle Detector
==================================================
Implements William O'Neil's Cup and Handle base pattern, as published in
"How to Make Money in Stocks". This is a SEPARATE pattern from the Darvas
Box and is detected/scored/reported independently.

O'Neil's Rules Encoded Here
----------------------------
1. PRIOR UPTREND: the stock must already be a market leader BEFORE the
   cup forms — at least a 30% advance over the preceding ~6 months.
   Cups that form after a long decline or sideways drift are NOT valid
   (O'Neil: "the prior trend... must be an uptrend").

2. CUP SHAPE: a rounded "U", not a sharp "V". The decline into the cup
   should be gradual, the bottom should round out (ideally with volume
   drying up), and the right side should climb back toward the old
   high in a controlled, decelerating manner.

3. CUP DEPTH: 12-33% off the high in normal markets (O'Neil's stated
   range), with up to 50% tolerated only in severe market corrections.
   Shallower cups (closer to 12-20%) are generally higher quality.

4. CUP DURATION: minimum 7 weeks (O'Neil's absolute floor), most
   reliable cups run 12-26 weeks (3-6 months); valid up to 65 weeks.

5. HANDLE: forms in the UPPER portion (upper half, ideally upper third)
   of the cup, after price has rounded back up near the old high. The
   handle itself drifts modestly downward on light/declining volume
   (a "shakeout" of weak holders) for at least 1-2 weeks, capped at a
   12% decline from the handle's own high. A handle that drops into
   the lower half of the cup, or that increases on heavy volume,
   invalidates the pattern.

6. BREAKOUT / PIVOT POINT: the buy signal is price breaking ABOVE the
   handle's high (the pivot) on volume at least 40-50% above the
   recent average — confirming institutional demand, not a low-volume
   drift through resistance. The buy zone is the pivot price up to 5%
   above it (O'Neil's "5% buy zone") — chasing far beyond that is
   discouraged.

Multi-Timeframe Integration
-----------------------------
The pattern is primarily evaluated on WEEKLY bars (O'Neil's own base
analysis was always done on weekly charts), then cross-checked against
daily bars (precise pivot/volume confirmation) and monthly bars (longer
trend context) for full multi-timeframe alignment, mirroring how the
Darvas Box scanner already cross-validates daily/weekly/monthly trend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    CUPHANDLE_BREAKOUT_VOL_SURGE_PCT, CUPHANDLE_BUY_ZONE_PCT,
    CUPHANDLE_HANDLE_MAX_DEPTH_PCT, CUPHANDLE_HANDLE_MAX_WEEKS,
    CUPHANDLE_HANDLE_MIN_WEEKS, CUPHANDLE_HANDLE_UPPER_ZONE,
    CUPHANDLE_HANDLE_VOL_DRYUP_RATIO, CUPHANDLE_IDEAL_MAX_WEEKS,
    CUPHANDLE_IDEAL_MIN_WEEKS, CUPHANDLE_MAX_DEPTH_PCT,
    CUPHANDLE_MAX_DEPTH_PCT_BEAR, CUPHANDLE_MAX_DURATION_WEEKS,
    CUPHANDLE_MIN_DEPTH_PCT, CUPHANDLE_MIN_DURATION_WEEKS,
    CUPHANDLE_MIN_HISTORY_WEEKS, CUPHANDLE_PRIOR_UPTREND_LOOKBACK_WEEKS,
    CUPHANDLE_PRIOR_UPTREND_MIN_PCT, MIN_AVG_VOLUME,
)
from logger_utils import get_logger

log = get_logger("scanner")


@dataclass
class CupHandlePattern:
    symbol:             str

    # Prior uptrend
    prior_uptrend_pct:  float
    prior_uptrend_ok:   bool

    # Cup geometry (all on WEEKLY bars)
    cup_start_date:     date
    cup_bottom_date:    date
    cup_end_date:       date            # where the right side rounds back to near the old high
    cup_high:           float           # left-side high (start of cup)
    cup_low:            float           # cup bottom
    cup_right_high:     float           # right-side high (where handle begins)
    cup_depth_pct:      float
    cup_duration_weeks: int
    cup_shape_ok:       bool            # rounded U-shape check passed
    cup_volume_dryup:   bool            # volume declined into the bottom

    # Handle geometry
    handle_start_date:  date
    handle_end_date:    date
    handle_high:        float           # = pivot point
    handle_low:         float
    handle_depth_pct:   float
    handle_duration_weeks: int
    handle_in_upper_zone: bool
    handle_volume_dryup:  bool

    # Breakout / pivot
    pivot_price:        float
    is_breaking_out:    bool            # latest bar closed above pivot with volume surge
    breakout_volume_ratio: float
    buy_zone_low:        float
    buy_zone_high:        float

    # Multi-timeframe alignment
    daily_confirms:      bool
    monthly_trend_ok:    bool

    # Scoring
    quality_score:        float = 0.0   # 0-100, O'Neil-rule-weighted composite
    is_valid:              bool = False
    rejection_reasons:    list = field(default_factory=list)


def detect_cup_and_handle(
    symbol: str,
    daily: pd.DataFrame,
    weekly: Optional[pd.DataFrame] = None,
    monthly: Optional[pd.DataFrame] = None,
) -> Optional[CupHandlePattern]:
    """
    Scan *daily* (with optional pre-computed *weekly*/*monthly*) for a
    valid O'Neil-style Cup and Handle pattern. Returns the most recent
    candidate pattern (valid or not — check `.is_valid` and
    `.rejection_reasons` to see why it failed) or None if there isn't
    even a candidate cup shape to evaluate.
    """
    if daily is None or len(daily) < 100:
        return None

    if weekly is None:
        from downloader import resample_weekly
        weekly = resample_weekly(daily)
    if monthly is None:
        from downloader import resample_monthly
        monthly = resample_monthly(daily)

    if len(weekly) < CUPHANDLE_MIN_HISTORY_WEEKS:
        return None

    avg_vol = daily["Volume"].iloc[-20:].mean()
    if avg_vol < MIN_AVG_VOLUME:
        return None

    w_high  = weekly["High"]
    w_low   = weekly["Low"]
    w_close = weekly["Close"]
    w_vol   = weekly["Volume"]
    n = len(weekly)

    rejection_reasons: list[str] = []

    # ── Step 1: Find the cup's left-side high (the peak before decline) ──────
    # Search the most recent CUPHANDLE_MAX_DURATION_WEEKS+handle window for
    # a peak, then validate everything relative to that peak.
    #
    # IMPORTANT: we must exclude a reserved tail of (min cup duration + min
    # handle duration) weeks from the search. Without this, on a stock that
    # has already broken out, the most recent bar (the breakout itself) is
    # often the single highest price in the whole window — argmax() would
    # then pick the BREAKOUT bar as the "cup's left-side high", leaving zero
    # room for any cup/handle to exist after it, and the function would
    # always bail out with "not enough room after left high".
    search_start = max(0, n - (CUPHANDLE_MAX_DURATION_WEEKS + CUPHANDLE_HANDLE_MAX_WEEKS + 5))
    reserved_tail = CUPHANDLE_MIN_DURATION_WEEKS + CUPHANDLE_HANDLE_MIN_WEEKS
    search_end = max(search_start + 1, n - reserved_tail)

    window_high = w_high.iloc[search_start:search_end]
    if window_high.empty:
        return None

    left_high_idx_rel = window_high.values.argmax()
    left_high_idx = search_start + left_high_idx_rel
    cup_high = float(w_high.iloc[left_high_idx])
    cup_high_date_idx = left_high_idx

    # Need enough bars after the left high to contain a full cup + handle
    if left_high_idx >= n - CUPHANDLE_MIN_DURATION_WEEKS - CUPHANDLE_HANDLE_MIN_WEEKS:
        return None

    # ── Step 2: Find the cup bottom (lowest low after the left high, before
    #    price has rounded back up near cup_high again) ───────────────────────
    post_high = w_low.iloc[left_high_idx:]
    cup_bottom_idx_rel = post_high.values.argmin()
    cup_bottom_idx = left_high_idx + cup_bottom_idx_rel
    cup_low = float(w_low.iloc[cup_bottom_idx])

    cup_depth_pct = (cup_high - cup_low) / cup_high * 100

    # ── Step 3: Find where the right side rounds back up NEAR THE OLD HIGH
    #    (the cup "end" / handle starting point) ─────────────────────────────
    # O'Neil's pattern requires the right side to fully round back up close
    # to the prior peak (cup_high) before a handle can form — NOT merely
    # halfway up the cup's range. Using a halfway threshold here was a real
    # bug: it let "cup_end" trigger far too early (as soon as price recovered
    # 50% of the depth), which then swept the rest of the actual rounding,
    # any small pullback, AND the eventual breakout bar all into what should
    # have been just the "handle" window — corrupting the handle's measured
    # high/low into picking up the breakout price instead of the true,
    # modest handle pullback.
    #
    # We require the right side to climb back to within RIGHT_SIDE_TOLERANCE
    # of cup_high (default: top 10% of the cup's range) before the cup is
    # considered "complete" and the handle search begins.
    RIGHT_SIDE_TOLERANCE = 0.90   # must reach at least 90% of the way back up
    cup_range = cup_high - cup_low
    right_side_floor = cup_low + cup_range * RIGHT_SIDE_TOLERANCE
    upper_zone_floor  = cup_low + cup_range * CUPHANDLE_HANDLE_UPPER_ZONE  # still used for the handle-zone check later

    right_side = w_close.iloc[cup_bottom_idx:]
    recovered = right_side[right_side >= right_side_floor]
    if recovered.empty:
        # Cup hasn't rounded back up near the old high yet — no handle
        # can have formed; this is a "cup still forming" case, not an error.
        return None
    cup_end_idx = weekly.index.get_loc(recovered.index[0])
    cup_right_high = float(w_high.iloc[cup_bottom_idx:cup_end_idx + 1].max())

    cup_duration_weeks = cup_end_idx - cup_high_date_idx

    # ── Step 4: Identify the handle (from cup_end_idx until either a
    #    genuine breakout above cup_right_high occurs, or
    #    CUPHANDLE_HANDLE_MAX_WEEKS is reached, whichever comes first) ────────
    #
    # IMPORTANT: the handle's high/low MUST be measured only over bars that
    # are still part of the consolidation — NOT over a bar that has already
    # broken out above the prior high. Using a fixed-width window without
    # this breakout boundary was a real bug: on any stock that has already
    # broken out, the breakout bar's price (now the highest in the window)
    # got included in the handle's own high/low calculation, corrupting
    # handle_high into being the breakout price itself rather than the
    # genuine, modest pullback high that defines the actual pivot point.
    handle_slice_high  = w_high.iloc[cup_end_idx:]
    handle_slice_low   = w_low.iloc[cup_end_idx:]
    handle_slice_close = w_close.iloc[cup_end_idx:]
    handle_slice_vol    = w_vol.iloc[cup_end_idx:]

    if len(handle_slice_high) < CUPHANDLE_HANDLE_MIN_WEEKS:
        # Not enough bars yet to even form a handle — report the cup only,
        # so the caller can still see "cup forming, handle pending"
        rejection_reasons.append("Handle has not formed yet (cup just completed)")
        handle_high = cup_right_high
        handle_low  = cup_right_high
        handle_depth_pct = 0.0
        handle_duration_weeks = 0
        handle_end_idx = cup_end_idx
        handle_in_upper_zone = True
        handle_volume_dryup = False
    else:
        # Find the first bar (if any) that breaks above cup_right_high —
        # everything from cup_end_idx up to (but excluding) that bar is the
        # true handle window. If no breakout has happened yet, fall back to
        # the fixed CUPHANDLE_HANDLE_MAX_WEEKS cap (handle still forming).
        max_window = min(CUPHANDLE_HANDLE_MAX_WEEKS, len(handle_slice_high))
        breakout_mask = handle_slice_high.iloc[:max_window] > cup_right_high
        if breakout_mask.any():
            handle_window_len = int(breakout_mask.values.argmax())
            handle_window_len = max(handle_window_len, CUPHANDLE_HANDLE_MIN_WEEKS) \
                if handle_window_len > 0 else CUPHANDLE_HANDLE_MIN_WEEKS
        else:
            handle_window_len = max_window

        handle_high = float(handle_slice_high.iloc[:handle_window_len].max())
        handle_low_window = handle_slice_low.iloc[:handle_window_len]
        handle_low = float(handle_low_window.min())
        handle_depth_pct = (handle_high - handle_low) / handle_high * 100 if handle_high > 0 else 100.0

        handle_end_idx = min(cup_end_idx + handle_window_len, n - 1)
        handle_duration_weeks = max(1, handle_end_idx - cup_end_idx)

        handle_in_upper_zone = handle_low >= upper_zone_floor

        cup_avg_vol = float(w_vol.iloc[cup_high_date_idx:cup_end_idx + 1].mean())
        handle_avg_vol = float(handle_slice_vol.iloc[:handle_window_len].mean())
        handle_volume_dryup = (
            handle_avg_vol <= cup_avg_vol * CUPHANDLE_HANDLE_VOL_DRYUP_RATIO
            if cup_avg_vol > 0 else False
        )

    # ── Step 5: Prior uptrend check (BEFORE the cup's left high) ──────────────
    prior_start_idx = max(0, cup_high_date_idx - CUPHANDLE_PRIOR_UPTREND_LOOKBACK_WEEKS)
    if prior_start_idx < cup_high_date_idx:
        prior_low = float(w_low.iloc[prior_start_idx:cup_high_date_idx + 1].min())
        prior_uptrend_pct = (cup_high - prior_low) / prior_low * 100 if prior_low > 0 else 0.0
    else:
        prior_uptrend_pct = 0.0
    prior_uptrend_ok = prior_uptrend_pct >= CUPHANDLE_PRIOR_UPTREND_MIN_PCT

    # ── Step 6: Cup shape validation (rounded U, not V; decelerating decline) ─
    cup_shape_ok, shape_reasons = _validate_cup_shape(
        w_close.iloc[cup_high_date_idx:cup_end_idx + 1],
        w_low.iloc[cup_high_date_idx:cup_end_idx + 1],
    )
    rejection_reasons.extend(shape_reasons)

    cup_avg_vol_check = float(w_vol.iloc[cup_high_date_idx:cup_bottom_idx + 1].mean())
    bottom_vol_check   = float(w_vol.iloc[max(cup_bottom_idx - 2, cup_high_date_idx):cup_bottom_idx + 3].mean())
    cup_volume_dryup = bottom_vol_check <= cup_avg_vol_check * 0.9 if cup_avg_vol_check > 0 else False

    # ── Step 7: Pivot, buy zone, breakout confirmation (use DAILY bars for
    #    precision — weekly bars are too coarse to confirm an exact breakout
    #    day/volume surge) ───────────────────────────────────────────────────
    pivot_price = handle_high
    buy_zone_low  = pivot_price
    buy_zone_high = pivot_price * (1 + CUPHANDLE_BUY_ZONE_PCT / 100)

    last_close = float(daily["Close"].iloc[-1])
    last_vol   = float(daily["Volume"].iloc[-1])
    avg_vol_20 = float(daily["Volume"].iloc[-21:-1].mean()) if len(daily) > 21 else avg_vol
    breakout_volume_ratio = last_vol / avg_vol_20 if avg_vol_20 > 0 else 0.0

    is_breaking_out = (
        last_close >= pivot_price and
        last_close <= buy_zone_high and
        breakout_volume_ratio >= (1 + CUPHANDLE_BREAKOUT_VOL_SURGE_PCT / 100)
    )

    # ── Step 8: Multi-timeframe alignment ─────────────────────────────────────
    daily_confirms = last_close >= float(daily["Close"].rolling(50).mean().iloc[-1]) if len(daily) > 50 else False
    monthly_trend_ok = False
    if monthly is not None and len(monthly) > 6:
        m_close = monthly["Close"]
        monthly_trend_ok = float(m_close.iloc[-1]) >= float(m_close.iloc[-6:].mean())

    # ── Step 9: Assemble validity checks against O'Neil's published rules ────
    if not prior_uptrend_ok:
        rejection_reasons.append(
            f"Prior uptrend only {prior_uptrend_pct:.1f}% "
            f"(O'Neil requires ≥{CUPHANDLE_PRIOR_UPTREND_MIN_PCT:.0f}%)"
        )
    if cup_duration_weeks < CUPHANDLE_MIN_DURATION_WEEKS:
        rejection_reasons.append(
            f"Cup duration {cup_duration_weeks}w below O'Neil's {CUPHANDLE_MIN_DURATION_WEEKS}w minimum"
        )
    if cup_duration_weeks > CUPHANDLE_MAX_DURATION_WEEKS:
        rejection_reasons.append(
            f"Cup duration {cup_duration_weeks}w exceeds O'Neil's {CUPHANDLE_MAX_DURATION_WEEKS}w maximum"
        )
    if not (CUPHANDLE_MIN_DEPTH_PCT <= cup_depth_pct <= CUPHANDLE_MAX_DEPTH_PCT_BEAR):
        rejection_reasons.append(
            f"Cup depth {cup_depth_pct:.1f}% outside O'Neil's "
            f"{CUPHANDLE_MIN_DEPTH_PCT:.0f}-{CUPHANDLE_MAX_DEPTH_PCT_BEAR:.0f}% range"
        )
    elif cup_depth_pct > CUPHANDLE_MAX_DEPTH_PCT:
        rejection_reasons.append(
            f"Cup depth {cup_depth_pct:.1f}% exceeds normal-market ceiling of "
            f"{CUPHANDLE_MAX_DEPTH_PCT:.0f}% (only acceptable in severe corrections)"
        )
    if handle_duration_weeks > 0 and handle_duration_weeks < CUPHANDLE_HANDLE_MIN_WEEKS:
        rejection_reasons.append(
            f"Handle duration {handle_duration_weeks}w below {CUPHANDLE_HANDLE_MIN_WEEKS}w minimum"
        )
    if handle_depth_pct > CUPHANDLE_HANDLE_MAX_DEPTH_PCT:
        rejection_reasons.append(
            f"Handle depth {handle_depth_pct:.1f}% exceeds "
            f"{CUPHANDLE_HANDLE_MAX_DEPTH_PCT:.0f}% maximum"
        )
    if not handle_in_upper_zone:
        rejection_reasons.append("Handle dropped into lower half of the cup (invalidates the pattern)")
    if not cup_shape_ok:
        rejection_reasons.append("Cup shape failed rounded-bottom validation (looks V-shaped or irregular)")

    is_valid = len(rejection_reasons) == 0

    quality_score = _score_pattern(
        prior_uptrend_pct=prior_uptrend_pct,
        cup_depth_pct=cup_depth_pct,
        cup_duration_weeks=cup_duration_weeks,
        cup_shape_ok=cup_shape_ok,
        cup_volume_dryup=cup_volume_dryup,
        handle_depth_pct=handle_depth_pct,
        handle_in_upper_zone=handle_in_upper_zone,
        handle_volume_dryup=handle_volume_dryup,
        breakout_volume_ratio=breakout_volume_ratio,
        is_breaking_out=is_breaking_out,
        daily_confirms=daily_confirms,
        monthly_trend_ok=monthly_trend_ok,
    )

    pattern = CupHandlePattern(
        symbol=symbol,
        prior_uptrend_pct=round(prior_uptrend_pct, 1),
        prior_uptrend_ok=prior_uptrend_ok,
        cup_start_date=weekly.index[cup_high_date_idx].date(),
        cup_bottom_date=weekly.index[cup_bottom_idx].date(),
        cup_end_date=weekly.index[cup_end_idx].date(),
        cup_high=round(cup_high, 2),
        cup_low=round(cup_low, 2),
        cup_right_high=round(cup_right_high, 2),
        cup_depth_pct=round(cup_depth_pct, 1),
        cup_duration_weeks=int(cup_duration_weeks),
        cup_shape_ok=cup_shape_ok,
        cup_volume_dryup=cup_volume_dryup,
        handle_start_date=weekly.index[cup_end_idx].date(),
        handle_end_date=weekly.index[handle_end_idx].date(),
        handle_high=round(handle_high, 2),
        handle_low=round(handle_low, 2),
        handle_depth_pct=round(handle_depth_pct, 1),
        handle_duration_weeks=int(handle_duration_weeks),
        handle_in_upper_zone=handle_in_upper_zone,
        handle_volume_dryup=handle_volume_dryup,
        pivot_price=round(pivot_price, 2),
        is_breaking_out=is_breaking_out,
        breakout_volume_ratio=round(breakout_volume_ratio, 2),
        buy_zone_low=round(buy_zone_low, 2),
        buy_zone_high=round(buy_zone_high, 2),
        daily_confirms=daily_confirms,
        monthly_trend_ok=monthly_trend_ok,
        quality_score=quality_score,
        is_valid=is_valid,
        rejection_reasons=rejection_reasons,
    )

    log.debug(
        "%s cup-handle: depth=%.1f%% duration=%dw handle_depth=%.1f%% valid=%s score=%.1f",
        symbol, cup_depth_pct, cup_duration_weeks, handle_depth_pct, is_valid, quality_score,
    )
    return pattern


# ─── Private helpers ──────────────────────────────────────────────────────────

def _validate_cup_shape(close: pd.Series, low: pd.Series) -> tuple[bool, list[str]]:
    """
    Validate the rounded "U" shape O'Neil insists on: the decline into
    the bottom should decelerate (not a sharp V crash), and the bottom
    itself should show some rounding (multiple bars near the low, not
    a single spike).
    """
    reasons: list[str] = []
    if len(close) < 5:
        return False, ["Cup too short to validate shape"]

    n = len(close)
    bottom_idx = low.values.argmin()

    # Left half slope should be negative (declining) but the decline rate
    # in the second half of the left side should be GENTLER than the first
    # half (deceleration) — a hallmark of a rounded bottom vs. a V-crash.
    left = close.iloc[: bottom_idx + 1]
    if len(left) >= 4:
        mid = len(left) // 2
        first_half_decline = (left.iloc[0] - left.iloc[mid]) / max(len(left.iloc[:mid]), 1)
        second_half_decline = (left.iloc[mid] - left.iloc[-1]) / max(len(left.iloc[mid:]), 1)
        if second_half_decline > first_half_decline * 1.5:
            reasons.append("Decline into the cup accelerated rather than decelerated (V-shape, not U-shape)")

    # Bottom rounding: at least 2 bars should sit within 5% of the absolute
    # low, rather than one single sharp spike low.
    abs_low = float(low.min())
    near_low_count = int((low <= abs_low * 1.05).sum())
    if near_low_count < 2:
        reasons.append("Cup bottom is a single sharp spike, not a rounded base")

    return len(reasons) == 0, reasons


def _score_pattern(
    prior_uptrend_pct: float, cup_depth_pct: float, cup_duration_weeks: int,
    cup_shape_ok: bool, cup_volume_dryup: bool, handle_depth_pct: float,
    handle_in_upper_zone: bool, handle_volume_dryup: bool,
    breakout_volume_ratio: float, is_breaking_out: bool,
    daily_confirms: bool, monthly_trend_ok: bool,
) -> float:
    """
    0-100 composite quality score weighted toward O'Neil's stated
    priorities: prior leadership (strength before the base), ideal cup
    depth/duration, proper handle formation, and a genuine volume-backed
    breakout — not just shape mechanics in isolation.

    FIXED 2026-06-20: the original per-criterion point allocations summed
    to 110, not 100, with only a final min(score, 100) clip at the end.
    That meant a pattern hitting every single criterion silently had its
    excess compressed away in a way that doesn't preserve the INTENDED
    relative weighting between criteria — effectively, whichever criteria
    got "clipped off" depended on arithmetic order, not actual priority.
    All weights below are proportionally rescaled (×100/110) so the true
    maximum is exactly 100, and every point allocation now means what it
    says.

    NOTE: unlike the Darvas Box scoring (which was rebalanced in this same
    session using real win/loss correlation data from a 5,175-trade
    backtest), there is not yet a comparably large Cup & Handle trade
    history to validate these weights against — backtest_cup_handle_symbol()
    was only just built this session. These weights remain based on
    O'Neil's published criteria emphasis, not yet empirically validated.
    Re-run --backtest-cup-handle-all once enough history accumulates and
    revisit this function the same way scanner.py's RSI/ADX weights were
    just revisited.
    """
    score = 0.0

    # Prior uptrend strength (18 pts) — O'Neil: this matters most, a cup
    # in a weak/lagging stock is much lower quality regardless of shape.
    score += min(prior_uptrend_pct / 50.0, 1.0) * 18

    # Cup depth quality (14 pts) — ideal zone 15-25%, O'Neil's sweet spot
    if 15 <= cup_depth_pct <= 25:
        score += 14
    elif CUPHANDLE_MIN_DEPTH_PCT <= cup_depth_pct <= CUPHANDLE_MAX_DEPTH_PCT:
        score += 9
    elif cup_depth_pct <= CUPHANDLE_MAX_DEPTH_PCT_BEAR:
        score += 4

    # Cup duration quality (9 pts) — ideal zone 12-26 weeks
    if CUPHANDLE_IDEAL_MIN_WEEKS <= cup_duration_weeks <= CUPHANDLE_IDEAL_MAX_WEEKS:
        score += 9
    elif CUPHANDLE_MIN_DURATION_WEEKS <= cup_duration_weeks <= CUPHANDLE_MAX_DURATION_WEEKS:
        score += 5

    # Cup shape (14 pts)
    if cup_shape_ok:
        score += 14

    # Cup volume dry-up at the bottom (9 pts) — confirms selling exhaustion
    if cup_volume_dryup:
        score += 9

    # Handle quality (13 pts split: depth + zone)
    if handle_depth_pct <= 8:
        score += 7
    elif handle_depth_pct <= CUPHANDLE_HANDLE_MAX_DEPTH_PCT:
        score += 4
    if handle_in_upper_zone:
        score += 6

    # Handle volume dry-up (5 pts) — O'Neil's "shakeout on light volume" tell
    if handle_volume_dryup:
        score += 5

    # Breakout confirmation (8 pts) — genuine volume-backed breakout.
    # Note this is now ALSO a hard entry gate in the backtest (and should
    # be treated as one live too — see scan_cup_handle), so its scoring
    # weight here mainly differentiates breakout STRENGTH among patterns
    # that already cleared the gate, not gate/no-gate itself.
    if is_breaking_out:
        score += 8
    elif breakout_volume_ratio >= 1.2:
        score += 3

    # Multi-timeframe alignment (up to 10 pts, daily + monthly trend)
    if daily_confirms:
        score += 5
    if monthly_trend_ok:
        score += 5

    return round(min(score, 100.0), 1)
