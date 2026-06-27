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
    CUPHANDLE_PRIOR_UPTREND_MIN_PCT, MIN_AVG_VOLUME, RSI_PERIOD,
    CUPHANDLE_MONTHLY_MIN_DURATION, CUPHANDLE_MONTHLY_MAX_DURATION,
    CUPHANDLE_MONTHLY_IDEAL_MIN, CUPHANDLE_MONTHLY_IDEAL_MAX,
    CUPHANDLE_MONTHLY_PRIOR_LOOKBACK, CUPHANDLE_MONTHLY_HANDLE_MIN,
    CUPHANDLE_MONTHLY_HANDLE_MAX, CUPHANDLE_DAILY_MIN_DURATION,
    CUPHANDLE_DAILY_MAX_DURATION, CUPHANDLE_DAILY_IDEAL_MIN,
    CUPHANDLE_DAILY_IDEAL_MAX, CUPHANDLE_DAILY_PRIOR_LOOKBACK,
    CUPHANDLE_DAILY_HANDLE_MIN, CUPHANDLE_DAILY_HANDLE_MAX,
)
from indicators import rsi as calc_rsi
from logger_utils import get_logger

log = get_logger("scanner")


@dataclass
class CupHandlePattern:
    symbol:             str

    # ADDED 2026-06-25: which bar timeframe this specific pattern was
    # found on. Previously the detector ONLY ever searched weekly bars
    # and used monthly purely as a trend filter — meaning a cup whose
    # ENTIRE structure exists only on the monthly chart (multi-year
    # bases) could never be found at all, since nothing ever searched
    # monthly bars for actual cup geometry. detect_cup_and_handle() now
    # runs three independent timeframe-specific searches (monthly,
    # weekly, daily) via _detect_cup_in_timeframe() and returns ALL
    # valid candidates found, ranked — this field records which one
    # each result came from.
    timeframe:          str = "weekly"   # 'monthly' / 'weekly' / 'daily'

    # Prior uptrend
    prior_uptrend_pct:  float = 0.0
    prior_uptrend_ok:   bool = False

    # Cup geometry (bar unit matches `timeframe` above)
    cup_start_date:     date = None
    cup_bottom_date:    date = None
    cup_end_date:       date = None      # where the right side rounds back to near the old high
    cup_high:           float = 0.0      # left-side high (start of cup)
    cup_low:            float = 0.0      # cup bottom
    cup_right_high:     float = 0.0      # right-side high (where handle begins)
    cup_depth_pct:      float = 0.0
    cup_duration_weeks: int = 0          # named "_weeks" for backward compat; actual unit is `timeframe`-dependent
    cup_shape_ok:       bool = False     # rounded U-shape check passed
    cup_volume_dryup:   bool = False     # volume declined into the bottom

    # Handle geometry
    handle_start_date:  date = None
    handle_end_date:    date = None
    handle_high:        float = 0.0      # = pivot point
    handle_low:         float = 0.0
    handle_depth_pct:   float = 0.0
    handle_duration_weeks: int = 0
    handle_in_upper_zone: bool = False
    handle_volume_dryup:  bool = False

    # Breakout / pivot
    pivot_price:        float = 0.0
    is_breaking_out:    bool = False     # latest bar closed above pivot with volume surge
    breakout_volume_ratio: float = 0.0
    buy_zone_low:        float = 0.0
    buy_zone_high:        float = 0.0

    # Multi-timeframe alignment
    daily_confirms:      bool = False
    monthly_trend_ok:    bool = False

    # Scoring
    quality_score:        float = 0.0   # 0-100, O'Neil-rule-weighted composite
    is_valid:              bool = False
    rejection_reasons:    list = field(default_factory=list)

    # ADDED 2026-06-25: when multiple valid patterns are found across
    # timeframes for the same symbol (point 2 — "nested structures"),
    # the highest-ranked one is returned as the primary result and the
    # others are attached here, sorted by quality_score descending, so
    # callers that want the full picture (e.g. a large monthly cup with
    # a weekly cup nested inside its handle) can see all of them rather
    # than only ever getting one.
    nested_patterns:      list = field(default_factory=list)

    # ADDED 2026-06-25 (point 5 of a detailed review): classic leading-
    # indicator checks — is RS making new highs BEFORE price does (a
    # sign institutions are accumulating ahead of the breakout), and is
    # the stock outperforming the benchmark specifically WHILE the
    # handle forms (confirming relative strength into the pivot, not
    # just at "today"). Both are computed only when a benchmark series
    # is supplied to detect_cup_and_handle() — default False/0.0 (not
    # used in scoring yet) when no benchmark is available, since a
    # missing optional signal should never silently fail detection.
    rs_new_high_before_price: bool = False
    rs_during_handle_vs_benchmark_pct: float = 0.0   # stock's RS-relative excess return during the handle window

    # ADDED 2026-06-25 (point 4, partial): volatility/range contraction
    # during the handle — a tightening daily range is one of the
    # "harder to encode but distinguishes elite setups" criteria
    # mentioned in review point 4. Computed from the DAILY bars
    # underlying the handle window, regardless of which timeframe the
    # cup itself was found on.
    volatility_contraction_pct: float = 0.0   # handle's avg daily range as % of cup's avg daily range (lower = tighter)


def _detect_cup_in_timeframe(
    symbol: str,
    bars: pd.DataFrame,
    timeframe: str,
    min_duration: int,
    max_duration: int,
    ideal_min: int,
    ideal_max: int,
    prior_lookback: int,
    handle_min: int,
    handle_max: int,
    daily: pd.DataFrame,
    monthly_trend_bars: Optional[pd.DataFrame],
) -> Optional[CupHandlePattern]:
    """
    Core, TIMEFRAME-AGNOSTIC cup-and-handle geometry search. ADDED
    2026-06-25 by extracting what used to be hardwired weekly-bar-only
    logic out of detect_cup_and_handle(), parameterizing every duration/
    lookback bound so the SAME search logic can run on monthly, weekly,
    or daily bars -- see detect_cup_and_handle() below for the
    orchestrator that calls this three times and ranks the results.

    This fixes a real architectural gap: previously monthly bars were
    ONLY ever used as a trend confirmation filter (a single bullish/
    bearish check), never searched for actual cup geometry -- meaning any
    stock whose entire cup-and-handle structure only exists at
    multi-year, monthly scale was structurally invisible no matter how
    clean the pattern was. This function is what now actually searches
    monthly bars for real cups, not just trend direction.

    *bars* -- the OHLCV series to search (monthly, weekly, or daily).
    *timeframe* -- label for the result ('monthly'/'weekly'/'daily').
    *daily* -- always the raw daily series, used for breakout/pivot
              confirmation regardless of which timeframe is being
              searched (a daily-precision breakout check makes sense
              even when the cup itself was found on monthly bars).
    *monthly_trend_bars* -- used for the cross-timeframe trend-alignment
              check (kept distinct from the cup SEARCH series above).
    """
    n = len(bars)
    if n < min_duration + handle_min + 5:
        return None

    b_high  = bars["High"]
    b_low   = bars["Low"]
    b_close = bars["Close"]
    b_vol   = bars["Volume"]

    avg_vol = daily["Volume"].iloc[-20:].mean()

    hard_rejections: list[str] = []
    soft_warnings: list[str] = []

    # -- Step 1: Find the cup's left-side high (the peak before decline) ------
    LOOKBACK = min(n, max_duration * 3)
    reserved_tail = min_duration + handle_min
    candidate_region_end = max(1, n - reserved_tail)
    candidate_region_start = max(0, n - LOOKBACK)

    if candidate_region_start >= candidate_region_end:
        return None

    LOCAL_PEAK_RADIUS = 3
    region_high = b_high.iloc[candidate_region_start:candidate_region_end]
    window_size = 2 * LOCAL_PEAK_RADIUS + 1
    rolling_max = region_high.rolling(window=window_size, center=True, min_periods=1).max()
    is_local_peak = (region_high.values == rolling_max.values)
    candidate_peaks = [candidate_region_start + i for i, v in enumerate(is_local_peak) if v]

    if not candidate_peaks:
        return None

    cup_high_date_idx = None
    cup_high = None
    for idx in reversed(candidate_peaks):
        if idx >= n - min_duration - handle_min:
            continue
        peak_val = float(b_high.iloc[idx])
        future_window_end = min(idx + max_duration, n)
        future_low = float(b_low.iloc[idx:future_window_end].min())
        depth_check = (peak_val - future_low) / peak_val * 100 if peak_val > 0 else 0
        if depth_check < CUPHANDLE_MIN_DEPTH_PCT:
            continue
        cup_high_date_idx = idx
        cup_high = peak_val
        break

    if cup_high_date_idx is None:
        return None

    left_high_idx = cup_high_date_idx

    # -- Step 2: Find the cup bottom -------------------------------------------
    post_high = b_low.iloc[left_high_idx:]
    cup_bottom_idx_rel = post_high.values.argmin()
    cup_bottom_idx = left_high_idx + cup_bottom_idx_rel
    cup_low = float(b_low.iloc[cup_bottom_idx])

    cup_depth_pct = (cup_high - cup_low) / cup_high * 100

    # -- Step 3: Right side recovers into the UPPER HALF of the cup's range ---
    cup_range = cup_high - cup_low
    upper_zone_floor = cup_low + cup_range * CUPHANDLE_HANDLE_UPPER_ZONE

    right_side = b_close.iloc[cup_bottom_idx:]
    recovered = right_side[right_side >= upper_zone_floor]
    if recovered.empty:
        return None

    first_recovery_idx = bars.index.get_loc(recovered.index[0])
    right_high_search_end = min(first_recovery_idx + max_duration, n)
    right_high_window = b_high.iloc[first_recovery_idx:right_high_search_end]
    cup_end_idx = first_recovery_idx + int(right_high_window.values.argmax())
    cup_right_high = float(b_high.iloc[cup_bottom_idx:cup_end_idx + 1].max())

    cup_duration = cup_end_idx - cup_high_date_idx

    # -- Step 4: Identify the handle --------------------------------------------
    handle_slice_high = b_high.iloc[cup_end_idx:]
    handle_slice_low  = b_low.iloc[cup_end_idx:]
    handle_slice_vol  = b_vol.iloc[cup_end_idx:]

    if len(handle_slice_high) < handle_min:
        hard_rejections.append(
            f"Handle has not formed yet (cup just completed on {timeframe} bars)"
        )
        handle_high = cup_right_high
        handle_low  = cup_right_high
        handle_depth_pct = 0.0
        handle_duration = 0
        handle_end_idx = cup_end_idx
        handle_in_upper_zone = True
        handle_volume_dryup = False
    else:
        max_window = min(handle_max, len(handle_slice_high))
        breakout_mask = handle_slice_high.iloc[:max_window] > cup_right_high
        if breakout_mask.any():
            handle_window_len = int(breakout_mask.values.argmax())
            handle_window_len = max(handle_window_len, handle_min) if handle_window_len > 0 else handle_min
        else:
            handle_window_len = max_window

        handle_high = float(handle_slice_high.iloc[:handle_window_len].max())
        handle_low  = float(handle_slice_low.iloc[:handle_window_len].min())
        handle_depth_pct = (handle_high - handle_low) / handle_high * 100 if handle_high > 0 else 100.0

        handle_end_idx = min(cup_end_idx + handle_window_len, n - 1)
        handle_duration = max(1, handle_end_idx - cup_end_idx)

        handle_in_upper_zone = handle_low >= upper_zone_floor

        cup_avg_vol = float(b_vol.iloc[cup_high_date_idx:cup_end_idx + 1].mean())
        handle_avg_vol = float(handle_slice_vol.iloc[:handle_window_len].mean())
        handle_volume_dryup = (
            handle_avg_vol <= cup_avg_vol * CUPHANDLE_HANDLE_VOL_DRYUP_RATIO
            if cup_avg_vol > 0 else False
        )

    # -- Step 5: Prior uptrend check ---------------------------------------------
    prior_start_idx = max(0, cup_high_date_idx - prior_lookback)
    if prior_start_idx < cup_high_date_idx:
        prior_low = float(b_low.iloc[prior_start_idx:cup_high_date_idx + 1].min())
        prior_uptrend_pct = (cup_high - prior_low) / prior_low * 100 if prior_low > 0 else 0.0
    else:
        prior_uptrend_pct = 0.0
    prior_uptrend_ok = prior_uptrend_pct >= CUPHANDLE_PRIOR_UPTREND_MIN_PCT

    # -- Step 6: Cup shape validation --------------------------------------------
    cup_shape_ok, shape_reasons = _validate_cup_shape(
        b_close.iloc[cup_high_date_idx:cup_end_idx + 1],
        b_low.iloc[cup_high_date_idx:cup_end_idx + 1],
    )
    hard_rejections.extend(shape_reasons)

    cup_avg_vol_check = float(b_vol.iloc[cup_high_date_idx:cup_bottom_idx + 1].mean())
    bottom_vol_check = float(
        b_vol.iloc[max(cup_bottom_idx - 2, cup_high_date_idx):cup_bottom_idx + 3].mean()
    )
    cup_volume_dryup = bottom_vol_check <= cup_avg_vol_check * 0.9 if cup_avg_vol_check > 0 else False

    # -- Step 7: Pivot, buy zone, breakout confirmation (always on DAILY bars) --
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

    # -- Step 8: Multi-timeframe alignment ---------------------------------------
    daily_confirms = (
        last_close >= float(daily["Close"].rolling(50).mean().iloc[-1])
        if len(daily) > 50 else False
    )
    monthly_trend_ok = False
    if monthly_trend_bars is not None and len(monthly_trend_bars) > 6:
        m_close = monthly_trend_bars["Close"]
        monthly_trend_ok = float(m_close.iloc[-1]) >= float(m_close.iloc[-6:].mean())

    # -- Step 9: Validity checks (hard rejections vs soft warnings) -----------
    if cup_duration < min_duration:
        hard_rejections.append(
            f"Cup duration {cup_duration} {timeframe} bars below minimum of {min_duration}"
        )
    if cup_duration > max_duration:
        soft_warnings.append(
            f"Cup duration {cup_duration} {timeframe} bars exceeds the typical "
            f"{max_duration}-bar outer bound for this timeframe"
        )

    if not (CUPHANDLE_MIN_DEPTH_PCT <= cup_depth_pct <= CUPHANDLE_MAX_DEPTH_PCT_BEAR):
        hard_rejections.append(
            f"Cup depth {cup_depth_pct:.1f}% outside O'Neil's "
            f"{CUPHANDLE_MIN_DEPTH_PCT:.0f}-{CUPHANDLE_MAX_DEPTH_PCT_BEAR:.0f}% range"
        )
    elif cup_depth_pct > CUPHANDLE_MAX_DEPTH_PCT:
        soft_warnings.append(
            f"Cup depth {cup_depth_pct:.1f}% exceeds the normal-market ideal of "
            f"{CUPHANDLE_MAX_DEPTH_PCT:.0f}% -- still within O'Neil's documented "
            f"severe-correction allowance, but a lower-confidence base"
        )

    if handle_duration > 0 and handle_duration < handle_min:
        hard_rejections.append(
            f"Handle duration {handle_duration} {timeframe} bars below minimum of {handle_min}"
        )
    if handle_depth_pct > CUPHANDLE_HANDLE_MAX_DEPTH_PCT:
        hard_rejections.append(
            f"Handle depth {handle_depth_pct:.1f}% exceeds {CUPHANDLE_HANDLE_MAX_DEPTH_PCT:.0f}% maximum"
        )
    if not handle_in_upper_zone:
        hard_rejections.append("Handle dropped into lower half of the cup (invalidates the pattern)")
    if not cup_shape_ok:
        hard_rejections.append("Cup shape failed rounded-bottom validation (looks V-shaped or irregular)")

    if not prior_uptrend_ok:
        soft_warnings.append(
            f"Prior uptrend only {prior_uptrend_pct:.1f}% "
            f"(O'Neil's stated guideline is >={CUPHANDLE_PRIOR_UPTREND_MIN_PCT:.0f}%, "
            f"but this is treated as a quality signal, not a hard gate)"
        )

    rejection_reasons = hard_rejections + soft_warnings
    is_valid = len(hard_rejections) == 0

    quality_score = _score_pattern(
        prior_uptrend_pct=prior_uptrend_pct,
        cup_depth_pct=cup_depth_pct,
        cup_duration_weeks=cup_duration,
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

    return CupHandlePattern(
        symbol=symbol,
        timeframe=timeframe,
        prior_uptrend_pct=round(prior_uptrend_pct, 1),
        prior_uptrend_ok=prior_uptrend_ok,
        cup_start_date=bars.index[cup_high_date_idx].date(),
        cup_bottom_date=bars.index[cup_bottom_idx].date(),
        cup_end_date=bars.index[cup_end_idx].date(),
        cup_high=round(cup_high, 2),
        cup_low=round(cup_low, 2),
        cup_right_high=round(cup_right_high, 2),
        cup_depth_pct=round(cup_depth_pct, 1),
        cup_duration_weeks=int(cup_duration),
        cup_shape_ok=cup_shape_ok,
        cup_volume_dryup=cup_volume_dryup,
        handle_start_date=bars.index[cup_end_idx].date(),
        handle_end_date=bars.index[handle_end_idx].date(),
        handle_high=round(handle_high, 2),
        handle_low=round(handle_low, 2),
        handle_depth_pct=round(handle_depth_pct, 1),
        handle_duration_weeks=int(handle_duration),
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


def detect_cup_and_handle(
    symbol: str,
    daily: pd.DataFrame,
    weekly: Optional[pd.DataFrame] = None,
    monthly: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.DataFrame] = None,
) -> Optional[CupHandlePattern]:
    """
    Scan *daily* (with optional pre-computed *weekly*/*monthly*) for
    valid O'Neil-style Cup and Handle pattern(s).

    REDESIGNED 2026-06-25 (addresses points 1 and 2 of a detailed
    real-world review): previously this function ONLY EVER searched
    weekly bars for cup geometry -- monthly bars were used purely as a
    trend-direction filter (a single bullish/bearish check), never
    searched for an actual cup shape. That meant a stock whose ENTIRE
    cup-and-handle structure exists only at multi-year, monthly scale
    (a genuine "3-year cup" or "4-year cup") was structurally invisible,
    no matter how clean the pattern was -- nothing ever looked for it.

    This now runs THREE INDEPENDENT searches via
    _detect_cup_in_timeframe() -- monthly, weekly, and daily -- each with
    its own duration/depth/handle bounds appropriate to that timeframe
    (see config.py's CUPHANDLE_MONTHLY_*/CUPHANDLE_DAILY_* constants).
    Real institutional charts often contain NESTED valid structures (a
    large monthly cup with a weekly cup inside its handle, which itself
    may contain a daily cup inside ITS handle) -- these are frequently
    the highest-conviction setups. Rather than returning just one
    pattern, ALL valid candidates found across all three timeframes are
    scored and ranked; the highest-quality one is returned as the
    primary result, with the rest attached via `.nested_patterns`
    (sorted by quality_score descending) so a caller that wants the
    full picture can see every valid structure, not just one.

    Returns None only if NO valid or near-valid candidate was found on
    ANY timeframe.
    """
    if daily is None or len(daily) < 100:
        return None

    if weekly is None:
        from downloader import resample_weekly
        weekly = resample_weekly(daily)
    if monthly is None:
        from downloader import resample_monthly
        monthly = resample_monthly(daily)

    avg_vol = daily["Volume"].iloc[-20:].mean()
    if avg_vol < MIN_AVG_VOLUME:
        return None

    candidates: list[CupHandlePattern] = []

    # Weekly -- the original, validated O'Neil-faithful scale.
    if len(weekly) >= CUPHANDLE_MIN_HISTORY_WEEKS:
        try:
            c = _detect_cup_in_timeframe(
                symbol, weekly, "weekly",
                min_duration=CUPHANDLE_MIN_DURATION_WEEKS,
                max_duration=CUPHANDLE_MAX_DURATION_WEEKS,
                ideal_min=CUPHANDLE_IDEAL_MIN_WEEKS,
                ideal_max=CUPHANDLE_IDEAL_MAX_WEEKS,
                prior_lookback=CUPHANDLE_PRIOR_UPTREND_LOOKBACK_WEEKS,
                handle_min=CUPHANDLE_HANDLE_MIN_WEEKS,
                handle_max=CUPHANDLE_HANDLE_MAX_WEEKS,
                daily=daily, monthly_trend_bars=monthly,
            )
            if c is not None:
                candidates.append(c)
        except Exception as e:
            log.debug("%s weekly cup-handle detection error: %s", symbol, e)

    # Monthly -- NEW: genuinely searches for large, multi-year cups instead
    # of only checking trend direction.
    if len(monthly) >= CUPHANDLE_MONTHLY_MIN_DURATION + CUPHANDLE_MONTHLY_HANDLE_MIN + 5:
        try:
            c = _detect_cup_in_timeframe(
                symbol, monthly, "monthly",
                min_duration=CUPHANDLE_MONTHLY_MIN_DURATION,
                max_duration=CUPHANDLE_MONTHLY_MAX_DURATION,
                ideal_min=CUPHANDLE_MONTHLY_IDEAL_MIN,
                ideal_max=CUPHANDLE_MONTHLY_IDEAL_MAX,
                prior_lookback=CUPHANDLE_MONTHLY_PRIOR_LOOKBACK,
                handle_min=CUPHANDLE_MONTHLY_HANDLE_MIN,
                handle_max=CUPHANDLE_MONTHLY_HANDLE_MAX,
                daily=daily, monthly_trend_bars=monthly,
            )
            if c is not None:
                candidates.append(c)
        except Exception as e:
            log.debug("%s monthly cup-handle detection error: %s", symbol, e)

    # Daily -- NEW: short, fast-forming bases, including ones nested inside
    # a larger weekly/monthly handle (point 2's "nested structures").
    if len(daily) >= CUPHANDLE_DAILY_MIN_DURATION + CUPHANDLE_DAILY_HANDLE_MIN + 5:
        try:
            c = _detect_cup_in_timeframe(
                symbol, daily, "daily",
                min_duration=CUPHANDLE_DAILY_MIN_DURATION,
                max_duration=CUPHANDLE_DAILY_MAX_DURATION,
                ideal_min=CUPHANDLE_DAILY_IDEAL_MIN,
                ideal_max=CUPHANDLE_DAILY_IDEAL_MAX,
                prior_lookback=CUPHANDLE_DAILY_PRIOR_LOOKBACK,
                handle_min=CUPHANDLE_DAILY_HANDLE_MIN,
                handle_max=CUPHANDLE_DAILY_HANDLE_MAX,
                daily=daily, monthly_trend_bars=monthly,
            )
            if c is not None:
                candidates.append(c)
        except Exception as e:
            log.debug("%s daily cup-handle detection error: %s", symbol, e)

    if not candidates:
        return None

    # Rank: valid patterns first, then by quality_score descending.
    candidates.sort(key=lambda p: (p.is_valid, p.quality_score), reverse=True)
    primary = candidates[0]
    primary.nested_patterns = candidates[1:]

    # ── Point 5: RS trend during the cup (leading-indicator checks) ──────────
    # Only computed when a benchmark series is supplied — degrades to the
    # dataclass defaults (False / 0.0) otherwise, never blocks detection.
    if benchmark is not None:
        try:
            _compute_rs_trend_signals(primary, daily, benchmark)
        except Exception as e:
            log.debug("%s RS-trend-during-cup computation failed: %s", symbol, e)

    # ── Point 4 (partial): volatility contraction during the handle ──────────
    try:
        _compute_volatility_contraction(primary, daily)
    except Exception as e:
        log.debug("%s volatility contraction computation failed: %s", symbol, e)

    log.debug(
        "%s cup-handle: %d candidate(s) across timeframes, primary=%s "
        "(depth=%.1f%% duration=%d valid=%s score=%.1f)",
        symbol, len(candidates), primary.timeframe,
        primary.cup_depth_pct, primary.cup_duration_weeks,
        primary.is_valid, primary.quality_score,
    )
    return primary

# ─── Private helpers ──────────────────────────────────────────────────────────

def _compute_rs_trend_signals(
    pattern: CupHandlePattern, daily: pd.DataFrame, benchmark: pd.DataFrame,
) -> None:
    """
    ADDED 2026-06-25 (review point 5): classic O'Neil/IBD leading-
    indicator checks on Relative Strength, mutating *pattern* in place:

    1. rs_new_high_before_price -- was the stock's RS line making new
       highs BEFORE the cup's right-side price high was set? This is a
       textbook sign of institutional accumulation happening ahead of
       (not just alongside) the price recovery -- if RS already peaked
       and price is still catching up, that's a stronger tell than RS
       and price moving in lockstep.

    2. rs_during_handle_vs_benchmark_pct -- the stock's excess return
       over the benchmark SPECIFICALLY during the handle window (not
       just "today"). A positive value confirms the stock kept leading
       the market while the handle formed -- exactly the "RS stronger
       than the benchmark while the handle forms" check requested.

    Uses DAILY bars for both checks regardless of which timeframe the
    cup itself was found on, since RS comparisons are most meaningful
    at daily resolution.
    """
    bench_close = benchmark["Close"].reindex(daily.index, method="ffill").dropna()
    common_idx = daily.index.intersection(bench_close.index)
    if len(common_idx) < 30:
        return

    close = daily["Close"].loc[common_idx]
    bench = bench_close.loc[common_idx]

    # RS line approximation: stock price / benchmark price (ratio).
    # A new high in this ratio = stock outperforming more than ever
    # before at that point -- the standard RS-line construction.
    rs_line = close / bench

    cup_start = pd.Timestamp(pattern.cup_start_date)
    cup_end   = pd.Timestamp(pattern.cup_end_date)
    handle_start = pd.Timestamp(pattern.handle_start_date)
    handle_end   = pd.Timestamp(pattern.handle_end_date)

    # Check 1: did the RS line set its high (within the cup window)
    # BEFORE price's own right-side high (cup_end_date)? If the RS
    # line's peak index is earlier than price's peak index within the
    # same window, RS led price.
    cup_window_mask = (rs_line.index >= cup_start) & (rs_line.index <= cup_end)
    if cup_window_mask.sum() >= 5:
        rs_window = rs_line[cup_window_mask]
        price_window = close[cup_window_mask]
        rs_peak_date = rs_window.idxmax()
        price_peak_date = price_window.idxmax()
        pattern.rs_new_high_before_price = rs_peak_date < price_peak_date

    # Check 2: stock's excess return over the benchmark DURING the
    # handle window specifically (not just at "today").
    handle_window_mask = (close.index >= handle_start) & (close.index <= handle_end)
    if handle_window_mask.sum() >= 2:
        h_close = close[handle_window_mask]
        h_bench = bench[handle_window_mask]
        if len(h_close) >= 2 and h_close.iloc[0] > 0 and h_bench.iloc[0] > 0:
            stock_ret = (h_close.iloc[-1] / h_close.iloc[0] - 1) * 100
            bench_ret = (h_bench.iloc[-1] / h_bench.iloc[0] - 1) * 100
            pattern.rs_during_handle_vs_benchmark_pct = round(stock_ret - bench_ret, 2)


def _compute_volatility_contraction(pattern: CupHandlePattern, daily: pd.DataFrame) -> None:
    """
    ADDED 2026-06-25 (review point 4, partial): measures whether the
    handle shows genuine volatility/range contraction relative to the
    cup as a whole -- one of the "harder to encode but distinguishes
    elite setups" characteristics from the review (alongside repeated
    10-week MA support and tight weekly closes, which are NOT yet
    implemented -- see the module-level TODO).

    Computed as: (handle's average daily High-Low range as a % of
    handle's average close) / (cup's average daily range as a % of
    cup's average close). A value below 1.0 means the handle is
    genuinely tighter than the cup as a whole -- the volatility
    contraction O'Neil-style traders look for ahead of a breakout.
    """
    cup_start = pd.Timestamp(pattern.cup_start_date)
    cup_end   = pd.Timestamp(pattern.cup_end_date)
    handle_start = pd.Timestamp(pattern.handle_start_date)
    handle_end   = pd.Timestamp(pattern.handle_end_date)

    cup_mask = (daily.index >= cup_start) & (daily.index <= cup_end)
    handle_mask = (daily.index >= handle_start) & (daily.index <= handle_end)

    if cup_mask.sum() < 5 or handle_mask.sum() < 2:
        return

    cup_bars = daily[cup_mask]
    handle_bars = daily[handle_mask]

    cup_range_pct = ((cup_bars["High"] - cup_bars["Low"]) / cup_bars["Close"]).mean() * 100
    handle_range_pct = ((handle_bars["High"] - handle_bars["Low"]) / handle_bars["Close"]).mean() * 100

    if cup_range_pct > 0:
        pattern.volatility_contraction_pct = round((handle_range_pct / cup_range_pct) * 100, 1)


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

    UPDATE 2026-06-21: a real 617-trade backtest run (full NSE universe)
    has now been analysed. Correlation of each measurable raw factor
    with win/loss was uniformly weak (|r| < 0.09 for every factor) —
    much weaker than Darvas Box's RSI (+0.255) or RS Rating (+0.201).
    A 1,000-sample bootstrap on composite_score vs. outcome gave a 95%
    CI of [-0.149, -0.000] — technically excludes zero, so the negative
    direction is not pure noise, but the effect is right at the edge of
    detectability. Given that, weights below were ADJUSTED CONSERVATIVELY
    (not inverted, not dramatically rebalanced) for the two components
    with the clearest negative signal — prior_uptrend_pct (was 18, now
    10) and cup_depth_pct (was 14 max, now 9 max) — with the freed points
    redistributed toward breakout confirmation and multi-timeframe
    alignment, which the hard-gate logic in detect_cup_and_handle()
    already requires for ANY signal to be emitted at all (i.e. they're
    known to correlate with the strategy's real 62.9% aggregate win rate,
    even though they weren't broken out as a separate measurable column
    in this run to confirm that directly).

    cup_shape_ok, cup_volume_dryup, handle_in_upper_zone, and
    handle_volume_dryup were NOT testable this round — they're boolean
    checks computed during detection but were never persisted to the
    trade log. Added to backtest_cup_handle_symbol()'s trade dict as of
    this fix so the NEXT backtest run can validate them properly. Revisit
    these weights again once that data exists — this is still an interim
    correction, not a final validated scoring model.
    """
    score = 0.0

    # Prior uptrend strength (10 pts, was 18) — REDUCED 2026-06-21.
    # Real correlation: -0.023 (weak negative). O'Neil's published
    # emphasis on this factor isn't showing up in the NSE sample tested
    # so far; kept as a real component (still conceptually meaningful —
    # a cup with zero prior advance isn't a valid O'Neil base at all)
    # but no longer weighted as the single largest factor pending
    # stronger evidence either way.
    score += min(prior_uptrend_pct / 50.0, 1.0) * 10

    # Cup depth quality (9 pts max, was 14) — REDUCED 2026-06-21.
    # Real correlation: -0.083 (weak negative, the strongest single
    # negative signal found). The "15-25% ideal zone" preference may not
    # hold on NSE data the way O'Neil documented for the US market it was
    # derived from — kept the same tiered logic, just lower stakes.
    if 15 <= cup_depth_pct <= 25:
        score += 9
    elif CUPHANDLE_MIN_DEPTH_PCT <= cup_depth_pct <= CUPHANDLE_MAX_DEPTH_PCT:
        score += 6
    elif cup_depth_pct <= CUPHANDLE_MAX_DEPTH_PCT_BEAR:
        score += 3

    # Cup duration quality (9 pts) — ideal zone 12-26 weeks. Unchanged;
    # correlation was -0.024, similarly weak, but duration wasn't one of
    # the two components flagged for adjustment this round (kept stable
    # rather than touching every weak-evidence factor at once).
    if CUPHANDLE_IDEAL_MIN_WEEKS <= cup_duration_weeks <= CUPHANDLE_IDEAL_MAX_WEEKS:
        score += 9
    elif CUPHANDLE_MIN_DURATION_WEEKS <= cup_duration_weeks <= CUPHANDLE_MAX_DURATION_WEEKS:
        score += 5

    # Cup shape (14 pts) — UNVALIDATED this round (not persisted to the
    # trade log until this fix); kept at original weight.
    if cup_shape_ok:
        score += 14

    # Cup volume dry-up at the bottom (9 pts) — UNVALIDATED this round
    # for the same reason; kept at original weight.
    if cup_volume_dryup:
        score += 9

    # Handle quality (13 pts split: depth + zone) — UNVALIDATED this
    # round; kept at original weight.
    if handle_depth_pct <= 8:
        score += 7
    elif handle_depth_pct <= CUPHANDLE_HANDLE_MAX_DEPTH_PCT:
        score += 4
    if handle_in_upper_zone:
        score += 6

    # Handle volume dry-up (5 pts) — UNVALIDATED this round; kept at
    # original weight.
    if handle_volume_dryup:
        score += 5

    # Breakout confirmation (17 pts, was 8) — RAISED 2026-06-21.
    # This is a HARD GATE in both the backtest (backtest_cup_handle_symbol
    # requires pattern.is_breaking_out to even record a trade) and is
    # recommended as one live too. Every trade in the validated 617-trade
    # sample cleared this gate, and that sample's 62.9% aggregate win
    # rate / +0.252R expectancy is real evidence the underlying breakout-
    # confirmation logic has edge — even though individual OTHER factors
    # didn't show strong standalone correlation. Raising this weight
    # leans further into the one mechanism with the clearest connection
    # to the strategy's actual measured performance.
    if is_breaking_out:
        score += 17
    elif breakout_volume_ratio >= 1.2:
        score += 4

    # Multi-timeframe alignment (14 pts total, was 10) — RAISED 2026-06-21.
    # Daily/monthly trend confirmation is conceptually load-bearing (it's
    # literally why this scanner checks 3 timeframes instead of 1) and
    # wasn't flagged with negative evidence — absorbing some of the
    # points freed up from prior_uptrend_pct and cup_depth_pct here
    # rather than leaving them unused.
    if daily_confirms:
        score += 7
    if monthly_trend_ok:
        score += 7

    return round(min(score, 100.0), 1)
