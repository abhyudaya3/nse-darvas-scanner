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
)
from indicators import rsi as calc_rsi
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

    # FIXED 2026-06-24: split into hard_rejections (pattern is genuinely
    # invalid) vs soft_warnings (valid but outside O'Neil's "ideal" zone).
    # See the full explanation further below, near where these lists are
    # combined into the final is_valid determination.
    hard_rejections: list[str] = []
    soft_warnings: list[str] = []

    # ── Step 1: Find the cup's left-side high (the peak before decline) ──────
    # FIXED 2026-06-24: this previously searched ONLY the most recent
    # ~(CUPHANDLE_MAX_DURATION_WEEKS + CUPHANDLE_HANDLE_MAX_WEEKS + 5) ≈ 82
    # weeks for the single highest price (argmax), then treated THAT as
    # the cup's left-side high. For a stock in a sustained uptrend, the
    # single highest price in any recent window is almost always very
    # close to "now" — which meant the search was structurally incapable
    # of ever finding a genuine large/long cup whose actual left-side
    # high occurred further back than ~82 weeks, EVEN THOUGH
    # CUPHANDLE_MAX_DURATION_WEEKS (65) was only ever meant to bound how
    # long ONE cup itself can run, not how far back the search is allowed
    # to look at all. Confirmed on a real-world case (a stock with ~4.5
    # years of history whose actual ~27%-deep, multi-month weekly base
    # started roughly 80+ weeks back) where this hard cap meant the
    # function would NEVER find the real cup — it would always lock onto
    # a smaller, more recent local high inside the ongoing uptrend instead,
    # producing a tiny, nonsensical "4% deep cup" that gets correctly
    # hard-rejected, but for the wrong reason: it isn't that no valid cup
    # exists, it's that the search never even looked at the real one.
    #
    # The fix: search a MUCH larger lookback (up to ~3 years / 156 weeks,
    # or the full available history if shorter) for every LOCAL peak (not
    # just the single global max), then test each candidate peak — most
    # recent first — to see if it produces a structurally valid cup
    # duration (CUPHANDLE_MIN/MAX_DURATION_WEEKS). The first candidate
    # (scanning backwards from most recent) that fits is used as the
    # cup's left-side high. This lets the detector find large, multi-
    # month-to-multi-year weekly/monthly-scale cups, not just ones that
    # happen to start within the last ~82 weeks.
    LOOKBACK_WEEKS = min(n, 160)   # ~3 years of weekly bars, or full history if shorter
    reserved_tail = CUPHANDLE_MIN_DURATION_WEEKS + CUPHANDLE_HANDLE_MIN_WEEKS
    candidate_region_end = max(1, n - reserved_tail)   # leave room for a cup+handle after the peak
    candidate_region_start = max(0, n - LOOKBACK_WEEKS)

    if candidate_region_start >= candidate_region_end:
        return None

    # Find local peaks: a bar whose High is the max within a small
    # surrounding window — these are the real candidate "cup starts",
    # not just whatever the single tallest bar happens to be.
    #
    # PERFORMANCE FIX 2026-06-24: the first version of this loop called
    # `.iloc[lo:hi].max()` once per bar in plain Python — each call carries
    # real pandas overhead (index lookups, slice-object construction,
    # method dispatch) even though the underlying comparison is trivial.
    # Profiling showed this turned a single detect_cup_and_handle() call
    # into ~300 pandas slice operations, and the walk-forward backtest
    # calls this function roughly once per WEEK of history — at full NSE
    # scale this extrapolated to nearly 3 hours just for Cup & Handle,
    # i.e. it would have meaningfully risked the workflow timeout. This
    # is now done with one vectorized rolling-window max (numpy/pandas
    # internals do the windowing in C, not a Python loop), then a single
    # elementwise comparison — ~280x fewer Python-level pandas calls for
    # the same logical result.
    LOCAL_PEAK_RADIUS = 3
    region_high = w_high.iloc[candidate_region_start:candidate_region_end]
    window_size = 2 * LOCAL_PEAK_RADIUS + 1
    rolling_max = region_high.rolling(window=window_size, center=True, min_periods=1).max()
    is_local_peak = (region_high.values == rolling_max.values)
    candidate_peaks = [candidate_region_start + i for i, v in enumerate(is_local_peak) if v]

    if not candidate_peaks:
        return None

    # Test candidates MOST RECENT FIRST — a cup-and-handle signal is most
    # actionable when it's recent, so prefer the latest valid candidate
    # rather than always defaulting to the oldest one that happens to fit.
    cup_high_date_idx = None
    cup_high = None
    for idx in reversed(candidate_peaks):
        if idx >= n - CUPHANDLE_MIN_DURATION_WEEKS - CUPHANDLE_HANDLE_MIN_WEEKS:
            continue   # not enough room after this peak for a full cup+handle
        # Quick structural pre-check: does the decline from this peak,
        # within the next MAX_DURATION_WEEKS, actually reach a valid cup
        # depth? If not, this peak is unlikely to be a real cup start —
        # skip it rather than committing to it and failing later.
        peak_val = float(w_high.iloc[idx])
        future_window_end = min(idx + CUPHANDLE_MAX_DURATION_WEEKS, n)
        future_low = float(w_low.iloc[idx:future_window_end].min())
        depth_check = (peak_val - future_low) / peak_val * 100 if peak_val > 0 else 0
        if depth_check < CUPHANDLE_MIN_DEPTH_PCT:
            continue
        cup_high_date_idx = idx
        cup_high = peak_val
        break

    if cup_high_date_idx is None:
        return None

    left_high_idx = cup_high_date_idx

    #    price has rounded back up near cup_high again) ───────────────────────
    post_high = w_low.iloc[left_high_idx:]
    cup_bottom_idx_rel = post_high.values.argmin()
    cup_bottom_idx = left_high_idx + cup_bottom_idx_rel
    cup_low = float(w_low.iloc[cup_bottom_idx])

    cup_depth_pct = (cup_high - cup_low) / cup_high * 100

    # ── Step 3: Find where the right side rounds back up into the UPPER
    #    HALF of the cup's range (the cup "end" / handle starting point) ────
    # FIXED 2026-06-24: an earlier session's fix required the right side to
    # climb back to 90% of the cup's range before acknowledging the cup as
    # "complete" — but that's NOT O'Neil's actual rule. His published
    # criterion is that the HANDLE forms in the upper HALF of the cup's
    # range (CUPHANDLE_HANDLE_UPPER_ZONE = 0.5), not that the cup itself
    # must nearly fully close before a handle search even begins. The 90%
    # threshold was so strict that any cup deeper than roughly 20-25% (cup
    # depth doesn't need to fully retrace to qualify) would essentially
    # never trigger — confirmed on a real-world case (a recently-listed
    # solar-sector NSE stock with an ~84-week trading history and a ~50%
    # deep cup that recovered to within ~13% of its high) where the
    # detector returned None outright because the price never climbed all
    # the way to 90% of the range, even though the actual setup — a
    # textbook rounded base with the handle sitting comfortably in the
    # upper half — was genuine and tradeable by O'Neil's real rule.
    #
    # The fix uses CUPHANDLE_HANDLE_UPPER_ZONE (0.5, i.e. the cup's
    # midpoint) as the recovery floor, matching the upper-half rule
    # directly, while still guarding against the original bug this
    # threshold was meant to prevent: once the right side first reaches
    # the upper-half floor, we look for the LOCAL PEAK in a forward window
    # before calling that the cup's end — this stops the handle search
    # from anchoring on the very first bar that barely crosses the
    # midpoint (which could just be normal volatility, not the true
    # right-side high) while still not requiring an unrealistic full
    # recovery to the old high.
    cup_range = cup_high - cup_low
    upper_zone_floor = cup_low + cup_range * CUPHANDLE_HANDLE_UPPER_ZONE

    right_side = w_close.iloc[cup_bottom_idx:]
    recovered = right_side[right_side >= upper_zone_floor]
    if recovered.empty:
        # Cup hasn't rounded back up into its upper half yet — no handle
        # can have formed; this is a "cup still forming" case, not an error.
        return None

    # From the first bar that reaches the upper half, scan forward up to
    # CUPHANDLE_MAX_DURATION_WEEKS more bars (or to the end of the series)
    # to find the actual right-side HIGH — the genuine peak before any
    # handle pullback begins — rather than freezing on the very first
    # bar that merely touched the midpoint.
    first_recovery_idx = weekly.index.get_loc(recovered.index[0])
    right_high_search_end = min(first_recovery_idx + CUPHANDLE_MAX_DURATION_WEEKS, n)
    right_high_window = w_high.iloc[first_recovery_idx:right_high_search_end]
    cup_end_idx = first_recovery_idx + int(right_high_window.values.argmax())
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
        hard_rejections.append("Handle has not formed yet (cup just completed)")
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
    hard_rejections.extend(shape_reasons)

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
    # ── FIXED 2026-06-24: separate HARD rejections (the pattern is
    # genuinely invalid — wrong shape, handle in the wrong place, outside
    # even O'Neil's stated outer bounds) from SOFT warnings (the pattern
    # IS valid by O'Neil's own published tolerance, just not in his
    # "ideal" zone). Previously every single item below went into one
    # list and ANY of them flipped is_valid to False — meaning a cup at
    # 49.5% depth (within O'Neil's documented "up to 50% in severe
    # corrections" allowance) was rejected with the exact same severity
    # as a cup with a V-shaped crash bottom or a handle that dropped into
    # the lower half. That's not what the published rules say: depth
    # 33-50%, duration slightly past the ideal window, and a prior
    # uptrend a few points under 30% are all things O'Neil's own
    # research treats as "less reliable, still valid" — not disqualifying.
    # Confirmed against a real-world case (a young, volatile NSE solar
    # stock) where this distinction is exactly what separates "this
    # genuinely isn't a cup and handle" from "this is a real, tradeable,
    # if slightly imperfect, cup and handle that the old logic discarded."
    hard_rejections: list[str] = []
    soft_warnings: list[str] = []

    if cup_duration_weeks < CUPHANDLE_MIN_DURATION_WEEKS:
        hard_rejections.append(
            f"Cup duration {cup_duration_weeks}w below O'Neil's {CUPHANDLE_MIN_DURATION_WEEKS}w minimum"
        )
    if cup_duration_weeks > CUPHANDLE_MAX_DURATION_WEEKS:
        # Soft: O'Neil's bound is itself described as a typical outer
        # limit, not an absolute law of physics — a pattern a few weeks
        # over is still worth surfacing, just flagged as non-ideal.
        soft_warnings.append(
            f"Cup duration {cup_duration_weeks}w exceeds O'Neil's typical {CUPHANDLE_MAX_DURATION_WEEKS}w outer bound"
        )

    if not (CUPHANDLE_MIN_DEPTH_PCT <= cup_depth_pct <= CUPHANDLE_MAX_DEPTH_PCT_BEAR):
        # Hard: outside even the documented "severe correction" allowance.
        hard_rejections.append(
            f"Cup depth {cup_depth_pct:.1f}% outside O'Neil's "
            f"{CUPHANDLE_MIN_DEPTH_PCT:.0f}-{CUPHANDLE_MAX_DEPTH_PCT_BEAR:.0f}% range"
        )
    elif cup_depth_pct > CUPHANDLE_MAX_DEPTH_PCT:
        # Soft: within the 33-50% "severe correction" allowance O'Neil
        # himself documented — valid, just not in the ideal 15-25% zone.
        soft_warnings.append(
            f"Cup depth {cup_depth_pct:.1f}% exceeds the normal-market "
            f"ideal of {CUPHANDLE_MAX_DEPTH_PCT:.0f}% — still within O'Neil's "
            f"documented severe-correction allowance, but a lower-confidence base"
        )

    if handle_duration_weeks > 0 and handle_duration_weeks < CUPHANDLE_HANDLE_MIN_WEEKS:
        hard_rejections.append(
            f"Handle duration {handle_duration_weeks}w below {CUPHANDLE_HANDLE_MIN_WEEKS}w minimum"
        )
    if handle_depth_pct > CUPHANDLE_HANDLE_MAX_DEPTH_PCT:
        hard_rejections.append(
            f"Handle depth {handle_depth_pct:.1f}% exceeds "
            f"{CUPHANDLE_HANDLE_MAX_DEPTH_PCT:.0f}% maximum"
        )
    if not handle_in_upper_zone:
        hard_rejections.append("Handle dropped into lower half of the cup (invalidates the pattern)")
    if not cup_shape_ok:
        hard_rejections.append("Cup shape failed rounded-bottom validation (looks V-shaped or irregular)")

    if not prior_uptrend_ok:
        # Soft: a weaker-than-ideal prior run is a quality signal, not a
        # disqualifier — O'Neil's emphasis is directional (stronger prior
        # leadership is better), not a hard binary gate in his own writing.
        soft_warnings.append(
            f"Prior uptrend only {prior_uptrend_pct:.1f}% "
            f"(O'Neil's stated guideline is ≥{CUPHANDLE_PRIOR_UPTREND_MIN_PCT:.0f}%, "
            f"but this is treated as a quality signal, not a hard gate)"
        )

    rejection_reasons = hard_rejections + soft_warnings
    is_valid = len(hard_rejections) == 0

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
