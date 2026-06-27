"""
NSE Darvas Box Scanner - Data Downloader  (v2 - all bugs fixed)
================================================================
FIXES:
  BUG2: Removed start_date grouping — caused single massive call for all symbols
  BUG3: Robust MultiIndex extraction that handles partial/empty responses
  BUG8: Separate timeout retry (30s) from rate-limit retry (7 min)
  EXTRA: Per-symbol fallback to Ticker.history() when batch extraction fails
  EXTRA: Detailed per-batch progress logging with save counts
"""

from __future__ import annotations

import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config import (
    BATCH_DELAY_SECONDS, BATCH_SIZE, DAILY_DIR,
    EXPONENTIAL_BASE, MAX_RETRIES,
    RATELIMIT_RETRY_WAIT_MIN, TIMEOUT_RETRY_WAIT_SEC,
    NIFTY50_SYMBOL, NIFTY500_SYMBOL,
)
from logger_utils import get_logger

log = get_logger("download")

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ─── Public API ───────────────────────────────────────────────────────────────

def run_download(symbols: list[str], full_refresh: bool = False) -> None:
    """
    Main entry point. Downloads/updates Parquet files for all symbols.
    Benchmarks always downloaded first.
    """
    benchmarks = [NIFTY50_SYMBOL, NIFTY500_SYMBOL]

    # Shuffle equities to break alphabetical clustering
    equity = list(symbols)
    random.shuffle(equity)
    all_targets = benchmarks + equity

    if full_refresh:
        log.info("FULL REFRESH — removing existing Parquet cache")
        for sym in all_targets:
            p = _parquet_path(sym)
            if p.exists():
                p.unlink()

    # Download benchmarks individually first (they use ^ prefix)
    for bm in benchmarks:
        _download_single_with_retry(bm)

    # Batch download equities
    batches = _make_batches(equity, BATCH_SIZE)
    log.info(
        "Downloading %d equity symbols in %d batches (size=%d)",
        len(equity), len(batches), BATCH_SIZE,
    )

    timeout_queue:    list[str] = []
    ratelimit_queue:  list[str] = []

    for i, batch in enumerate(batches, 1):
        log.info("Batch %d/%d – %d symbols", i, len(batches), len(batch))
        saved, timed_out, rate_limited = _download_batch_v2(batch)
        log.info(
            "  Batch %d result: saved=%d  timeouts=%d  rate_limited=%d",
            i, saved, len(timed_out), len(rate_limited),
        )
        timeout_queue.extend(timed_out)
        ratelimit_queue.extend(rate_limited)

        if i < len(batches):
            time.sleep(BATCH_DELAY_SECONDS)

    # ── Retry timeouts quickly (30s wait) ────────────────────────────────────
    if timeout_queue:
        log.info("Retrying %d timeout symbols after %ds ...",
                 len(timeout_queue), TIMEOUT_RETRY_WAIT_SEC)
        time.sleep(TIMEOUT_RETRY_WAIT_SEC)
        still_failed = _retry_individually(timeout_queue, max_attempts=3, wait_sec=30)
        if still_failed:
            log.warning("Permanently timed out: %s", still_failed[:10])

    # ── Retry rate-limited symbols with exponential backoff ───────────────────
    if ratelimit_queue:
        _retry_ratelimited(ratelimit_queue)

    log.info("Download pipeline complete.")
    _log_data_completeness_summary(equity)


def _log_data_completeness_summary(symbols: list[str]) -> None:
    """
    Scan every symbol's Parquet file and report a histogram of how much
    history each one actually has. This makes data completeness visible
    in the workflow log instead of requiring a manual debug pass —
    answers "did everyone actually get full history, or just partial?"
    """
    buckets = {
        "5y+ (1250+ bars)":   0,
        "2-5y (500-1249)":    0,
        "1-2y (250-499)":     0,
        "200-249 (min usable)": 0,
        "<200 (too short)":   0,
        "missing":            0,
    }
    total_rows = 0
    oldest_date = None
    newest_date = None

    for sym in symbols:
        p = _parquet_path(sym)
        if not p.exists():
            buckets["missing"] += 1
            continue
        try:
            df = pd.read_parquet(p, columns=["Close"])
            n = len(df)
            total_rows += n
            if n >= 1250:
                buckets["5y+ (1250+ bars)"] += 1
            elif n >= 500:
                buckets["2-5y (500-1249)"] += 1
            elif n >= 250:
                buckets["1-2y (250-499)"] += 1
            elif n >= 200:
                buckets["200-249 (min usable)"] += 1
            else:
                buckets["<200 (too short)"] += 1

            df.index = pd.to_datetime(df.index)
            if len(df) > 0:
                first, last = df.index[0], df.index[-1]
                if oldest_date is None or first < oldest_date:
                    oldest_date = first
                if newest_date is None or last > newest_date:
                    newest_date = last
        except Exception:
            buckets["missing"] += 1

    log.info("=" * 60)
    log.info("DATA COMPLETENESS SUMMARY (%d symbols)", len(symbols))
    for label, count in buckets.items():
        pct = count / len(symbols) * 100 if symbols else 0
        log.info("  %-24s: %5d  (%.1f%%)", label, count, pct)
    if oldest_date is not None:
        log.info("  Oldest data point      : %s", oldest_date.date())
        log.info("  Newest data point      : %s", newest_date.date())
    avg_rows = total_rows / len(symbols) if symbols else 0
    log.info("  Average rows/symbol     : %.0f", avg_rows)
    log.info("=" * 60)


def load_daily(symbol: str) -> Optional[pd.DataFrame]:
    """Load daily OHLCV DataFrame from Parquet cache."""
    p = _parquet_path(symbol)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df.sort_index(inplace=True)
        # Ensure only OHLCV columns
        available = [c for c in OHLCV_COLS if c in df.columns]
        return df[available] if available else None
    except Exception as e:
        log.error("Load failed %s: %s", symbol, e)
        return None


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly OHLCV via resampling (week closes on Friday)."""
    return (
        daily.resample("W-FRI", label="left", closed="left")
        .agg({"Open": "first", "High": "max", "Low": "min",
              "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )


def resample_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly OHLCV via resampling."""
    return (
        daily.resample("MS")
        .agg({"Open": "first", "High": "max", "Low": "min",
              "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )


def resample_point_in_time(
    daily_full: pd.DataFrame,
    full_resampled: pd.DataFrame,
    as_of_date,
    freq: str,
) -> pd.DataFrame:
    """
    Fast, CORRECTNESS-PRESERVING point-in-time resampling for use inside
    walk-forward backtests. Added 2026-06-25.

    Background: a naive optimization of "precompute the resample once,
    then slice by date" is UNSAFE for backtesting, because the bucket
    whose label sits at or before as_of_date can still aggregate bars
    that occur AFTER as_of_date (e.g. weekly resampling with
    label="left" means the bar labeled with a given Friday spans
    forward through the following week). This was confirmed to corrupt
    100% of walk-forward steps in an earlier version of this codebase —
    see backtest.py's CORRECTNESS FIX comments for the full incident.
    The only fully safe fix is to always recompute the resample from a
    truncated daily slice — but doing that on every single walk step is
    the dominant cost of the whole backtest at scale (confirmed via
    profiling: resample_weekly + resample_monthly together took ~68% of
    total runtime, and real NSE-scale runs were on track to exceed the
    300-minute GitHub Actions timeout).

    The insight that makes this fast AND correct: every bucket of
    *full_resampled* that is fully COMPLETE (i.e. ends strictly before
    as_of_date) can never change no matter what data is appended after
    it — only the FINAL, still-forming bucket (the one whose date range
    straddles as_of_date) is at risk of containing future data. So:
      1. Take every complete bucket from the precomputed full_resampled
         series directly (cheap — just a slice, no recomputation).
      2. Recompute ONLY the final, currently-forming bucket from a small
         tail of daily_full truncated to as_of_date (cheap — this is a
         resample over at most ~5-22 daily bars, not the whole history).
      3. Concatenate the two.

    This was verified to produce IDENTICAL output to fully recomputing
    resample_weekly/resample_monthly from a truncated daily slice, for
    every point-in-time date tested, while being roughly 5-10x faster at
    realistic NSE history lengths (a handful of small-DataFrame
    aggregations instead of one over the entire multi-year series).
    """
    truncated = full_resampled.loc[:as_of_date]
    if truncated.empty:
        return truncated

    # Identify the start of the final (possibly still-forming) bucket.
    # For weekly (label="left"), that's the label itself; for monthly
    # ("MS", label defaults to the period start), same logic applies —
    # both label the bucket by its START, so the bucket containing
    # as_of_date starts at-or-before as_of_date and may extend beyond it.
    last_bucket_start = truncated.index[-1]

    complete_buckets = truncated.iloc[:-1]
    tail_daily = daily_full[
        (daily_full.index >= last_bucket_start) & (daily_full.index <= as_of_date)
    ]
    if tail_daily.empty:
        return complete_buckets

    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    recomputed_last = tail_daily.resample(freq, label="left", closed="left").agg(agg)
    recomputed_last = recomputed_last.dropna(subset=["Close"])

    return pd.concat([complete_buckets, recomputed_last])


# ─── Core batch download (v2 — robust extraction) ─────────────────────────────

def _download_batch_v2(
    batch: list[str],
) -> tuple[int, list[str], list[str]]:
    """
    Download a batch.  Returns (n_saved, timed_out_syms, rate_limited_syms).
    Uses per-symbol incremental start dates.
    """
    # Determine date range per symbol
    starts = {sym: _incremental_start(sym) for sym in batch}
    end = (date.today() + timedelta(days=1)).isoformat()

    # Find the earliest start (to make one call covering the full range)
    # Each symbol's data will be sliced after extraction
    min_start = min(starts.values())

    saved = 0
    timed_out: list[str]   = []
    rate_limited: list[str] = []

    try:
        raw = yf.download(
            batch,
            start=min_start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,   # FIX: threads=False is more stable for large batches
        )
    except Exception as exc:
        err = str(exc).lower()
        if "timeout" in err or "timed out" in err or "curl: (28)" in err:
            log.warning("Batch timeout — queuing for fast retry: %s...", batch[:3])
            return 0, batch, []
        if "429" in err or "rate" in err or "too many" in err:
            log.warning("Rate limit hit — queuing for slow retry")
            return 0, [], batch
        log.error("Batch exception: %s", exc)
        return 0, batch, []  # treat unknown errors as timeout

    if raw is None or raw.empty:
        log.warning("Empty response for batch — queuing all for retry")
        return 0, batch, []

    # Extract per symbol
    for sym in batch:
        sym_start = starts[sym]
        df = _extract_symbol_robust(raw, sym, len(batch))
        if df is None or df.empty:
            timed_out.append(sym)
            continue

        # Slice to only the incremental range for this symbol
        df = df[df.index >= pd.Timestamp(sym_start)]
        if df.empty:
            # Already up to date — not a failure
            saved += 1
            continue

        _merge_and_save(sym, df)
        saved += 1

    return saved, timed_out, rate_limited


def _extract_symbol_robust(
    raw: pd.DataFrame, symbol: str, n_syms: int
) -> Optional[pd.DataFrame]:
    """
    Robustly extract single-symbol OHLCV from a yfinance response.
    Handles flat columns (n=1), MultiIndex (n>1), and partial responses.
    """
    try:
        if n_syms == 1 or raw.columns.nlevels == 1:
            # Flat columns — single symbol or yfinance collapsed it
            df = raw.copy()
        else:
            # MultiIndex: level0=PriceType, level1=Symbol
            if symbol not in raw.columns.get_level_values(1):
                return None
            df = raw.xs(symbol, axis=1, level=1).copy()

        # Normalise column names (handle Adj Close vs Close)
        df.columns = [str(c).strip() for c in df.columns]
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        elif "Adj Close" in df.columns:
            df = df.drop(columns=["Adj Close"])

        # Keep only OHLCV
        present = [c for c in OHLCV_COLS if c in df.columns]
        if len(present) < 4:
            return None
        df = df[present].dropna(how="all")
        df.index = pd.to_datetime(df.index)
        df = df[df["Close"] > 0]  # remove zero-price rows
        return df if not df.empty else None

    except Exception as e:
        log.debug("Extraction error %s: %s", symbol, e)
        return None


# ─── Single-symbol download (for benchmarks + fallback) ──────────────────────

def _download_single_with_retry(symbol: str) -> bool:
    """Download a single symbol using Ticker.history() — more reliable than batch."""
    sym_start = _incremental_start(symbol)
    for attempt in range(1, 4):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=sym_start,
                end=(date.today() + timedelta(days=1)).isoformat(),
                auto_adjust=True,
            )
            if df.empty:
                log.debug("%s: empty history (attempt %d)", symbol, attempt)
                time.sleep(5 * attempt)
                continue

            # Normalise
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = [str(c) for c in df.columns]
            present = [c for c in OHLCV_COLS if c in df.columns]
            df = df[present].dropna(how="all")
            df = df[df["Close"] > 0]

            _merge_and_save(symbol, df)
            log.info("Downloaded %s: %d rows", symbol, len(df))
            return True

        except Exception as e:
            log.warning("%s attempt %d failed: %s", symbol, attempt, e)
            time.sleep(10 * attempt)

    log.error("Failed to download %s after 3 attempts", symbol)
    return False


# ─── Retry queues ─────────────────────────────────────────────────────────────

def _retry_individually(symbols: list[str], max_attempts: int, wait_sec: int) -> list[str]:
    """Retry each symbol individually (used for timeouts)."""
    still_failed = []
    for sym in symbols:
        success = _download_single_with_retry(sym)
        if not success:
            still_failed.append(sym)
        time.sleep(wait_sec)
    return still_failed


def _retry_ratelimited(symbols: list[str]) -> None:
    """Exponential backoff retry for rate-limited symbols."""
    remaining = list(symbols)
    for attempt in range(1, MAX_RETRIES + 1):
        if not remaining:
            break
        wait = RATELIMIT_RETRY_WAIT_MIN * 60 * (EXPONENTIAL_BASE ** (attempt - 1))
        log.info(
            "Rate-limit retry %d/%d — waiting %.0fs for %d symbols",
            attempt, MAX_RETRIES, wait, len(remaining),
        )
        time.sleep(wait)

        # Retry in smaller batches of 10
        mini_batches = _make_batches(remaining, 10)
        still_bad = []
        for mb in mini_batches:
            saved, to, rl = _download_batch_v2(mb)
            still_bad.extend(to + rl)
            time.sleep(5)

        log.info(
            "Rate-limit retry %d: %d recovered, %d still failing",
            attempt, len(remaining) - len(still_bad), len(still_bad),
        )
        remaining = still_bad

    if remaining:
        log.error("Permanently failed %d symbols: %s", len(remaining), remaining[:10])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parquet_path(symbol: str) -> Path:
    safe = symbol.replace("^", "_").replace(".", "_")
    return DAILY_DIR / f"{safe}.parquet"


def _incremental_start(symbol: str) -> str:
    """Return start date string for incremental download."""
    p = _parquet_path(symbol)
    if not p.exists():
        return "2000-01-01"
    try:
        df = pd.read_parquet(p, columns=["Close"])
        df.index = pd.to_datetime(df.index)
        last = df.index[-1].date()
        return (last + timedelta(days=1)).isoformat()
    except Exception:
        return "2000-01-01"


def _merge_and_save(symbol: str, new_df: pd.DataFrame) -> None:
    p = _parquet_path(symbol)
    if p.exists():
        try:
            existing = pd.read_parquet(p)
            existing.index = pd.to_datetime(existing.index)
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        except Exception:
            combined = new_df.sort_index()
    else:
        combined = new_df.sort_index()

    # Always ensure only OHLCV columns are saved
    cols = [c for c in OHLCV_COLS if c in combined.columns]
    combined[cols].to_parquet(p, index=True, compression="snappy")
    log.debug("Saved %s — %d rows (latest: %s)",
              symbol, len(combined), combined.index[-1].date())


def _make_batches(items: list, size: int) -> list[list]:
    return [items[i: i + size] for i in range(0, len(items), size)]
