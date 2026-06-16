"""
NSE Darvas Box Scanner - Main Orchestrator
===========================================
Full pipeline:
  1. Init database
  2. Fetch/update universe
  3. Download / update daily data
  4. Scan all symbols for bottom-of-box setups
  5. Persist signals + watchlist
  6. Verify existing signals
  7. Compute performance metrics
  8. Generate Excel report
  9. Send Telegram notification

Designed to run unattended via GitHub Actions after market close.
Fully restartable (checkpoint via Parquet files + SQLite).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import date, timedelta

from config import WATCHLIST_EXPIRY_DAYS, NIFTY50_SYMBOL, NIFTY500_SYMBOL
from database import init_db, signal_exists, upsert_signal, upsert_watchlist
from downloader import load_daily, run_download
from logger_utils import get_logger
from report import generate_report
from scanner import scan_symbol
from signal_tracker import compute_effectiveness, verify_all_signals
from telegram_notify import send_report
from universe import fetch_nse_symbols

log = get_logger("scanner")


def main(args: argparse.Namespace) -> int:
    start_time = time.time()
    log.info("=" * 60)
    log.info("NSE DARVAS BOX SCANNER  |  %s", date.today().isoformat())
    log.info("=" * 60)

    # ── 1. Initialise database ────────────────────────────────────────────────
    init_db()

    # ── 2. Fetch universe ─────────────────────────────────────────────────────
    log.info("[1/8] Fetching NSE symbol universe …")
    try:
        symbols = fetch_nse_symbols(force_refresh=args.refresh_universe)
        log.info("Universe: %d symbols", len(symbols))
    except Exception as e:
        log.error("Universe fetch failed: %s", e)
        return 1

    # ── 3. Download / update data ─────────────────────────────────────────────
    log.info("[2/8] Updating price data (full_refresh=%s) …", args.full_refresh)
    try:
        run_download(symbols, full_refresh=args.full_refresh)
    except Exception as e:
        log.error("Download failed: %s\n%s", e, traceback.format_exc())
        # Non-fatal: proceed with existing data

    # ── 4. Load benchmarks ────────────────────────────────────────────────────
    log.info("[3/8] Loading benchmark data …")
    nifty50  = load_daily(NIFTY50_SYMBOL)
    nifty500 = load_daily(NIFTY500_SYMBOL)
    benchmark = nifty50 if nifty50 is not None else nifty500

    if benchmark is None:
        log.error("No benchmark data available – RS Rating will be degraded")
        # Fallback: create a flat benchmark
        import pandas as pd
        benchmark = pd.DataFrame({"Close": [1.0]})

    # ── 5. Scan symbols ───────────────────────────────────────────────────────
    log.info("[4/8] Scanning %d symbols for Darvas bottom-of-box setups …", len(symbols))
    signals_today: list = []
    errors = 0

    for i, sym in enumerate(symbols, 1):
        if i % 100 == 0:
            log.info("  Progress: %d/%d (signals so far: %d)", i, len(symbols), len(signals_today))

        try:
            daily = load_daily(sym)
            if daily is None or daily.empty:
                continue

            sig = scan_symbol(sym, daily, benchmark, sector=_get_sector(sym))
            if sig is None:
                continue

            # Skip if duplicate signal for this symbol today
            if signal_exists(sig.signal_id):
                log.debug("%s already in DB for today – skipping", sym)
                continue

            signals_today.append(sig)

        except KeyboardInterrupt:
            log.info("Interrupted by user at symbol %d", i)
            break
        except Exception as e:
            errors += 1
            log.debug("Error scanning %s: %s", sym, e)
            if errors > 50:
                log.warning("Too many scan errors (%d) – continuing anyway", errors)

    log.info("[4/8] Scan complete: %d signals found (%d errors)", len(signals_today), errors)

    # ── 6. Persist signals ────────────────────────────────────────────────────
    log.info("[5/8] Persisting %d signals …", len(signals_today))
    expiry = date.today() + timedelta(days=WATCHLIST_EXPIRY_DAYS)
    for sig in signals_today:
        upsert_signal(sig)
        upsert_watchlist(sig, expiry_date=expiry)

    # ── 7. Verify existing signals ────────────────────────────────────────────
    log.info("[6/8] Verifying open signals …")
    try:
        verify_all_signals()
    except Exception as e:
        log.error("Signal verification error: %s", e)

    # ── 8. Compute performance ────────────────────────────────────────────────
    log.info("[7/8] Computing strategy effectiveness …")
    try:
        compute_effectiveness()
    except Exception as e:
        log.error("Effectiveness calc error: %s", e)

    # ── 9. Generate report ────────────────────────────────────────────────────
    log.info("[8/8] Generating Excel report …")
    report_path = None
    try:
        report_path = generate_report(signals_today)
    except Exception as e:
        log.error("Report generation failed: %s\n%s", e, traceback.format_exc())

    # ── 10. Send Telegram ─────────────────────────────────────────────────────
    if report_path and not args.no_telegram:
        try:
            send_report(signals_today, report_path)
        except Exception as e:
            log.error("Telegram notification failed: %s", e)

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info("SCAN COMPLETE in %.1fs | %d signals | Report: %s",
             elapsed, len(signals_today), report_path or "N/A")
    log.info("=" * 60)
    return 0


def _get_sector(symbol: str) -> str:
    """Best-effort sector lookup from yfinance (non-blocking, cached)."""
    # Sector fetching is expensive for 2000+ stocks.
    # In production, pre-build a sector map from NSE sector files.
    # Here we return Unknown to keep the scanner fast.
    return "Unknown"


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE Darvas Box Bottom-of-Box Scanner")
    p.add_argument("--full-refresh",      action="store_true",
                   help="Re-download full price history for all symbols")
    p.add_argument("--refresh-universe",  action="store_true",
                   help="Force refresh of NSE symbol list")
    p.add_argument("--no-telegram",       action="store_true",
                   help="Skip Telegram notification")
    p.add_argument("--backtest-symbol",   type=str, default=None,
                   help="Run backtest for a specific symbol and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.backtest_symbol:
        from backtest import backtest_symbol
        result = backtest_symbol(args.backtest_symbol.upper() + ".NS")
        if result:
            for k, v in result.items():
                print(f"  {k:20s}: {v}")
        else:
            print("No backtest results (insufficient data or no boxes)")
        sys.exit(0)

    sys.exit(main(args))
