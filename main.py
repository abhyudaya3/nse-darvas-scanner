"""
NSE Darvas Box Scanner - Main Orchestrator  (v2)
=================================================
FIXES:
  - Added per-symbol rejection stats to diagnose 0-signal runs
  - Benchmark fallback improved (Nifty50 → Nifty500 → flat)
  - Added --debug-symbol flag for single-stock deep diagnosis
  - Scan errors capped at 200 (not 50) before warning
  - Progress logging every 50 symbols (not 100)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import defaultdict
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
    t0 = time.time()
    log.info("=" * 60)
    log.info("NSE DARVAS BOX SCANNER  |  %s", date.today().isoformat())
    log.info("=" * 60)

    # ── 1. Init DB ────────────────────────────────────────────────────────────
    init_db()

    # ── 2. Universe ───────────────────────────────────────────────────────────
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
        log.error("Download error: %s\n%s", e, traceback.format_exc())

    # ── 4. Load benchmarks ────────────────────────────────────────────────────
    log.info("[3/8] Loading benchmark data …")
    benchmark = _load_best_benchmark()

    # ── 5. Scan ───────────────────────────────────────────────────────────────
    log.info("[4/8] Scanning %d symbols …", len(symbols))
    signals_today: list = []
    errors       = 0
    skip_reasons = defaultdict(int)

    for i, sym in enumerate(symbols, 1):
        if i % 50 == 0:
            log.info(
                "  [%d/%d] signals=%d  errors=%d",
                i, len(symbols), len(signals_today), errors,
            )

        try:
            daily = load_daily(sym)
            if daily is None or daily.empty:
                skip_reasons["no_data"] += 1
                continue
            if len(daily) < 200:
                skip_reasons["short_history"] += 1
                continue

            sig = scan_symbol(sym, daily, benchmark, sector=_get_sector(sym))
            if sig is None:
                skip_reasons["filter_failed"] += 1
                continue

            if signal_exists(sig.signal_id):
                skip_reasons["duplicate"] += 1
                continue

            signals_today.append(sig)

        except KeyboardInterrupt:
            log.info("Interrupted at symbol %d", i)
            break
        except Exception as e:
            errors += 1
            log.debug("Error scanning %s: %s", sym, e)
            if errors > 200:
                log.warning("High error count (%d) – check error.log", errors)

    # ── Skip reason summary (KEY for diagnosing 0-signal runs) ───────────────
    log.info(
        "[4/8] Scan complete: %d signals | %d errors | skip breakdown: %s",
        len(signals_today), errors, dict(skip_reasons),
    )

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

    # ── 8. Effectiveness ──────────────────────────────────────────────────────
    log.info("[7/8] Computing strategy effectiveness …")
    try:
        compute_effectiveness()
    except Exception as e:
        log.error("Effectiveness error: %s", e)

    # ── 9. Report ─────────────────────────────────────────────────────────────
    log.info("[8/8] Generating Excel report …")
    report_path = None
    try:
        report_path = generate_report(signals_today)
    except Exception as e:
        log.error("Report generation failed: %s\n%s", e, traceback.format_exc())

    # ── 10. Telegram ──────────────────────────────────────────────────────────
    if report_path and not args.no_telegram:
        try:
            send_report(signals_today, report_path)
        except Exception as e:
            log.error("Telegram error: %s", e)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(
        "COMPLETE in %.1fs | %d signals | Report: %s",
        elapsed, len(signals_today), report_path or "N/A",
    )
    log.info("=" * 60)
    return 0


# ─── Benchmark loader ─────────────────────────────────────────────────────────

def _load_best_benchmark():
    """Try Nifty50 → Nifty500 → flat fallback."""
    import pandas as pd

    for sym in [NIFTY50_SYMBOL, NIFTY500_SYMBOL]:
        df = load_daily(sym)
        if df is not None and len(df) > 100:
            log.info("Benchmark: %s (%d bars)", sym, len(df))
            return df

    log.warning("No benchmark data — RS Rating will default to 50")
    idx = pd.date_range(end=date.today(), periods=300, freq="B")
    return pd.DataFrame({"Close": [10000.0] * 300}, index=idx)


# ─── Debug single symbol ──────────────────────────────────────────────────────

def _debug_symbol(symbol: str) -> None:
    """Print detailed filter-by-filter diagnosis for one symbol."""
    log.info("=== DEBUG: %s ===", symbol)
    init_db()
    daily = load_daily(symbol)
    if daily is None:
        print(f"No data found for {symbol}")
        return

    print(f"Data rows : {len(daily)}")
    print(f"Date range: {daily.index[0].date()} → {daily.index[-1].date()}")
    print(f"Last close: {daily['Close'].iloc[-1]:.2f}")

    from darvas import get_active_box
    from indicators import rsi as calc_rsi, adx as calc_adx, ema, volume_ratio as vr_fn, higher_highs_higher_lows
    from config import RSI_MIN, RSI_MAX, ADX_MIN, EMA_TREND, VOLUME_RATIO_MIN, VOLUME_MA_FAST, VOLUME_MA_PERIOD, ENTRY_ZONE_PCT

    close, high, low, vol = daily["Close"], daily["High"], daily["Low"], daily["Volume"]

    box = get_active_box(symbol, daily)
    if box is None:
        print("FAIL: No active Darvas box found")
        return
    print(f"Box       : low={box.box_low:.2f}  high={box.box_high:.2f}  age={box.age_bars}  quality={box.quality_score:.1f}")

    cp = float(close.iloc[-1])
    bh = box.box_high - box.box_low
    ez_hi = box.box_low + bh * ENTRY_ZONE_PCT
    in_zone = box.box_low <= cp <= ez_hi
    print(f"Entry zone: [{box.box_low:.2f} – {ez_hi:.2f}]  price={cp:.2f}  IN_ZONE={in_zone}")

    rsi_v   = float(calc_rsi(close).iloc[-1])
    adx_v   = float(calc_adx(high, low, close)["ADX"].iloc[-1])
    ema200  = float(ema(close, EMA_TREND).iloc[-1])
    vf      = float(vol.iloc[-VOLUME_MA_FAST:].mean())
    vs      = float(vol.rolling(VOLUME_MA_PERIOD).mean().iloc[-1])
    vol_rat = vf / vs if vs > 0 else 0
    hh_hl   = higher_highs_higher_lows(high, low, 20)

    print(f"EMA200    : {ema200:.2f}  price_above={cp > ema200}")
    print(f"RSI       : {rsi_v:.1f}  pass={RSI_MIN <= rsi_v <= RSI_MAX}  range=[{RSI_MIN},{RSI_MAX}]")
    print(f"ADX       : {adx_v:.1f}  pass={adx_v >= ADX_MIN}  min={ADX_MIN}")
    print(f"Vol ratio : {vol_rat:.2f}  pass={vol_rat >= VOLUME_RATIO_MIN}  min={VOLUME_RATIO_MIN}")
    print(f"HH-HL     : {hh_hl}")


def _get_sector(symbol: str) -> str:
    return "Unknown"


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE Darvas Box Bottom-of-Box Scanner")
    p.add_argument("--full-refresh",     action="store_true",
                   help="Re-download full history for all symbols")
    p.add_argument("--refresh-universe", action="store_true",
                   help="Force refresh of NSE symbol list")
    p.add_argument("--no-telegram",      action="store_true",
                   help="Skip Telegram notification")
    p.add_argument("--backtest-symbol",  type=str, default=None,
                   help="Run backtest for one symbol (e.g. RELIANCE)")
    p.add_argument("--debug-symbol",     type=str, default=None,
                   help="Print filter-by-filter diagnosis for one symbol")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.debug_symbol:
        sym = args.debug_symbol.upper()
        if not sym.endswith(".NS"):
            sym += ".NS"
        _debug_symbol(sym)
        sys.exit(0)

    if args.backtest_symbol:
        from backtest import backtest_symbol
        sym = args.backtest_symbol.upper()
        if not sym.endswith(".NS"):
            sym += ".NS"
        result = backtest_symbol(sym)
        if result:
            for k, v in result.items():
                print(f"  {k:22s}: {v}")
        else:
            print("No results — insufficient data or no boxes found")
        sys.exit(0)

    sys.exit(main(args))
