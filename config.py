"""
NSE Darvas Box Scanner - Configuration
Production-grade configuration for all modules.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DAILY_DIR       = DATA_DIR / "daily"
SIGNALS_DIR     = DATA_DIR / "signals"
WATCHLIST_DIR   = DATA_DIR / "watchlist"
PERF_DIR        = DATA_DIR / "performance"
LOGS_DIR        = BASE_DIR / "logs"
REPORTS_DIR     = BASE_DIR / "reports"

for d in [DAILY_DIR, SIGNALS_DIR, WATCHLIST_DIR, PERF_DIR, LOGS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database files
SIGNALS_DB   = DATA_DIR / "signals.db"
WATCHLIST_DB = DATA_DIR / "watchlist.db"
PERFORMANCE_DB = DATA_DIR / "performance.db"

# ─── Data Sources ─────────────────────────────────────────────────────────────
DATA_SOURCES    = ["yfinance", "stooq"]
PRIMARY_SOURCE  = "yfinance"
FALLBACK_SOURCE = "stooq"

# ─── Universe ─────────────────────────────────────────────────────────────────
NIFTY50_SYMBOL  = "^NSEI"
NIFTY500_SYMBOL = "^CRSLDX"

# Minimum liquidity filters
MIN_AVG_VOLUME   = 50_000
MIN_PRICE        = 10.0
MAX_PRICE        = 100_000.0
MIN_HISTORY_DAYS = 200          # FIX: was 252, many NSE stocks have ~200 days history

# ─── Account Settings ─────────────────────────────────────────────────────────
ACCOUNT_SIZE       = float(os.getenv("ACCOUNT_SIZE", "1_000_000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PCT",     "1.0"))

# ─── Darvas Box Parameters ────────────────────────────────────────────────────
DARVAS_LOOKBACK          = 252
DARVAS_HIGH_LOOKBACK     = 3
DARVAS_MIN_CONSOLIDATION = 5
DARVAS_MAX_WIDTH_PCT     = 40.0
DARVAS_MIN_WIDTH_PCT     = 2.0
DARVAS_BOX_TOUCH_MIN     = 2

# ─── Cup and Handle Parameters (William O'Neil, "How to Make Money in
# Stocks") ──────────────────────────────────────────────────────────────────
# Prior uptrend: stock must already be a leader BEFORE the cup forms.
CUPHANDLE_PRIOR_UPTREND_LOOKBACK_WEEKS = 26      # look back this many weeks for the prior run
CUPHANDLE_PRIOR_UPTREND_MIN_PCT        = 30.0    # minimum prior advance, O'Neil's stated floor

# Cup shape: duration and depth.
CUPHANDLE_MIN_DURATION_WEEKS = 7      # O'Neil's absolute minimum
# CONFIGURABLE 2026-06-25: was a hard-coded 65 with no override. Today's
# market has leaders building longer bases (70/90/120+ week cups have
# been observed) — raise this via the CUPHANDLE_MAX_DURATION_WEEKS_OVERRIDE
# env var (or just edit the value directly) without needing a code change.
# Set to None to use O'Neil's classic 65w bound unchanged.
CUPHANDLE_MAX_DURATION_WEEKS_OVERRIDE = (
    int(os.getenv("CUPHANDLE_MAX_DURATION_WEEKS_OVERRIDE", "0")) or None
)
CUPHANDLE_MAX_DURATION_WEEKS = CUPHANDLE_MAX_DURATION_WEEKS_OVERRIDE or 65
CUPHANDLE_IDEAL_MIN_WEEKS    = 12     # most reliable cups (~3 months+)
CUPHANDLE_IDEAL_MAX_WEEKS    = 26     # most reliable cups (~6 months)
CUPHANDLE_MIN_DEPTH_PCT      = 12.0   # shallow cups are weaker but valid
CUPHANDLE_MAX_DEPTH_PCT      = 33.0   # O'Neil's normal-market ceiling
CUPHANDLE_MAX_DEPTH_PCT_BEAR = 50.0   # allowed in severe market corrections

# ─── Per-Timeframe Detection Bounds — ADDED 2026-06-25 ───────────────────────
# detect_cup_and_handle() now runs THREE independent searches (monthly,
# weekly, daily) instead of only ever searching weekly bars and treating
# monthly as a trend FILTER rather than a true detector. Each timeframe
# needs its own duration/lookback scale — a "65-bar" cup means something
# completely different on monthly bars (65 months ≈ 5.4 years) vs daily
# bars (65 days ≈ 3 months), so the weekly constants above are NOT simply
# reused for the other two timeframes.
#
# WEEKLY uses the constants above directly (this is the original,
# validated O'Neil-faithful scale and is unchanged).
#
# MONTHLY: scaled for genuinely large, multi-year bases. A monthly cup
# needs far fewer bars to span a long real-world duration (e.g. a 3-year
# cup is only ~36 monthly bars but ~156 weekly bars), so the bar-count
# bounds are much smaller in absolute terms even though the real-world
# duration is much larger.
CUPHANDLE_MONTHLY_MIN_DURATION = 4     # ~4 months minimum
CUPHANDLE_MONTHLY_MAX_DURATION = 60    # ~5 years outer bound (configurable like weekly, see below)
CUPHANDLE_MONTHLY_MAX_DURATION_OVERRIDE = (
    int(os.getenv("CUPHANDLE_MONTHLY_MAX_DURATION_OVERRIDE", "0")) or None
)
if CUPHANDLE_MONTHLY_MAX_DURATION_OVERRIDE:
    CUPHANDLE_MONTHLY_MAX_DURATION = CUPHANDLE_MONTHLY_MAX_DURATION_OVERRIDE
CUPHANDLE_MONTHLY_IDEAL_MIN    = 6     # ~6 months
CUPHANDLE_MONTHLY_IDEAL_MAX    = 18    # ~1.5 years
CUPHANDLE_MONTHLY_PRIOR_LOOKBACK = 12  # 12 months of prior-uptrend runway
CUPHANDLE_MONTHLY_HANDLE_MIN   = 1     # ~1 month
CUPHANDLE_MONTHLY_HANDLE_MAX   = 4     # ~4 months

# DAILY: scaled for short, fast-forming bases (often inside a weekly
# handle, per point 2's "nested cups" observation) — these resolve in
# weeks, not months.
CUPHANDLE_DAILY_MIN_DURATION   = 25    # ~5 weeks minimum (5 trading days/week)
CUPHANDLE_DAILY_MAX_DURATION   = 130   # ~26 weeks outer bound
CUPHANDLE_DAILY_IDEAL_MIN      = 40    # ~8 weeks
CUPHANDLE_DAILY_IDEAL_MAX      = 90    # ~18 weeks
CUPHANDLE_DAILY_PRIOR_LOOKBACK = 60    # ~12 weeks of prior-uptrend runway
CUPHANDLE_DAILY_HANDLE_MIN     = 3     # ~3-5 trading days
CUPHANDLE_DAILY_HANDLE_MAX     = 25    # ~5 weeks

# ─── NOT YET IMPLEMENTED — honest TODO from a detailed real-world review
# (2026-06-25) ─────────────────────────────────────────────────────────────
# Point 4 of that review asked for several "elite setup" signals beyond
# basic geometry. Implemented this session: volatility/range contraction
# during the handle (CupHandlePattern.volatility_contraction_pct) and RS
# trend during the cup (point 5 — rs_new_high_before_price,
# rs_during_handle_vs_benchmark_pct). NOT yet implemented, and not scored
# into quality_score at all yet:
#   - Repeated support at the 10-week moving average during the cup/handle
#   - Multi-week volume CONTRACTION pattern (a trend across several weeks,
#     not just "is handle volume below cup volume" which IS already checked)
#   - Tight weekly closes (consecutive small-range closing-price clusters)
#   - General "accumulation before breakout" scoring beyond volume dry-up
# These are real, harder-to-encode signals genuinely worth adding — flagged
# here rather than silently left out, so it's clear what's done vs pending.

# Handle: forms in the upper portion of the cup, drifts down on light volume.
CUPHANDLE_HANDLE_MIN_WEEKS       = 1     # O'Neil: at least 1-2 weeks
CUPHANDLE_HANDLE_MAX_WEEKS       = 12    # beyond this it's a separate base, not a handle
CUPHANDLE_HANDLE_MAX_DEPTH_PCT   = 12.0  # handle decline from its own high
CUPHANDLE_HANDLE_UPPER_ZONE      = 0.5   # handle must form in upper half of the cup
CUPHANDLE_HANDLE_VOL_DRYUP_RATIO = 0.75  # handle avg volume vs cup avg volume (must be lower)

# Breakout / pivot confirmation.
CUPHANDLE_BREAKOUT_VOL_SURGE_PCT = 40.0   # O'Neil: 40-50%+ above average volume on breakout day
CUPHANDLE_BUY_ZONE_PCT           = 5.0    # O'Neil's 5% buy zone above the pivot point

# Liquidity / data sufficiency (reuses MIN_AVG_VOLUME from above).
CUPHANDLE_MIN_HISTORY_WEEKS = 60   # need enough weekly history to find the prior run + cup + handle

# ─── Technical Filters ────────────────────────────────────────────────────────
RSI_PERIOD          = 14
RSI_MIN             = 25.0      # FIX: 25–55 — at box lows RSI can be deeply oversold
RSI_MAX             = 55.0      # FIX: was 50 — widened to catch more reversals

ADX_PERIOD          = 14
ADX_MIN             = 15.0      # FIX: was 20 — NSE mid/small caps have lower ADX
ADX_PREFER_MIN      = 20.0
ADX_PREFER_MAX      = 40.0

ATR_PERIOD          = 14
ATR_STOP_MULTIPLIER = 1.5

EMA_SHORT  = 20
EMA_MID    = 50
EMA_LONG   = 150
EMA_TREND  = 200

VOLUME_MA_PERIOD  = 20
VOLUME_MA_FAST    = 5           # FIX: 5-day avg > 20-day avg = accumulation signal
VOLUME_RATIO_MIN  = 0.8         # FIX: was 1.0 — 80% of avg is fine for entry zone

# Entry zone width (% of box height from box_low)
ENTRY_ZONE_PCT = 0.40           # FIX: was 0.30 — widened to 40% of box

# ─── RS Rating ────────────────────────────────────────────────────────────────
RS_WEIGHTS = {"3m": 0.40, "6m": 0.20, "9m": 0.20, "12m": 0.20}
RS_MIN_PREFERRED = 70           # FIX: was 80 — relaxed for correction markets
RS_MIN_STRONG    = 85           # FIX: was 90

# ─── Scoring Weights (total = 100) ────────────────────────────────────────────
SCORE_WEIGHTS = {
    # ── Rebalanced 2026-06-20 based on real correlation analysis from the
    # bt_2026-06-19_1bd36aa6 backtest run (5,175 historical Darvas trades).
    # Correlation of each raw factor with win/loss outcome (1=win, 0=loss):
    #   rsi_at_entry   : +0.255  (STRONGEST predictor — was the most
    #                              underweighted factor at only 5 pts)
    #   rs_rating      : +0.201  (second strongest — already well-weighted)
    #   sepa_score     : +0.150  (solid — bumped up modestly)
    #   composite_score: +0.153  (the OLD blended score itself, for reference)
    #   box_width_pct  : +0.104  (weak positive, part of box_quality already)
    #   box_age_bars   : +0.083  (weak positive, part of box_quality already)
    #   adx_at_entry   : +0.052  (NEAR-ZERO — bucket win rates were flat
    #                              50-59% across ALL ADX ranges tested;
    #                              was previously overweighted at 10 pts)
    # Weekly/monthly trend weren't directly measurable as raw columns in
    # this trade log (they feed into composite_score, not stored
    # separately) — trimmed modestly to fund the RSI/SEPA increases
    # rather than left untouched on no specific evidence either way.
    #
    # Re-run --backtest-all periodically and revisit this breakdown —
    # these weights should be periodically re-validated against fresh
    # trade data, not treated as permanently fixed.
    "rs_rating":        25,   # unchanged — strong, already correct
    "weekly_trend":     10,   # was 15 — trimmed, no direct evidence either way
    "monthly_trend":     7,   # was 10 — trimmed, no direct evidence either way
    "volume_expansion":  8,   # was 10 — trimmed slightly
    "box_quality":      13,   # was 15 — trimmed slightly (still meaningful)
    "adx_strength":      5,   # was 10 — cut significantly, near-zero correlation
    "rsi_reversal":     20,   # was 5  — RAISED significantly, strongest predictor
    "sepa_score":       12,   # was 10 — raised modestly, solid correlation
}

SCORE_THRESHOLDS = {
    "elite":       90,
    "very_strong": 80,
    "strong":      70,
    "watch":       60,          # band label floor — still used for backtest diagnostics
}

# Minimum composite score required for the LIVE scanner to actually emit
# a signal. This is INTENTIONALLY separate from SCORE_THRESHOLDS["watch"]
# above — that dict defines classification LABELS (Elite/Very Strong/
# Strong/Watch), this defines the live output GATE.
#
# Backtest evidence (2,374-symbol NSE universe, 5,214 historical setups):
#   cutoff 60: 2000 trades, 61.5% win rate, PF 3.08
#   cutoff 70: 1301 trades, 65.0% win rate, PF 3.71
#   cutoff 80: 428 trades,  67.8% win rate, PF 4.50
#   cutoff 85: 120 trades,  75.8% win rate, PF 7.13  <- current default
#   cutoff 88: 37 trades,   86.5% win rate, PF 19.31 (very few signals/year)
# Raise this if you want fewer, higher-conviction signals; lower it
# (down to "watch" = 60) if you want more trade frequency at the cost
# of a lower win rate. Re-run --backtest-all after any change to verify
# the new cutoff still holds up against fresh data.
MIN_SIGNAL_SCORE = 85

# Minimum Cup & Handle quality score for the live scanner to emit a
# signal. Moved here from cup_handle_scanner.py (2026-06-24) for
# consistency with MIN_SIGNAL_SCORE above — both gates now live in one
# place. NOTE: unlike MIN_SIGNAL_SCORE, this has NOT been validated
# against a fresh backtest run since the 2026-06-24 fixes to
# detect_cup_and_handle() (the right-side-recovery threshold fix and the
# hard/soft rejection split) — those changes meaningfully shift what
# range of quality_score values real patterns now produce, so this
# number should be revisited once a backtest run reflects the new
# detection logic. Real-world check: a recently-listed, ~84-week-history
# solar-sector stock with a genuine (if imperfect — deep cup, slightly
# long duration, weaker prior uptrend) cup and handle scored 49.4 under
# the fixed logic — below this gate, so it's surfaced as detected and
# valid internally but not yet emitted as a live signal. Lower this if
# you want more borderline-but-real setups surfaced; that's a deliberate
# tradeoff, not a bug, until backtest evidence says otherwise.
CH_MIN_QUALITY_SCORE = 60.0

# ─── Targets ──────────────────────────────────────────────────────────────────
TARGET1_LABEL = "Box High"
TARGET2_LABEL = "1× Height Above Box"
TARGET3_LABEL = "2× Height Above Box"
TARGET4_LABEL = "ATR Trailing Stop"

# ─── Rate Limiting & Download ─────────────────────────────────────────────────
BATCH_SIZE                = 50
BATCH_DELAY_SECONDS       = 2.0
TIMEOUT_RETRY_WAIT_SEC    = 30      # FIX: short wait for simple timeouts
RATELIMIT_RETRY_WAIT_MIN  = 7       # FIX: longer wait only for 429 rate limits
MAX_RETRIES               = 5
EXPONENTIAL_BASE          = 2

# Legacy alias used in retry code
RETRY_WAIT_MINUTES        = RATELIMIT_RETRY_WAIT_MIN

# ─── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")
TELEGRAM_MAX_MSG   = 4000

# ─── Watchlist Expiry ─────────────────────────────────────────────────────────
WATCHLIST_EXPIRY_DAYS = 30

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"

LOG_FILES = {
    "download":       LOGS_DIR / "download.log",
    "update":         LOGS_DIR / "update.log",
    "scanner":        LOGS_DIR / "scanner.log",
    "error":          LOGS_DIR / "error.log",
    "performance":    LOGS_DIR / "performance.log",
    "signal_tracker": LOGS_DIR / "signal_tracker.log",
}
