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
SIGNALS_DB      = DATA_DIR / "signals.db"
WATCHLIST_DB    = DATA_DIR / "watchlist.db"
PERFORMANCE_DB  = DATA_DIR / "performance.db"

# ─── Data Sources ─────────────────────────────────────────────────────────────
DATA_SOURCES    = ["yfinance", "stooq"]  # Priority order
PRIMARY_SOURCE  = "yfinance"
FALLBACK_SOURCE = "stooq"

# ─── Universe ─────────────────────────────────────────────────────────────────
# NSE indices for benchmarks
NIFTY50_SYMBOL  = "^NSEI"
NIFTY500_SYMBOL = "^CRSLDX"  # NSE 500 proxy (use Nifty 500 TRI if available)

# Minimum liquidity filters
MIN_AVG_VOLUME      = 50_000       # shares/day
MIN_PRICE           = 10.0         # INR
MAX_PRICE           = 100_000.0    # INR
MIN_HISTORY_DAYS    = 252          # 1 year minimum

# ─── Account Settings ─────────────────────────────────────────────────────────
ACCOUNT_SIZE        = float(os.getenv("ACCOUNT_SIZE",  "1_000_000"))   # INR
RISK_PER_TRADE_PCT  = float(os.getenv("RISK_PCT",      "1.0"))         # %

# ─── Darvas Box Parameters ────────────────────────────────────────────────────
DARVAS_LOOKBACK         = 252      # days to scan for box formation
DARVAS_HIGH_LOOKBACK    = 3        # days of equal/lower highs to confirm box high
DARVAS_MIN_CONSOLIDATION= 5        # minimum bars inside box
DARVAS_MAX_WIDTH_PCT    = 40.0     # max box width as % of box low
DARVAS_MIN_WIDTH_PCT    = 2.0      # min box width %
DARVAS_BOX_TOUCH_MIN    = 2        # minimum support touches for valid box

# ─── Technical Filters ────────────────────────────────────────────────────────
RSI_PERIOD          = 14
RSI_MIN             = 35.0
RSI_MAX             = 50.0
ADX_PERIOD          = 14
ADX_MIN             = 20.0
ADX_PREFER_MIN      = 25.0
ADX_PREFER_MAX      = 40.0
ATR_PERIOD          = 14
ATR_STOP_MULTIPLIER = 1.5         # ATR × multiplier below box low for stop loss
EMA_SHORT           = 20
EMA_MID             = 50
EMA_LONG            = 150
EMA_TREND           = 200
VOLUME_MA_PERIOD    = 20
VOLUME_RATIO_MIN    = 1.0

# ─── RS Rating ────────────────────────────────────────────────────────────────
RS_WEIGHTS = {
    "3m":  0.40,
    "6m":  0.20,
    "9m":  0.20,
    "12m": 0.20,
}
RS_MIN_PREFERRED    = 80
RS_MIN_STRONG       = 90

# ─── Scoring Weights (total = 100) ────────────────────────────────────────────
SCORE_WEIGHTS = {
    "rs_rating":        25,
    "weekly_trend":     15,
    "monthly_trend":    10,
    "volume_expansion": 10,
    "box_quality":      15,
    "adx_strength":     10,
    "rsi_reversal":      5,
    "sepa_score":       10,
}

SCORE_THRESHOLDS = {
    "elite":       90,
    "very_strong": 80,
    "strong":      70,
}

# ─── Targets ──────────────────────────────────────────────────────────────────
TARGET1_LABEL = "Box High"
TARGET2_LABEL = "1× Height Above Box"
TARGET3_LABEL = "2× Height Above Box"
TARGET4_LABEL = "ATR Trailing Stop"

# ─── Rate Limiting ────────────────────────────────────────────────────────────
BATCH_SIZE              = 50       # symbols per yfinance batch download
BATCH_DELAY_SECONDS     = 2.0      # seconds between batches
RETRY_WAIT_MINUTES      = 7        # minutes to wait on 429
MAX_RETRIES             = 5
EXPONENTIAL_BASE        = 2        # backoff multiplier

# ─── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID",   "")
TELEGRAM_MAX_MSG    = 4000         # characters before splitting

# ─── Watchlist Expiry ─────────────────────────────────────────────────────────
WATCHLIST_EXPIRY_DAYS = 30         # signal expires if not triggered

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
