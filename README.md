# NSE Darvas Box Scanner — Bottom-of-Box Edition

A production-grade, fully automated Darvas Box scanner for NSE (National Stock Exchange of India) equities. Designed to run **unattended for years** through GitHub Actions, maintain its own historical database, and continuously verify signal performance.

---

## Philosophy

Rather than chasing breakouts (the classic Darvas approach), this system enters **near the bottom of the box** — buying support, not momentum. This is inspired by the observation that stocks with strong RS leaders often return to the lower boundary of a consolidation box before the next leg up, offering a superior risk-reward entry.

---

## Architecture

```
darvas_scanner/
├── main.py              ← Orchestrator (entry point)
├── config.py            ← All tuneable parameters
├── universe.py          ← NSE symbol list (live + cached)
├── downloader.py        ← Batch download, incremental update, Parquet storage
├── indicators.py        ← RSI, ATR, ADX, EMA, volume, RS Rating, SEPA
├── darvas.py            ← Darvas Box detection engine
├── scanner.py           ← Entry logic, multi-TF, composite scoring
├── database.py          ← SQLite persistence layer
├── signal_tracker.py    ← Signal verification + effectiveness + adaptive analysis
├── backtest.py          ← Traditional backtest + forward validation report
├── report.py            ← Professional Excel workbook generator
├── telegram_notify.py   ← Telegram bot integration
├── logger_utils.py      ← Named loggers → dedicated log files
├── requirements.txt
└── .github/workflows/scanner.yml
```

---

## Strategy Logic

### Darvas Box Detection
1. Identify a new 52-week high (the *pivot*).
2. Confirm the **Box High** when no subsequent close exceeds it for 3 bars.
3. Confirm the **Box Low** as the lowest intraday low where no close breaks below for 3 bars.
4. The box must contain at least **5 consolidation bars** and **2 support touches**.
5. Box width: **2% – 40%** of box low.

### Bottom-of-Box Entry Zone
Price must be in the **bottom 30% of the box** (entry zone low = box low, entry zone high = box low + 30% of box height).

### Hard Filters (all must pass)
| Filter | Condition |
|--------|-----------|
| Trend | Price above 200 EMA |
| RSI(14) | 35 – 50 (bullish reversal zone) |
| ADX(14) | > 20 |
| Volume | Current > 20-day average |
| Structure | Higher Highs + Higher Lows over 30 bars |

### Composite Score (100 points)
| Component | Weight |
|-----------|--------|
| RS Rating (O'Neil) | 25 |
| Weekly Trend | 15 |
| Box Quality (age, touches, tightness) | 15 |
| Monthly Trend | 10 |
| Volume Expansion | 10 |
| ADX Strength | 10 |
| SEPA Score (Minervini) | 10 |
| RSI Reversal Quality | 5 |

Signals scoring **≥ 70** are output. **< 70** are silently discarded.

### Targets
- **T1** = Box High
- **T2** = Box High + 1× Box Height
- **T3** = Box High + 2× Box Height
- **Stop Loss** = Box Low − 1.5 × ATR(14)

---

## Setup

### 1. Clone repository
```bash
git clone https://github.com/YOUR_USERNAME/nse-darvas-scanner.git
cd nse-darvas-scanner
pip install -r requirements.txt
```

### 2. Configure secrets (GitHub repository settings → Secrets)
| Secret | Value |
|--------|-------|
| `TELEGRAM_BOT_TOKEN` | Your bot token from @BotFather |
| `TELEGRAM_CHAT_ID`   | Your chat/channel ID |

### 3. Configure variables (optional)
| Variable | Default | Description |
|----------|---------|-------------|
| `ACCOUNT_SIZE` | 1000000 | Account size in INR |
| `RISK_PCT` | 1.0 | Risk per trade (%) |

### 4. First run (local)
```bash
# Download full history + scan
python main.py --refresh-universe

# Dry run (no Telegram)
python main.py --no-telegram
```

### 5. Force full re-download
```bash
python main.py --full-refresh
```

### 6. Backtest a single symbol
```bash
python main.py --backtest-symbol RELIANCE
```

---

## GitHub Actions

The workflow in `.github/workflows/scanner.yml` runs automatically **Monday to Friday at 4:30 PM IST** (30 minutes after NSE close).

**Manual trigger:**  
Go to Actions → NSE Darvas Box Scanner → Run workflow.

**Artifacts uploaded per run:**
- `darvas-report-{N}` – Excel report (retained 90 days)
- `darvas-logs-{N}` – All log files (retained 30 days)

**Cache strategy:**
- Daily Parquet files are cached between runs (avoids re-downloading 2000+ stocks).
- SQLite databases (signals, watchlist) persist across runs via `actions/cache`.
- Incremental downloads only fetch the **missing candles** since last run.

---

## Log Files

| File | Contents |
|------|----------|
| `logs/download.log` | Download progress, retry counts |
| `logs/update.log` | Incremental update details |
| `logs/scanner.log` | Scan progress, signals found |
| `logs/error.log` | All WARNING+ messages |
| `logs/performance.log` | Strategy effectiveness results |
| `logs/signal_tracker.log` | Signal outcome updates |

---

## Excel Report (4 sheets)

1. **Today's Signals** — Sorted by Composite Score (colour-coded by classification)
2. **Active Watchlist** — All pending/active signals being tracked
3. **Performance** — Strategy effectiveness by score band
4. **Adaptive Analysis** — Best-performing RS/ADX/RSI/box-width combinations

---

## Signal Statuses

| Status | Meaning |
|--------|---------|
| Waiting | Generated; entry not yet triggered |
| Active | Entry triggered; monitoring for exit |
| Target 1 Achieved | Price reached box high |
| Target 2 Achieved | Price reached box high + 1× height |
| Target 3 Achieved | Price reached box high + 2× height |
| Stopped Out | Price hit stop loss |
| Expired | Not triggered within 30 days |

---

## Key Parameters (config.py)

```python
DARVAS_MIN_CONSOLIDATION = 5       # bars inside box
DARVAS_MIN_WIDTH_PCT     = 2.0     # minimum box width %
DARVAS_MAX_WIDTH_PCT     = 40.0    # maximum box width %
ATR_STOP_MULTIPLIER      = 1.5     # stop = box_low − 1.5×ATR
RSI_MIN / RSI_MAX        = 35 / 50 # entry RSI range
ADX_MIN                  = 20      # minimum trend strength
BATCH_SIZE               = 50      # symbols per yfinance call
WATCHLIST_EXPIRY_DAYS    = 30      # days before signal expires
```

All parameters can be tuned without touching the logic modules.

---

## Extending the System

- **Add sector data:** Populate `_get_sector()` in `main.py` using an NSE sector CSV.
- **Add earnings filter:** Placeholder in SEPA — hook into NSE earnings calendar API.
- **Add F-score:** Extend `indicators.py` with Piotroski F-Score using quarterly data.
- **Add alternate data source:** Implement `load_daily_stooq()` in `downloader.py`.

---

## Disclaimer

This software is for **educational and research purposes only**.  
It does not constitute financial advice. Trading involves significant risk.  
Always conduct your own analysis before making investment decisions.
