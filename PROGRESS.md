# Refactor: Local-First Data Architecture

## Goal
Replace the yfinance-heavy pipeline with a local-first data store that eliminates
redundant downloads, rate limiting, and unnecessary API calls.

## Architecture

### Local Data Store
```
.cache/ohlcv/{TICKER}.pkl        ← permanent, append-only OHLCV DataFrame
.cache/fundamentals/{TICKER}.json ← P/E, market cap, shares_outstanding (quarterly TTL)
.cache/universe.json              ← filtered ticker list (weekly TTL)
```

### Data Sources (after refactor)
| Source | Purpose | Frequency |
|--------|---------|-----------|
| Kaggle (svaningelgem) | Bootstrap NYSE + NASDAQ OHLCV history | One-time |
| SimFin bulk CSV | Bootstrap additional US stock OHLCV | One-time |
| Tiingo API | Daily incremental OHLCV (1 bar per ticker) | Daily |
| yfinance | Fallback for tickers not in Kaggle/SimFin | As needed |
| yfinance `.info` | Shares outstanding, P/E, market cap | Quarterly |

### API Call Reduction
| Call | Before | After |
|------|--------|-------|
| OHLCV per ticker | Full 10yr re-download every 24h | 1-bar append daily |
| Universe price/volume filter | Separate 3-month OHLCV batch download | Read from local store (0 API calls) |
| Fundamentals | Every daily scan | Quarterly only |

---

## Tasks

| # | Task | Owner | Status |
|---|------|-------|--------|
| 1 | `--bootstrap` CLI command + `src/data/bootstrap.py` | agent/task-1-bootstrap | ✅ Complete |
| 2 | Tiingo incremental fetch + quarterly fundamentals TTL | agent/task-2-tiingo | 🔄 In Progress |
| 3 | Universe filter from local store (zero OHLCV API calls) | agent/task-3-universe | ✅ Complete |

---

## Task Details

### Task 1 — Bootstrap
**Files:** `src/data/bootstrap.py` (new), `src/scheduler/main.py`

Downloads Kaggle NYSE + NASDAQ datasets and SimFin bulk CSV, parses into
per-ticker `.cache/ohlcv/{TICKER}.pkl` files. Only processes tickers that
don't already have a local pkl file.

CLI: `python -m src.scheduler.main --bootstrap`

### Task 2 — Tiingo Incremental Fetch
**Files:** `src/data/fetcher.py`, `requirements.txt`

- `fetch_ohlcv()`: if pkl exists, fetch only bars since last stored date via Tiingo;
  if no pkl, fall back to yfinance full history
- `fetch_fundamentals()`: change cache TTL from 24h to 90 days (quarterly)
- Add `TIINGO_API_KEY` to `.env.example`; fall back to yfinance if key absent

### Task 3 — Universe Filter from Local Store
**Files:** `src/universe/builder.py`

- Remove `_ohlcv_prefilter()` (no more batch API calls for price/volume)
- Replace with `_local_ohlcv_prefilter()` that reads existing pkl files to check
  last close price and 60-day avg volume — zero API calls
- Only call `fetch_fundamentals()` for tickers that pass local price/volume filter
