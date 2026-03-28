"""Fetcher module: wraps yfinance with disk caching, retries, and error handling."""

import json
import logging
import os
import pickle
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from tiingo import TiingoClient

    _TIINGO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIINGO_AVAILABLE = False

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours — for OHLCV freshness checks
_FUNDAMENTALS_CACHE_TTL_SECONDS = 90 * 24 * 60 * 60  # 90 days (quarterly)
_RETRY_COUNT = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubled each retry: 2, 4, 8
_FUNDAMENTALS_REQUEST_DELAY = 0.2  # seconds between per-ticker fundamentals calls
_TIINGO_REQUEST_DELAY = 0.1  # seconds between Tiingo calls (free tier: 500 req/hour)


def _is_cache_fresh(path: Path, ttl_seconds: float = _CACHE_TTL_SECONDS) -> bool:
    """Return True if the file at *path* exists and was modified within ttl_seconds.

    Args:
        path: Path to the cache file.
        ttl_seconds: Maximum age in seconds for the cache to be considered fresh.
            Defaults to 24 hours; pass _FUNDAMENTALS_CACHE_TTL_SECONDS for
            quarterly fundamentals.
    """
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_seconds


def _get_last_stored_date(ticker: str, cache_dir: str) -> "date | None":
    """Return the last date stored in the ticker's pkl, or None if no pkl exists.

    Args:
        ticker: Ticker symbol (e.g. "AAPL").
        cache_dir: Directory where pkl files are stored.

    Returns:
        The most recent date in the stored DataFrame index, or None if the
        pkl file does not exist or cannot be read.
    """
    pkl_file = Path(cache_dir) / f"{ticker}.pkl"
    if not pkl_file.exists():
        return None
    try:
        with pkl_file.open("rb") as fh:
            df: pd.DataFrame = pickle.load(fh)  # noqa: S301
        if df.empty:
            return None
        last_ts = df.index.max()
        # Handle both Timestamp and plain date index entries
        if hasattr(last_ts, "date"):
            result: date = last_ts.date()
            return result
        return date.fromisoformat(str(last_ts)[:10])
    except Exception as exc:
        logger.warning("Could not read last stored date for %s: %s", ticker, exc)
        return None


def _fetch_tiingo_since(tickers: list[str], since_date: date) -> "dict[str, pd.DataFrame]":
    """Fetch daily OHLCV for tickers from since_date to today using Tiingo.

    Requires TIINGO_API_KEY in environment. Falls back gracefully if key is absent.
    Returns dict of ticker -> DataFrame with same columns as the pkl store.

    Args:
        tickers: List of ticker symbols to fetch.
        since_date: Earliest date (inclusive) to fetch bars from.

    Returns:
        Mapping of ticker -> OHLCV DataFrame with columns
        Open, High, Low, Close, Volume. Tickers with no data are omitted.
    """
    api_key = os.environ.get("TIINGO_API_KEY", "")
    if not api_key:
        logger.warning(
            "Tiingo API key not set — falling back to yfinance for incremental fetch. "
            "Set TIINGO_API_KEY in .env for better reliability."
        )
        return _batch_download_since(tickers, since_date)

    if not _TIINGO_AVAILABLE:
        logger.warning(
            "tiingo package not installed — falling back to yfinance for incremental fetch. "
            "Run: pip install 'tiingo>=0.14.0'"
        )
        return _batch_download_since(tickers, since_date)

    client = TiingoClient({"api_key": api_key})
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    col_map = {
        "adjOpen": "Open",
        "adjHigh": "High",
        "adjLow": "Low",
        "adjClose": "Close",
        "adjVolume": "Volume",
        # Also accept unadjusted names as fallback
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            raw_df: pd.DataFrame = client.get_dataframe(
                ticker,
                frequency="daily",
                startDate=since_date,
                endDate=date.today(),
            )
            if raw_df is None or raw_df.empty:
                logger.debug("No new Tiingo data for %s.", ticker)
                continue
            # Rename columns to standard names
            raw_df = raw_df.rename(columns=col_map)
            available = [c for c in ohlcv_cols if c in raw_df.columns]
            if not available:
                logger.warning("Tiingo response for %s has no recognised OHLCV columns.", ticker)
                continue
            df = raw_df[available].copy()
            # Ensure DatetimeIndex is tz-naive for consistency with yfinance output
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            result[ticker] = df
        except Exception as exc:
            logger.warning("Tiingo fetch failed for %s: %s — skipping.", ticker, exc)
        time.sleep(_TIINGO_REQUEST_DELAY)

    return result


def _batch_download_since(tickers: list[str], since_date: date) -> "dict[str, pd.DataFrame]":
    """Fetch daily OHLCV from since_date to today using yfinance.

    Used as a fallback when Tiingo is unavailable. Fetches only from since_date
    rather than a full multi-year history to minimise data transfer.

    Args:
        tickers: List of ticker symbols.
        since_date: Start date (inclusive) for the fetch.

    Returns:
        Mapping of ticker -> DataFrame with OHLCV columns.
    """
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    end_date = date.today() + timedelta(days=1)  # yfinance end date is exclusive

    last_exc: Exception = RuntimeError("Unknown error")
    raw = None
    for attempt in range(_RETRY_COUNT):
        try:
            raw = yf.download(
                tickers,
                start=since_date.isoformat(),
                end=end_date.isoformat(),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception as exc:
            last_exc = exc
            delay = _RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "Attempt %d/%d failed (incremental yfinance download): %s. Retrying in %.0fs.",
                attempt + 1,
                _RETRY_COUNT,
                exc,
                delay,
            )
            time.sleep(delay)
    else:
        logger.warning("Incremental yfinance download failed after retries: %s", last_exc)
        return {}

    if raw is None or raw.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                ticker_df = raw.xs(ticker, axis=1, level=1)
                cols = [c for c in ohlcv_cols if c in ticker_df.columns]
                df = ticker_df[cols].dropna(how="all").copy()
                if not df.empty:
                    result[ticker] = df
            except KeyError:
                logger.debug("No incremental data for %s.", ticker)
    else:
        ticker = tickers[0]
        cols = [c for c in ohlcv_cols if c in raw.columns]
        if cols:
            df = raw[cols].dropna(how="all").copy()
            if not df.empty:
                result[ticker] = df

    return result


def _batch_download(
    tickers: list[str], period_years: int, interval: str
) -> "dict[str, pd.DataFrame]":
    """Download OHLCV for multiple tickers in a single yf.download() call.

    Batching avoids per-ticker HTTP requests and reduces Yahoo Finance rate
    limiting compared to calling yf.Ticker().history() in a loop.

    Args:
        tickers: List of ticker symbols to download.
        period_years: Years of history to fetch.
        interval: Bar interval string accepted by yfinance (e.g. "1d").

    Returns:
        Mapping of ticker -> OHLCV DataFrame. Tickers with no data are omitted.
    """
    period_str = f"{period_years}y"
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]

    last_exc: Exception = RuntimeError("Unknown error")
    for attempt in range(_RETRY_COUNT):
        try:
            raw = yf.download(
                tickers,
                period=period_str,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception as exc:
            last_exc = exc
            delay = _RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "Attempt %d/%d failed (batch download): %s. Retrying in %.0fs.",
                attempt + 1,
                _RETRY_COUNT,
                exc,
                delay,
            )
            time.sleep(delay)
    else:
        raise last_exc

    if raw is None or raw.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        # yf.download returns MultiIndex (metric, ticker) — extract per-ticker slices
        for ticker in tickers:
            try:
                ticker_df = raw.xs(ticker, axis=1, level=1)
                cols = [c for c in ohlcv_cols if c in ticker_df.columns]
                df = ticker_df[cols].dropna(how="all").copy()
                if not df.empty:
                    result[ticker] = df
            except KeyError:
                logger.debug("No data for ticker %s in batch download.", ticker)
    else:
        # Flat column index (single ticker, older yfinance behaviour)
        ticker = tickers[0]
        cols = [c for c in ohlcv_cols if c in raw.columns]
        if cols:
            df = raw[cols].dropna(how="all").copy()
            if not df.empty:
                result[ticker] = df

    return result


def fetch_ohlcv(
    tickers: list[str],
    period_years: int = 10,
    interval: str = "1d",
    cache_dir: str = ".cache/ohlcv",
) -> "dict[str, pd.DataFrame]":
    """Fetch daily OHLCV for each ticker using an append-only local cache.

    For each ticker:
    - If a pkl exists and today's date is already stored: return cached data.
    - If a pkl exists but is not current: fetch only new bars since the last
      stored date via Tiingo (or yfinance fallback) and append them.
    - If no pkl exists: fall back to a full yfinance history download.

    Args:
        tickers: List of ticker symbols to fetch.
        period_years: Years of historical data when doing a full yfinance
            download (used only when no pkl exists).
        interval: Bar interval string accepted by yfinance (default "1d").
        cache_dir: Directory used for pickle cache files.

    Returns:
        Mapping of ticker symbol -> DataFrame with columns
        Open, High, Low, Close, Volume.  Tickers that cannot be fetched
        (invalid symbol, empty result, or persistent network error) are
        omitted from the result.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    result: dict[str, pd.DataFrame] = {}
    needs_full_download: list[str] = []
    needs_incremental: dict[str, date] = {}  # ticker -> last_stored_date

    today = date.today()
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]

    for ticker in tickers:
        last_date = _get_last_stored_date(ticker, cache_dir)

        if last_date is None:
            # No pkl — needs full yfinance download
            needs_full_download.append(ticker)
            continue

        if last_date >= today:
            # Today is already stored — cache hit, load and return
            pkl_file = cache_path / f"{ticker}.pkl"
            logger.debug("Cache up-to-date for %s (last date: %s).", ticker, last_date)
            with pkl_file.open("rb") as fh:
                df: pd.DataFrame = pickle.load(fh)  # noqa: S301
            if isinstance(df.columns, pd.MultiIndex):
                df = (
                    df.xs(ticker, axis=1, level=1)
                    if ticker in df.columns.get_level_values(1)
                    else df
                )
                df.columns = pd.Index([str(c) for c in df.columns])
            result[ticker] = df[[c for c in ohlcv_cols if c in df.columns]]
        else:
            # pkl exists but is stale — needs incremental update
            needs_incremental[ticker] = last_date

    # --- Full downloads (no pkl) ---
    if needs_full_download:
        logger.info("Fetching %d ticker(s) from yfinance (full history).", len(needs_full_download))
        try:
            fetched = _batch_download(needs_full_download, period_years, interval)
        except Exception as exc:
            logger.warning("Batch download failed after retries: %s", exc)
            fetched = {}

        for ticker in needs_full_download:
            df = fetched.get(ticker)
            if df is None or df.empty:
                logger.warning("Empty DataFrame returned for ticker %s — skipping.", ticker)
                continue
            pkl_file = cache_path / f"{ticker}.pkl"
            with pkl_file.open("wb") as fh:
                pickle.dump(df, fh)
            result[ticker] = df

    # --- Incremental updates (pkl exists, append new bars) ---
    if needs_incremental:
        # Find the earliest since_date to use as the Tiingo/yfinance start
        # We fetch each ticker from its own last_date + 1 day, but batch by
        # earliest common date to minimise API calls.
        earliest_since = min(needs_incremental.values()) + timedelta(days=1)
        tickers_to_update = list(needs_incremental.keys())

        logger.info(
            "Fetching incremental updates for %d ticker(s) since %s.",
            len(tickers_to_update),
            earliest_since,
        )

        incremental = _fetch_tiingo_since(tickers_to_update, earliest_since)

        for ticker in tickers_to_update:
            last_date = needs_incremental[ticker]
            pkl_file = cache_path / f"{ticker}.pkl"

            # Load existing data
            with pkl_file.open("rb") as fh:
                existing_df: pd.DataFrame = pickle.load(fh)  # noqa: S301

            new_df = incremental.get(ticker)
            if new_df is not None and not new_df.empty:
                # Filter to only rows after last_date to avoid duplicates
                cutoff = pd.Timestamp(last_date)
                new_df = new_df[new_df.index > cutoff]

            if new_df is None or new_df.empty:
                logger.debug("No new bars for %s (last stored: %s).", ticker, last_date)
                # Return existing data as-is
            else:
                # Align columns before concatenation
                existing_cols = [c for c in ohlcv_cols if c in existing_df.columns]
                new_cols = [c for c in ohlcv_cols if c in new_df.columns]
                shared_cols = [c for c in existing_cols if c in new_cols]
                merged = pd.concat(
                    [existing_df[shared_cols], new_df[shared_cols]], axis=0
                )
                merged = merged[~merged.index.duplicated(keep="last")]
                merged.sort_index(inplace=True)
                existing_df = merged
                with pkl_file.open("wb") as fh:
                    pickle.dump(existing_df, fh)
                logger.debug(
                    "Appended %d new bar(s) to %s.", len(new_df), ticker
                )

            cols = [c for c in ohlcv_cols if c in existing_df.columns]
            result[ticker] = existing_df[cols]

    return result


def _fetch_fundamentals_with_retry(ticker: str) -> "dict[str, object]":
    """Fetch fundamental data from yfinance with up to 3 retries.

    Args:
        ticker: Ticker symbol (e.g. "AAPL").

    Returns:
        Raw ``yf.Ticker.info`` dict.

    Raises:
        Exception: Re-raises the last exception if all retries are exhausted.
    """
    last_exc: Exception = RuntimeError("Unknown error")
    for attempt in range(_RETRY_COUNT):
        try:
            info: dict[str, object] = yf.Ticker(ticker).info
            return info
        except Exception as exc:
            last_exc = exc
            delay = _RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "Attempt %d/%d failed fetching fundamentals for %s: %s. Retrying in %.0fs.",
                attempt + 1,
                _RETRY_COUNT,
                ticker,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc


def fetch_fundamentals(
    tickers: list[str],
    cache_dir: str = ".cache/fundamentals",
) -> "dict[str, dict[str, object]]":
    """Fetch fundamental data (P/E, market cap, avg volume, price) per ticker.

    Uses a 90-day (quarterly) cache TTL so that fundamentals are only re-fetched
    once per quarter rather than on every daily scan.

    Args:
        tickers: List of ticker symbols to fetch.
        cache_dir: Directory used for JSON cache files.

    Returns:
        Mapping of ticker symbol -> dict with keys:
        ``pe_ratio``, ``market_cap``, ``avg_volume``, ``price``.
        Values are ``None`` when the field is not available from yfinance.
        Tickers that fail after retries are omitted from the result.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    result: dict[str, dict[str, object]] = {}
    total = len(tickers)
    fetched_count = 0
    t0 = time.time()
    _LOG_EVERY = max(1, min(50, total // 20))  # log ~20 times across the run

    for i, ticker in enumerate(tickers, start=1):
        json_file = cache_path / f"{ticker}.json"

        if _is_cache_fresh(json_file, ttl_seconds=_FUNDAMENTALS_CACHE_TTL_SECONDS):
            logger.debug("Cache hit for fundamentals %s.", ticker)
            with json_file.open() as fh:
                result[ticker] = json.load(fh)
            continue

        time.sleep(_FUNDAMENTALS_REQUEST_DELAY)
        try:
            info = _fetch_fundamentals_with_retry(ticker)
        except Exception as exc:
            logger.warning(
                "Skipping fundamentals for %s after retries exhausted: %s", ticker, exc
            )
            continue

        fundamentals: dict[str, object] = {
            "pe_ratio": info.get("trailingPE"),
            "market_cap": info.get("marketCap"),
            "avg_volume": info.get("averageVolume"),
            "price": info.get("currentPrice"),
        }

        with json_file.open("w") as fh:
            json.dump(fundamentals, fh)

        result[ticker] = fundamentals
        fetched_count += 1

        if i % _LOG_EVERY == 0 or i == total:
            elapsed = time.time() - t0
            rate = fetched_count / elapsed if elapsed > 0 else 0
            remaining = total - i
            eta_s = remaining / rate if rate > 0 else 0
            logger.info(
                "Fundamentals: %d / %d  |  fetched=%d  |  elapsed=%.0fs  |  ETA=%.0fs",
                i,
                total,
                fetched_count,
                elapsed,
                eta_s,
            )

    return result


def clear_cache(cache_dir: str = ".cache") -> None:
    """Delete all cached data files by removing the cache directory tree.

    Args:
        cache_dir: Root cache directory to remove.  Does nothing if the
            directory does not exist.
    """
    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        logger.info("Cleared cache directory: %s", cache_dir)
    else:
        logger.debug("Cache directory %s does not exist — nothing to clear.", cache_dir)


__all__ = [
    "fetch_ohlcv",
    "fetch_fundamentals",
    "clear_cache",
    "_get_last_stored_date",
    "_fetch_tiingo_since",
]
