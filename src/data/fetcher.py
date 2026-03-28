"""Fetcher module: wraps yfinance with disk caching, retries, and error handling."""

import json
import logging
import pickle
import shutil
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours
_RETRY_COUNT = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubled each retry: 2, 4, 8


def _is_cache_fresh(path: Path) -> bool:
    """Return True if the file at *path* exists and was modified less than 24 h ago."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < _CACHE_TTL_SECONDS


def _batch_download(
    tickers: list[str], period_years: int, interval: str
) -> dict[str, pd.DataFrame]:
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
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for each ticker. Uses disk cache if < 24 h old.

    Cache-fresh tickers are loaded from disk. The remaining tickers are
    fetched in a single batched yf.download() call to avoid rate limiting.

    Args:
        tickers: List of ticker symbols to fetch.
        period_years: Years of historical data to retrieve.
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
    to_fetch: list[str] = []

    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]

    for ticker in tickers:
        pkl_file = cache_path / f"{ticker}.pkl"
        if _is_cache_fresh(pkl_file):
            logger.debug("Cache hit for %s — loading from %s.", ticker, pkl_file)
            with pkl_file.open("rb") as fh:
                df = pickle.load(fh)  # noqa: S301
            # Flatten MultiIndex columns in case cache was written by an older code path
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.get_level_values(1) else df
                df.columns = [str(c) for c in df.columns]
            result[ticker] = df[[c for c in ohlcv_cols if c in df.columns]]
        else:
            to_fetch.append(ticker)

    if not to_fetch:
        return result

    logger.info("Fetching %d ticker(s) from yfinance (batched).", len(to_fetch))
    try:
        fetched = _batch_download(to_fetch, period_years, interval)
    except Exception as exc:
        logger.warning("Batch download failed after retries: %s", exc)
        return result

    for ticker in to_fetch:
        df = fetched.get(ticker)
        if df is None or df.empty:
            logger.warning("Empty DataFrame returned for ticker %s — skipping.", ticker)
            continue
        pkl_file = cache_path / f"{ticker}.pkl"
        with pkl_file.open("wb") as fh:
            pickle.dump(df, fh)
        result[ticker] = df

    return result


def _fetch_fundamentals_with_retry(ticker: str) -> dict[str, object]:
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
) -> dict[str, dict[str, object]]:
    """Fetch fundamental data (P/E, market cap, avg volume, price) per ticker.

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

    for ticker in tickers:
        json_file = cache_path / f"{ticker}.json"

        if _is_cache_fresh(json_file):
            logger.debug("Cache hit for fundamentals %s — loading from %s.", ticker, json_file)
            with json_file.open() as fh:
                result[ticker] = json.load(fh)
            continue

        logger.debug("Cache miss for fundamentals %s — fetching from yfinance.", ticker)
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


__all__ = ["fetch_ohlcv", "fetch_fundamentals", "clear_cache"]
