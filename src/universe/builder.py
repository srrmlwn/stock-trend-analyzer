"""Universe builder: fetches S&P 500 + NYSE/NASDAQ listings, applies pre-filters, caches result."""

import json
import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NASDAQ_FTP_URL = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
_OTHER_FTP_URL = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia.

    Returns:
        List of ticker symbols (e.g. ["AAPL", "MSFT", "BRK-B", ...]).
        Returns an empty list if the scrape fails.
    """
    try:
        tables = pd.read_html(
            _SP500_WIKIPEDIA_URL,
            storage_options={"User-Agent": "Mozilla/5.0 (compatible; stock-trend-analyzer/1.0)"},
        )
        df = tables[0]
        symbols: list[str] = df["Symbol"].tolist()
        cleaned: list[str] = []
        for sym in symbols:
            if isinstance(sym, str):
                sym = sym.strip().replace(".", "-")
                cleaned.append(sym)
        logger.info("Fetched %d S&P 500 tickers from Wikipedia.", len(cleaned))
        return cleaned
    except Exception as exc:
        logger.warning("Failed to scrape S&P 500 tickers from Wikipedia: %s", exc)
        return []


def _fetch_ftp_symbols(url: str) -> list[str]:
    """Fetch pipe-delimited symbol file from NASDAQ FTP and return cleaned symbols.

    Args:
        url: FTP URL of the pipe-delimited symbol directory file.

    Returns:
        List of valid ticker symbols, empty list on failure.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
    except Exception:
        # Try urllib for ftp:// scheme since requests may not handle it
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Failed to fetch symbol list from %s: %s", url, exc)
            return []

    symbols: list[str] = []
    for line in StringIO(text):
        line = line.strip()
        if not line or line.startswith("Symbol"):
            continue
        parts = line.split("|")
        if not parts:
            continue
        sym = parts[0].strip()
        # Filter out test symbols: those ending in $ or containing spaces
        if sym.endswith("$") or " " in sym:
            continue
        if sym:
            symbols.append(sym)

    return symbols


def get_nyse_nasdaq_tickers() -> list[str]:
    """Fetch NYSE and NASDAQ listings from NASDAQ FTP symbol directory files.

    Attempts to download pipe-delimited files from NASDAQ FTP.
    Returns an empty list with a warning log if both fetches fail.

    Returns:
        Deduplicated list of ticker symbols from NASDAQ and other (NYSE) listings.
    """
    nasdaq_symbols = _fetch_ftp_symbols(_NASDAQ_FTP_URL)
    other_symbols = _fetch_ftp_symbols(_OTHER_FTP_URL)

    if not nasdaq_symbols and not other_symbols:
        logger.warning(
            "Failed to fetch NYSE/NASDAQ listings from NASDAQ FTP. Returning empty list."
        )
        return []

    combined = list(dict.fromkeys(nasdaq_symbols + other_symbols))
    logger.info(
        "Fetched %d NYSE/NASDAQ tickers (%d NASDAQ, %d other).",
        len(combined),
        len(nasdaq_symbols),
        len(other_symbols),
    )
    return combined


def _local_ohlcv_prefilter(
    tickers: list[str],
    min_price: float = 10.0,
    min_avg_volume: int = 500_000,
    cache_dir: str = ".cache/ohlcv",
) -> list[str]:
    """Filter tickers by price and volume using the local OHLCV store.

    Reads existing pkl files. Tickers with no pkl file are kept (not excluded)
    so they can still be bootstrapped. Uses last closing price and 60-day avg volume.
    Makes zero API calls.

    Args:
        tickers: Candidate ticker list.
        min_price: Minimum last closing price.
        min_avg_volume: Minimum 60-day average daily volume.
        cache_dir: Directory containing per-ticker pkl files.

    Returns:
        Filtered ticker list. Tickers without local data are passed through.
    """
    t0 = time.time()
    total = len(tickers)
    logger.info("Local OHLCV pre-filter: checking %d tickers against local store", total)

    store = Path(cache_dir)
    passed: list[str] = []
    failed = 0
    no_data = 0

    for ticker in tickers:
        pkl_path = store / f"{ticker}.pkl"
        if not pkl_path.exists():
            # No local data yet — pass through so it can be bootstrapped later
            no_data += 1
            passed.append(ticker)
            continue

        try:
            df: pd.DataFrame = pd.read_pickle(str(pkl_path))
            if df.empty:
                # Empty file — pass through
                no_data += 1
                passed.append(ticker)
                continue

            # Normalise column names to title-case for consistent access
            df.columns = [str(c).title() for c in df.columns]

            last_close = float(df["Close"].iloc[-1])
            # Use up to last 60 rows for average volume
            avg_vol = float(df["Volume"].iloc[-60:].mean())

            if last_close >= min_price and avg_vol >= min_avg_volume:
                passed.append(ticker)
            else:
                failed += 1
                logger.debug(
                    "Excluding %s: close=%.2f (min %.2f), avg_vol=%.0f (min %d)",
                    ticker,
                    last_close,
                    min_price,
                    avg_vol,
                    min_avg_volume,
                )
        except Exception as exc:
            # Corrupt or unreadable file — pass through conservatively
            logger.warning("Could not read local OHLCV for %s: %s — passing through.", ticker, exc)
            no_data += 1
            passed.append(ticker)

    elapsed = time.time() - t0
    logger.info(
        "Local OHLCV pre-filter: %d passed, %d failed, %d passed through (no local data) — %.1fs",
        len(passed) - no_data,
        failed,
        no_data,
        elapsed,
    )
    return passed


def apply_prefilter(
    tickers: list[str],
    fundamentals: dict[str, dict],  # type: ignore[type-arg]
    min_market_cap_B: float = 1.0,
    min_avg_volume: int = 500_000,
    min_price: float = 10.0,
) -> list[str]:
    """Apply market cap filter to a list of tickers.

    Filters out tickers that do not meet the minimum market cap threshold.
    Tickers with missing/None market_cap values are excluded conservatively.
    Price and volume filtering is handled upstream by _local_ohlcv_prefilter.

    Args:
        tickers: List of ticker symbols to filter.
        fundamentals: Mapping of ticker -> dict with key ``market_cap``.
        min_market_cap_B: Minimum market capitalisation in billions of USD.
        min_avg_volume: Unused; kept for interface compatibility.
        min_price: Unused; kept for interface compatibility.

    Returns:
        Filtered list of tickers passing the market cap criterion.
    """
    passed: list[str] = []
    min_market_cap = min_market_cap_B * 1e9

    for ticker in tickers:
        info = fundamentals.get(ticker)
        if info is None:
            logger.debug("Excluding %s: no fundamentals data.", ticker)
            continue

        market_cap = info.get("market_cap")
        if market_cap is None or market_cap < min_market_cap:
            logger.debug("Excluding %s: market_cap below threshold.", ticker)
            continue

        passed.append(ticker)

    logger.info(
        "Market cap filter: %d / %d tickers passed (market_cap >= %.1fB).",
        len(passed),
        len(tickers),
        min_market_cap_B,
    )
    return passed


def get_universe(
    force_refresh: bool = False,
    cache_path: str = ".cache/universe.json",
    cache_ttl_hours: int = 168,
    include_nyse_nasdaq: bool = False,
) -> list[str]:
    """Return the filtered stock universe, using a weekly cache.

    Loads from cache if the file exists and is younger than ``cache_ttl_hours``
    (default 168 h = 7 days) and ``force_refresh`` is False. Otherwise
    rebuilds by:

    1. Fetching S&P 500 tickers (and NYSE/NASDAQ if ``include_nyse_nasdaq``).
    2. Running a local OHLCV pre-filter (price + volume from pkl files, 0 API calls).
    3. Fetching fundamentals only for survivors.
    4. Applying the market-cap filter.

    Args:
        force_refresh: When True, bypass the cache and always re-fetch.
        cache_path: Path to the JSON cache file.
        cache_ttl_hours: Cache time-to-live in hours.
        include_nyse_nasdaq: When True, merge NYSE/NASDAQ listings (~12k tickers)
            into the pool. Rebuild takes ~15–20 min but runs only once a week.

    Returns:
        Filtered list of ticker symbols.
    """
    from src.data.fetcher import fetch_fundamentals

    cache = Path(cache_path)
    cache_ttl_seconds = cache_ttl_hours * 3600

    if not force_refresh and cache.exists():
        age = time.time() - cache.stat().st_mtime
        if age < cache_ttl_seconds:
            logger.info(
                "Loading universe from cache: %s (age %.1fh).", cache_path, age / 3600
            )
            with cache.open() as fh:
                data: list[str] = json.load(fh)
            return data

    logger.info("Rebuilding universe (force_refresh=%s).", force_refresh)

    sp500 = get_sp500_tickers()

    if include_nyse_nasdaq:
        nyse_nasdaq = get_nyse_nasdaq_tickers()
        all_tickers = list(dict.fromkeys(sp500 + nyse_nasdaq))
        logger.info(
            "Ticker pool: %d unique tickers (%d S&P 500, %d NYSE/NASDAQ).",
            len(all_tickers),
            len(sp500),
            len(nyse_nasdaq),
        )
    else:
        if not sp500:
            logger.warning(
                "S&P 500 fetch failed and include_nyse_nasdaq=False — universe will be empty."
            )
        all_tickers = sp500
        logger.info("Ticker pool: %d S&P 500 tickers.", len(all_tickers))

    if not all_tickers:
        return []

    settings = _load_universe_settings()
    min_price: float = settings.get("min_price", 10.0)
    min_avg_volume: int = int(settings.get("min_avg_volume", 500_000))
    min_market_cap_B: float = settings.get("min_market_cap_B", 1.0)

    t_start = time.time()

    # Stage 1: local OHLCV pre-filter (price + volume from pkl files) — 0 API calls
    logger.info(
        "Stage 1/3: local OHLCV pre-filter (price + volume, 0 API calls) for %d tickers.",
        len(all_tickers),
    )
    ohlcv_passed = _local_ohlcv_prefilter(
        all_tickers, min_price=min_price, min_avg_volume=min_avg_volume
    )
    logger.info(
        "Stage 1/3 complete in %.0fs — %d tickers remain.", time.time() - t_start, len(ohlcv_passed)
    )

    # Stage 2: fetch fundamentals for survivors — quarterly TTL
    t2 = time.time()
    logger.info("Stage 2/3: fetching fundamentals for %d tickers.", len(ohlcv_passed))
    fundamentals = fetch_fundamentals(ohlcv_passed)
    logger.info("Stage 2/3 complete in %.0fs.", time.time() - t2)

    # Stage 3: market cap filter
    logger.info("Stage 3/3: applying market cap filter (>= %.1fB).", min_market_cap_B)
    filtered = apply_prefilter(ohlcv_passed, fundamentals, min_market_cap_B=min_market_cap_B)
    logger.info(
        "Universe rebuild complete in %.0fs total — %d tickers.",
        time.time() - t_start,
        len(filtered),
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as fh:
        json.dump(filtered, fh, indent=2)
    logger.info("Universe cached to %s (%d tickers).", cache_path, len(filtered))

    return filtered


def _load_universe_settings() -> dict[str, float]:
    """Load universe filter thresholds from config/settings.yaml if available.

    Returns:
        Dict with keys min_price, min_avg_volume, min_market_cap_B.
        Falls back to defaults if file is missing or unparseable.
    """
    try:
        import yaml  # type: ignore[import-untyped]

        with open("config/settings.yaml") as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg.get("universe", {})
    except Exception:
        return {}


__all__ = [
    "get_sp500_tickers",
    "get_nyse_nasdaq_tickers",
    "_local_ohlcv_prefilter",
    "apply_prefilter",
    "get_universe",
]
