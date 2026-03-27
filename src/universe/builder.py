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
        tables = pd.read_html(_SP500_WIKIPEDIA_URL)
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


def apply_prefilter(
    tickers: list[str],
    fundamentals: dict[str, dict],  # type: ignore[type-arg]
    min_market_cap_B: float = 1.0,
    min_avg_volume: int = 500_000,
    min_price: float = 10.0,
) -> list[str]:
    """Apply pre-filter criteria to a list of tickers.

    Filters out tickers that do not meet minimum market cap, average volume,
    and price thresholds. Tickers with missing/None values are excluded
    conservatively.

    Args:
        tickers: List of ticker symbols to filter.
        fundamentals: Mapping of ticker -> dict with keys ``market_cap``,
            ``avg_volume``, and ``price``.
        min_market_cap_B: Minimum market capitalisation in billions of USD.
        min_avg_volume: Minimum average daily trading volume.
        min_price: Minimum share price in USD.

    Returns:
        Filtered list of tickers passing all criteria.
    """
    passed: list[str] = []
    min_market_cap = min_market_cap_B * 1e9

    for ticker in tickers:
        info = fundamentals.get(ticker)
        if info is None:
            logger.debug("Excluding %s: no fundamentals data.", ticker)
            continue

        market_cap = info.get("market_cap")
        avg_volume = info.get("avg_volume")
        price = info.get("price")

        if market_cap is None or avg_volume is None or price is None:
            logger.debug("Excluding %s: missing fundamental field(s).", ticker)
            continue

        if market_cap < min_market_cap:
            logger.debug(
                "Excluding %s: market_cap %.2fB < %.2fB.",
                ticker,
                market_cap / 1e9,
                min_market_cap_B,
            )
            continue

        if avg_volume < min_avg_volume:
            logger.debug(
                "Excluding %s: avg_volume %d < %d.", ticker, avg_volume, min_avg_volume
            )
            continue

        if price < min_price:
            logger.debug("Excluding %s: price %.2f < %.2f.", ticker, price, min_price)
            continue

        passed.append(ticker)

    logger.info(
        "Pre-filter: %d/%d tickers passed (market_cap>=%.1fB, volume>=%d, price>=%.2f).",
        len(passed),
        len(tickers),
        min_market_cap_B,
        min_avg_volume,
        min_price,
    )
    return passed


def get_universe(
    force_refresh: bool = False,
    cache_path: str = ".cache/universe.json",
    cache_ttl_hours: int = 168,
) -> list[str]:
    """Return the filtered stock universe, using a weekly cache.

    Loads from cache if the file exists and is younger than ``cache_ttl_hours``
    (default 168 h = 7 days) and ``force_refresh`` is False. Otherwise,
    re-fetches tickers from S&P 500 and NYSE/NASDAQ sources, fetches
    fundamentals, applies the pre-filter, and writes the result to cache.

    Args:
        force_refresh: When True, bypass the cache and always re-fetch.
        cache_path: Path to the JSON cache file.
        cache_ttl_hours: Cache time-to-live in hours.

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
    nyse_nasdaq = get_nyse_nasdaq_tickers()

    all_tickers = list(dict.fromkeys(sp500 + nyse_nasdaq))
    logger.info(
        "Combined ticker pool: %d unique tickers (%d S&P500, %d NYSE/NASDAQ).",
        len(all_tickers),
        len(sp500),
        len(nyse_nasdaq),
    )

    fundamentals = fetch_fundamentals(all_tickers)

    filtered = apply_prefilter(all_tickers, fundamentals)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as fh:
        json.dump(filtered, fh, indent=2)
    logger.info("Universe cached to %s (%d tickers).", cache_path, len(filtered))

    return filtered


__all__ = [
    "get_sp500_tickers",
    "get_nyse_nasdaq_tickers",
    "apply_prefilter",
    "get_universe",
]
