"""Technical indicator calculator for stock price DataFrames.

Computes RSI, MACD, SMA, EMA, volume ratio, and price change percentage using
pandas-ta (when available) or a built-in fallback implementation.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import pandas_ta; fall back to manual implementations
# ---------------------------------------------------------------------------
if importlib.util.find_spec("pandas_ta") is not None:
    try:
        import pandas_ta  # noqa: F401  # type: ignore[import-untyped]

        _HAS_PANDAS_TA = True
        logger.debug("pandas_ta loaded successfully.")
    except Exception:  # pragma: no cover
        _HAS_PANDAS_TA = False
        logger.warning("pandas_ta found but failed to import. Using built-in implementations.")
else:
    _HAS_PANDAS_TA = False
    logger.warning("pandas_ta is not installed. Using built-in indicator implementations.")


# ---------------------------------------------------------------------------
# Internal helpers (manual implementations that match pandas-ta output)
# ---------------------------------------------------------------------------


def _rsi_manual(series: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing (matches pandas-ta output)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Initial averages via simple mean over the first `period` values
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _ema_manual(series: pd.Series, period: int) -> pd.Series:
    """Compute EMA (matches pandas-ta / standard EMA)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _sma_manual(series: pd.Series, period: int) -> pd.Series:
    """Compute SMA."""
    return series.rolling(window=period).mean()


def _macd_manual(
    series: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, histogram, and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, histogram, signal_line


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column to df.

    Parameters
    ----------
    df:
        OHLCV DataFrame with at least a ``Close`` column.
    period:
        Look-back window for RSI calculation (default 14).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added ``RSI_{period}`` column.
    """
    col = f"RSI_{period}"
    if _HAS_PANDAS_TA:
        df.ta.rsi(length=period, append=True)
        # pandas-ta may name it RSI_14 — rename if needed
        if col not in df.columns:
            candidates = [c for c in df.columns if c.upper().startswith("RSI_")]
            if candidates:
                df.rename(columns={candidates[-1]: col}, inplace=True)
    else:
        df[col] = _rsi_manual(df["Close"], period)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD columns to df.

    Adds ``MACD_{fast}_{slow}_{signal}``, ``MACDh_{fast}_{slow}_{signal}``, and
    ``MACDs_{fast}_{slow}_{signal}``.

    Parameters
    ----------
    df:
        OHLCV DataFrame with at least a ``Close`` column.
    fast:
        Fast EMA period (default 12).
    slow:
        Slow EMA period (default 26).
    signal:
        Signal EMA period (default 9).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with added MACD columns.
    """
    macd_col = f"MACD_{fast}_{slow}_{signal}"
    hist_col = f"MACDh_{fast}_{slow}_{signal}"
    sig_col = f"MACDs_{fast}_{slow}_{signal}"

    if _HAS_PANDAS_TA:
        df.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
        # Ensure columns exist with expected names (pandas-ta 0.3.x uses these names)
        for expected in (macd_col, hist_col, sig_col):
            if expected not in df.columns:
                logger.warning("Expected column %s not found after pandas_ta.macd().", expected)
    else:
        macd_line, histogram, signal_line = _macd_manual(df["Close"], fast, slow, signal)
        df[macd_col] = macd_line
        df[hist_col] = histogram
        df[sig_col] = signal_line
    return df


def add_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add SMA column to df.

    Parameters
    ----------
    df:
        OHLCV DataFrame with at least a ``Close`` column.
    period:
        Look-back window for the simple moving average.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added ``SMA_{period}`` column.
    """
    col = f"SMA_{period}"
    if _HAS_PANDAS_TA:
        df.ta.sma(length=period, append=True)
        if col not in df.columns:
            candidates = [c for c in df.columns if c.upper().startswith("SMA_")]
            if candidates:
                df.rename(columns={candidates[-1]: col}, inplace=True)
    else:
        df[col] = _sma_manual(df["Close"], period)
    return df


def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add EMA column to df.

    Parameters
    ----------
    df:
        OHLCV DataFrame with at least a ``Close`` column.
    period:
        Look-back window for the exponential moving average.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added ``EMA_{period}`` column.
    """
    col = f"EMA_{period}"
    if _HAS_PANDAS_TA:
        df.ta.ema(length=period, append=True)
        if col not in df.columns:
            candidates = [c for c in df.columns if c.upper().startswith("EMA_")]
            if candidates:
                df.rename(columns={candidates[-1]: col}, inplace=True)
    else:
        df[col] = _ema_manual(df["Close"], period)
    return df


def add_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add volume ratio column to df.

    The ratio is ``Volume / rolling_mean(Volume, period)``.

    Parameters
    ----------
    df:
        OHLCV DataFrame with at least a ``Volume`` column.
    period:
        Rolling window size for the mean volume (default 20).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added ``volume_ratio_{period}`` column.
    """
    df[f"volume_ratio_{period}"] = df["Volume"] / df["Volume"].rolling(period).mean()
    return df


def add_price_change_pct(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add price change percentage column to df.

    Computes ``(Close - Close.shift(period)) / Close.shift(period) * 100``.

    Parameters
    ----------
    df:
        OHLCV DataFrame with at least a ``Close`` column.
    period:
        Number of periods to look back.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added ``price_change_pct_{period}`` column.
    """
    shifted = df["Close"].shift(period)
    df[f"price_change_pct_{period}"] = (df["Close"] - shifted) / shifted * 100
    return df


# ---------------------------------------------------------------------------
# Config-driven indicator computation
# ---------------------------------------------------------------------------


def _collect_indicator_requests(items: list[Any]) -> list[dict[str, Any]]:
    """Recursively collect indicator request dicts from condition items.

    Each item can be either:
    - a leaf condition: ``{"indicator": "RSI", "period": 14, ...}``
    - a nested group: ``{"operator": "OR", "items": [...]}``
    """
    requests: list[dict[str, Any]] = []
    for item in items:
        if "indicator" in item:
            requests.append(item)
        elif "items" in item:
            requests.extend(_collect_indicator_requests(item["items"]))
    return requests


def add_all_indicators(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Compute only the indicators referenced in the active rules config.

    Walks all rules and their (possibly nested) condition items to discover which
    indicators are needed, then calls the corresponding ``add_*`` function for
    each unique (indicator, params) combination.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    config:
        Rules configuration dict matching the rules.yaml schema.  Expected
        top-level key: ``"rules"``, each rule containing a ``"conditions"``
        dict with an ``"items"`` list.

    Returns
    -------
    pd.DataFrame
        The input DataFrame enriched with only the required indicator columns.
    """
    rules: list[dict[str, Any]] = config.get("rules", [])
    seen: set[tuple[str, ...]] = set()

    for rule in rules:
        conditions = rule.get("conditions", {})
        top_items: list[Any] = conditions.get("items", [])
        indicator_requests = _collect_indicator_requests(top_items)

        for req in indicator_requests:
            indicator: str = req.get("indicator", "").upper()

            if indicator == "RSI":
                period: int = int(req.get("period", 14))
                key: tuple[str, ...] = ("RSI", str(period))
                if key not in seen:
                    seen.add(key)
                    add_rsi(df, period=period)

            elif indicator == "MACD":
                fast: int = int(req.get("fast", 12))
                slow: int = int(req.get("slow", 26))
                signal: int = int(req.get("signal", 9))
                key = ("MACD", str(fast), str(slow), str(signal))
                if key not in seen:
                    seen.add(key)
                    add_macd(df, fast=fast, slow=slow, signal=signal)

            elif indicator == "SMA":
                period = int(req.get("period", 20))
                key = ("SMA", str(period))
                if key not in seen:
                    seen.add(key)
                    add_sma(df, period=period)
                # Also add the comparison SMA if this is a crossover condition
                compare_period = req.get("compare_period")
                if compare_period is not None:
                    cmp_period = int(compare_period)
                    cmp_key = ("SMA", str(cmp_period))
                    if cmp_key not in seen:
                        seen.add(cmp_key)
                        add_sma(df, period=cmp_period)

            elif indicator == "EMA":
                period = int(req.get("period", 20))
                key = ("EMA", str(period))
                if key not in seen:
                    seen.add(key)
                    add_ema(df, period=period)

            elif indicator in ("VOLUME_RATIO", "VOLUME_SPIKE"):
                period = int(req.get("period", req.get("vs_period", 20)))
                key = ("VOLUME_RATIO", str(period))
                if key not in seen:
                    seen.add(key)
                    add_volume_ratio(df, period=period)

            elif indicator == "PRICE_CHANGE_PCT":
                period = int(req.get("period", 1))
                key = ("PRICE_CHANGE_PCT", str(period))
                if key not in seen:
                    seen.add(key)
                    add_price_change_pct(df, period=period)

            else:
                logger.warning("Unknown indicator %r in config — skipping.", indicator)

    return df
