"""Tests for src/indicators/calculator.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.calculator import (
    add_all_indicators,
    add_ema,
    add_macd,
    add_price_change_pct,
    add_rsi,
    add_sma,
    add_volume_ratio,
)


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------


def test_add_rsi_column_exists(sample_ohlcv: pd.DataFrame) -> None:
    """RSI_14 column must be present after calling add_rsi."""
    df = sample_ohlcv.copy()
    result = add_rsi(df)
    assert "RSI_14" in result.columns


def test_add_rsi_bounded(sample_ohlcv: pd.DataFrame) -> None:
    """All non-NaN RSI values must lie in [0, 100]."""
    df = sample_ohlcv.copy()
    result = add_rsi(df)
    rsi = result["RSI_14"].dropna()
    assert not rsi.empty, "RSI series is entirely NaN"
    assert (rsi >= 0).all(), "RSI value below 0 found"
    assert (rsi <= 100).all(), "RSI value above 100 found"


def test_add_rsi_custom_period(sample_ohlcv: pd.DataFrame) -> None:
    """Custom period produces the correctly named column."""
    df = sample_ohlcv.copy()
    result = add_rsi(df, period=7)
    assert "RSI_7" in result.columns


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------


def test_add_macd_columns_exist(sample_ohlcv: pd.DataFrame) -> None:
    """All three MACD columns must be present after calling add_macd."""
    df = sample_ohlcv.copy()
    result = add_macd(df)
    assert "MACD_12_26_9" in result.columns
    assert "MACDh_12_26_9" in result.columns
    assert "MACDs_12_26_9" in result.columns


def test_add_macd_histogram(sample_ohlcv: pd.DataFrame) -> None:
    """MACDh must equal MACD minus MACDs within floating-point tolerance."""
    df = sample_ohlcv.copy()
    result = add_macd(df)
    mask = result[["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]].notna().all(axis=1)
    valid = result[mask]
    expected = valid["MACD_12_26_9"] - valid["MACDs_12_26_9"]
    actual = valid["MACDh_12_26_9"]
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# SMA tests
# ---------------------------------------------------------------------------


def test_add_sma_column_exists(sample_ohlcv: pd.DataFrame) -> None:
    """SMA_20 column must be present after calling add_sma(period=20)."""
    df = sample_ohlcv.copy()
    result = add_sma(df, period=20)
    assert "SMA_20" in result.columns


def test_add_sma_values(sample_ohlcv: pd.DataFrame) -> None:
    """SMA values must match a manual rolling mean calculation."""
    df = sample_ohlcv.copy()
    result = add_sma(df, period=10)
    expected = df["Close"].rolling(10).mean()
    pd.testing.assert_series_equal(
        result["SMA_10"].dropna(),
        expected.dropna(),
        check_names=False,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------


def test_add_ema_column_exists(sample_ohlcv: pd.DataFrame) -> None:
    """EMA_20 column must be present after calling add_ema(period=20)."""
    df = sample_ohlcv.copy()
    result = add_ema(df, period=20)
    assert "EMA_20" in result.columns


def test_add_ema_values(sample_ohlcv: pd.DataFrame) -> None:
    """EMA values must be finite (non-NaN) after the warm-up period."""
    df = sample_ohlcv.copy()
    result = add_ema(df, period=10)
    non_nan = result["EMA_10"].dropna()
    assert len(non_nan) > 0
    assert np.isfinite(non_nan.values).all()


# ---------------------------------------------------------------------------
# Volume ratio tests
# ---------------------------------------------------------------------------


def test_add_volume_ratio_column_exists(sample_ohlcv: pd.DataFrame) -> None:
    """volume_ratio_20 column must exist after calling add_volume_ratio."""
    df = sample_ohlcv.copy()
    result = add_volume_ratio(df)
    assert "volume_ratio_20" in result.columns


def test_add_volume_ratio_equal_mean(sample_ohlcv: pd.DataFrame) -> None:
    """When volume is constant, volume_ratio must be 1.0 after warm-up period."""
    df = sample_ohlcv.copy()
    period = 5
    df["Volume"] = 1_000_000.0  # constant volume
    result = add_volume_ratio(df, period=period)
    col = f"volume_ratio_{period}"
    post_warmup = result[col].iloc[period:]
    np.testing.assert_allclose(post_warmup.values, 1.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# Price change pct tests
# ---------------------------------------------------------------------------


def test_add_price_change_pct_column_exists(sample_ohlcv: pd.DataFrame) -> None:
    """price_change_pct_5 column must be present after calling add_price_change_pct."""
    df = sample_ohlcv.copy()
    result = add_price_change_pct(df, period=5)
    assert "price_change_pct_5" in result.columns


def test_add_price_change_pct_values(sample_ohlcv: pd.DataFrame) -> None:
    """price_change_pct values must match manual calculation."""
    df = sample_ohlcv.copy()
    period = 3
    result = add_price_change_pct(df, period=period)
    shifted = df["Close"].shift(period)
    expected = (df["Close"] - shifted) / shifted * 100
    pd.testing.assert_series_equal(
        result[f"price_change_pct_{period}"],
        expected,
        check_names=False,
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# add_all_indicators tests
# ---------------------------------------------------------------------------


def test_add_all_indicators_only_computes_referenced(sample_ohlcv: pd.DataFrame) -> None:
    """When config only references RSI, no MACD or SMA columns should be added."""
    df = sample_ohlcv.copy()
    config = {
        "rules": [
            {
                "name": "rsi_signal",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {"indicator": "RSI", "period": 14, "operator": "less_than", "value": 30}
                    ],
                },
            }
        ]
    }
    result = add_all_indicators(df, config)

    assert "RSI_14" in result.columns
    macd_cols = [c for c in result.columns if c.startswith("MACD")]
    sma_cols = [c for c in result.columns if c.startswith("SMA_")]
    assert len(macd_cols) == 0, f"Unexpected MACD columns: {macd_cols}"
    assert len(sma_cols) == 0, f"Unexpected SMA columns: {sma_cols}"


def test_add_all_indicators_handles_nested_or(sample_ohlcv: pd.DataFrame) -> None:
    """Indicators inside a nested OR block must be computed correctly."""
    df = sample_ohlcv.copy()
    config = {
        "rules": [
            {
                "name": "complex_rule",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {
                            "operator": "OR",
                            "items": [
                                {
                                    "indicator": "RSI",
                                    "period": 14,
                                    "operator": "less_than",
                                    "value": 30,
                                },
                                {
                                    "indicator": "SMA",
                                    "period": 20,
                                    "operator": "greater_than",
                                    "value": 50,
                                },
                            ],
                        }
                    ],
                },
            }
        ]
    }
    result = add_all_indicators(df, config)

    assert "RSI_14" in result.columns, "RSI_14 not found after nested OR"
    assert "SMA_20" in result.columns, "SMA_20 not found after nested OR"
    # EMA should NOT be present
    ema_cols = [c for c in result.columns if c.startswith("EMA_")]
    assert len(ema_cols) == 0, f"Unexpected EMA columns: {ema_cols}"


def test_add_all_indicators_deduplicates(sample_ohlcv: pd.DataFrame) -> None:
    """The same indicator referenced in multiple rules is only computed once."""
    df = sample_ohlcv.copy()
    config = {
        "rules": [
            {
                "name": "rule1",
                "conditions": {
                    "operator": "AND",
                    "items": [{"indicator": "RSI", "period": 14, "operator": "lt", "value": 30}],
                },
            },
            {
                "name": "rule2",
                "conditions": {
                    "operator": "AND",
                    "items": [{"indicator": "RSI", "period": 14, "operator": "gt", "value": 70}],
                },
            },
        ]
    }
    result = add_all_indicators(df, config)
    rsi_cols = [c for c in result.columns if c.startswith("RSI_")]
    assert rsi_cols.count("RSI_14") == 1 or len(rsi_cols) == 1


def test_add_all_indicators_empty_config(sample_ohlcv: pd.DataFrame) -> None:
    """An empty config produces no additional columns."""
    df = sample_ohlcv.copy()
    original_cols = set(df.columns)
    result = add_all_indicators(df, {})
    assert set(result.columns) == original_cols


def test_add_all_indicators_multiple_indicators(sample_ohlcv: pd.DataFrame) -> None:
    """Config with RSI and MACD produces both sets of columns."""
    df = sample_ohlcv.copy()
    config = {
        "rules": [
            {
                "name": "multi",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {"indicator": "RSI", "period": 14, "operator": "lt", "value": 30},
                        {"indicator": "MACD", "fast": 12, "slow": 26, "signal": 9},
                    ],
                },
            }
        ]
    }
    result = add_all_indicators(df, config)
    assert "RSI_14" in result.columns
    assert "MACD_12_26_9" in result.columns
    assert "MACDh_12_26_9" in result.columns
    assert "MACDs_12_26_9" in result.columns


def test_add_all_indicators_ema(sample_ohlcv: pd.DataFrame) -> None:
    """Config with EMA produces EMA column via add_all_indicators."""
    df = sample_ohlcv.copy()
    config = {
        "rules": [
            {
                "name": "ema_rule",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {"indicator": "EMA", "period": 20, "operator": "gt", "value": 100}
                    ],
                },
            }
        ]
    }
    result = add_all_indicators(df, config)
    assert "EMA_20" in result.columns


def test_add_all_indicators_volume_ratio(sample_ohlcv: pd.DataFrame) -> None:
    """Config with VOLUME_RATIO produces volume_ratio column via add_all_indicators."""
    df = sample_ohlcv.copy()
    config = {
        "rules": [
            {
                "name": "vol_rule",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {"indicator": "VOLUME_RATIO", "period": 20, "operator": "gt", "value": 1.5}
                    ],
                },
            }
        ]
    }
    result = add_all_indicators(df, config)
    assert "volume_ratio_20" in result.columns


def test_add_all_indicators_price_change_pct(sample_ohlcv: pd.DataFrame) -> None:
    """Config with PRICE_CHANGE_PCT produces price_change_pct column via add_all_indicators."""
    df = sample_ohlcv.copy()
    config = {
        "rules": [
            {
                "name": "pcp_rule",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {
                            "indicator": "PRICE_CHANGE_PCT",
                            "period": 5,
                            "operator": "gt",
                            "value": 2,
                        }
                    ],
                },
            }
        ]
    }
    result = add_all_indicators(df, config)
    assert "price_change_pct_5" in result.columns


def test_add_all_indicators_unknown_indicator(sample_ohlcv: pd.DataFrame) -> None:
    """Unknown indicator in config logs a warning and does not raise."""
    df = sample_ohlcv.copy()
    config = {
        "rules": [
            {
                "name": "unknown_rule",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {"indicator": "BOLLINGER", "period": 20, "operator": "gt", "value": 0}
                    ],
                },
            }
        ]
    }
    original_cols = set(df.columns)
    result = add_all_indicators(df, config)
    # No new columns should be added for an unknown indicator
    assert set(result.columns) == original_cols
