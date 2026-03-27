"""Shared pytest fixtures for the stock-trend-analyzer test suite."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Return a synthetic 60-row daily OHLCV DataFrame for a single ticker."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range(end=date.today(), periods=n, freq="B")  # business days
    close = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
    close = np.maximum(close, 1.0)  # ensure positive
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = (np.random.randint(500_000, 5_000_000, n)).astype(float)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def sample_rules_config() -> dict:
    """Return a minimal rules config dict matching the rules.yaml schema."""
    return {
        "rules": [
            {
                "name": "rsi_oversold",
                "type": "buy",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {
                            "indicator": "RSI",
                            "period": 14,
                            "operator": "less_than",
                            "value": 30,
                        }
                    ],
                },
            },
            {
                "name": "rsi_overbought",
                "type": "sell",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {
                            "indicator": "RSI",
                            "period": 14,
                            "operator": "greater_than",
                            "value": 70,
                        }
                    ],
                },
            },
        ]
    }


@pytest.fixture
def sample_fundamentals() -> dict:
    """Return a sample fundamentals dict for a single ticker."""
    return {
        "pe_ratio": 25.0,
        "market_cap": 500_000_000_000,  # $500B
        "avg_volume": 80_000_000,
        "price": 182.45,
    }
