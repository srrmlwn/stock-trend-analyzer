"""Integration tests wiring together real modules end-to-end."""

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from src.backtest.engine import run_backtest
from src.rules.parser import load_rules
from src.scanner.scanner import run_scan_on_tickers


@pytest.mark.integration  # type: ignore[untyped-decorator]
def test_rules_yaml_is_valid() -> None:
    """Assert config/rules.yaml parses without errors."""
    rules = load_rules("config/rules.yaml")
    assert len(rules) > 0
    for rule in rules:
        assert rule.name
        assert rule.type in ("buy", "sell")
        assert rule.conditions is not None


@pytest.mark.integration  # type: ignore[untyped-decorator]
def test_settings_yaml_is_valid() -> None:
    """Assert config/settings.yaml parses without errors."""
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    assert "universe" in settings
    assert "data" in settings
    assert "scanner" in settings
    assert "backtest" in settings
    assert "email" in settings


@pytest.mark.integration  # type: ignore[untyped-decorator]
def test_full_scan_pipeline_dry_run() -> None:
    """Run full pipeline on [AAPL, MSFT, NVDA] with mocked data. Assert no exceptions."""
    # Use mocked yfinance to avoid network calls but exercise full code path
    rules = load_rules("config/rules.yaml")
    settings = {
        "data": {
            "period_years": 1,
            "cache_ttl_hours": 24,
            "retry_attempts": 3,
            "retry_backoff_seconds": 1,
        },
        "universe": {"cache_ttl_hours": 168},
    }

    # Build realistic 252-row synthetic OHLCV DataFrames
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "NVDA"]

    def make_ohlcv(n: int = 252) -> pd.DataFrame:
        dates = pd.date_range(end=date.today(), periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
        close = np.maximum(close, 10.0)
        high = close * 1.01
        low = close * 0.99
        open_ = close * (1 + np.random.randn(n) * 0.003)
        volume = np.random.randint(500_000, 5_000_000, n).astype(float)
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
        df.index.name = "Date"
        return df

    mock_ohlcv = {t: make_ohlcv() for t in tickers}
    mock_fundamentals = {
        t: {"pe_ratio": 25.0, "market_cap": 2e12, "avg_volume": 50_000_000, "price": 150.0}
        for t in tickers
    }

    with (
        patch("src.scanner.scanner.fetch_ohlcv", return_value=mock_ohlcv),
        patch("src.scanner.scanner.fetch_fundamentals", return_value=mock_fundamentals),
        patch("src.scanner.scanner.get_universe", return_value=tickers),
    ):
        signals = run_scan_on_tickers(tickers, rules, settings)

    # No exception = pass. Signals may or may not fire depending on synthetic data.
    assert isinstance(signals, list)
    for signal in signals:
        assert signal.ticker in tickers
        assert signal.action in ("BUY", "SELL")
        assert signal.rule_name
        assert signal.price > 0


@pytest.mark.integration  # type: ignore[untyped-decorator]
def test_backtest_on_small_universe() -> None:
    """Run backtest on [AAPL, MSFT] with mocked data for 2020-2022.

    Assert result fields within sane ranges.
    """
    rules = load_rules("config/rules.yaml")

    np.random.seed(123)

    def make_ohlcv(n: int = 504) -> pd.DataFrame:  # ~2 years
        dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
        close = np.maximum(close, 10.0)
        high = close * 1.01
        low = close * 0.99
        open_ = close * (1 + np.random.randn(n) * 0.003)
        volume = np.random.randint(500_000, 5_000_000, n).astype(float)
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
        df.index.name = "Date"
        return df

    tickers = ["AAPL", "MSFT"]
    mock_ohlcv = {t: make_ohlcv() for t in tickers}
    mock_ohlcv["SPY"] = make_ohlcv()

    with patch("src.backtest.engine.fetch_ohlcv", return_value=mock_ohlcv):
        result = run_backtest(
            tickers=tickers,
            rules=rules,
            start_date="2020-01-01",
            end_date="2022-12-31",
            position_size_usd=2000.0,
            initial_capital=50_000.0,
        )

    # Sane range checks
    assert -100.0 <= result.total_return_pct <= 1000.0
    assert -100.0 <= result.benchmark_return_pct <= 1000.0
    assert 0.0 <= result.win_rate_pct <= 100.0
    assert result.max_drawdown_pct <= 0.0
    assert result.total_trades >= 0
    assert result.avg_holding_days >= 0.0
    assert isinstance(result.trades, pd.DataFrame)


@pytest.mark.integration  # type: ignore[untyped-decorator]
def test_signal_fields_are_complete() -> None:
    """Ensure signals from the pipeline have all required fields populated."""
    rules = load_rules("config/rules.yaml")
    settings = {
        "data": {
            "period_years": 1,
            "cache_ttl_hours": 24,
            "retry_attempts": 3,
            "retry_backoff_seconds": 1,
        },
    }

    np.random.seed(7)
    n = 252
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    # Construct data that guarantees RSI oversold (low, trending down)
    close = np.linspace(100, 60, n) + np.random.randn(n) * 0.5
    close = np.maximum(close, 10.0)
    df = pd.DataFrame(
        {
            "Open": close * 1.001,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.ones(n) * 2_000_000,
        },
        index=dates,
    )
    df.index.name = "Date"

    mock_ohlcv = {"TSLA": df}
    mock_fundamentals = {
        "TSLA": {
            "pe_ratio": 20.0,
            "market_cap": 5e11,
            "avg_volume": 10_000_000,
            "price": 60.0,
        }
    }

    with (
        patch("src.scanner.scanner.fetch_ohlcv", return_value=mock_ohlcv),
        patch("src.scanner.scanner.fetch_fundamentals", return_value=mock_fundamentals),
    ):
        signals = run_scan_on_tickers(["TSLA"], rules, settings)

    for signal in signals:
        assert signal.ticker == "TSLA"
        assert signal.action in ("BUY", "SELL")
        assert signal.rule_name != ""
        assert signal.reason != ""
        assert signal.price > 0
        assert signal.triggered_at is not None


@pytest.mark.integration  # type: ignore[untyped-decorator]
def test_email_report_builds_without_error() -> None:
    """Build an HTML report from real Signal objects. No SMTP call."""
    from src.report.builder import _build_subject, build_report
    from src.rules.models import Signal

    signals = [
        Signal(
            ticker="AAPL",
            action="BUY",
            rule_name="rsi_oversold_with_volume",
            reason="RSI(14)=27.3 < 30, volume ratio=1.8x",
            triggered_at=date(2025, 3, 27),
            price=182.45,
        ),
        Signal(
            ticker="MSFT",
            action="SELL",
            rule_name="death_cross",
            reason="SMA(50) crossed below SMA(200)",
            triggered_at=date(2025, 3, 27),
            price=415.20,
        ),
    ]

    html = build_report(signals, date(2025, 3, 27))
    assert "AAPL" in html
    assert "MSFT" in html
    assert "#d4edda" in html  # green BUY
    assert "#f8d7da" in html  # red SELL
    assert "$182.45" in html

    subject = _build_subject(signals, date(2025, 3, 27))
    assert "2025-03-27" in subject
    assert "2 signals" in subject

    empty_html = build_report([], date(2025, 3, 27))
    assert "No signals" in empty_html
