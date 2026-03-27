"""Tests for src/scanner/scanner.py — all external I/O is mocked."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.rules.models import ConditionBlock, ConditionItem, FundamentalFilter, Rule, Signal
from src.scanner.scanner import (
    _enrich_dataframes,
    _load_settings,
    run_scan,
    run_scan_on_tickers,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_sample_df() -> pd.DataFrame:
    """Return a small synthetic OHLCV DataFrame."""
    import numpy as np

    np.random.seed(0)
    n = 60
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": [1_000_000.0] * n,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def _make_sample_rule() -> Rule:
    """Return a minimal buy Rule for testing."""
    condition = ConditionItem(
        indicator="RSI",
        operator="less_than",
        period=14,
        value=30.0,
    )
    block = ConditionBlock(operator="AND", items=[condition])
    return Rule(
        name="rsi_oversold",
        type="buy",
        conditions=block,
        fundamentals=FundamentalFilter(),
    )


def _make_sample_signal() -> Signal:
    """Return a sample Signal for test assertions."""
    return Signal(
        ticker="AAPL",
        action="BUY",
        rule_name="rsi_oversold",
        reason="RSI(14)=28.0 < 30",
        triggered_at=date(2026, 3, 27),
        price=150.0,
    )


SAMPLE_TICKERS = ["AAPL", "MSFT"]
SAMPLE_OHLCV = {"AAPL": _make_sample_df(), "MSFT": _make_sample_df()}
SAMPLE_FUNDAMENTALS = {
    "AAPL": {"pe_ratio": 25.0, "market_cap": 3_000_000_000_000, "avg_volume": 80_000_000, "price": 182.45},
    "MSFT": {"pe_ratio": 30.0, "market_cap": 2_500_000_000_000, "avg_volume": 30_000_000, "price": 415.0},
}
SAMPLE_RULES = [_make_sample_rule()]
SAMPLE_SIGNALS = [_make_sample_signal()]
SAMPLE_SETTINGS = {
    "universe": {"cache_ttl_hours": 168},
    "data": {"period_years": 10},
}


# ---------------------------------------------------------------------------
# test_run_scan_returns_signals
# ---------------------------------------------------------------------------

@patch("src.scanner.scanner.load_rules", return_value=SAMPLE_RULES)
@patch("src.scanner.scanner._load_settings", return_value=SAMPLE_SETTINGS)
@patch("src.scanner.scanner.get_universe", return_value=SAMPLE_TICKERS)
@patch("src.scanner.scanner.fetch_ohlcv", return_value=SAMPLE_OHLCV)
@patch("src.scanner.scanner.fetch_fundamentals", return_value=SAMPLE_FUNDAMENTALS)
@patch("src.scanner.scanner.add_all_indicators", side_effect=lambda df, cfg: df)
@patch("src.scanner.scanner.evaluate_universe", return_value=SAMPLE_SIGNALS)
def test_run_scan_returns_signals(
    mock_eval,
    mock_indicators,
    mock_fundamentals,
    mock_ohlcv,
    mock_universe,
    mock_settings,
    mock_rules,
):
    """Full pipeline runs and returns the expected signals."""
    signals = run_scan(
        config_path="config/rules.yaml",
        settings_path="config/settings.yaml",
        dry_run=False,
    )

    assert signals == SAMPLE_SIGNALS
    mock_rules.assert_called_once_with("config/rules.yaml")
    mock_universe.assert_called_once_with(cache_ttl_hours=168)
    mock_ohlcv.assert_called_once_with(SAMPLE_TICKERS, period_years=10)
    mock_fundamentals.assert_called_once_with(SAMPLE_TICKERS)
    mock_eval.assert_called_once()


# ---------------------------------------------------------------------------
# test_run_scan_dry_run_prints_to_stdout
# ---------------------------------------------------------------------------

@patch("src.scanner.scanner.load_rules", return_value=SAMPLE_RULES)
@patch("src.scanner.scanner._load_settings", return_value=SAMPLE_SETTINGS)
@patch("src.scanner.scanner.get_universe", return_value=SAMPLE_TICKERS)
@patch("src.scanner.scanner.fetch_ohlcv", return_value=SAMPLE_OHLCV)
@patch("src.scanner.scanner.fetch_fundamentals", return_value=SAMPLE_FUNDAMENTALS)
@patch("src.scanner.scanner.add_all_indicators", side_effect=lambda df, cfg: df)
@patch("src.scanner.scanner.evaluate_universe", return_value=SAMPLE_SIGNALS)
def test_run_scan_dry_run_prints_to_stdout(
    mock_eval,
    mock_indicators,
    mock_fundamentals,
    mock_ohlcv,
    mock_universe,
    mock_settings,
    mock_rules,
    capsys,
):
    """dry_run=True should print each signal to stdout."""
    signals = run_scan(
        config_path="config/rules.yaml",
        settings_path="config/settings.yaml",
        dry_run=True,
    )

    captured = capsys.readouterr()
    assert "AAPL" in captured.out
    assert "BUY" in captured.out
    assert "rsi_oversold" in captured.out
    assert len(signals) == 1


# ---------------------------------------------------------------------------
# test_run_scan_on_tickers
# ---------------------------------------------------------------------------

@patch("src.scanner.scanner.fetch_ohlcv", return_value=SAMPLE_OHLCV)
@patch("src.scanner.scanner.fetch_fundamentals", return_value=SAMPLE_FUNDAMENTALS)
@patch("src.scanner.scanner.add_all_indicators", side_effect=lambda df, cfg: df)
@patch("src.scanner.scanner.evaluate_universe", return_value=SAMPLE_SIGNALS)
def test_run_scan_on_tickers(
    mock_eval,
    mock_indicators,
    mock_fundamentals,
    mock_ohlcv,
):
    """Providing explicit tickers bypasses get_universe."""
    with patch("src.scanner.scanner.get_universe") as mock_universe:
        signals = run_scan_on_tickers(
            tickers=SAMPLE_TICKERS,
            rules=SAMPLE_RULES,
            settings=SAMPLE_SETTINGS,
        )

        # get_universe must NOT be called
        mock_universe.assert_not_called()

    mock_ohlcv.assert_called_once_with(SAMPLE_TICKERS, period_years=10)
    mock_fundamentals.assert_called_once_with(SAMPLE_TICKERS)
    mock_eval.assert_called_once()
    assert signals == SAMPLE_SIGNALS


# ---------------------------------------------------------------------------
# test_load_settings_reads_yaml
# ---------------------------------------------------------------------------

def test_load_settings_reads_yaml():
    """_load_settings should correctly parse config/settings.yaml."""
    settings = _load_settings("config/settings.yaml")

    assert isinstance(settings, dict)
    assert "universe" in settings
    assert "data" in settings
    assert settings["universe"]["cache_ttl_hours"] == 168
    assert settings["data"]["period_years"] == 10


def test_load_settings_missing_file():
    """_load_settings should raise FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        _load_settings("nonexistent/path/settings.yaml")


# ---------------------------------------------------------------------------
# test_enrich_dataframes_calls_add_all_indicators
# ---------------------------------------------------------------------------

@patch("src.scanner.scanner.add_all_indicators", side_effect=lambda df, cfg: df)
def test_enrich_dataframes_calls_add_all_indicators(mock_add):
    """_enrich_dataframes must call add_all_indicators once per ticker."""
    ohlcv = {"AAPL": _make_sample_df(), "MSFT": _make_sample_df()}
    rules = SAMPLE_RULES

    result = _enrich_dataframes(ohlcv, rules)

    # Called once for each ticker
    assert mock_add.call_count == 2
    assert set(result.keys()) == {"AAPL", "MSFT"}


@patch("src.scanner.scanner.add_all_indicators", side_effect=lambda df, cfg: df)
def test_enrich_dataframes_empty_universe(mock_add):
    """_enrich_dataframes with empty ohlcv should make no calls and return empty dict."""
    result = _enrich_dataframes({}, SAMPLE_RULES)
    mock_add.assert_not_called()
    assert result == {}
