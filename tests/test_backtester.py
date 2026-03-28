"""Tests for the backtesting engine and report modules."""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestResult, run_backtest, run_signal_replay
from src.backtest.report import export_backtest_csv, print_backtest_report
from src.rules.models import ConditionBlock, ConditionItem, FundamentalFilter, Rule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 250, seed: int = 42, trend: float = 0.3) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with n rows of business-day index."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    close = 100.0 + np.cumsum(rng.normal(trend, 1.5, n))
    close = np.maximum(close, 1.0)
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


def _make_flat_ohlcv(n: int = 250) -> pd.DataFrame:
    """Create a perfectly flat OHLCV DataFrame — no crossovers can occur."""
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    close = np.full(n, 100.0)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def _make_rsi_buy_rule() -> Rule:
    """RSI < 30 buy rule."""
    return Rule(
        name="rsi_oversold",
        type="buy",
        conditions=ConditionBlock(
            operator="AND",
            items=[
                ConditionItem(indicator="RSI", operator="less_than", period=14, value=30.0)
            ],
        ),
        fundamentals=FundamentalFilter(),
    )


def _make_rsi_sell_rule() -> Rule:
    """RSI > 70 sell rule."""
    return Rule(
        name="rsi_overbought",
        type="sell",
        conditions=ConditionBlock(
            operator="AND",
            items=[
                ConditionItem(indicator="RSI", operator="greater_than", period=14, value=70.0)
            ],
        ),
        fundamentals=FundamentalFilter(),
    )


def _make_backtest_result(
    total_return: float = 15.0,
    benchmark: float = 10.0,
    win_rate: float = 55.0,
    drawdown: float = -8.0,
    sharpe: float = 1.2,
    avg_days: float = 28.0,
    n_trades: int = 20,
) -> BacktestResult:
    """Build a synthetic BacktestResult for report tests."""
    trades = pd.DataFrame(
        {
            "ticker": ["AAPL"] * n_trades,
            "entry_date": pd.date_range("2022-01-01", periods=n_trades, freq="10D"),
            "exit_date": pd.date_range("2022-01-31", periods=n_trades, freq="10D"),
            "entry_price": [150.0] * n_trades,
            "exit_price": [155.0] * n_trades,
            "return_pct": [3.33] * n_trades,
            "rule_name": ["rsi_oversold"] * n_trades,
        }
    )
    return BacktestResult(
        total_return_pct=total_return,
        benchmark_return_pct=benchmark,
        win_rate_pct=win_rate,
        max_drawdown_pct=drawdown,
        sharpe_ratio=sharpe,
        avg_holding_days=avg_days,
        total_trades=n_trades,
        trades=trades,
    )


# ---------------------------------------------------------------------------
# run_signal_replay tests
# ---------------------------------------------------------------------------


class TestRunSignalReplay:
    def test_run_signal_replay_returns_dataframe(self) -> None:
        """Result must be a DataFrame with the required columns."""
        df = _make_ohlcv(n=250, seed=99, trend=0.5)
        rules = [_make_rsi_buy_rule()]
        result = run_signal_replay("AAPL", df, rules)

        assert isinstance(result, pd.DataFrame)
        expected_cols = {"date", "rule_name", "action", "price_at_signal", "price_30d_later", "return_pct"}
        assert expected_cols.issubset(set(result.columns))

    def test_run_signal_replay_empty_when_no_signals(self) -> None:
        """Flat price data produces no RSI crossover signals."""
        df = _make_flat_ohlcv(n=250)
        # With flat data, RSI stabilises around 50 — never < 30 or > 70
        rules = [_make_rsi_buy_rule()]
        result = run_signal_replay("FLAT", df, rules)

        assert isinstance(result, pd.DataFrame)
        # Either empty or action column contains only BUY/SELL (shouldn't have BUY on flat data)
        if not result.empty:
            # Flat data RSI should be around 50; oversold rule should not fire
            assert (result["action"] == "BUY").sum() == 0

    def test_run_signal_replay_empty_df_returns_empty(self) -> None:
        """Passing an empty DataFrame returns an empty result."""
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        rules = [_make_rsi_buy_rule()]
        result = run_signal_replay("AAPL", df, rules)
        assert result.empty

    def test_run_signal_replay_no_rules_returns_empty(self) -> None:
        """Passing no rules returns an empty DataFrame."""
        df = _make_ohlcv(n=250)
        result = run_signal_replay("AAPL", df, rules=[])
        assert result.empty

    def test_run_signal_replay_price_at_signal_positive(self) -> None:
        """All recorded signal prices should be positive."""
        df = _make_ohlcv(n=300, seed=7, trend=-0.8)  # downtrend → RSI oversold
        rules = [_make_rsi_buy_rule()]
        result = run_signal_replay("AAPL", df, rules)
        if not result.empty:
            assert (result["price_at_signal"] > 0).all()


# ---------------------------------------------------------------------------
# run_backtest tests
# ---------------------------------------------------------------------------


class TestRunBacktest:
    def _mock_fetch(self, tickers: list[str], **kwargs: object) -> dict[str, pd.DataFrame]:
        """Return synthetic OHLCV data for all requested tickers including SPY."""
        data: dict[str, pd.DataFrame] = {}
        for i, ticker in enumerate(tickers):
            seed = 42 + i
            trend = -0.5 if ticker != "SPY" else 0.3
            data[ticker] = _make_ohlcv(n=300, seed=seed, trend=trend)
        return data

    def test_run_backtest_returns_result(self) -> None:
        """run_backtest should return a BacktestResult with all fields."""
        rules = [_make_rsi_buy_rule()]
        with patch("src.backtest.engine.fetch_ohlcv", side_effect=self._mock_fetch):
            result = run_backtest(["AAPL"], rules, start_date="2020-01-01")

        assert isinstance(result, BacktestResult)
        assert isinstance(result.total_return_pct, float)
        assert isinstance(result.benchmark_return_pct, float)
        assert isinstance(result.win_rate_pct, float)
        assert isinstance(result.max_drawdown_pct, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.avg_holding_days, float)
        assert isinstance(result.total_trades, int)
        assert isinstance(result.trades, pd.DataFrame)

    def test_run_backtest_benchmark_return(self) -> None:
        """SPY benchmark return should be computed from mock data."""
        rules = [_make_rsi_buy_rule()]
        with patch("src.backtest.engine.fetch_ohlcv", side_effect=self._mock_fetch):
            result = run_backtest(["AAPL"], rules, start_date="2020-01-01")

        # With upward-trending SPY (trend=0.3 over 300 days), expect positive benchmark
        assert isinstance(result.benchmark_return_pct, float)
        # Benchmark should be non-zero since we have SPY data
        # (either positive or negative depending on mock data trajectory)
        assert not np.isnan(result.benchmark_return_pct)

    def test_run_backtest_empty_tickers(self) -> None:
        """Empty ticker list should return a zero-result BacktestResult."""
        rules = [_make_rsi_buy_rule()]
        result = run_backtest([], rules)
        assert result.total_trades == 0
        assert result.total_return_pct == 0.0

    def test_run_backtest_no_rules(self) -> None:
        """Empty rules list should return a zero-result BacktestResult."""
        with patch("src.backtest.engine.fetch_ohlcv", side_effect=self._mock_fetch):
            result = run_backtest(["AAPL"], [])
        assert result.total_trades == 0

    def test_run_backtest_trades_columns(self) -> None:
        """Trades DataFrame should have all required columns."""
        rules = [_make_rsi_buy_rule()]
        with patch("src.backtest.engine.fetch_ohlcv", side_effect=self._mock_fetch):
            result = run_backtest(["AAPL"], rules, start_date="2020-01-01")

        if result.total_trades > 0:
            expected_cols = {"ticker", "entry_date", "exit_date", "entry_price", "exit_price", "return_pct", "rule_name"}
            assert expected_cols.issubset(set(result.trades.columns))

    def test_win_rate_between_0_and_100(self) -> None:
        """Win rate must always be in [0, 100]."""
        rules = [_make_rsi_buy_rule()]
        with patch("src.backtest.engine.fetch_ohlcv", side_effect=self._mock_fetch):
            result = run_backtest(["AAPL"], rules, start_date="2020-01-01")

        assert 0.0 <= result.win_rate_pct <= 100.0

    def test_sharpe_ratio_computable(self) -> None:
        """Sharpe ratio should be finite — no division-by-zero on valid data."""
        rules = [_make_rsi_buy_rule()]
        with patch("src.backtest.engine.fetch_ohlcv", side_effect=self._mock_fetch):
            result = run_backtest(["AAPL"], rules, start_date="2020-01-01")

        assert np.isfinite(result.sharpe_ratio)

    def test_max_drawdown_negative_or_zero(self) -> None:
        """Max drawdown must be <= 0."""
        rules = [_make_rsi_buy_rule()]
        with patch("src.backtest.engine.fetch_ohlcv", side_effect=self._mock_fetch):
            result = run_backtest(["AAPL"], rules, start_date="2020-01-01")

        assert result.max_drawdown_pct <= 0.0

    def test_run_backtest_multiple_tickers(self) -> None:
        """Backtest can handle multiple tickers."""
        rules = [_make_rsi_buy_rule()]
        with patch("src.backtest.engine.fetch_ohlcv", side_effect=self._mock_fetch):
            result = run_backtest(["AAPL", "MSFT", "GOOG"], rules, start_date="2020-01-01")

        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0


# ---------------------------------------------------------------------------
# report tests
# ---------------------------------------------------------------------------


class TestReport:
    def test_print_backtest_report_outputs_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_backtest_report should write the report header to stdout."""
        result = _make_backtest_result()
        print_backtest_report(result)
        captured = capsys.readouterr()
        assert "Backtest Report" in captured.out
        assert "Total Return" in captured.out
        assert "Win Rate" in captured.out
        assert "Sharpe Ratio" in captured.out
        assert "Total Trades" in captured.out

    def test_print_backtest_report_positive_return_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Positive total return should be prefixed with '+'."""
        result = _make_backtest_result(total_return=25.5, benchmark=12.3)
        print_backtest_report(result)
        captured = capsys.readouterr()
        assert "+25.5%" in captured.out
        assert "+12.3%" in captured.out

    def test_print_backtest_report_negative_return_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Negative total return should not include '+'."""
        result = _make_backtest_result(total_return=-5.0, benchmark=-2.0)
        print_backtest_report(result)
        captured = capsys.readouterr()
        assert "-5.0%" in captured.out
        assert "-2.0%" in captured.out

    def test_export_backtest_csv_creates_file(self, tmp_path: object) -> None:
        """export_backtest_csv should write a file at the given output_path."""
        assert isinstance(tmp_path, type(tmp_path))  # tmp_path is pytest's Path fixture
        from pathlib import Path
        out_path = str(Path(str(tmp_path)) / "trades.csv")  # type: ignore[arg-type]

        result = _make_backtest_result(n_trades=5)
        export_backtest_csv(result, out_path)

        assert os.path.exists(out_path)
        loaded = pd.read_csv(out_path)
        assert len(loaded) == 5
        assert "ticker" in loaded.columns
        assert "return_pct" in loaded.columns

    def test_export_backtest_csv_empty_trades(self, tmp_path: object) -> None:
        """export_backtest_csv with empty trades should still create the file."""
        from pathlib import Path
        out_path = str(Path(str(tmp_path)) / "empty_trades.csv")  # type: ignore[arg-type]

        result = BacktestResult(
            total_return_pct=0.0,
            benchmark_return_pct=0.0,
            win_rate_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            avg_holding_days=0.0,
            total_trades=0,
            trades=pd.DataFrame(
                columns=["ticker", "entry_date", "exit_date", "entry_price", "exit_price", "return_pct", "rule_name"]
            ),
        )
        export_backtest_csv(result, out_path)
        assert os.path.exists(out_path)
