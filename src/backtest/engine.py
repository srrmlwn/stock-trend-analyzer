"""Backtesting engine for replaying trading rules against historical data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.fetcher import fetch_ohlcv
from src.indicators.calculator import add_all_indicators
from src.rules.engine import evaluate_ticker
from src.rules.models import Rule

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a completed backtest run.

    Attributes:
        total_return_pct: Portfolio total return as a percentage.
        benchmark_return_pct: SPY buy-and-hold return over the same period.
        win_rate_pct: Percentage of trades with a positive return.
        max_drawdown_pct: Worst peak-to-trough equity drawdown (negative or zero).
        sharpe_ratio: Annualized Sharpe ratio (mean daily return / std * sqrt(252)).
        avg_holding_days: Mean holding period in calendar days.
        total_trades: Total number of completed trades.
        trades: DataFrame with columns: ticker, entry_date, exit_date, entry_price,
            exit_price, return_pct, rule_name.
    """

    total_return_pct: float
    benchmark_return_pct: float
    win_rate_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_holding_days: float
    total_trades: int
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


def _build_rules_config_from_rules(rules: list[Rule]) -> dict[str, Any]:
    """Convert Rule objects into the config dict format expected by add_all_indicators.

    Args:
        rules: List of Rule dataclass objects.

    Returns:
        Config dict with structure matching the rules.yaml schema.
    """

    def _condition_item_to_dict(item: Any) -> dict[str, Any]:
        """Convert a ConditionItem or ConditionBlock to a dict."""
        from src.rules.models import ConditionBlock, ConditionItem

        if isinstance(item, ConditionItem):
            d: dict[str, Any] = {"indicator": item.indicator, "operator": item.operator}
            if item.period is not None:
                d["period"] = item.period
            if item.value is not None:
                d["value"] = item.value
            if item.multiplier is not None:
                d["multiplier"] = item.multiplier
            if item.vs_period is not None:
                d["vs_period"] = item.vs_period
            if item.fast is not None:
                d["fast"] = item.fast
            if item.slow is not None:
                d["slow"] = item.slow
            if item.signal is not None:
                d["signal"] = item.signal
            if item.compare_to_indicator is not None:
                d["compare_to_indicator"] = item.compare_to_indicator
            if item.compare_period is not None:
                d["compare_period"] = item.compare_period
            return d
        elif isinstance(item, ConditionBlock):
            return {
                "operator": item.operator,
                "items": [_condition_item_to_dict(sub) for sub in item.items],
            }
        return {}

    rules_list = []
    for rule in rules:
        cond_items = [_condition_item_to_dict(item) for item in rule.conditions.items]
        rules_list.append(
            {
                "name": rule.name,
                "type": rule.type,
                "conditions": {
                    "operator": rule.conditions.operator,
                    "items": cond_items,
                },
            }
        )
    return {"rules": rules_list}


def run_signal_replay(
    ticker: str,
    df: pd.DataFrame,
    rules: list[Rule],
) -> pd.DataFrame:
    """Return a DataFrame of historical signals for one ticker.

    Replays each rule against each row of historical data. For each BUY signal
    fired, records the price at signal and the price 30 trading days later.

    Args:
        ticker: Ticker symbol (e.g. "AAPL").
        df: OHLCV DataFrame with DatetimeIndex.
        rules: List of Rule objects to evaluate.

    Returns:
        DataFrame with columns: date, rule_name, action, price_at_signal,
        price_30d_later, return_pct. Returns empty DataFrame if no signals fire.
    """
    _replay_cols = [
        "date", "rule_name", "action", "price_at_signal", "price_30d_later", "return_pct"
    ]

    if df.empty or len(df) < 3:
        logger.debug("Ticker %s: DataFrame too short for signal replay.", ticker)
        return pd.DataFrame(columns=_replay_cols)

    if not rules:
        return pd.DataFrame(columns=_replay_cols)

    # Pre-compute all indicators on the full df once
    config = _build_rules_config_from_rules(rules)
    df = df.copy()
    df = add_all_indicators(df, config)

    records: list[dict[str, Any]] = []

    # Walk from row index 2 onward so crossover conditions have context
    for i in range(2, len(df)):
        sub_df = df.iloc[: i + 1]
        signals = evaluate_ticker(ticker, sub_df, {}, rules)

        current_date = df.index[i]
        current_close = float(df["Close"].iloc[i])

        for signal in signals:
            # Look up price 30 trading days later
            future_idx = i + 30
            if future_idx < len(df):
                price_30d = float(df["Close"].iloc[future_idx])
            else:
                price_30d = float("nan")

            if not np.isnan(price_30d) and current_close > 0:
                ret_pct = (price_30d - current_close) / current_close * 100.0
            else:
                ret_pct = float("nan")

            records.append(
                {
                    "date": current_date,
                    "rule_name": signal.rule_name,
                    "action": signal.action,
                    "price_at_signal": current_close,
                    "price_30d_later": price_30d,
                    "return_pct": ret_pct,
                }
            )

    if not records:
        return pd.DataFrame(columns=_replay_cols)

    return pd.DataFrame(records)


def _compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum peak-to-trough drawdown percentage.

    Args:
        equity_curve: Series of cumulative equity values.

    Returns:
        Max drawdown as a negative percentage (e.g. -12.4), or 0.0 if no drawdown.
    """
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak * 100.0
    return float(drawdown.min())


def _compute_sharpe(daily_returns: pd.Series) -> float:
    """Compute annualized Sharpe ratio from daily returns.

    Args:
        daily_returns: Series of daily portfolio return values (fractional, not percent).

    Returns:
        Annualized Sharpe ratio, or 0.0 if standard deviation is zero or insufficient data.
    """
    if len(daily_returns) < 2:
        return 0.0
    std = float(daily_returns.std())
    if std == 0.0:
        return 0.0
    mean = float(daily_returns.mean())
    return float(mean / std * np.sqrt(252))


def run_backtest(
    tickers: list[str],
    rules: list[Rule],
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    position_size_usd: float = 2000.0,
    initial_capital: float = 50_000.0,
) -> BacktestResult:
    """Replay rules against historical data and return performance metrics.

    Fetches OHLCV data for each ticker plus SPY (benchmark), runs signal replay
    for each ticker, simulates trades, and computes portfolio-level metrics.

    Args:
        tickers: List of ticker symbols to backtest.
        rules: List of Rule objects to evaluate.
        start_date: ISO date string for the start of the backtest period.
        end_date: ISO date string for the end of the backtest period (defaults to today).
        position_size_usd: Dollar amount allocated per trade.
        initial_capital: Starting portfolio capital in USD.

    Returns:
        BacktestResult with performance metrics and a trades DataFrame.
    """
    _trade_cols = [
        "ticker", "entry_date", "exit_date", "entry_price", "exit_price", "return_pct", "rule_name"
    ]
    _empty_trades = pd.DataFrame(columns=_trade_cols)
    _empty_result = BacktestResult(
        total_return_pct=0.0,
        benchmark_return_pct=0.0,
        win_rate_pct=0.0,
        max_drawdown_pct=0.0,
        sharpe_ratio=0.0,
        avg_holding_days=0.0,
        total_trades=0,
        trades=_empty_trades,
    )

    if not tickers:
        logger.warning("run_backtest called with empty ticker list.")
        return _empty_result

    if not rules:
        logger.warning("run_backtest called with no rules.")
        return _empty_result

    # Fetch OHLCV for tickers + SPY benchmark
    all_symbols = list(dict.fromkeys(tickers + ["SPY"]))
    logger.info("Fetching OHLCV data for %d symbols.", len(all_symbols))
    raw_data = fetch_ohlcv(all_symbols)

    if not raw_data:
        logger.warning("No OHLCV data returned. Returning empty backtest result.")
        return _empty_result

    # Filter by date range
    def _filter_dates(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        return df

    filtered_data: dict[str, pd.DataFrame] = {}
    for sym, df in raw_data.items():
        fdf = _filter_dates(df)
        if not fdf.empty:
            filtered_data[sym] = fdf

    # Compute SPY benchmark return
    benchmark_return_pct = 0.0
    spy_df = filtered_data.get("SPY")
    if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
        spy_start = float(spy_df["Close"].iloc[0])
        spy_end = float(spy_df["Close"].iloc[-1])
        if spy_start > 0:
            benchmark_return_pct = (spy_end - spy_start) / spy_start * 100.0
        logger.info("SPY benchmark return: %.2f%%", benchmark_return_pct)

    # Run signal replay for each ticker and simulate trades
    all_trade_records: list[dict[str, Any]] = []

    for ticker in tickers:
        df = filtered_data.get(ticker)
        if df is None or df.empty:
            logger.debug("No data for ticker %s — skipping.", ticker)
            continue

        logger.info("Running signal replay for %s (%d rows).", ticker, len(df))
        signals_df = run_signal_replay(ticker, df, rules)

        if signals_df.empty:
            logger.debug("No signals for ticker %s.", ticker)
            continue

        # Simulate trades: BUY signals → enter next day open, exit after 30 calendar days
        buy_signals = signals_df[signals_df["action"] == "BUY"].copy()

        for _, sig_row in buy_signals.iterrows():
            signal_date = sig_row["date"]
            rule_name = sig_row["rule_name"]

            # Find next trading day after signal for entry
            future_rows = df[df.index > signal_date]
            if future_rows.empty:
                continue

            entry_row = future_rows.iloc[0]
            entry_date = entry_row.name
            entry_price = float(entry_row.get("Open", entry_row.get("Close", 0.0)))

            if entry_price <= 0:
                continue

            # Exit: 30 calendar days after entry (find closest trading day)
            exit_target = entry_date + pd.Timedelta(days=30)
            future_exit_rows = df[df.index >= exit_target]
            if future_exit_rows.empty:
                # Use last available row
                exit_row = df.iloc[-1]
            else:
                exit_row = future_exit_rows.iloc[0]

            exit_date = exit_row.name
            exit_price = float(exit_row.get("Open", exit_row.get("Close", 0.0)))

            if exit_price <= 0:
                continue

            ret_pct = (exit_price - entry_price) / entry_price * 100.0

            all_trade_records.append(
                {
                    "ticker": ticker,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": ret_pct,
                    "rule_name": rule_name,
                }
            )

    if not all_trade_records:
        logger.info("No trades generated in backtest.")
        return BacktestResult(
            total_return_pct=0.0,
            benchmark_return_pct=benchmark_return_pct,
            win_rate_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            avg_holding_days=0.0,
            total_trades=0,
            trades=_empty_trades,
        )

    trades_df = pd.DataFrame(all_trade_records)

    # Sort trades by entry date for equity curve construction
    trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

    total_trades = len(trades_df)

    # Win rate
    positive_trades = (trades_df["return_pct"] > 0).sum()
    win_rate_pct = float(positive_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    # Average holding days
    holding_days = (trades_df["exit_date"] - trades_df["entry_date"]).dt.days
    avg_holding_days = float(holding_days.mean()) if not holding_days.empty else 0.0

    # Build equity curve: start with initial_capital, apply each trade's P&L
    equity = initial_capital
    equity_series_values: list[float] = [initial_capital]
    for _, trade in trades_df.iterrows():
        trade_return = float(trade["return_pct"]) / 100.0
        pnl = position_size_usd * trade_return
        equity += pnl
        equity_series_values.append(equity)

    equity_series = pd.Series(equity_series_values)
    total_return_pct = (equity - initial_capital) / initial_capital * 100.0

    # Max drawdown
    max_drawdown_pct = _compute_max_drawdown(equity_series)

    # Sharpe: use per-trade returns as proxy for daily returns
    # Convert trade returns to fractional returns
    trade_returns_frac = trades_df["return_pct"] / 100.0
    sharpe_ratio = _compute_sharpe(trade_returns_frac)

    logger.info(
        "Backtest complete: %d trades, total return=%.2f%%, benchmark=%.2f%%",
        total_trades,
        total_return_pct,
        benchmark_return_pct,
    )

    return BacktestResult(
        total_return_pct=total_return_pct,
        benchmark_return_pct=benchmark_return_pct,
        win_rate_pct=win_rate_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        avg_holding_days=avg_holding_days,
        total_trades=total_trades,
        trades=trades_df,
    )
