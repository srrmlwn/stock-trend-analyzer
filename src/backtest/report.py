"""Backtest reporting utilities: console output and CSV export."""

from __future__ import annotations

from src.backtest.engine import BacktestResult


def print_backtest_report(result: BacktestResult) -> None:
    """Print a formatted backtest summary to stdout.

    Args:
        result: BacktestResult object from run_backtest.
    """
    total_sign = "+" if result.total_return_pct >= 0 else ""
    bench_sign = "+" if result.benchmark_return_pct >= 0 else ""
    drawdown_sign = "+" if result.max_drawdown_pct > 0 else ""

    print("=== Backtest Report ===")
    print(
        f"Total Return:      {total_sign}{result.total_return_pct:.1f}%"
        f"   (Benchmark: {bench_sign}{result.benchmark_return_pct:.1f}%)"
    )
    print(f"Win Rate:          {result.win_rate_pct:.1f}%")
    print(f"Max Drawdown:      {drawdown_sign}{result.max_drawdown_pct:.1f}%")
    print(f"Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"Avg Holding Days:  {result.avg_holding_days:.1f}")
    print(f"Total Trades:      {result.total_trades}")


def export_backtest_csv(result: BacktestResult, output_path: str) -> None:
    """Export the trades DataFrame to a CSV file at output_path.

    Args:
        result: BacktestResult object from run_backtest.
        output_path: Filesystem path where the CSV file will be written.
    """
    result.trades.to_csv(output_path, index=False)
