"""CLI entrypoint and optional blocking scheduler for stock-trend-analyzer.

Usage:
    python -m src.scheduler.main --run-now
    python -m src.scheduler.main --run-now --dry-run
    python -m src.scheduler.main --backtest --tickers AAPL MSFT NVDA
    python -m src.scheduler.main --backtest --tickers AAPL MSFT \
        --start-date 2018-01-01 --end-date 2023-12-31
    python -m src.scheduler.main --schedule
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime
from typing import Any

import schedule
import yaml  # type: ignore[import-untyped]

from src.backtest.engine import run_backtest
from src.backtest.report import print_backtest_report
from src.report.sender import send_daily_report
from src.rules.parser import load_rules
from src.scanner.scanner import run_scan

logger = logging.getLogger(__name__)

# Module-level config paths (can be overridden by CLI args)
_config_path: str = "config/rules.yaml"
_settings_path: str = "config/settings.yaml"


def _load_settings(settings_path: str) -> dict[str, Any]:
    """Load and return settings.yaml as a dict.

    Args:
        settings_path: Path to the settings YAML file.

    Returns:
        Parsed settings dictionary.
    """
    from pathlib import Path

    path = Path(settings_path)
    if not path.exists():
        logger.warning("Settings file not found: %s — using defaults", settings_path)
        return {}
    with path.open("r") as fh:
        result: dict[str, Any] = yaml.safe_load(fh) or {}
    return result


def _run_now(dry_run: bool = False) -> None:
    """Run the full daily scan pipeline once.

    Calls run_scan and, when not in dry-run mode, sends the daily email report.

    Args:
        dry_run: When True, print signals to stdout and skip email delivery.
    """
    logger.info("Running scan — dry_run=%s", dry_run)
    signals = run_scan(
        config_path=_config_path,
        settings_path=_settings_path,
        dry_run=dry_run,
    )
    logger.info("Scan produced %d signal(s)", len(signals))

    if not dry_run:
        send_daily_report(signals, date.today())
        logger.info("Daily report sent")


def _run_backtest(
    tickers: list[str],
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> None:
    """Run backtest on the given tickers and print the report.

    Loads rules from the configured config path, runs the backtester, and
    prints a formatted summary to stdout.

    Args:
        tickers: List of ticker symbols to backtest.
        start_date: ISO date string for the backtest start (default ``"2015-01-01"``).
        end_date: ISO date string for the backtest end, or None for today.
    """
    logger.info(
        "Starting backtest for %d ticker(s): %s (start=%s, end=%s)",
        len(tickers),
        tickers,
        start_date,
        end_date,
    )
    rules = load_rules(_config_path)
    result = run_backtest(tickers, rules, start_date=start_date, end_date=end_date)
    print_backtest_report(result)


def _job() -> None:
    """The scheduled job — runs the full scan and sends the email report.

    Called by the scheduler on each configured weekday at the configured time.
    """
    logger.info("Scheduled job starting at %s", datetime.utcnow().isoformat())
    signals = run_scan(
        config_path=_config_path,
        settings_path=_settings_path,
        dry_run=False,
    )
    logger.info("Scan produced %d signal(s)", len(signals))
    send_daily_report(signals, date.today())
    logger.info("Scheduled job complete — report sent")


def _run_schedule() -> None:
    """Block and run the scan daily at the configured time on weekdays.

    Reads ``scanner.run_time_et`` from settings.yaml (default ``"10:00"``),
    schedules the job for Monday through Friday, and enters an infinite loop
    checking for pending jobs every 60 seconds.
    """
    settings = _load_settings(_settings_path)
    run_time: str = settings.get("scanner", {}).get("run_time_et", "10:00")

    schedule.every().monday.at(run_time).do(_job)
    schedule.every().tuesday.at(run_time).do(_job)
    schedule.every().wednesday.at(run_time).do(_job)
    schedule.every().thursday.at(run_time).do(_job)
    schedule.every().friday.at(run_time).do(_job)

    logger.info("Scheduler started. Next run: %s", schedule.next_run())

    while True:
        schedule.run_pending()
        time.sleep(60)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate action.

    Supports three mutually exclusive modes:
    - ``--run-now``: Run the full scan pipeline immediately.
    - ``--backtest``: Run the backtester on specified tickers.
    - ``--schedule``: Start the blocking daily scheduler.
    """
    global _config_path, _settings_path  # noqa: PLW0603

    parser = argparse.ArgumentParser(
        description="Stock Trend Analyzer — CLI entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--run-now",
        action="store_true",
        help="Run the scan immediately",
    )
    mode_group.add_argument(
        "--backtest",
        action="store_true",
        help="Run the backtester",
    )
    mode_group.add_argument(
        "--schedule",
        action="store_true",
        help="Start the daily scheduler (blocks)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Combined with --run-now: print signals, skip email",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="One or more ticker symbols (required with --backtest)",
    )
    parser.add_argument(
        "--start-date",
        default="2015-01-01",
        metavar="YYYY-MM-DD",
        help="Backtest start date (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Backtest end date (default: today)",
    )
    parser.add_argument(
        "--config",
        default="config/rules.yaml",
        metavar="PATH",
        help="Path to rules.yaml (default: config/rules.yaml)",
    )
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        metavar="PATH",
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Apply config/settings paths globally so helper functions pick them up
    _config_path = args.config
    _settings_path = args.settings

    if args.run_now:
        _run_now(dry_run=args.dry_run)

    elif args.backtest:
        if not args.tickers:
            parser.error("--tickers is required when using --backtest")
        _run_backtest(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    elif args.schedule:
        _run_schedule()


if __name__ == "__main__":
    main()
