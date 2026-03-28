"""Daily scanner: orchestrates the full stock-trend-analyzer pipeline.

Loads rules, fetches universe/OHLCV/fundamentals, computes indicators,
evaluates rules, and returns triggered signals.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]

from src.data.fetcher import fetch_fundamentals, fetch_ohlcv
from src.indicators.calculator import add_all_indicators
from src.rules.engine import evaluate_universe
from src.rules.models import ConditionBlock, ConditionItem, Rule, Signal
from src.rules.parser import load_rules
from src.universe.builder import get_universe

logger = logging.getLogger(__name__)


def _load_settings(settings_path: str) -> dict[str, Any]:
    """Load and return settings.yaml as a dict.

    Args:
        settings_path: Path to the settings YAML file.

    Returns:
        Parsed settings dictionary.

    Raises:
        FileNotFoundError: If the settings file does not exist.
    """
    path = Path(settings_path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    with path.open("r") as fh:
        settings: dict[str, Any] = yaml.safe_load(fh)
    return settings


def _condition_block_to_dict(
    block: ConditionBlock | ConditionItem,
) -> dict[str, Any]:
    """Recursively convert a ConditionBlock or ConditionItem to a plain dict.

    Args:
        block: A ConditionBlock or ConditionItem dataclass instance.

    Returns:
        Dictionary representation suitable for ``add_all_indicators``.
    """
    if isinstance(block, ConditionItem):
        d: dict[str, Any] = {
            "indicator": block.indicator,
            "operator": block.operator,
        }
        if block.period is not None:
            d["period"] = block.period
        if block.value is not None:
            d["value"] = block.value
        if block.multiplier is not None:
            d["multiplier"] = block.multiplier
        if block.vs_period is not None:
            d["vs_period"] = block.vs_period
        if block.fast is not None:
            d["fast"] = block.fast
        if block.slow is not None:
            d["slow"] = block.slow
        if block.signal is not None:
            d["signal"] = block.signal
        if block.compare_to_indicator is not None:
            d["compare_to_indicator"] = block.compare_to_indicator
        if block.compare_period is not None:
            d["compare_period"] = block.compare_period
        return d

    # ConditionBlock
    return {
        "operator": block.operator,
        "items": [_condition_block_to_dict(item) for item in block.items],
    }


def _enrich_dataframes(
    ohlcv: dict[str, pd.DataFrame],
    rules: list[Rule],
) -> dict[str, pd.DataFrame]:
    """Add technical indicators to each ticker's DataFrame.

    Builds a minimal rules config dict from the given Rule objects and passes
    it to ``add_all_indicators`` so only required indicators are computed.

    Args:
        ohlcv: Mapping of ticker symbol -> OHLCV DataFrame.
        rules: List of Rule objects whose conditions determine which
            indicators are needed.

    Returns:
        The same mapping with each DataFrame enriched with indicator columns.
    """
    config: dict[str, Any] = {"rules": []}
    for rule in rules:
        rule_dict: dict[str, Any] = {
            "name": rule.name,
            "type": rule.type,
            "conditions": _condition_block_to_dict(rule.conditions),
        }
        config["rules"].append(rule_dict)

    for ticker, df in ohlcv.items():
        ohlcv[ticker] = add_all_indicators(df, config)

    return ohlcv


def run_scan_on_tickers(
    tickers: list[str],
    rules: list[Rule],
    settings: dict[str, Any],
) -> list[Signal]:
    """Scan a specific ticker list against the provided rules.

    Useful for testing — bypasses ``get_universe`` and uses the provided
    ticker list directly.

    Args:
        tickers: List of ticker symbols to scan.
        rules: List of Rule objects to evaluate against each ticker.
        settings: Settings dict (as loaded from settings.yaml).

    Returns:
        List of Signal objects triggered by the evaluation.
    """
    period_years: int = settings.get("data", {}).get("period_years", 10)

    ohlcv = fetch_ohlcv(tickers, period_years=period_years)
    logger.debug("OHLCV fetched for %d tickers", len(ohlcv))

    fundamentals = fetch_fundamentals(tickers)
    logger.debug("Fundamentals fetched for %d tickers", len(fundamentals))

    ohlcv = _enrich_dataframes(ohlcv, rules)
    logger.debug("Indicators computed")

    signals = evaluate_universe(ohlcv, fundamentals, rules)
    return signals


def run_scan(
    config_path: str = "config/rules.yaml",
    settings_path: str = "config/settings.yaml",
    dry_run: bool = False,
) -> list[Signal]:
    """Run the full daily scan pipeline.

    Loads rules and settings, builds the stock universe, fetches OHLCV and
    fundamental data, computes technical indicators, evaluates all rules, and
    returns triggered signals. When ``dry_run`` is True the signals are also
    printed to stdout and no email is sent.

    Args:
        config_path: Path to the rules YAML configuration file.
        settings_path: Path to the settings YAML configuration file.
        dry_run: When True, print each signal to stdout and skip email delivery.

    Returns:
        List of Signal objects triggered during the scan.
    """
    rules = load_rules(config_path)
    settings = _load_settings(settings_path)

    scan_date = date.today().isoformat()
    logger.info("Starting daily scan — %s", scan_date)
    t0 = time.time()

    universe_cfg: dict[str, Any] = settings.get("universe", {})
    cache_ttl_hours: int = universe_cfg.get("cache_ttl_hours", 168)
    include_nyse_nasdaq: bool = bool(universe_cfg.get("include_nyse_nasdaq", False))
    universe = get_universe(cache_ttl_hours=cache_ttl_hours, include_nyse_nasdaq=include_nyse_nasdaq)
    logger.info("Universe: %d tickers", len(universe))

    period_years: int = settings.get("data", {}).get("period_years", 10)
    ohlcv = fetch_ohlcv(universe, period_years=period_years)
    logger.info("OHLCV fetched in %.1fs", time.time() - t0)

    fundamentals = fetch_fundamentals(universe)
    logger.info("Fundamentals fetched")

    ohlcv = _enrich_dataframes(ohlcv, rules)
    logger.info("Indicators computed")

    signals = evaluate_universe(ohlcv, fundamentals, rules)
    logger.info("Scan complete: %d signals in %.1fs", len(signals), time.time() - t0)

    if dry_run:
        for signal in signals:
            print(
                f"{signal.ticker:6s} | {signal.action:4s} | "
                f"{signal.rule_name:30s} | {signal.reason}"
            )

    return signals


__all__ = [
    "run_scan",
    "run_scan_on_tickers",
    "_load_settings",
    "_enrich_dataframes",
]
