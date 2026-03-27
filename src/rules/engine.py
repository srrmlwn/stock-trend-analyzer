"""Rules engine for evaluating trading rules against ticker data."""

import logging
from datetime import date
from typing import Any

import pandas as pd

from src.rules.models import ConditionBlock, ConditionItem, FundamentalFilter, Rule, Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _col_exists(df: pd.DataFrame, col: str) -> bool:
    """Return True if column exists in DataFrame."""
    return col in df.columns


def _get_last(df: pd.DataFrame, col: str, offset: int = 0) -> float | None:
    """Return the value at position -(1+offset) for *col*, or None if missing."""
    if col not in df.columns:
        return None
    idx = -(1 + offset)
    if abs(idx) > len(df):
        return None
    val = df[col].iloc[idx]
    if pd.isna(val):
        return None
    return float(val)


def _evaluate_condition_item(
    item: ConditionItem,
    df: pd.DataFrame,
) -> tuple[bool, str]:
    """Evaluate a single ConditionItem against the DataFrame.

    Args:
        item: The condition to evaluate.
        df: DataFrame containing indicator columns.

    Returns:
        A (result, reason) tuple. reason is non-empty only when result is True.
    """
    indicator = item.indicator
    operator = item.operator

    # -----------------------------------------------------------------------
    # RSI
    # -----------------------------------------------------------------------
    if indicator == "RSI":
        col = f"RSI_{item.period}"
        val = _get_last(df, col)
        if val is None:
            logger.warning("Column '%s' not found; condition evaluates to False", col)
            return False, ""
        if operator == "less_than":
            result = val < (item.value or 0.0)
            reason = f"RSI({item.period})={val:.1f} < {item.value}"
            return result, reason if result else ""
        if operator == "greater_than":
            result = val > (item.value or 0.0)
            reason = f"RSI({item.period})={val:.1f} > {item.value}"
            return result, reason if result else ""
        logger.warning("Unknown RSI operator '%s'; condition evaluates to False", operator)
        return False, ""

    # -----------------------------------------------------------------------
    # MACD
    # -----------------------------------------------------------------------
    if indicator == "MACD":
        macd_col = f"MACD_{item.fast}_{item.slow}_{item.signal}"
        macds_col = f"MACDs_{item.fast}_{item.slow}_{item.signal}"

        macd_now = _get_last(df, macd_col, offset=0)
        macds_now = _get_last(df, macds_col, offset=0)
        macd_prev = _get_last(df, macd_col, offset=1)
        macds_prev = _get_last(df, macds_col, offset=1)

        if any(v is None for v in [macd_now, macds_now, macd_prev, macds_prev]):
            missing = [c for c in [macd_col, macds_col] if c not in df.columns]
            if missing:
                logger.warning(
                    "Columns %s not found; condition evaluates to False", missing
                )
            return False, ""

        # mypy narrowing
        assert macd_now is not None
        assert macds_now is not None
        assert macd_prev is not None
        assert macds_prev is not None

        if operator == "crossed_above_signal":
            result = (macd_now > macds_now) and (macd_prev <= macds_prev)
            return result, "MACD crossed above signal line" if result else ""
        if operator == "crossed_below_signal":
            result = (macd_now < macds_now) and (macd_prev >= macds_prev)
            return result, "MACD crossed below signal line" if result else ""
        logger.warning("Unknown MACD operator '%s'; condition evaluates to False", operator)
        return False, ""

    # -----------------------------------------------------------------------
    # SMA / EMA
    # -----------------------------------------------------------------------
    if indicator in ("SMA", "EMA"):
        col = f"{indicator}_{item.period}"

        if operator in ("crossed_above", "crossed_below"):
            # Compare to another indicator
            cmp_ind = item.compare_to_indicator or indicator
            cmp_col = f"{cmp_ind}_{item.compare_period}"

            val_now = _get_last(df, col, offset=0)
            cmp_now = _get_last(df, cmp_col, offset=0)
            val_prev = _get_last(df, col, offset=1)
            cmp_prev = _get_last(df, cmp_col, offset=1)

            missing = [c for c in [col, cmp_col] if c not in df.columns]
            if missing:
                logger.warning(
                    "Columns %s not found; condition evaluates to False", missing
                )
                return False, ""

            if any(v is None for v in [val_now, cmp_now, val_prev, cmp_prev]):
                return False, ""

            assert val_now is not None
            assert cmp_now is not None
            assert val_prev is not None
            assert cmp_prev is not None

            if operator == "crossed_above":
                result = (val_now > cmp_now) and (val_prev <= cmp_prev)
                reason = (
                    f"{indicator}({item.period}) crossed above "
                    f"{cmp_ind}({item.compare_period})"
                )
                return result, reason if result else ""
            else:  # crossed_below
                result = (val_now < cmp_now) and (val_prev >= cmp_prev)
                reason = (
                    f"{indicator}({item.period}) crossed below "
                    f"{cmp_ind}({item.compare_period})"
                )
                return result, reason if result else ""

        # Numeric comparison
        val = _get_last(df, col)
        if val is None:
            logger.warning("Column '%s' not found; condition evaluates to False", col)
            return False, ""
        if operator == "less_than":
            result = val < (item.value or 0.0)
            return result, f"{indicator}({item.period})={val:.2f} < {item.value}" if result else ""
        if operator == "greater_than":
            result = val > (item.value or 0.0)
            return result, f"{indicator}({item.period})={val:.2f} > {item.value}" if result else ""
        logger.warning(
            "Unknown %s operator '%s'; condition evaluates to False", indicator, operator
        )
        return False, ""

    # -----------------------------------------------------------------------
    # volume_spike
    # -----------------------------------------------------------------------
    if indicator == "volume_spike":
        col = f"volume_ratio_{item.vs_period}"
        val = _get_last(df, col)
        if val is None:
            logger.warning("Column '%s' not found; condition evaluates to False", col)
            return False, ""
        if operator == "greater_than":
            result = val > (item.multiplier or 0.0)
            reason = f"volume ratio={val:.1f}x > {item.multiplier}x"
            return result, reason if result else ""
        logger.warning(
            "Unknown volume_spike operator '%s'; condition evaluates to False", operator
        )
        return False, ""

    # -----------------------------------------------------------------------
    # price_change_pct
    # -----------------------------------------------------------------------
    if indicator == "price_change_pct":
        col = f"price_change_pct_{item.period}"
        val = _get_last(df, col)
        if val is None:
            logger.warning("Column '%s' not found; condition evaluates to False", col)
            return False, ""
        if operator == "less_than":
            result = val < (item.value or 0.0)
            reason = f"price change({item.period}d)={val:.1f}% < {item.value:.1f}%"
            return result, reason if result else ""
        if operator == "greater_than":
            result = val > (item.value or 0.0)
            reason = f"price change({item.period}d)={val:.1f}% > {item.value:.1f}%"
            return result, reason if result else ""
        logger.warning(
            "Unknown price_change_pct operator '%s'; condition evaluates to False", operator
        )
        return False, ""

    logger.warning("Unknown indicator '%s'; condition evaluates to False", indicator)
    return False, ""


def _evaluate_condition_block(
    block: ConditionBlock,
    df: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """Recursively evaluate a ConditionBlock against the DataFrame.

    Args:
        block: The condition block to evaluate.
        df: DataFrame containing indicator columns.

    Returns:
        A (result, reasons) tuple. reasons collects non-empty reason strings
        from triggered leaf conditions.
    """
    if block.operator == "AND":
        all_reasons: list[str] = []
        for item in block.items:
            if isinstance(item, ConditionItem):
                result, reason = _evaluate_condition_item(item, df)
                if not result:
                    return False, []
                if reason:
                    all_reasons.append(reason)
            else:  # ConditionBlock
                result, reasons = _evaluate_condition_block(item, df)
                if not result:
                    return False, []
                all_reasons.extend(reasons)
        return True, all_reasons

    else:  # OR
        for item in block.items:
            if isinstance(item, ConditionItem):
                result, reason = _evaluate_condition_item(item, df)
                if result:
                    return True, [reason] if reason else []
            else:  # ConditionBlock
                result, reasons = _evaluate_condition_block(item, df)
                if result:
                    return True, reasons
        return False, []


def _check_fundamentals(fundamentals: FundamentalFilter, fund_data: dict[str, Any]) -> bool:
    """Return True if the ticker passes the fundamental filter.

    Args:
        fundamentals: The FundamentalFilter to apply.
        fund_data: Fundamental data dict for the ticker.

    Returns:
        True if all fundamental conditions pass, False otherwise.
    """
    if fundamentals.max_pe is not None:
        pe = fund_data.get("pe_ratio")
        if pe is not None and pe > fundamentals.max_pe:
            return False

    if fundamentals.min_market_cap_B is not None:
        mktcap = fund_data.get("market_cap")
        if mktcap is None or mktcap < fundamentals.min_market_cap_B * 1e9:
            return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_ticker(
    ticker: str,
    df: pd.DataFrame,
    fundamentals: dict[str, Any],
    rules: list[Rule],
) -> list[Signal]:
    """Evaluate all rules against one ticker and return triggered signals.

    Args:
        ticker: Ticker symbol (e.g. "AAPL").
        df: DataFrame with OHLCV data and pre-computed indicator columns.
        fundamentals: Dict of fundamental data for the ticker
            (keys: pe_ratio, market_cap, avg_volume, price, …).
        rules: List of Rule objects to evaluate.

    Returns:
        List of Signal objects for each rule that fired.
    """
    if df.empty or len(df) < 2:
        return []

    signals: list[Signal] = []
    last_row = df.iloc[-1]
    price = float(last_row.get("Close", 0.0))
    triggered_at: date = last_row.name.date() if hasattr(last_row.name, "date") else date.today()

    for rule in rules:
        # Step 1: fundamental filter
        if not _check_fundamentals(rule.fundamentals, fundamentals):
            logger.debug(
                "Ticker %s failed fundamental filter for rule '%s'", ticker, rule.name
            )
            continue

        # Step 2: evaluate condition block
        passed, reasons = _evaluate_condition_block(rule.conditions, df)
        if not passed:
            continue

        reason_str = ", ".join(r for r in reasons if r) or rule.name
        action: str = "BUY" if rule.type == "buy" else "SELL"

        signal = Signal(
            ticker=ticker,
            action=action,  # type: ignore[arg-type]
            rule_name=rule.name,
            reason=reason_str,
            triggered_at=triggered_at,
            price=price,
        )
        signals.append(signal)
        logger.info(
            "Signal: %s %s — %s (rule: %s)", action, ticker, reason_str, rule.name
        )

    return signals


def evaluate_universe(
    universe: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict[str, Any]],
    rules: list[Rule],
) -> list[Signal]:
    """Evaluate all rules across all tickers and return all triggered signals.

    Args:
        universe: Mapping of ticker symbol → enriched DataFrame.
        fundamentals: Mapping of ticker symbol → fundamentals dict.
        rules: List of Rule objects to evaluate.

    Returns:
        Flat list of all Signal objects triggered across the universe.
    """
    all_signals: list[Signal] = []
    tickers = list(universe.keys())
    total = len(tickers)

    for i, ticker in enumerate(tickers, start=1):
        if i % 100 == 0:
            logger.info("Evaluating rules: %d / %d tickers processed", i, total)

        df = universe[ticker]
        fund_data = fundamentals.get(ticker, {})
        signals = evaluate_ticker(ticker, df, fund_data, rules)
        all_signals.extend(signals)

    logger.info(
        "Universe evaluation complete: %d tickers, %d signals", total, len(all_signals)
    )
    return all_signals
