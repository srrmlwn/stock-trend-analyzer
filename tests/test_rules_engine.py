"""Tests for the rules engine module."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.rules.engine import evaluate_ticker, evaluate_universe
from src.rules.models import (
    ConditionBlock,
    ConditionItem,
    FundamentalFilter,
    Rule,
    Signal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 10, **extra_cols: list[float]) -> pd.DataFrame:
    """Return a minimal DataFrame with Close/Volume columns plus any extras."""
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    close = [100.0 + i for i in range(n)]
    df = pd.DataFrame(
        {
            "Open": close,
            "High": [c * 1.01 for c in close],
            "Low": [c * 0.99 for c in close],
            "Close": close,
            "Volume": [1_000_000.0] * n,
        },
        index=dates,
    )
    df.index.name = "Date"
    for col, vals in extra_cols.items():
        df[col] = vals
    return df


def _rsi_rule(
    rsi_op: str = "less_than",
    rsi_val: float = 30.0,
    rule_type: str = "buy",
    fundamentals: FundamentalFilter | None = None,
) -> Rule:
    """Return a simple RSI rule."""
    return Rule(
        name="test_rsi",
        type=rule_type,  # type: ignore[arg-type]
        conditions=ConditionBlock(
            operator="AND",
            items=[
                ConditionItem(indicator="RSI", operator=rsi_op, period=14, value=rsi_val)
            ],
        ),
        fundamentals=fundamentals or FundamentalFilter(),
    )


@pytest.fixture
def enriched_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return sample_ohlcv enriched with all indicator columns needed by tests."""
    df = sample_ohlcv.copy()
    n = len(df)

    # RSI indicators (14-period)
    df["RSI_14"] = [50.0] * n  # neutral default

    # MACD columns (12_26_9)
    df["MACD_12_26_9"] = [0.5] * n
    df["MACDs_12_26_9"] = [0.3] * n  # MACD > signal → bullish

    # SMA columns
    df["SMA_50"] = df["Close"] * 1.01   # slightly above
    df["SMA_200"] = df["Close"] * 0.99  # slightly below

    # EMA columns
    df["EMA_50"] = df["Close"] * 1.01
    df["EMA_200"] = df["Close"] * 0.99

    # Volume ratio
    df["volume_ratio_20"] = [1.8] * n

    # Price change pct
    df["price_change_pct_5"] = [-3.5] * n

    return df


@pytest.fixture
def basic_fundamentals() -> dict:
    """Return a fundamentals dict that should pass standard filters."""
    return {
        "pe_ratio": 20.0,
        "market_cap": 10_000_000_000,  # $10B
        "avg_volume": 1_000_000,
        "price": 100.0,
    }


# ---------------------------------------------------------------------------
# AND logic tests
# ---------------------------------------------------------------------------


def test_and_both_conditions_true_fires_signal(basic_fundamentals: dict) -> None:
    """AND rule where both conditions are True should produce a signal."""
    df = _make_df(
        n=5,
        RSI_14=[25.0] * 5,
        volume_ratio_20=[2.0] * 5,
    )
    rule = Rule(
        name="and_rule",
        type="buy",
        conditions=ConditionBlock(
            operator="AND",
            items=[
                ConditionItem(indicator="RSI", operator="less_than", period=14, value=30.0),
                ConditionItem(
                    indicator="volume_spike",
                    operator="greater_than",
                    vs_period=20,
                    multiplier=1.5,
                ),
            ],
        ),
    )
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert len(signals) == 1
    assert signals[0].action == "BUY"
    assert signals[0].rule_name == "and_rule"


def test_and_one_condition_false_no_signal(basic_fundamentals: dict) -> None:
    """AND rule where one condition is False should produce no signal."""
    df = _make_df(
        n=5,
        RSI_14=[50.0] * 5,  # RSI > 30 → fails less_than 30
        volume_ratio_20=[2.0] * 5,
    )
    rule = Rule(
        name="and_rule",
        type="buy",
        conditions=ConditionBlock(
            operator="AND",
            items=[
                ConditionItem(indicator="RSI", operator="less_than", period=14, value=30.0),
                ConditionItem(
                    indicator="volume_spike",
                    operator="greater_than",
                    vs_period=20,
                    multiplier=1.5,
                ),
            ],
        ),
    )
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert signals == []


# ---------------------------------------------------------------------------
# OR logic tests
# ---------------------------------------------------------------------------


def test_or_one_condition_true_fires_signal(basic_fundamentals: dict) -> None:
    """OR rule where at least one condition is True should produce a signal."""
    df = _make_df(
        n=5,
        RSI_14=[25.0] * 5,      # triggers less_than 30
        volume_ratio_20=[0.5] * 5,  # does NOT trigger > 1.5
    )
    rule = Rule(
        name="or_rule",
        type="buy",
        conditions=ConditionBlock(
            operator="OR",
            items=[
                ConditionItem(indicator="RSI", operator="less_than", period=14, value=30.0),
                ConditionItem(
                    indicator="volume_spike",
                    operator="greater_than",
                    vs_period=20,
                    multiplier=1.5,
                ),
            ],
        ),
    )
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert len(signals) == 1
    assert signals[0].action == "BUY"


def test_or_no_condition_true_no_signal(basic_fundamentals: dict) -> None:
    """OR rule where no condition is True should produce no signal."""
    df = _make_df(
        n=5,
        RSI_14=[50.0] * 5,
        volume_ratio_20=[0.5] * 5,
    )
    rule = Rule(
        name="or_rule",
        type="buy",
        conditions=ConditionBlock(
            operator="OR",
            items=[
                ConditionItem(indicator="RSI", operator="less_than", period=14, value=30.0),
                ConditionItem(
                    indicator="volume_spike",
                    operator="greater_than",
                    vs_period=20,
                    multiplier=1.5,
                ),
            ],
        ),
    )
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert signals == []


# ---------------------------------------------------------------------------
# Fundamental filter tests
# ---------------------------------------------------------------------------


def test_fundamental_filter_blocks_high_pe(basic_fundamentals: dict) -> None:
    """Rule with max_pe=15 should not fire when pe_ratio=20."""
    df = _make_df(n=5, RSI_14=[25.0] * 5)
    rule = _rsi_rule(fundamentals=FundamentalFilter(max_pe=15.0))
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert signals == []


def test_fundamental_filter_passes_within_pe_limit(basic_fundamentals: dict) -> None:
    """Rule with max_pe=25 should fire when pe_ratio=20."""
    df = _make_df(n=5, RSI_14=[25.0] * 5)
    rule = _rsi_rule(fundamentals=FundamentalFilter(max_pe=25.0))
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert len(signals) == 1


def test_fundamental_filter_blocks_small_market_cap(basic_fundamentals: dict) -> None:
    """Rule requiring min_market_cap_B=50 should not fire with $10B cap."""
    df = _make_df(n=5, RSI_14=[25.0] * 5)
    rule = _rsi_rule(fundamentals=FundamentalFilter(min_market_cap_B=50.0))
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert signals == []


def test_fundamental_filter_pe_none_skips_check(basic_fundamentals: dict) -> None:
    """When pe_ratio is absent from fundamentals, the max_pe check is skipped."""
    funds_no_pe = {k: v for k, v in basic_fundamentals.items() if k != "pe_ratio"}
    df = _make_df(n=5, RSI_14=[25.0] * 5)
    rule = _rsi_rule(fundamentals=FundamentalFilter(max_pe=5.0))  # very strict
    signals = evaluate_ticker("TEST", df, funds_no_pe, [rule])
    # pe_ratio absent → check is skipped → rule fires
    assert len(signals) == 1


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_empty_df_returns_empty_list(basic_fundamentals: dict) -> None:
    """An empty DataFrame should return no signals."""
    df = pd.DataFrame()
    rule = _rsi_rule()
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert signals == []


def test_single_row_df_returns_empty_list(basic_fundamentals: dict) -> None:
    """A DataFrame with only 1 row should return no signals (need >= 2)."""
    df = _make_df(n=1, RSI_14=[25.0])
    rule = _rsi_rule()
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert signals == []


def test_missing_column_returns_false_no_crash(basic_fundamentals: dict) -> None:
    """If a required column is absent, the condition should evaluate to False without error."""
    df = _make_df(n=5)  # no RSI_14 column
    rule = _rsi_rule()
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert signals == []


def test_sell_rule_produces_sell_signal(basic_fundamentals: dict) -> None:
    """A sell-type rule should produce a SELL signal."""
    df = _make_df(n=5, RSI_14=[80.0] * 5)
    rule = _rsi_rule(rsi_op="greater_than", rsi_val=75.0, rule_type="sell")
    signals = evaluate_ticker("TEST", df, basic_fundamentals, [rule])
    assert len(signals) == 1
    assert signals[0].action == "SELL"


# ---------------------------------------------------------------------------
# evaluate_universe tests
# ---------------------------------------------------------------------------


def test_evaluate_universe_aggregates_signals(basic_fundamentals: dict) -> None:
    """evaluate_universe should collect signals from multiple tickers."""
    df_a = _make_df(n=5, RSI_14=[25.0] * 5)
    df_b = _make_df(n=5, RSI_14=[25.0] * 5)
    df_c = _make_df(n=5, RSI_14=[60.0] * 5)  # won't trigger

    universe = {"A": df_a, "B": df_b, "C": df_c}
    funds = {
        "A": basic_fundamentals,
        "B": basic_fundamentals,
        "C": basic_fundamentals,
    }
    rule = _rsi_rule()
    signals = evaluate_universe(universe, funds, [rule])
    tickers_fired = {s.ticker for s in signals}
    assert "A" in tickers_fired
    assert "B" in tickers_fired
    assert "C" not in tickers_fired


def test_evaluate_universe_empty_universe() -> None:
    """evaluate_universe with an empty dict should return an empty list."""
    signals = evaluate_universe({}, {}, [_rsi_rule()])
    assert signals == []


# ---------------------------------------------------------------------------
# Reason string tests
# ---------------------------------------------------------------------------


def test_reason_string_rsi(basic_fundamentals: dict) -> None:
    """RSI triggered condition should produce the expected reason string."""
    df = _make_df(n=5, RSI_14=[27.3] * 5)
    rule = _rsi_rule(rsi_val=30.0)
    signals = evaluate_ticker("AAPL", df, basic_fundamentals, [rule])
    assert len(signals) == 1
    assert "RSI(14)" in signals[0].reason
    assert "27.3" in signals[0].reason
    assert "30" in signals[0].reason


def test_reason_string_volume_spike(basic_fundamentals: dict) -> None:
    """volume_spike triggered condition should produce the expected reason string."""
    df = _make_df(n=5, volume_ratio_20=[1.8] * 5)
    rule = Rule(
        name="vol_spike",
        type="buy",
        conditions=ConditionBlock(
            operator="AND",
            items=[
                ConditionItem(
                    indicator="volume_spike",
                    operator="greater_than",
                    vs_period=20,
                    multiplier=1.5,
                )
            ],
        ),
    )
    signals = evaluate_ticker("AAPL", df, basic_fundamentals, [rule])
    assert len(signals) == 1
    assert "volume ratio" in signals[0].reason
    assert "1.8x" in signals[0].reason


def test_reason_string_price_change_pct(basic_fundamentals: dict) -> None:
    """price_change_pct triggered condition should produce the expected reason string."""
    df = _make_df(n=5, price_change_pct_5=[-3.2] * 5)
    rule = Rule(
        name="price_drop",
        type="sell",
        conditions=ConditionBlock(
            operator="AND",
            items=[
                ConditionItem(
                    indicator="price_change_pct",
                    operator="less_than",
                    period=5,
                    value=-2.0,
                )
            ],
        ),
    )
    signals = evaluate_ticker("AAPL", df, basic_fundamentals, [rule])
    assert len(signals) == 1
    assert "price change" in signals[0].reason
    assert "-3.2%" in signals[0].reason


def test_reason_string_sma_crossover(basic_fundamentals: dict) -> None:
    """SMA crossover should produce the expected reason string."""
    # SMA_50 crossed above SMA_200: prev SMA_50 <= SMA_200, now SMA_50 > SMA_200
    df = _make_df(
        n=5,
        SMA_50=[99.0, 99.0, 99.0, 99.0, 101.0],
        SMA_200=[100.0, 100.0, 100.0, 100.0, 100.0],
    )
    rule = Rule(
        name="golden_cross",
        type="buy",
        conditions=ConditionBlock(
            operator="AND",
            items=[
                ConditionItem(
                    indicator="SMA",
                    operator="crossed_above",
                    period=50,
                    compare_to_indicator="SMA",
                    compare_period=200,
                )
            ],
        ),
    )
    signals = evaluate_ticker("AAPL", df, basic_fundamentals, [rule])
    assert len(signals) == 1
    assert "SMA(50)" in signals[0].reason
    assert "crossed above" in signals[0].reason
    assert "SMA(200)" in signals[0].reason


def test_reason_string_macd(basic_fundamentals: dict) -> None:
    """MACD crossed_above_signal should produce the expected reason string."""
    # prev: MACD <= signal; now: MACD > signal
    df = _make_df(
        n=5,
        MACD_12_26_9=[0.1, 0.1, 0.1, 0.1, 0.5],
        MACDs_12_26_9=[0.2, 0.2, 0.2, 0.2, 0.3],
    )
    rule = Rule(
        name="macd_cross",
        type="buy",
        conditions=ConditionBlock(
            operator="AND",
            items=[
                ConditionItem(
                    indicator="MACD",
                    operator="crossed_above_signal",
                    fast=12,
                    slow=26,
                    signal=9,
                )
            ],
        ),
    )
    signals = evaluate_ticker("AAPL", df, basic_fundamentals, [rule])
    assert len(signals) == 1
    assert "MACD" in signals[0].reason
    assert "signal" in signals[0].reason
