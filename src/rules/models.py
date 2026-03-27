"""Data models for the rules engine."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal


@dataclass
class ConditionItem:
    """Represents a single technical or fundamental condition."""

    indicator: str  # RSI, MACD, SMA, EMA, volume_spike, price_change_pct
    operator: str  # less_than, greater_than, crossed_above, crossed_below, crossed_above_signal
    period: int | None = None
    value: float | None = None
    multiplier: float | None = None
    vs_period: int | None = None
    fast: int | None = None
    slow: int | None = None
    signal: int | None = None
    compare_to_indicator: str | None = None
    compare_period: int | None = None


@dataclass
class ConditionBlock:
    """A group of conditions combined with AND or OR logic."""

    operator: Literal["AND", "OR"]
    items: list["ConditionItem | ConditionBlock"] = field(default_factory=list)


@dataclass
class FundamentalFilter:
    """Optional fundamental pre-screen applied before technical conditions."""

    max_pe: float | None = None
    min_market_cap_B: float | None = None


@dataclass
class Rule:
    """A complete trading rule with conditions and optional fundamental filter."""

    name: str
    type: Literal["buy", "sell"]
    conditions: ConditionBlock
    fundamentals: FundamentalFilter = field(default_factory=FundamentalFilter)


@dataclass
class Signal:
    """A triggered trading signal for a specific ticker."""

    ticker: str
    action: Literal["BUY", "SELL"]
    rule_name: str
    reason: str
    triggered_at: date
    price: float
