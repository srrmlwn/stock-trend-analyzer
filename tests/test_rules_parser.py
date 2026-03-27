"""Tests for the rules parser module."""

import textwrap
from pathlib import Path

import pytest
import yaml

from src.rules.models import ConditionBlock, ConditionItem, FundamentalFilter, Rule
from src.rules.parser import load_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_rules_yaml(tmp_path: Path, content: dict) -> str:
    """Write a rules config dict as YAML and return the file path."""
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.dump(content))
    return str(p)


def minimal_rule(
    name: str = "test_rule",
    rule_type: str = "buy",
    conditions: dict | None = None,
) -> dict:
    """Return a minimal valid rule dict."""
    if conditions is None:
        conditions = {
            "operator": "AND",
            "items": [
                {"indicator": "RSI", "period": 14, "operator": "less_than", "value": 30}
            ],
        }
    return {"name": name, "type": rule_type, "conditions": conditions}


# ---------------------------------------------------------------------------
# Valid config tests
# ---------------------------------------------------------------------------


def test_valid_rules_parses_correctly(tmp_path: Path) -> None:
    """A well-formed rules.yaml should parse into the expected Rule objects."""
    config = {
        "rules": [
            minimal_rule("buy_rsi", "buy"),
            minimal_rule("sell_rsi", "sell"),
        ]
    }
    rules = load_rules(write_rules_yaml(tmp_path, config))

    assert len(rules) == 2
    assert isinstance(rules[0], Rule)
    assert rules[0].name == "buy_rsi"
    assert rules[0].type == "buy"
    assert isinstance(rules[0].conditions, ConditionBlock)
    assert rules[0].conditions.operator == "AND"
    assert len(rules[0].conditions.items) == 1

    item = rules[0].conditions.items[0]
    assert isinstance(item, ConditionItem)
    assert item.indicator == "RSI"
    assert item.period == 14
    assert item.operator == "less_than"
    assert item.value == 30


def test_and_or_nesting(tmp_path: Path) -> None:
    """Nested AND/OR condition blocks should parse recursively."""
    config = {
        "rules": [
            {
                "name": "nested_rule",
                "type": "buy",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {"indicator": "RSI", "period": 14, "operator": "less_than", "value": 30},
                        {
                            "operator": "OR",
                            "items": [
                                {
                                    "indicator": "volume_spike",
                                    "vs_period": 20,
                                    "operator": "greater_than",
                                    "multiplier": 1.5,
                                },
                                {
                                    "indicator": "price_change_pct",
                                    "period": 5,
                                    "operator": "less_than",
                                    "value": -3.0,
                                },
                            ],
                        },
                    ],
                },
            }
        ]
    }
    rules = load_rules(write_rules_yaml(tmp_path, config))
    assert len(rules) == 1
    block = rules[0].conditions
    assert block.operator == "AND"
    assert len(block.items) == 2

    nested = block.items[1]
    assert isinstance(nested, ConditionBlock)
    assert nested.operator == "OR"
    assert len(nested.items) == 2


def test_fundamentals_block_is_optional(tmp_path: Path) -> None:
    """Rules without a fundamentals block should default to FundamentalFilter()."""
    config = {"rules": [minimal_rule()]}
    rules = load_rules(write_rules_yaml(tmp_path, config))
    assert rules[0].fundamentals == FundamentalFilter()
    assert rules[0].fundamentals.max_pe is None
    assert rules[0].fundamentals.min_market_cap_B is None


def test_fundamentals_block_parsed(tmp_path: Path) -> None:
    """Rules with a fundamentals block should parse max_pe and min_market_cap_B."""
    rule_dict = minimal_rule()
    rule_dict["fundamentals"] = {"max_pe": 40, "min_market_cap_B": 2.0}
    config = {"rules": [rule_dict]}
    rules = load_rules(write_rules_yaml(tmp_path, config))
    assert rules[0].fundamentals.max_pe == 40.0
    assert rules[0].fundamentals.min_market_cap_B == 2.0


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_missing_name_raises_value_error(tmp_path: Path) -> None:
    """A rule missing 'name' should raise ValueError."""
    config = {
        "rules": [
            {
                "type": "buy",
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {"indicator": "RSI", "period": 14, "operator": "less_than", "value": 30}
                    ],
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="missing 'name'"):
        load_rules(write_rules_yaml(tmp_path, config))


def test_invalid_type_raises_value_error(tmp_path: Path) -> None:
    """A rule with type other than buy/sell should raise ValueError."""
    config = {"rules": [minimal_rule(rule_type="hold")]}
    with pytest.raises(ValueError, match="invalid type"):
        load_rules(write_rules_yaml(tmp_path, config))


def test_empty_items_raises_value_error(tmp_path: Path) -> None:
    """A condition block with an empty items list should raise ValueError."""
    config = {
        "rules": [
            {
                "name": "bad_rule",
                "type": "buy",
                "conditions": {
                    "operator": "AND",
                    "items": [],
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="non-empty"):
        load_rules(write_rules_yaml(tmp_path, config))


def test_missing_conditions_raises_value_error(tmp_path: Path) -> None:
    """A rule without 'conditions' should raise ValueError."""
    config = {"rules": [{"name": "bad", "type": "buy"}]}
    with pytest.raises(ValueError, match="missing 'conditions'"):
        load_rules(write_rules_yaml(tmp_path, config))


def test_missing_conditions_operator_raises_value_error(tmp_path: Path) -> None:
    """A conditions block without 'operator' should raise ValueError."""
    config = {
        "rules": [
            {
                "name": "bad",
                "type": "buy",
                "conditions": {
                    "items": [
                        {"indicator": "RSI", "period": 14, "operator": "less_than", "value": 30}
                    ]
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="missing 'operator'"):
        load_rules(write_rules_yaml(tmp_path, config))


def test_invalid_conditions_operator_raises_value_error(tmp_path: Path) -> None:
    """A conditions block with operator other than AND/OR should raise ValueError."""
    config = {
        "rules": [
            {
                "name": "bad",
                "type": "buy",
                "conditions": {
                    "operator": "XOR",
                    "items": [
                        {"indicator": "RSI", "period": 14, "operator": "less_than", "value": 30}
                    ],
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="invalid operator"):
        load_rules(write_rules_yaml(tmp_path, config))


def test_file_not_found_raises() -> None:
    """Requesting a non-existent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_rules("/tmp/does_not_exist_rules.yaml")


def test_real_rules_yaml_parses() -> None:
    """The actual config/rules.yaml in the repo should parse without errors."""
    rules = load_rules("config/rules.yaml")
    assert len(rules) > 0
    for rule in rules:
        assert isinstance(rule, Rule)
        assert rule.type in ("buy", "sell")
