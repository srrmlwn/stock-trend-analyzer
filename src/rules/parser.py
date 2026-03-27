"""Parser for loading and validating rules.yaml configuration."""

import logging
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.rules.models import ConditionBlock, ConditionItem, FundamentalFilter, Rule

logger = logging.getLogger(__name__)

VALID_TYPES = {"buy", "sell"}
VALID_OPERATORS = {"AND", "OR"}


def _parse_condition_item(item_data: dict[str, Any]) -> ConditionItem:
    """Parse a single condition item dict into a ConditionItem dataclass.

    Args:
        item_data: Dictionary containing condition item fields.

    Returns:
        Parsed ConditionItem instance.
    """
    return ConditionItem(
        indicator=item_data["indicator"],
        operator=item_data["operator"],
        period=item_data.get("period"),
        value=item_data.get("value"),
        multiplier=item_data.get("multiplier"),
        vs_period=item_data.get("vs_period"),
        fast=item_data.get("fast"),
        slow=item_data.get("slow"),
        signal=item_data.get("signal"),
        compare_to_indicator=item_data.get("compare_to_indicator"),
        compare_period=item_data.get("compare_period"),
    )


def _parse_condition_block(block_data: dict[str, Any], path: str = "conditions") -> ConditionBlock:
    """Recursively parse a condition block into a ConditionBlock dataclass.

    Args:
        block_data: Dictionary containing operator and items.
        path: Dotted path string for descriptive error messages.

    Returns:
        Parsed ConditionBlock instance.

    Raises:
        ValueError: If the block is invalid (missing operator, empty items, etc.).
    """
    if "operator" not in block_data:
        raise ValueError(f"Condition block at '{path}' is missing 'operator'")

    operator = block_data["operator"]
    if operator not in VALID_OPERATORS:
        raise ValueError(
            f"Condition block at '{path}' has invalid operator '{operator}'. "
            f"Must be one of: {sorted(VALID_OPERATORS)}"
        )

    if "items" not in block_data:
        raise ValueError(f"Condition block at '{path}' is missing 'items'")

    raw_items = block_data["items"]
    if not isinstance(raw_items, list) or len(raw_items) == 0:
        raise ValueError(
            f"Condition block at '{path}' must have a non-empty list of items"
        )

    parsed_items: list[ConditionItem | ConditionBlock] = []
    for idx, item in enumerate(raw_items):
        item_path = f"{path}.items[{idx}]"
        if not isinstance(item, dict):
            raise ValueError(f"Item at '{item_path}' must be a mapping, got {type(item).__name__}")

        if "indicator" in item:
            # It's a leaf ConditionItem
            if "operator" not in item:
                raise ValueError(f"ConditionItem at '{item_path}' is missing 'operator'")
            parsed_items.append(_parse_condition_item(item))
        elif "operator" in item:
            # It's a nested ConditionBlock
            parsed_items.append(_parse_condition_block(item, path=item_path))
        else:
            raise ValueError(
                f"Item at '{item_path}' must have either 'indicator' (ConditionItem) "
                f"or 'operator' (nested ConditionBlock)"
            )

    return ConditionBlock(operator=operator, items=parsed_items)


def _parse_fundamentals(fund_data: dict[str, Any]) -> FundamentalFilter:
    """Parse a fundamentals dict into a FundamentalFilter dataclass.

    Args:
        fund_data: Dictionary containing fundamental filter fields.

    Returns:
        Parsed FundamentalFilter instance.
    """
    return FundamentalFilter(
        max_pe=fund_data.get("max_pe"),
        min_market_cap_B=fund_data.get("min_market_cap_B"),
    )


def _parse_rule(rule_data: dict[str, Any], index: int) -> Rule:
    """Parse a single rule dict into a Rule dataclass.

    Args:
        rule_data: Dictionary containing rule fields.
        index: Index of the rule in the list (for error messages).

    Returns:
        Parsed Rule instance.

    Raises:
        ValueError: If the rule is invalid.
    """
    path = f"rules[{index}]"

    if "name" not in rule_data:
        raise ValueError(f"Rule at '{path}' is missing 'name'")

    name = rule_data["name"]

    if "type" not in rule_data:
        raise ValueError(f"Rule '{name}' is missing 'type'")

    rule_type = rule_data["type"]
    if rule_type not in VALID_TYPES:
        raise ValueError(
            f"Rule '{name}' has invalid type '{rule_type}'. Must be one of: {sorted(VALID_TYPES)}"
        )

    if "conditions" not in rule_data:
        raise ValueError(f"Rule '{name}' is missing 'conditions'")

    conditions_data = rule_data["conditions"]
    if not isinstance(conditions_data, dict):
        raise ValueError(
            f"Rule '{name}' conditions must be a mapping, got {type(conditions_data).__name__}"
        )

    conditions = _parse_condition_block(conditions_data, path=f"{path}.conditions")

    fundamentals = FundamentalFilter()
    if "fundamentals" in rule_data and rule_data["fundamentals"] is not None:
        fund_data = rule_data["fundamentals"]
        if not isinstance(fund_data, dict):
            raise ValueError(
                f"Rule '{name}' fundamentals must be a mapping, got {type(fund_data).__name__}"
            )
        fundamentals = _parse_fundamentals(fund_data)

    return Rule(
        name=name,
        type=rule_type,
        conditions=conditions,
        fundamentals=fundamentals,
    )


def load_rules(config_path: str = "config/rules.yaml") -> list[Rule]:
    """Parse and validate rules.yaml, returning a list of Rule objects.

    Args:
        config_path: Path to the rules YAML configuration file.

    Returns:
        List of validated Rule instances.

    Raises:
        ValueError: If the config is missing, malformed, or fails validation.
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Rules config not found: {config_path}")

    logger.info("Loading rules from %s", config_path)

    with path.open("r") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Rules config must be a YAML mapping, got {type(raw).__name__}")

    if "rules" not in raw:
        raise ValueError("Rules config is missing top-level 'rules' key")

    rules_list = raw["rules"]
    if not isinstance(rules_list, list):
        raise ValueError(
            f"'rules' must be a list, got {type(rules_list).__name__}"
        )

    rules: list[Rule] = []
    for idx, rule_data in enumerate(rules_list):
        if not isinstance(rule_data, dict):
            raise ValueError(
                f"Rule at index {idx} must be a mapping, got {type(rule_data).__name__}"
            )
        rule = _parse_rule(rule_data, idx)
        rules.append(rule)
        logger.debug("Loaded rule: %s (%s)", rule.name, rule.type)

    logger.info("Loaded %d rules from %s", len(rules), config_path)
    return rules
