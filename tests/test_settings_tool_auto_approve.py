import pytest

from vocode.settings import (
    ToolAutoApproveRule,
)
from vocode.lib.validators import tool_auto_approve_matches


def test_tool_auto_approve_rule_valid_regex():
    rule = ToolAutoApproveRule(key="action", pattern=r"^deploy$")
    assert rule.key == "action"


def test_tool_auto_approve_rule_invalid_regex_raises():
    with pytest.raises(ValueError):
        ToolAutoApproveRule(key="action", pattern=r"[*invalid")


def test_tool_auto_approve_matches_simple_key():
    rules = [ToolAutoApproveRule(key="action", pattern=r"^deploy$")]
    assert tool_auto_approve_matches(rules, {"action": "deploy"}) is True
    assert tool_auto_approve_matches(rules, {"action": "plan"}) is False


def test_tool_auto_approve_matches_dotted_key():
    rules = [ToolAutoApproveRule(key="resource.name", pattern=r"^prod-")]
    args_ok = {"resource": {"name": "prod-api"}}
    args_miss = {"resource": {"name": "dev-api"}}
    args_missing = {"resource": {}}

    assert tool_auto_approve_matches(rules, args_ok) is True
    assert tool_auto_approve_matches(rules, args_miss) is False
    assert tool_auto_approve_matches(rules, args_missing) is False


def test_tool_auto_approve_matches_non_string_values_are_stringified():
    rules = [ToolAutoApproveRule(key="count", pattern=r"^10$")]
    assert tool_auto_approve_matches(rules, {"count": 10}) is True
    assert tool_auto_approve_matches(rules, {"count": 5}) is False


def test_tool_auto_approve_matches_no_rules():
    assert tool_auto_approve_matches([], {"any": "value"}) is False
