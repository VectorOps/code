from __future__ import annotations

from typing import Any, Dict
import re


def get_value_by_dotted_key(data: Dict[str, Any], key: str) -> Any | None:
    """Resolve a dotted key path (e.g. "a.b.c") inside a nested dict.

    Returns None when any path segment is missing or encounters a non-dict.
    """

    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def regex_matches_value(pattern: str, value: Any) -> bool:
    """Return True when the given regex pattern matches the value string.

    Assumes the pattern has already been validated/compiled elsewhere.
    """

    text = str(value)
    return re.search(pattern, text) is not None
