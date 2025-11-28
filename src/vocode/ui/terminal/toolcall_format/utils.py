from __future__ import annotations

from typing import Any, Iterable, List, Sequence

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.formatted_text.utils import fragment_list_width


def _extract_field_value(payload: Any, field: str) -> List[Any]:
    """
    Return [payload[field]] when payload is a mapping and contains the field.
    Otherwise return an empty list, indicating 'absent'.
    """
    from collections.abc import Mapping as _Mapping

    if isinstance(payload, _Mapping) and field in payload:  # type: ignore[arg-type]
        return [payload[field]]  # type: ignore[index]
    return []


def _stringify_value(v: Any) -> str:
    if isinstance(v, str):
        return f"\"{v}\""
    return str(v)


def _flatten_params(extracted: Iterable[Any]) -> List[str]:
    """
    Flatten a mixture of scalars and lists into a flat list of displayable parameter strings.
    - Lists are expanded into individual parameters.
    - Strings are wrapped in double quotes.
    """
    out: List[str] = []
    for item in extracted:
        if isinstance(item, (list, tuple)):
            for sub in item:
                out.append(_stringify_value(sub))
        else:
            out.append(_stringify_value(item))
    return out


def _build_param_fragments(params: Sequence[str]) -> FormattedText:
    """
    Build fragments for parameters: value (toolcall.parameter) and comma+space
    separators (toolcall.separator).
    """
    fragments: FormattedText = []
    for idx, p in enumerate(params):
        fragments.append(("class:toolcall.parameter", p))
        if idx < len(params) - 1:
            fragments.append(("class:toolcall.separator", ", "))
    return fragments


def _truncate_params_to_width(
    prefix: FormattedText,
    params: FormattedText,
    suffix: FormattedText,
    max_total_width: int,
) -> FormattedText:
    """
    Truncate only the parameter fragments to ensure the whole line does not exceed
    max_total_width. Adds an ellipsis "..." styled as toolcall.parameter before the
    closing parenthesis when truncation occurs. Keeps the closing parenthesis.
    """
    prefix_w = fragment_list_width(prefix)
    suffix_w = fragment_list_width(suffix)
    remaining_for_params = max(0, max_total_width - prefix_w - suffix_w)
    if remaining_for_params <= 0:
        ellipsis = [("class:toolcall.parameter", "...")]
        return list(prefix) + ellipsis + list(suffix)

    acc: FormattedText = []
    used = 0
    for style, text in params:
        w = len(text)
        if used + w <= remaining_for_params:
            acc.append((style, text))
            used += w
            continue

        acc.append(("class:toolcall.parameter", "..."))
        break

    if not acc:
        acc = [("class:toolcall.parameter", "...")]

    return list(prefix) + acc + list(suffix)
