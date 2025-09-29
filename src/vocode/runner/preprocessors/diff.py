from __future__ import annotations

from typing import List, Dict, Any, Optional
from vocode.runner.executors.apply_patch import SUPPORTED_PATCH_FORMATS

from vocode.runner.preprocessors.base import register_preprocessor


def _diff_preprocessor(
    text: str, options: Optional[Dict[str, Any]] = None, **_: Any
) -> str:
    """
    Inject additional system instructions for diff patching formats.
    Options:
      - format: str, defaults to "v4a"
    Behavior:
      - If options['format'] matches a key in SUPPORTED_PATCH_FORMATS, appends that format's system_prompt.
      - Otherwise, returns the text unchanged.
    """
    fmt = (options or {}).get("format", "v4a")
    if isinstance(fmt, str):
        fmt = fmt.lower().strip()
    else:
        fmt = "v4a"

    entry = SUPPORTED_PATCH_FORMATS.get(fmt)
    if not entry:
        return text
    instruction = entry.system_prompt

    base_text = text or ""
    new_text = (base_text + ("\n\n" if base_text else "") + instruction).strip()
    return new_text


# Register at import time
register_preprocessor(
    name="diff",
    func=_diff_preprocessor,
    description=f"Injects system instructions for diff patches. Options: {{'format': one of {', '.join(sorted(SUPPORTED_PATCH_FORMATS.keys()))}}}",
)
