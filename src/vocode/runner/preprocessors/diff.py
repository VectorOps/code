from __future__ import annotations

from typing import List, Dict, Any, Optional
from vocode.runner.executors.apply_patch import SUPPORTED_PATCH_FORMATS
from vocode.models import PreprocessorSpec

from vocode.runner.preprocessors.base import register_preprocessor


def _diff_preprocessor(
    project, spec: PreprocessorSpec, text: str, **_: Any
) -> str:
    """
    Inject additional system instructions for diff patching formats.
    Options:
      - format: str, defaults to "v4a"
    Behavior:
      - If options['format'] matches a key in SUPPORTED_PATCH_FORMATS, merges that format's system_prompt with input text.
      - Honors spec.prepend to place instruction before or after the input text.
      - Otherwise, returns the text unchanged.
    """
    fmt = (spec.options or {}).get("format", "v4a")
    if isinstance(fmt, str):
        fmt = fmt.lower().strip()
    else:
        fmt = "v4a"

    entry = SUPPORTED_PATCH_FORMATS.get(fmt)
    if not entry:
        return text
    instruction = entry.system_prompt

    base_text = text or ""
    if spec.prepend:
        merged = (instruction + ("\n\n" if base_text else "") + base_text).strip()
    else:
        merged = (base_text + ("\n\n" if base_text else "") + instruction).strip()
    return merged


# Register at import time
register_preprocessor(
    name="diff",
    func=_diff_preprocessor,
    description=f"Injects system instructions for diff patches. Options: {{'format': one of {', '.join(sorted(SUPPORTED_PATCH_FORMATS.keys()))}}}",
)
