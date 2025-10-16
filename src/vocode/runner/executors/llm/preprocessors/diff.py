from __future__ import annotations

from typing import List, Dict, Any, Optional
from vocode.runner.executors.apply_patch import SUPPORTED_PATCH_FORMATS
from vocode.models import PreprocessorSpec, Mode
from vocode.state import Message

from vocode.runner.executors.llm.preprocessors.base import register_preprocessor


def _diff_preprocessor(project, spec: PreprocessorSpec, messages: List[Message]) -> List[Message]:
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
        return messages

    instruction = entry.system_prompt

    suffix = (spec.options or {}).get("suffix", "")
    target_message = None

    if spec.mode == Mode.System:
        for msg in messages:
            if msg.role == "system":
                target_message = msg
                break
    elif spec.mode == Mode.User:
        for msg in reversed(messages):
            if msg.role == "user":
                target_message = msg
                break

    if target_message:
        if spec.prepend:
            target_message.text = f"{instruction}{suffix}{target_message.text}"
        else:
            target_message.text = f"{target_message.text}{suffix}{instruction}"

    return messages


# Register at import time
register_preprocessor(
    name="diff",
    func=_diff_preprocessor,
    description=f"Injects system instructions for diff patches. Options: {{'format': one of {', '.join(sorted(SUPPORTED_PATCH_FORMATS.keys()))}}}",
)
