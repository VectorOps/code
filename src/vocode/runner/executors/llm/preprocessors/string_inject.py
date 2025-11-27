from __future__ import annotations

from typing import Any, List

from vocode.models import Mode, PreprocessorSpec
from vocode.runner.executors.llm.preprocessors.base import register_preprocessor
from vocode.state import Message


def _string_inject_preprocessor(
    project: Any, spec: PreprocessorSpec, messages: List[Message]
) -> List[Message]:
    """
    Injects a literal string from options['text'] into either the system or user
    message, based on spec.mode.

    Behavior:
    - Reads options['text']; if it's not a non-empty string, returns messages unchanged.
    - Chooses the target message by spec.mode:
      * Mode.System: first system message in messages.
      * Mode.User: last user message in messages.
    - If messages is empty, creates a new message with role derived from spec.mode.
    - Honors spec.prepend to place the injected text before or after existing text.
    - Uses options['separator'] (default: "\\n\\n") between existing and injected text.
    - Skips injection if the text is already present in the target message.
    """
    opts = spec.options or {}
    raw = opts.get("text")

    if not isinstance(raw, str):
        return messages

    inject = raw.strip()
    if not inject:
        return messages

    target_message: Message | None = None

    if not messages:
        role = "system" if spec.mode == Mode.System else "user"
        target_message = Message(text="", role=role)
        messages.append(target_message)
    else:
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

    if not target_message:
        return messages

    base_text = target_message.text or ""
    # Simple dedupe: if the injected text is already present, skip.
    if inject in base_text:
        return messages

    separator = opts.get("separator", "\n\n")

    if spec.prepend:
        if base_text:
            target_message.text = f"{inject}{separator}{base_text}"
        else:
            target_message.text = inject
    else:
        if base_text:
            target_message.text = f"{base_text}{separator}{inject}"
        else:
            target_message.text = inject

    return messages


register_preprocessor(
    name="string_inject",
    func=_string_inject_preprocessor,
    description=(
        "Injects a literal string from options.text into the system or user "
        "message, chosen by spec.mode; supports prepend/append and a configurable separator."
    ),
)