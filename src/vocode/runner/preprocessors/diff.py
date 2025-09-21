from __future__ import annotations

from typing import List, Dict, Any, Optional

from vocode.runner.preprocessors.base import register_preprocessor

DIFF_V4A_SYSTEM_INSTRUCTION = r"""You must output exactly one fenced code block labeled patch. No prose before or after. Do not wrap the patch in JSON/YAML/strings. Do not add backslash-escapes (\n, \t, \") unless they exist in the original file contents. Never double-escape.

Required envelope:
```patch
*** Begin Patch
[YOUR_PATCH]
*** End Patch
```

[YOUR_PATCH] contains file sections (each file appears exactly once):
*** Add File: <relative/path>
*** Update File: <relative/path>
*** Delete File: <relative/path>

For Update/Add files, changes are expressed with context blocks:

[Up to 3 lines of context]
- <old line prefixed with minus and one space>
+ <new line prefixed with plus and one space>
[Up to 3 lines of context]

Update/Add blocks: exact context and edits
- Context must be an exact copy of the file lines.
- Do not escape any quotes, backslashes, or newlines. Produce the file content literally as it appears in the source, character for character.
- Preserve blank lines in context. Represent a blank context line as a completely empty line (no leading space).
- For non-blank context lines, start with a single space, then the exact text.
- Include at least one line of pre- and post-context; add more if helpful. Be conservative, do not include whole file.
- Use @@ anchor to separate multiple changes when needed:
  @@
- If insufficient to disambiguate, add an @@ anchor naming the class or function:
  @@ class BaseClass
  @@     def method_name(...):

Change lines:
- Use '-' for the old line, '+' for the new line.
- The text after the sign must be exact (including whitespace).

Backslashes and escapes (important):
- Write raw file text. Never add Markdown/JSON escaping.
- Never double-escape backslashes in strings.
  Example (Python):
    Source:
        assert s == "a\\b\n"
    Correct:
        - assert s == "a\\b\n"
        + assert s == "a\\c\n"
    Wrong (double-escaped) — do NOT output:
        - assert s == "a\\\\b\\n"

Blank line context example (do not omit the blank line):
 header1

- old
+ new
 footer1

Minimal example:
```patch
*** Begin Patch
*** Update File: pkg/mod.py
 header1

- old
+ new
 footer1
*** Add File: pkg/new.txt
+ hello
*** Delete File: pkg/unused.txt
*** End Patch
```"""


def _diff_preprocessor(text: str, options: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    """
    Inject additional system instructions for diff patching formats.
    Options:
      - format: str, defaults to "v4a"
    Behavior:
      - For format == "v4a", appends DIFF_V4A_SYSTEM_INSTRUCTION to the input text.
    """
    fmt = (options or {}).get("format", "v4a")

    if isinstance(fmt, str):
        fmt = fmt.lower().strip()
    else:
        fmt = "v4a"

    if fmt != "v4a":
        return text

    base_text = text or ""
    new_text = (base_text + ("\n\n" if base_text else "") + DIFF_V4A_SYSTEM_INSTRUCTION).strip()
    return new_text


# Register at import time
register_preprocessor(
    name="diff",
    func=_diff_preprocessor,
    description="Injects simplified system instructions for V4A diff patches (exact context, preserve blank lines, no double-escaping). Options: {'format': 'v4a'}",
)
