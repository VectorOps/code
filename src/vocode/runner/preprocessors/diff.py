from __future__ import annotations

from typing import List, Dict, Any, Optional

from vocode.runner.preprocessors.base import register_preprocessor

DIFF_V4A_SYSTEM_INSTRUCTION = r"""You must output *exactly one* fenced code block labeled patch.
No prose before or after.
Do not wrap the patch in JSON/YAML/strings.
Do not add backslash-escapes (\n, \t, \") or html-escapes (&quot; and similar) unless they *literally* present in the source file.
*Never* double-escape.

Required envelope:
```patch
*** Begin Patch
[YOUR_PATCH]
*** End Patch
```

[YOUR_PATCH] is a concatenation of file sections.
Each file appears exactly once in [YOUR_PATCH].

Allowed section headers per file:
- `*** Add File: <relative/path>`
- `*** Update File: <relative/path>`
- `*** Delete File: <relative/path>`

For Update/Add files, changes are expressed with context blocks:

[0-3 lines of context before]
-<old line>
+<new line>
[0-3 lines of context after]

Update/Add blocks: exact context and edits
- Context must be an exact copy of the file lines with leading space.
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

Rules:

1. Literal text only. Emit the file’s exact bytes as text lines.
 * Do not add Markdown/JSON escaping.
 * Preserve quotes, backslashes, tabs, Unicode, and blank lines exactly.

2. Line prefixes:
 * Non-blank context lines start with one leading space followed by the exact text.
 * Change lines start with - (old) or + (new) followed by the exact text.
 * A blank context line is completely empty (no spaces).

3. Context must match the current file character for character.
 * Provide at least one line of pre- and post-context when updating. Add up to 3 lines if it helps disambiguate.

4. Multiple changes in the same file: separate blocks with a line containing exactly:
```patch
@@
```
 Optionally disambiguate with a labeled anchor on its own line:
```patch
@@ class ClassName
@@     def method_name(...):
```

5. Ordering & uniqueness:
 * Include each file path once. Merge all changes for that file into its single section.
 * Within a file, order context blocks top-to-bottom by their occurrence in the file.
 * Across files, sort sections lexicographically by path (recommended).

6. Newlines:
 * Preserve each line’s trailing newline semantics.
 * If the file ends without a trailing newline, represent the last line exactly as it exists (no extra newline).

7. Tabs & spaces: Preserve indentation exactly; do not convert tabs to spaces or vice versa.

8. Binary or non-text files: Do not attempt to inline binary data. Omit such files unless your toolchain supports textual diffs for them.

## Minimal example:
```patch
*** Begin Patch
*** Update File: pkg/mod.py
 header1

-old
+new
 footer1
*** End Patch
```

## Add + Update + Delete together:
```patch
*** Begin Patch
*** Update File: src/core/runner.py
 class Runner:
-    def run(self):
+    def run(self) -> None:
         self._init()
@@
@@     def _init(self):
-        setup()
+        setup(verbose=True)
*** Add File: tests/test_runner.py
+import pytest
+
+def test_runner_smoke():
+    assert True
*** Delete File: scripts/old_tool.py
*** End Patch
```

## Escapes (emit literally, never double-escape):
Source contains assert s == "a\\b\n"

```patch
*** Begin Patch
*** Update File: src/checks.py
     def test_str():
-        assert s == "a\\b\n"
+        assert s == "a\\c\n"
*** End Patch
```

(Incorrect/double-escaped forms like \\n or \\\\ are forbidden unless they exist in the source.)

# Self-Check (must pass before you output)

* Exactly one fenced code block labeled patch, nothing else.
* Envelope lines are present and exact: *** Begin Patch … *** End Patch.
* Each file path appears once; Add/Update/Delete headers are correct.
* For Update sections: at least one pre- and post-context line; context matches file exactly.
* No JSON/Markdown escaping added; quotes/backslashes/tabs preserved literally.
* Blank context lines are truly empty; non-blank context lines start with one space.
* Multiple edits within a file are separated by @@ (and optional labeled anchors if needed).
* No trailing commentary or extra fences outside the single patch block.
"""

DIFF_PATCH_SYSTEM_INSTRUCTION = r"""# Patch format: SEARCH/REPLACE blocks

**IMPORTANT:** You implement exactly the ARCHITECT PLAN. Follow repo style. Keep edits minimal. No speculation. No reformatting. No new deps unless the plan says so.

**OUTPUT:** Only patch blocks. No prose before/between/after.

## Format
Emit exactly one SEARCH/REPLACE fenced block per change using the file’s language tag:

```<lang>
<full/path/to/file>
<<<<<<< SEARCH
<contiguous lines that EXACTLY match current file content>
=======
<replacement lines>
>>>>>>> REPLACE
```

Edits: use the format above.
Adds (new file): leave SEARCH empty; put full file contents in REPLACE.
Deletes: put entire current file in SEARCH; leave REPLACE empty.

## Rules
1. SEARCH must match character-for-character (whitespace, quotes, comments, docstrings).
2. Include enough lines in SEARCH to uniquely identify lines being replaced.
3. No other diff headers, line numbers, or markers.
4. Keep changes narrowly scoped; avoid touching unrelated code.
5. Use existing libs/patterns; keep imports/types/names consistent.
6. SEARCH/REPLACE will only change first occurence.
7. Keep changes small. Break larger changes into series of SEARCH/REPLACE blocks.
8. You are allowed to emit multiple blocks per file, but blocks should not overlap.

## Self-check before emitting
1. All planned changes covered?
2. SEARCH sections exact? Imports/types/tests correct? Unrelated edits avoided?
"""


def _diff_preprocessor(text: str, options: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    """
    Inject additional system instructions for diff patching formats.
    Options:
      - format: str, defaults to "v4a"
    Behavior:
      - For format == "v4a", appends DIFF_V4A_SYSTEM_INSTRUCTION.
      - For format in {'patch','fenced','search_replace','sr'}, appends DIFF_PATCH_SYSTEM_INSTRUCTION.
    """
    fmt = (options or {}).get("format", "v4a")

    if isinstance(fmt, str):
        fmt = fmt.lower().strip()
    else:
        fmt = "v4a"

    instruction: Optional[str] = None
    if fmt == "v4a":
        instruction = DIFF_V4A_SYSTEM_INSTRUCTION
    elif fmt == "patch":
        instruction = DIFF_PATCH_SYSTEM_INSTRUCTION
    else:
        return text

    base_text = text or ""
    new_text = (base_text + ("\n\n" if base_text else "") + instruction).strip()
    return new_text


# Register at import time
register_preprocessor(
    name="diff",
    func=_diff_preprocessor,
    description="Injects system instructions for diff patches. Options: {'format': 'v4a'|'patch'} — 'v4a' => V4A context diffs; 'patch' => fenced SEARCH/REPLACE blocks.",
)
