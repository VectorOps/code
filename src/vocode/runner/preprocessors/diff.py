from __future__ import annotations

from typing import List, Dict, Any, Optional

from vocode.runner.preprocessors.base import register_preprocessor


# Alternative V4A prompt as original adds random escapes
DIFF_V4A_SYSTEM_INSTRUCTION = """You must output exactly one fenced code block containing a raw V4A patch. No prose before or after. Do not wrap the patch in JSON/YAML/strings. Do not emit backslash-escapes (`\n`, `\t`, `\"`) unless those characters exist in the original file contents.

Required envelope:
```patch
*** Begin Patch
[YOUR_PATCH]
*** End Patch
V4A format inside the envelope:

Each file appears exactly once as one of:
*** Add File: <relative/path>
*** Update File: <relative/path>
*** Delete File: <relative/path>

For Update/Add files, changes are expressed with context blocks:

[3 lines of pre-context EXACTLY matching file contents]
- <old line>
+ <new line>
[3 lines of post-context EXACTLY matching file contents]

Context rules:

* Show no more than 3 lines of context above and below each change by default.
* If insufficient to disambiguate, add an @@ anchor naming the class or function:
  @@ class BaseClass
  @@ def method_name(...):

* If two changes contexts would overlap, do not duplicate overlapping lines.

Absolute paths are forbidden; use relative paths only.

Validation checklist (you must pass all before emitting):
1. Output is exactly one fenced code block labeled patch.
2. Contains one *** Begin Patch and one *** End Patch.
3. No JSON, no quotes around the patch, no extra text outside the fence.
4. *No* backslash-escapes unless they exist in the source file.
5. Each changed file appears exactly once.
6. When creating the patch, ensure the number of blank lines in the context sections exactly matches the source file, as any mismatch will cause the patch to fail.

Minimal example (fenced, raw):

```patch
*** Begin Patch
*** Update File: pygorithm/searching/binary_search.py
@@ class BaseClass
@@     def search(self):
-        pass
+        raise NotImplementedError()
@@ class Subclass
@@     def search(self):
-        pass
+        raise NotImplementedError()
*** Update File: pygorithm/searching/binary_search_test.py
@@ class TestSubclass
@@     def test_search(self):
-        pass
+        raise NotImplementedError()
*** Delete File: pygorithm/searching/dummy.py
*** End Patch
```
"""


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
    description="Injects system instructions for V4A diff patch format. Options: {'format': 'v4a'}",
)
