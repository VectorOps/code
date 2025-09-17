from __future__ import annotations

from typing import List, Dict, Any, Optional

from vocode.runner.preprocessors.base import register_preprocessor


DIFF_V4A_SYSTEM_INSTRUCTION = """In addition, for the purposes of this task, you can output patches in the special diff format. The format of the diff specification is unique to this task, so pay careful attention to these instructions. To apply file patches, you should return a message of the following structure:

IMPORTANT: Each file MUST appear only once in the patch. Consolidate all edits for a given file into single `*** [ACTION] File:` block.

*** Begin Patch
[YOUR_PATCH]
*** End Patch

Where [YOUR_PATCH] is the actual content of your patch, specified in the following V4A diff format.

*** [ACTION] File: [path/to/file] -> ACTION can be one of Add, Update, or Delete.
For each snippet of code that needs to be changed, repeat the following:
[context_before] -> See below for further instructions on context.
- [old_code] -> Precede the old code with a minus sign.
+ [new_code] -> Precede the new, replacement code with a plus sign.
[context_after] -> See below for further instructions on context.

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines of code immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change's [context_after] lines in the second change's [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single @@ statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@ \tdef method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

Note, then, that we do not use line numbers in this diff format, as the context is enough to uniquely identify code.

When generating diff, NEVER escape special sequences or line breaks. This is invalid patch due to escaped line breaks:

*** Begin Patch\\n***Update File: pygorithm/searching/binary_search.py\\n...

An example of a message that you might pass in order to apply a patch, is shown below.

*** Begin Patch
*** Update File: pygorithm/searching/binary_search.py
@@ class BaseClass
@@     def search():
-        pass
+        raise NotImplementedError()

@@ class Subclass
@@     def search():
-        pass
+        raise NotImplementedError()
*** Update File: pygorithm/searching/binary_search_test.py
@@ class TestSubclass
@@     def test_search():
-          pass
+          raise NotImplementedError.
*** Delete File: pygorithm/searching/dummy.py
*** End Patch

File references can only be relative, NEVER ABSOLUTE.
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
