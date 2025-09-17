import pytest

from vocode.runner.executors.apply_patch import v4a
from vocode.runner.executors.apply_patch.v4a import DiffError


@pytest.fixture
def initial_files():
    """A common set of initial files for testing."""
    return {
        "file_a.txt": "line 1\nline 2\nline 3\nline 4\nline 5\n",
        "file_b.txt": "this file will be deleted",
        "file_c.txt": "alpha\nbravo\ncharlie\n",
    }


def test_process_patch_happy_path(initial_files):
    """Tests a successful patch with add, update, delete, and move operations."""
    patch_text = """*** Begin Patch
*** Update File: file_a.txt
@@ line 2
-line 3
+line three
@@ line 5
+line 6
*** Delete File: file_b.txt
*** Add File: new_file.txt
+hello new world
+this is a new line
*** Update File: file_c.txt
*** Move to: file_d.txt
@@ bravo
-charlie
+delta
*** End Patch"""

    # Mock file system operations
    files = initial_files.copy()

    def open_fn(path):
        return files[path]

    def write_fn(path, content):
        files[path] = content

    def remove_fn(path):
        del files[path]

    v4a.process_patch(patch_text, open_fn, write_fn, remove_fn)

    assert files == {
        "file_a.txt": "line 1\nline 2\nline three\nline 4\nline 5\nline 6\n",
        "new_file.txt": "hello new world\nthis is a new line",
        "file_d.txt": "alpha\nbravo\ndelta\n",
    }
    assert "file_b.txt" not in files
    assert "file_c.txt" not in files


@pytest.mark.parametrize(
    "patch_text, files, error_match",
    [
        # --- Sentinel and format errors ---
        (
            "*** Update File: file_a.txt\n@@ line 1\n-line 2\n*** End Patch",
            {"file_a.txt": "line 1\nline 2"},
            r"missing '.* Begin Patch' and/or '.* End Patch' sentinels",
        ),
        (
            "*** Begin Patch\n*** Update File: file_a.txt\n@@ line 1\n-line 2",
            {"file_a.txt": "line 1\nline 2"},
            r"missing '.* Begin Patch' and/or '.* End Patch' sentinels",
        ),
        (
            '*** Begin Patch\\n*** Update File: file_a.txt\\n@@ line 1\\n-line 2\\n*** End Patch',
            {"file_a.txt": "line 1\nline 2"},
            r"contains literal escape sequences",
        ),
        # --- Parser errors ---
        (
            "*** Begin Patch\n*** Update File: a.txt\n@@ 1\n-2\n*** Update File: a.txt\n@@ 3\n-4\n*** End Patch",
            {"a.txt": "1\n2\n3\n4"},
            "Duplicate update for file: a.txt",
        ),
        (
            "*** Begin Patch\n*** Delete File: a.txt\n*** Delete File: a.txt\n*** End Patch",
            {"a.txt": "1"},
            "Duplicate delete for file: a.txt",
        ),
        (
            "*** Begin Patch\n*** Add File: a.txt\n+foo\n*** Add File: a.txt\n+bar\n*** End Patch",
            {},
            "Duplicate add for file: a.txt",
        ),
        (
            "*** Begin Patch\n*** Update File: nonexistent.txt\n*** End Patch",
            {"a.txt": "1"},
            "Update File Error - missing file: nonexistent.txt",
        ),
        (
            "*** Begin Patch\n*** Delete File: nonexistent.txt\n*** End Patch",
            {"a.txt": "1"},
            "Delete File Error - missing file: nonexistent.txt",
        ),
        (
            "*** Begin Patch\n*** Add File: a.txt\n+foo\n*** End Patch",
            {"a.txt": "1"},
            "Add File Error - file already exists: a.txt",
        ),
        (
            "*** Begin Patch\n*** Bogus Command: a.txt\n*** End Patch",
            {},
            "Unknown line while parsing",
        ),
        # --- Section parsing errors ---
        (
            "*** Begin Patch\n*** Update File: a.txt\n@@ 1\n  2\ninvalid line\n*** End Patch",
            {"a.txt": "1\n2\n3"},
            "Invalid hunk line prefix",
        ),
        (
            # This tests context that cannot be found in the file
            "*** Begin Patch\n*** Update File: a.txt\n@@ 1\n  non-existent-context\n-3\n*** End Patch",
            {"a.txt": "1\n2\n3"},
            "Invalid context",
        ),
        (
            "*** Begin Patch\n*** Add File: a.txt\n+hello\nworld\n*** End Patch",
            {},
            r"Invalid Add File line \(missing '\+'\)",
        ),
        (
            # This tests a header appearing inside a hunk
            "*** Begin Patch\n*** Update File: a.txt\n@@ 1\n  2\n*** Bogus Header\n*** End Patch",
            {"a.txt": "1\n2\n3"},
            "Unexpected section/header marker inside a hunk",
        ),
        (
            "*** Begin Patch\n*** Update File: a.txt\n@@ 1\nx 2\n*** End Patch",
            {"a.txt": "1\n2"},
            "Invalid hunk line prefix",
        ),
        (
            "*** Begin Patch\n*** Update File: a.txt\n@@ \n*** End Patch",
            {"a.txt": "1"},
            "Nothing in this section",
        ),
        # --- Application errors (caught during parsing in current implementation) ---
        (
            # This test case for overlapping chunks now fails earlier on invalid context
            "*** Begin Patch\n*** Update File: a.txt\n@@ 1\n  2\n@@ 1\n  2\n*** End Patch",
            {"a.txt": "1\n2\n3"},
            "Invalid context",
        ),
    ],
)
def test_parsing_errors(patch_text, files, error_match):
    """Tests a wide range of parsing and structural errors in a patch."""
    with pytest.raises(DiffError, match=error_match):
        v4a.text_to_patch(patch_text, files)


def test_process_patch_wrapper_errors():
    """Tests errors from the top-level process_patch wrapper function."""
    with pytest.raises(DiffError, match="Patch text must start with .* Begin Patch"):
        v4a.process_patch("invalid patch", lambda p: "", lambda p, c: None, lambda p: None)

    with pytest.raises(DiffError, match="contains literal escape sequences"):
        v4a.process_patch(
            "*** Begin Patch\\n", lambda p: "", lambda p, c: None, lambda p: None
        )
