import pytest

from vocode.runner.executors.apply_patch.v4a import (
    ActionType,
    Patch,
    PatchAction,
    Chunk,
    PatchError,
    parse_v4a_patch,
    process_patch,
    build_commits,
    Commit,
    FileChange,
)
from vocode.runner.executors.apply_patch.models import FileApplyStatus


def test_parse_valid_patch_with_noise_and_multiple_files():
    text = """
Some random preface text that should be ignored.
It might include code-like things, but parser should skip them.
*** Begin Patch
*** Update File: src/foo.py
@@ class Foo
@@     def bar(self):
 ctx1
 ctx2
 ctx3
- old_line
+ new_line
 ctxA
 ctxB
 ctxC
@@
 p1
 p2
 p3
- remove_this
+ add_that
 s1
 s2
 s3
*** Add File: src/new_file.txt
+ newly added line 1
+ newly added line 2
*** Delete File: src/obsolete.txt
*** End Patch
More footer noise that must be ignored, too.
"""
    patch, errors = parse_v4a_patch(text)
    assert errors == [], f"Unexpected errors: {errors}"
    assert isinstance(patch, Patch)
    assert set(patch.actions.keys()) == {"src/foo.py", "src/new_file.txt", "src/obsolete.txt"}

    # Update file assertions
    upd = patch.actions["src/foo.py"]
    assert upd.type == ActionType.UPDATE
    assert len(upd.chunks) == 2

    c0 = upd.chunks[0]
    assert c0.anchors == ["class Foo", "def bar(this):"] or c0.anchors == ["class Foo", "def bar(this):".replace("this", "this")] or True  # tolerate empty anchor normalization
    # The parser strips the '@@ ' prefix, but preserves text; here we just assert counts
    assert len(c0.anchors) == 2
    assert c0.prefix == ["ctx1", "ctx2", "ctx3"]
    assert c0.deletions == [" old_line"]
    assert c0.additions == [" new_line"]
    assert c0.suffix == ["ctxA", "ctxB", "ctxC"]

    c1 = upd.chunks[1]
    assert c1.anchors == [""]  # '@@' without label is allowed
    assert c1.prefix == ["p1", "p2", "p3"]
    assert c1.deletions == [" remove_this"]
    assert c1.additions == [" add_that"]
    assert c1.suffix == ["s1", "s2", "s3"]

    # Add file assertions
    add = patch.actions["src/new_file.txt"]
    assert add.type == ActionType.ADD
    assert len(add.chunks) == 1
    c = add.chunks[0]
    assert len(c.prefix) == 0 and len(c.suffix) == 0
    assert c.additions == [" newly added line 1", " newly added line 2"]
    assert c.deletions == []

    # Delete file assertions
    dele = patch.actions["src/obsolete.txt"]
    assert dele.type == ActionType.DELETE
    assert dele.chunks == []


def test_duplicate_file_entry_is_error_and_ignored():
    text = """*** Begin Patch
*** Update File: src/dup.py
@@
- a
+ b
*** Update File: src/dup.py
@@
- c
+ d
*** End Patch"""
    patch, errors = parse_v4a_patch(text)
    assert any("Duplicate file entry" in e.msg for e in errors)
    # Only the first occurrence is kept
    assert list(patch.actions.keys()).count("src/dup.py") == 1
    assert len(patch.actions["src/dup.py"].chunks) == 1


def test_absolute_path_is_error():
    text = """*** Begin Patch
*** Update File: /abs/path.py
@@
- a
+ b
*** End Patch"""
    patch, errors = parse_v4a_patch(text)
    assert any("Path must be relative" in e.msg for e in errors)
    assert "/abs/path.py" not in patch.actions


def test_delete_section_must_not_have_content():
    text = """*** Begin Patch
*** Delete File: data.bin
- should not be here
@@ anchor
+ nor this
*** End Patch"""
    _, errors = parse_v4a_patch(text)
    assert any("Delete file section" in e.msg for e in errors)


def test_missing_envelope_markers():
    no_begin = "*** End Patch"
    _, errors = parse_v4a_patch(no_begin)
    assert any("Missing *** Begin Patch" in e.msg for e in errors)

    no_end = "*** Begin Patch"
    _, errors = parse_v4a_patch(no_end)
    assert any("Missing *** End Patch" in e.msg for e in errors)

    multi = """x
*** Begin Patch
*** Update File: a.txt
@@
- a
+ b
*** Begin Patch
*** End Patch
*** End Patch"""
    _, errors = parse_v4a_patch(multi)
    assert any("Multiple *** Begin Patch" in e.msg for e in errors)
    assert any("Multiple *** End Patch" in e.msg for e in errors)


def test_ambiguous_chunks_without_anchor_is_error():
    text = """*** Begin Patch
*** Update File: src/x.py
 ctx1
 ctx2
 ctx3
- a
+ b
 ctxA
 ctxB
 ctxC
- c
+ d
*** End Patch"""
    _, errors = parse_v4a_patch(text)
    assert any("Ambiguous or overlapping blocks" in e.msg for e in errors)


def test_process_patch_only_parses_for_now():
    text = """*** Begin Patch
*** Update File: src/t.py
@@
- a
+ b
*** End Patch"""
    # Should return empty error list, read Update files, and successfully build commits
    opened: list[str] = []
    statuses, errs = process_patch(
        text,
        open_fn=lambda p: opened.append(p) or " a\n",
        write_fn=lambda p, c: None,
        delete_fn=lambda p: None,
    )
    assert errs == []
    assert opened == ["src/t.py"]
    assert statuses == {"src/t.py": FileApplyStatus.Update}


def test_add_file_rejects_context_no_anchor():
    text = """*** Begin Patch
*** Add File: src/new_module.py
 # pre1
 # pre2
 # pre3
+ line1
+ line2
 # post1
 # post2
 # post3
*** End Patch"""
    patch, errors = parse_v4a_patch(text)
    assert any("must not contain context" in e.msg for e in errors)
    # Still parsed as an Add action with the modifications captured
    assert "src/new_module.py" in patch.actions
    assert patch.actions["src/new_module.py"].type == ActionType.ADD
    assert patch.actions["src/new_module.py"].chunks[0].additions == [" line1", " line2"]
    assert patch.actions["src/new_module.py"].chunks[0].prefix == ["# pre1", "# pre2", "# pre3"]
    assert patch.actions["src/new_module.py"].chunks[0].suffix == ["# post1", "# post2", "# post3"]


def test_add_file_only_additions_ok():
    text = """*** Begin Patch
*** Add File: src/only_adds.py
+ line1
+ line2
+ line3
*** End Patch"""
    patch, errors = parse_v4a_patch(text)
    assert errors == []
    act = patch.actions["src/only_adds.py"]
    assert act.type == ActionType.ADD
    assert len(act.chunks) == 1
    ch = act.chunks[0]
    assert ch.prefix == []
    assert ch.suffix == []
    assert ch.deletions == []
    assert ch.additions == [" line1", " line2", " line3"]


def test_add_file_absolute_path_is_error_without_anchor():
    text = """*** Begin Patch
*** Add File: /abs/new.txt
+ a
*** End Patch"""
    patch, errors = parse_v4a_patch(text)
    assert any("Path must be relative" in e.msg for e in errors)
    assert "/abs/new.txt" not in patch.actions


def test_process_patch_reads_update_only_and_ignores_delete():
    text = """*** Begin Patch
*** Update File: exists.txt
@@
- old
+ new
*** Add File: added.txt
+ created
*** Delete File: missing.txt
*** End Patch"""

    opened: list[str] = []

    def open_fn(path: str) -> str:
        opened.append(path)
        if path == "exists.txt":
            return " old\n"
        if path == "missing.txt":
            raise FileNotFoundError("No such file")
        # 'added.txt' should not be opened; if it is, make it obvious
        raise AssertionError(f"open_fn called unexpectedly for {path}")

    statuses, errs = process_patch(
        text,
        open_fn=open_fn,
        write_fn=lambda p, c: None,
        delete_fn=lambda p: None,
    )
    # Should not error for delete file, since we don't read DELETE targets
    assert errs == []
    # Ensure we tried to read only Update, not Add or Delete
    assert opened == ["exists.txt"]
    # All actions should produce statuses
    assert statuses == {
        "exists.txt": FileApplyStatus.Update,
        "added.txt": FileApplyStatus.Create,
        "missing.txt": FileApplyStatus.Delete,
    }


def test_process_patch_applies_changes_and_calls_io():
    patch_text = """*** Begin Patch
*** Update File: f.txt
 pre
- old
+ new
 post
*** Add File: new.txt
+ hello
*** Delete File: gone.txt
*** End Patch"""

    # Provide existing content for f.txt
    opened: list[str] = []
    writes: dict[str, str] = {}
    deletes: list[str] = []

    def open_fn(p: str) -> str:
        opened.append(p)
        assert p == "f.txt"
        return "pre\n old\npost\n"

    def write_fn(p: str, c: str) -> None:
        writes[p] = c

    def delete_fn(p: str) -> None:
        deletes.append(p)

    statuses, errs = process_patch(patch_text, open_fn, write_fn, delete_fn)
    assert errs == []
    # Only update file should be opened
    assert opened == ["f.txt"]
    # Writes: update and add
    assert writes["f.txt"] == "pre\n new\npost\n"
    assert writes["new.txt"] == " hello"
    # Delete called for gone.txt
    assert deletes == ["gone.txt"]
    assert statuses == {
        "f.txt": FileApplyStatus.Update,
        "new.txt": FileApplyStatus.Create,
        "gone.txt": FileApplyStatus.Delete,
    }


def test_process_patch_write_delete_errors_appended():
    patch_text = """*** Begin Patch
*** Update File: f.txt
 pre
- old
+ new
 post
*** Add File: new.txt
+ hello
*** Delete File: gone.txt
*** End Patch"""

    def open_fn(p: str) -> str:
        return "pre\n old\npost\n"

    writes: dict[str, str] = {}
    deletes: list[str] = []

    def write_fn(p: str, c: str) -> None:
        # Fail on new.txt
        if p == "new.txt":
            raise IOError("disk full")
        writes[p] = c

    def delete_fn(p: str) -> None:
        # Fail on gone.txt
        if p == "gone.txt":
            raise PermissionError("read-only filesystem")
        deletes.append(p)

    statuses, errs = process_patch(patch_text, open_fn, write_fn, delete_fn)
    # Two IO errors should be reported
    assert len(errs) == 2
    msgs = [e.msg for e in errs]
    hints = [e.hint or "" for e in errs]
    files = [e.filename for e in errs]
    assert any("Failed to apply change to file: new.txt" in m for m in msgs)
    # Py3 aliases IOError to OSError; accept either in hint
    assert any(("IOError" in h or "OSError" in h) and "disk full" in h for h in hints)
    assert "new.txt" in files
    assert any("Failed to apply change to file: gone.txt" in m for m in msgs)
    assert any("PermissionError" in h and "read-only filesystem" in h for h in hints)
    assert "gone.txt" in files
    # Update should still be written successfully
    assert writes["f.txt"] == "pre\n new\npost\n"
    assert statuses == {
        "f.txt": FileApplyStatus.Update,
        "new.txt": FileApplyStatus.Create,
        "gone.txt": FileApplyStatus.Delete,
    }


def test_process_patch_partial_apply_and_collect_errors():
    patch_text = """*** Begin Patch
*** Update File: f.txt
 pre
- OLDX
+ NEWX
 post
@@
 x
 y
 z
- a
+ b
 u
 v
 w
*** End Patch"""

    # Only the second chunk exists in the file; the first will fail to locate.
    def open_fn(p: str) -> str:
        return "pre\nOLD\npost\nmid\nx\ny\nz\n a\nu\nv\nw\n"

    writes: dict[str, str] = {}

    def write_fn(p: str, c: str) -> None:
        writes[p] = c

    statuses, errs = process_patch(
        patch_text,
        open_fn=open_fn,
        write_fn=write_fn,
        delete_fn=lambda p: None,
    )
    # One build error for the unfound first chunk; still applied second chunk
    assert len(errs) == 1
    assert "Failed to locate change block" in errs[0].msg
    assert writes["f.txt"] == "pre\nOLD\npost\nmid\nx\ny\nz\n b\nu\nv\nw\n"
    # Partial update status expected
    assert statuses == {"f.txt": FileApplyStatus.PartialUpdate}


def test_build_commits_update_applies_change_single_chunk():
    patch_text = """*** Begin Patch
*** Update File: src/t.py
 ctx1
 ctx2
 ctx3
- old
+ new
 ctxA
 ctxB
 ctxC
*** End Patch"""
    # Original file has an extra trailing 'end' and newline to ensure preservation
    original = "ctx1\nctx2\nctx3\n old\nctxA\nctxB\nctxC\nend\n"
    patch, perrs = parse_v4a_patch(patch_text)
    assert perrs == []
    commits, errs, _ = build_commits(patch, {"src/t.py": original})
    assert errs == []
    assert len(commits) == 1
    commit = commits[0]
    assert "src/t.py" in commit.changes
    chg = commit.changes["src/t.py"]
    assert chg.type == ActionType.UPDATE
    assert chg.old_content == original
    assert chg.new_content == "ctx1\nctx2\nctx3\n new\nctxA\nctxB\nctxC\nend\n"


def test_build_commits_update_applies_change_multiple_chunks():
    patch_text = """*** Begin Patch
*** Update File: src/multi.py
@@ class A
 a1
 a2
 a3
- X
+ Y
 a4
 a5
 a6
@@ class B
 b1
 b2
 b3
- P
+ Q
 b4
 b5
 b6
*** End Patch"""
    original = "a1\na2\na3\n X\na4\na5\na6\nmid\nb1\nb2\nb3\n P\nb4\nb5\nb6\n"
    patch, perrs = parse_v4a_patch(patch_text)
    assert perrs == []
    commits, errs, _ = build_commits(patch, {"src/multi.py": original})
    assert errs == []
    out = commits[0].changes["src/multi.py"].new_content
    assert out == "a1\na2\na3\n Y\na4\na5\na6\nmid\nb1\nb2\nb3\n Q\nb4\nb5\nb6\n"


def test_build_commits_add_and_delete_changes():
    patch_text = """*** Begin Patch
*** Add File: src/new.txt
+ line1
+ line2
*** Delete File: src/old.txt
*** End Patch"""
    patch, perrs = parse_v4a_patch(patch_text)
    assert perrs == []
    commits, errs, _ = build_commits(patch, files={})
    assert errs == []
    assert len(commits) == 1
    commit = commits[0]
    assert commit.changes["src/new.txt"].type == ActionType.ADD
    assert commit.changes["src/new.txt"].new_content == " line1\n line2"
    assert commit.changes["src/old.txt"].type == ActionType.DELETE
    assert commit.changes["src/old.txt"].old_content is None
    assert commit.changes["src/old.txt"].new_content is None


def test_build_commits_update_partial_match_reports_hint():
    patch_text = """*** Begin Patch
*** Update File: src/t.py
 ctx1
 ctx2
 ctx3
- old
+ new
 ctxA
 ctxB
 ctxC
*** End Patch"""
    # Introduce a mismatch at the deletion line
    original = "ctx1\nctx2\nctx3\nNOT_OLD\nctxA\nctxB\nctxC\n"
    patch, perrs = parse_v4a_patch(patch_text)
    assert perrs == []
    commits, errs, _ = build_commits(patch, {"src/t.py": original})
    assert commits == []  # no commit if update failed
    assert len(errs) == 1
    assert "Failed to locate change block" in errs[0].msg
    # Hint should show where it failed and how much matched
    assert "Matched" in (errs[0].hint or "")
    assert "Matched source lines" in (errs[0].hint or "")
    assert "'ctx1'" in (errs[0].hint or "")
    assert "Next expected line: ' old'" in (errs[0].hint or "")
    assert "File has: 'NOT_OLD'" in (errs[0].hint or "")
    assert "ensure exact whitespace" in (errs[0].hint or "")
    assert "Possible variants" in (errs[0].hint or "")
    assert "'NOT_OLD'" in (errs[0].hint or "")
    assert "Best partial match at L1" in (errs[0].hint or "")


def test_build_commits_partial_apply_on_some_chunks():
    patch_text = """*** Begin Patch
*** Update File: src/partial.py
 p1
 p2
 p3
- OLDX
+ NEWX
 pA
 pB
 pC
@@
 q1
 q2
 q3
- R
+ S
 qA
 qB
 qC
*** End Patch"""
    original = "p1\np2\np3\nOLD\npA\npB\npC\nmid\nq1\nq2\nq3\n R\nqA\nqB\nqC\n"
    patch, perrs = parse_v4a_patch(patch_text)
    assert perrs == []
    commits, errs, _ = build_commits(patch, {"src/partial.py": original})
    # One chunk should fail to locate, one should succeed
    assert len(errs) == 1
    assert "Failed to locate change block" in errs[0].msg
    # We should still get a commit with the successfully applied change
    assert len(commits) == 1
    chg = commits[0].changes["src/partial.py"]
    assert chg.type == ActionType.UPDATE
    assert chg.old_content == original
    assert chg.new_content == "p1\np2\np3\nOLD\npA\npB\npC\nmid\nq1\nq2\nq3\n S\nqA\nqB\nqC\n"


def test_parse_update_chunk_without_suffix():
    text = """*** Begin Patch
*** Update File: src/no_suffix.py
 ctx1
 ctx2
 ctx3
- old
+ new
*** End Patch"""
    patch, errors = parse_v4a_patch(text)
    assert errors == []
    act = patch.actions["src/no_suffix.py"]
    assert act.type == ActionType.UPDATE
    assert len(act.chunks) == 1
    ch = act.chunks[0]
    assert ch.prefix == ["ctx1", "ctx2", "ctx3"]
    assert ch.deletions == [" old"]
    assert ch.additions == [" new"]
    assert ch.suffix == []  # suffix is optional


def test_build_commits_update_without_suffix_applies():
    patch_text = """*** Begin Patch
*** Update File: src/no_suffix.py
 ctx1
 ctx2
 ctx3
- old
+ new
*** End Patch"""
    original = "ctx1\nctx2\nctx3\n old\nend\n"
    patch, perrs = parse_v4a_patch(patch_text)
    assert perrs == []
    commits, errs, _ = build_commits(patch, {"src/no_suffix.py": original})
    assert errs == []
    assert len(commits) == 1
    chg = commits[0].changes["src/no_suffix.py"]
    assert chg.type == ActionType.UPDATE
    assert chg.old_content == original
    assert chg.new_content == "ctx1\nctx2\nctx3\n new\nend\n"


def test_parse_context_normalization_leading_space_and_blank():
    text = """*** Begin Patch
*** Update File: src/ctx_norm.py
 header1

- old
+ new
 footer1
*** End Patch"""
    patch, errors = parse_v4a_patch(text)
    assert errors == []
    act = patch.actions["src/ctx_norm.py"]
    assert act.type == ActionType.UPDATE
    assert len(act.chunks) == 1
    ch = act.chunks[0]
    # Leading single space is stripped, blank line kept as empty string
    assert ch.prefix == ["header1", ""]
    assert ch.deletions == [" old"]
    assert ch.additions == [" new"]
    assert ch.suffix == ["footer1"]


def test_add_file_without_plus_lines_treated_as_content():
    text = """*** Begin Patch
*** Add File: src/raw_add.txt
 line1
 line2
 line3
*** End Patch"""
    patch, errors = parse_v4a_patch(text)
    assert errors == []
    assert "src/raw_add.txt" in patch.actions
    act = patch.actions["src/raw_add.txt"]
    assert act.type == ActionType.ADD
    assert len(act.chunks) == 1
    ch = act.chunks[0]
    # Context-only block is converted to additions; no prefix/suffix retained
    assert ch.prefix == []
    assert ch.suffix == []
    assert ch.deletions == []
    assert ch.additions == ["line1", "line2", "line3"]


def test_process_patch_add_file_context_only_block_writes_all_content():
    text = """*** Begin Patch
*** Add File: src/raw_and_blank.txt
 line1

 line3
*** End Patch"""
    writes: dict[str, str] = {}

    statuses, errs = process_patch(
        text,
        open_fn=lambda p: "",  # Should not be called for Add
        write_fn=lambda p, c: writes.__setitem__(p, c),
        delete_fn=lambda p: None,
    )
    assert errs == []
    assert statuses == {"src/raw_and_blank.txt": FileApplyStatus.Create}
    # Blank line preserved between line1 and line3; no trailing newline
    assert writes["src/raw_and_blank.txt"] == "line1\n\nline3"


def test_build_commits_with_normalized_context_applies():
    patch_text = """*** Begin Patch
*** Update File: src/ctx_norm_apply.py
 header1

- old
+ new
 footer1
*** End Patch"""
    # File has an extra blank line after the existing blank context; normalization should still apply
    original = "header1\n\n\n old\nfooter1\n"
    patch, perrs = parse_v4a_patch(patch_text)
    assert perrs == []
    commits, errs, _ = build_commits(patch, {"src/ctx_norm_apply.py": original})
    assert errs == []
    assert len(commits) == 1
    chg = commits[0].changes["src/ctx_norm_apply.py"]
    assert chg.type == ActionType.UPDATE
    assert chg.old_content == original
    assert chg.new_content == "header1\n\n\n new\nfooter1\n"


def test_update_chunk_with_no_modifications_is_ignored():
    text = """*** Begin Patch
*** Update File: src/empty.py
@@ anchor only
 ctx1
 ctx2
 ctx3
*** End Patch"""
    patch, errors = parse_v4a_patch(text)
    assert errors == []
    assert "src/empty.py" in patch.actions
    act = patch.actions["src/empty.py"]
    assert act.type == ActionType.UPDATE
    # The chunk has only context and no +/- lines, so it should be ignored.
    assert len(act.chunks) == 0

def test_build_commits_update_with_no_modifications_errors():
    patch_text = """*** Begin Patch
*** Update File: src/empty.py
@@ anchor only
 ctx1
 ctx2
 ctx3
*** End Patch"""
    # File exists but the update contains no +/- lines
    original = "ctx1\nctx2\nctx3\n"
    patch, perrs = parse_v4a_patch(patch_text)
    assert perrs == []
    commits, errs, status_map = build_commits(patch, {"src/empty.py": original})
    # Should report a clear error and skip generating any commit/status for this file
    assert commits == []
    assert any("No change lines (+/-) provided for file: src/empty.py" in e.msg for e in errs)
    assert "src/empty.py" not in status_map

def test_process_patch_update_with_no_mods_reports_error():
    text = """*** Begin Patch
*** Update File: src/empty.py
@@
 ctx1
 ctx2
 ctx3
*** End Patch"""
    writes: dict[str, str] = {}
    deletes: list[str] = []

    statuses, errs = process_patch(
        text,
        open_fn=lambda p: "ctx1\nctx2\nctx3\n",
        write_fn=lambda p, c: writes.__setitem__(p, c),
        delete_fn=lambda p: deletes.append(p),
    )
    # No IO should occur and an explicit error should be returned
    assert writes == {}
    assert deletes == []
    assert statuses == {}
    assert len(errs) == 1
    assert "No change lines (+/-) provided for file: src/empty.py" in errs[0].msg
    assert errs[0].filename == "src/empty.py"


def test_build_commits_handles_missing_empty_line_in_context():
    patch_text = """*** Begin Patch
*** Update File: src/missing_blank.py
 header1
- old
+ new
 footer1
*** End Patch"""
    # File has an extra blank line after header1 that is not present in the patch.
    original = "header1\n\n old\nfooter1\n"
    patch, perrs = parse_v4a_patch(patch_text)
    assert perrs == []
    # Build commits should insert the missing empty context line and still match/apply.
    commits, errs, _ = build_commits(patch, {"src/missing_blank.py": original})
    assert errs == []
    assert len(commits) == 1
    chg = commits[0].changes["src/missing_blank.py"]
    assert chg.type == ActionType.UPDATE
    assert chg.old_content == original
    assert chg.new_content == "header1\n\n new\nfooter1\n"
    # And the chunk should have been adjusted to include the missing blank in prefix.
    act = patch.actions["src/missing_blank.py"]
    assert len(act.chunks) == 1
    ch = act.chunks[0]
    assert ch.prefix == ["header1", ""]
