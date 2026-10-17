import { expect, test } from "vitest";
import { fileMention, insertFileMention } from "../src/models/file-mention";

test.each([
  ["@", 1, ""],
  ["Fix @src/Comp", 13, "src/Comp"],
  ["First line\n@file", 16, "file"],
  ['Use @"my file', 13, "my file"],
])("detects a file mention at the caret in %s", (text, caret, query) => {
  expect(fileMention(text, caret)?.query).toBe(query);
});

test.each([
  "me@example.com",
  "escaped \\@file",
  "@file ",
  "@@file",
  'Use @"my file" ',
])("does not complete non-mentions: %s", (text) => {
  expect(fileMention(text, text.length)).toBeUndefined();
});

test("selections do not trigger completion", () => {
  expect(fileMention("@file", 1, 5)).toBeUndefined();
});

test("inserts only the active mention and preserves surrounding text", () => {
  const text = "Compare @old.py with @new.py today";
  const mention = fileMention(text, text.indexOf("@new") + 4)!;
  expect(mention.query).toBe("new");
  expect(insertFileMention(text, mention, "src/new.py")).toEqual({
    text: "Compare @old.py with @src/new.py today",
    caret: 32,
  });
});

test("quotes whitespace and special characters without losing suffix text", () => {
  const text = "Read @fi then fix";
  const mention = fileMention(text, 8)!;
  const result = insertFileMention(text, mention, 'docs/my "file".md');
  expect(result.text).toBe('Read @"docs/my \\"file\\".md" then fix');
  expect(fileMention(result.text, result.caret)).toBeUndefined();
});

test("replaces a quoted mention when editing its middle", () => {
  const text = 'Read @"my old file.md" please';
  const mention = fileMention(text, 9)!;
  expect(insertFileMention(text, mention, "new.md").text).toBe(
    "Read @new.md please",
  );
});
