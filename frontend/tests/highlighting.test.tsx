import assert from "node:assert/strict";
import { test } from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import { HighlightedCode } from "../src/components/HighlightedCode";

test("highlights explicit languages and aliases", () => {
  for (const language of ["python", "py", "Python"]) {
    const html = renderToStaticMarkup(
      <HighlightedCode language={language}>
        {"def example():\n    return 1"}
      </HighlightedCode>,
    );
    assert.match(html, /hljs-keyword/);
  }
});

test("highlights patch additions, removals and headings", () => {
  const html = renderToStaticMarkup(
    <HighlightedCode language="diff">
      {"*** Update File: example.py\n@@\n-old\n+new\n context"}
    </HighlightedCode>,
  );
  assert.match(html, /hljs-addition/);
  assert.match(html, /hljs-deletion/);
  assert.match(html, /hljs-comment/);
});

test("unknown and unspecified languages stay plain and escaped", () => {
  for (const language of [undefined, "unknown-language", "text"]) {
    const html = renderToStaticMarkup(
      <HighlightedCode language={language}>
        {'<script>alert("test")</script>\n'}
      </HighlightedCode>,
    );
    assert.doesNotMatch(html, /<script|<span/);
    assert.match(html, /&lt;script&gt;/);
    assert.ok(html.endsWith("\n</code>"));
  }
});

test("highlighted content escapes raw HTML", () => {
  const html = renderToStaticMarkup(
    <HighlightedCode language="python">{'value = "<script>"'}</HighlightedCode>,
  );
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&lt;script&gt;/);
});
