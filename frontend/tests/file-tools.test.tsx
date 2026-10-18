import assert from "node:assert/strict";
import { test } from "vitest";
import { fireEvent, render } from "@testing-library/react";
import type { Json } from "../src/models/json";
import type { ToolView } from "../src/components/tools/factory";
import { DirectorySummary, FileDetails } from "../src/components/tools/views";

function output(blocks: Json[]) {
  const tool: ToolView = {
    name: "read_file",
    arguments: { path: "example.png" },
    output: { content: blocks },
    state: "completed",
    failed: false,
    error: "",
    processCommand: undefined,
  };
  const view = render(<FileDetails tool={tool} />);
  const disclosure = view.getByText("Result / output").closest("details");
  assert.ok(disclosure);
  disclosure.open = true;
  fireEvent(disclosure, new Event("toggle"));
  return view;
}

test("directory summaries show the effective default path", () => {
  const tool: ToolView = {
    name: "list_files",
    arguments: {},
    output: undefined,
    state: "pending",
    failed: false,
    error: "",
    processCommand: undefined,
  };
  const view = render(<DirectorySummary tool={tool} />);
  assert.equal(view.getByText(".").textContent, ".");
  view.rerender(
    <DirectorySummary tool={{ ...tool, arguments: { path: "frontend" } }} />,
  );
  assert.equal(view.getByText("frontend").textContent, "frontend");
});

test("file output preserves trailing images and text without repeating structured content", () => {
  const view = output([
    {
      type: "text",
      text: JSON.stringify({
        path: "example.png",
        content: "File content",
        truncated: true,
      }),
    },
    { type: "image", mime_type: "image/png", data: "cHJldmlldw==" },
    { type: "text", text: "Additional information" },
  ]);
  assert.equal(view.getAllByText("File content").length, 1);
  assert.equal(view.getAllByText("example.png", { selector: "dd" }).length, 2);
  assert.ok(view.getByText("Additional information"));
  assert.ok(
    view.getByText(
      "Partial content. More data is available beyond this range.",
    ),
  );
  assert.equal(
    view.getByRole("img", { name: "Tool result preview" }).getAttribute("src"),
    "data:image/png;base64,cHJldmlldw==",
  );
  assert.equal(view.queryByText("No result recorded yet."), null);
});

test("metadata-only file results do not repeat the metadata before an image", () => {
  const view = output([
    {
      type: "text",
      text: JSON.stringify({ mime_type: "image/png", bytes_read: 42 }),
    },
    { type: "image", mime_type: "image/png", data: "cHJldmlldw==" },
  ]);
  assert.ok(view.getByText("42"));
  assert.equal(view.container.querySelectorAll(".tool-code").length, 0);
  assert.equal(view.getAllByRole("img").length, 1);
});

test.each(["png", "jpeg", "gif", "webp"])(
  "image-only file output renders %s without dumping the envelope",
  (format) => {
    const view = output([
      { type: "image", mime_type: `image/${format}`, data: "cHJldmlldw==" },
    ]);
    assert.equal(view.getAllByRole("img").length, 1);
    assert.equal(view.queryByText("content", { selector: "dt" }), null);
    assert.equal(view.container.querySelectorAll(".tool-code").length, 0);
  },
);

test("binary file output retains its disclosure and trailing result blocks", () => {
  const view = output([
    {
      type: "text",
      text: JSON.stringify({ encoding: "base64", content: "YmluYXJ5" }),
    },
    { type: "text", text: "Binary file note" },
  ]);
  assert.ok(view.getByText("Binary file note"));
  assert.equal(view.queryByText("YmluYXJ5"), null);
  const disclosure = view
    .getByText("Binary content (base64)")
    .closest("details");
  assert.ok(disclosure);
  disclosure.open = true;
  fireEvent(disclosure, new Event("toggle"));
  assert.equal(view.getAllByText("YmluYXJ5").length, 1);
});

test("unstructured text and unsupported images remain inspectable", () => {
  const view = output([
    { type: "text", text: "Unstructured output" },
    { type: "image", mime_type: "image/svg+xml", data: "PHN2Zy8+" },
  ]);
  assert.ok(view.getByText("Unstructured output"));
  assert.equal(view.queryByRole("img"), null);
  assert.ok(view.getByText(/image\/svg\+xml/));
});
