import assert from "node:assert/strict";
import { fireEvent, render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, test, vi } from "vitest";
import { CodeBlock } from "../src/components/CodeBlock";
import { HighlightedCode } from "../src/components/HighlightedCode";
import { MarkdownBody } from "../src/components/MarkdownBody";

const originalClipboard = Object.getOwnPropertyDescriptor(
  navigator,
  "clipboard",
);

afterEach(() => {
  if (originalClipboard)
    Object.defineProperty(navigator, "clipboard", originalClipboard);
  else Reflect.deleteProperty(navigator, "clipboard");
});

function clipboard() {
  const writeText = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText },
  });
  return writeText;
}

test("each Markdown block copies its own plain text, excluding inline code", async () => {
  const writeText = clipboard();
  const snippets = [
    'def example():\n    return "<value>"\n',
    "plain & unhighlighted\n",
    "unknown language\n",
    "indented block\n",
  ];
  const view = render(
    <MarkdownBody
      message={[
        "Text with `inline code`.",
        `\`\`\`python\n${snippets[0]}\`\`\``,
        `\`\`\`\n${snippets[1]}\`\`\``,
        `\`\`\`unknown\n${snippets[2]}\`\`\``,
        "    indented block",
      ].join("\n\n")}
    />,
  );
  const buttons = view.getAllByRole("button", {
    name: "Copy code to clipboard",
  });
  assert.equal(buttons.length, snippets.length);
  for (const [index, button] of buttons.entries()) {
    fireEvent.click(button);
    await waitFor(() => assert.equal(writeText.mock.calls.length, index + 1));
    assert.deepEqual(writeText.mock.calls[index], [snippets[index]]);
  }
  await waitFor(() => assert.equal(view.getAllByText("Copied").length, 4));
});

test("copy works from the keyboard without triggering the parent action", async () => {
  const user = userEvent.setup();
  const writeText = clipboard();
  const inspect = vi.fn();
  const view = render(
    <div onClick={inspect}>
      <CodeBlock className="tool-diff">
        <HighlightedCode language="diff">{"-old\n+new\n"}</HighlightedCode>
      </CodeBlock>
    </div>,
  );
  await user.tab();
  assert.equal(document.activeElement, view.getByRole("button"));
  await user.keyboard("{Enter}");
  assert.deepEqual(writeText.mock.calls, [["-old\n+new\n"]]);
  assert.equal(inspect.mock.calls.length, 0);
  assert.equal(
    view.container.querySelector("pre")?.textContent,
    "-old\n+new\n",
  );
});

test("failed clipboard writes show feedback and can be retried", async () => {
  const writeText = clipboard();
  writeText.mockRejectedValueOnce(new Error("Permission denied"));
  const view = render(<CodeBlock>{"  snippet\n\n"}</CodeBlock>);
  fireEvent.click(view.getByRole("button"));
  await view.findByText("Copy failed. Please try again.");
  assert.equal(view.queryByText("Copied"), null);
  fireEvent.click(view.getByRole("button"));
  await view.findByText("Copied");
  assert.deepEqual(writeText.mock.calls, [
    ["  snippet\n\n"],
    ["  snippet\n\n"],
  ]);
});

test("unavailable clipboard API reports failure without throwing", async () => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: undefined,
  });
  const view = render(<CodeBlock>snippet</CodeBlock>);
  fireEvent.click(view.getByRole("button"));
  await view.findByText("Copy failed. Please try again.");
});
