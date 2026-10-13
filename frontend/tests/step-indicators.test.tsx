import assert from "node:assert/strict";
import { test, vi } from "vitest";
import { fireEvent, render } from "@testing-library/react";
import { renderToStaticMarkup } from "react-dom/server";
import { StepIndicators } from "../src/components/steps/StepIndicators";
import { AssistantStep } from "../src/components/steps/AssistantStep";
import { ToolStep } from "../src/components/steps/ToolStep";
import { UserStep } from "../src/components/steps/UserStep";
import type { StepActions } from "../src/components/steps/registry";
import { stepView } from "../src/models/steps";
import { StepCard } from "../src/components/steps/StepCard";

const actions: StepActions = {
  pending: [],
  enabled: true,
  inspect: () => {},
  reply: async () => true,
};

test("status uses icons with accessible labels and hover titles without info icons", () => {
  for (const state of [
    "failed",
    "streaming",
    "pending",
    "Needs approval",
    "cancelled",
    "unknown",
  ]) {
    const label = state[0].toUpperCase() + state.slice(1);
    const html = renderToStaticMarkup(<StepIndicators state={state} />);
    assert.match(
      html,
      new RegExp(`role="img" aria-label="${label}" title="${label}"`),
    );
    assert.doesNotMatch(html, /aria-label="View details"/);
    assert.equal((html.match(/<svg/g) ?? []).length, 1);
    assert.doesNotMatch(html, />[^<]*(completed|Details)[^<]*</);
    assert.equal(
      html.includes("animate-spin"),
      ["streaming", "pending"].includes(state),
    );
  }
});

test("completed steps omit indicators entirely", () => {
  for (const state of [undefined, "completed", "Authorized"]) {
    const html = renderToStaticMarkup(<StepIndicators state={state} />);
    assert.equal(html, "");
  }
});

test("chat entries omit generic headings and info buttons but open details from the card", () => {
  const step = stepView({
    position: 1,
    step: {
      id: "step",
      workflow_id: "root",
      operation: {
        kind: "llm.tool",
        input: { call: { name: "exec", arguments: { command: "echo hello" } } },
      },
      outcome: { kind: "completed", result: "Hello" },
    },
  });
  for (const Component of [AssistantStep, ToolStep, UserStep]) {
    const html = renderToStaticMarkup(
      <Component step={step} actions={actions} />,
    );
    assert.doesNotMatch(
      html,
      /activity-heading|>Assistant<|>Run Command<|>You<|aria-label="Completed"/,
    );
    assert.match(
      html,
      /<section[^>]*role="button"[^>]*aria-label="View details"/,
    );
    assert.doesNotMatch(html, /<button[^>]*aria-label="View details"|<svg/);
    assert.doesNotMatch(html, />completed|\/ Details|>Details</);
    const inspect = vi.fn();
    const view = render(
      <Component step={step} actions={{ ...actions, inspect }} />,
    );
    const card = view.getByRole("button", { name: "View details" });
    fireEvent.click(
      card.querySelector(".message-text, .markdown-content, code")!,
    );
    assert.deepEqual(inspect.mock.calls, [[step.id]]);
    fireEvent.keyDown(card, { key: "Enter" });
    fireEvent.keyDown(card, { key: " " });
    assert.deepEqual(inspect.mock.calls, [[step.id], [step.id], [step.id]]);
    view.unmount();
  }
  const tool = renderToStaticMarkup(<ToolStep step={step} actions={actions} />);
  assert.match(tool, />Ran<\/span>/);
  assert.match(tool, /<code>echo hello<\/code>/);
});

test("nested controls and selected text do not open step details", () => {
  const inspect = vi.fn();
  const click = vi.fn();
  const view = render(
    <StepCard className="user-step" inspect={inspect}>
      <p>Selectable text</p>
      <a href="#link">Link</a>
      <button onClick={click}>
        <span>Action</span>
      </button>
      <input aria-label="Input" />
      <details>
        <summary>Expand</summary>Content
      </details>
    </StepCard>,
  );
  fireEvent.click(view.getByText("Link"));
  fireEvent.click(view.getByText("Action"));
  fireEvent.keyDown(view.getByRole("button", { name: "Action" }), {
    key: "Enter",
  });
  fireEvent.click(view.getByRole("textbox"));
  fireEvent.click(view.getByText("Expand"));
  assert.equal(click.mock.calls.length, 1);
  assert.equal(inspect.mock.calls.length, 0);

  const selection = window.getSelection()!;
  const range = document.createRange();
  range.selectNodeContents(view.getByText("Selectable text"));
  selection.addRange(range);
  try {
    fireEvent.click(view.getByText("Selectable text"));
    assert.equal(inspect.mock.calls.length, 0);
  } finally {
    selection.removeAllRanges();
  }
  fireEvent.click(view.getByText("Selectable text"));
  assert.equal(inspect.mock.calls.length, 1);
});
