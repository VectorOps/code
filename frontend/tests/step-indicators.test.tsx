import assert from "node:assert/strict";
import { test, vi } from "vitest";
import { fireEvent, render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderToStaticMarkup } from "react-dom/server";
import { StepIndicators } from "../src/components/steps/StepIndicators";
import { AssistantStep } from "../src/components/steps/AssistantStep";
import { ToolStep } from "../src/components/steps/ToolStep";
import { UserStep } from "../src/components/steps/UserStep";
import type { StepActions } from "../src/components/steps/registry";
import { stepView } from "../src/models/steps";
import { StepCard } from "../src/components/steps/StepCard";

test("step cards preserve rich content and keep controls outside button semantics", () => {
  const view = render(
    <StepCard className="assistant-step" inspect={vi.fn()}>
      <h2>Deployment failed</h2>
      <p>Do not proceed.</p>
      <ul>
        <li>Check the logs</li>
      </ul>
      <a href="#logs">Logs</a>
      <button>Approve once</button>
      <textarea aria-label="Edit input" />
    </StepCard>,
  );
  const card = view.container.querySelector(".step-card")!;
  assert.equal(card.getAttribute("role"), null);
  assert.equal(card.getAttribute("tabindex"), null);
  assert.equal(card.getAttribute("aria-label"), null);
  assert.equal(
    view.getByRole("heading", { name: "Deployment failed" }).tagName,
    "H2",
  );
  assert.equal(view.getByRole("listitem").textContent, "Check the logs");
  for (const control of [
    view.getByRole("link", { name: "Logs" }),
    view.getByRole("button", { name: "Approve once" }),
    view.getByRole("textbox", { name: "Edit input" }),
  ]) {
    assert.equal(
      control.parentElement?.closest('[role="button"], button'),
      null,
    );
  }
  const details = view.getByRole("button", { name: "View details" });
  assert.equal(details.tagName, "BUTTON");
  assert.equal(details.contains(view.getByText("Do not proceed.")), false);
});

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

test("chat entries open details by card click or a separate keyboard control", async () => {
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
    assert.doesNotMatch(
      html,
      /<section[^>]*(role="button"|aria-label="View details")|<svg/,
    );
    assert.match(
      html,
      /<button[^>]*class="step-card-details"[^>]*>View details<\/button>/,
    );
    assert.doesNotMatch(html, />completed|\/ Details|>Details</);
    const inspect = vi.fn();
    const view = render(
      <Component step={step} actions={{ ...actions, inspect }} />,
    );
    const details = view.getByRole("button", { name: "View details" });
    const card = details.closest(".step-card")!;
    fireEvent.click(
      card.querySelector(".message-text, .markdown-content, code")!,
    );
    assert.deepEqual(inspect.mock.calls, [[step.id]]);
    assert.equal(document.activeElement, details);
    await userEvent.setup().keyboard("{Enter} ");
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
  selection.removeAllRanges();
  const range = document.createRange();
  range.selectNodeContents(view.getByText("Selectable text"));
  selection.addRange(range);
  assert.equal(selection.isCollapsed, false);
  try {
    fireEvent.click(view.getByText("Selectable text"));
    assert.equal(inspect.mock.calls.length, 0);
  } finally {
    selection.removeAllRanges();
  }
  fireEvent.click(view.getByText("Selectable text"));
  assert.equal(inspect.mock.calls.length, 1);
});
