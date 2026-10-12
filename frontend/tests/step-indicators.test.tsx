import assert from "node:assert/strict";
import { test } from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import { StepIndicators } from "../src/components/steps/StepIndicators";
import { AssistantStep } from "../src/components/steps/AssistantStep";
import { ToolStep } from "../src/components/steps/ToolStep";
import { UserStep } from "../src/components/steps/UserStep";
import type { StepActions } from "../src/components/steps/registry";
import { stepView } from "../src/models/steps";

const actions: StepActions = {
  pending: [],
  enabled: true,
  inspect: () => {},
  reply: async () => true,
};

test("status and details use icons with accessible labels and hover titles", () => {
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
    assert.match(html, /aria-label="View details" title="View details"/);
    assert.equal((html.match(/<svg/g) ?? []).length, 2);
    assert.doesNotMatch(html, />[^<]*(completed|Details)[^<]*</);
    assert.equal(
      html.includes("animate-spin"),
      ["streaming", "pending"].includes(state),
    );
  }
});

test("details-only indicators omit the status icon", () => {
  for (const state of [undefined, "completed", "Authorized"]) {
    const html = renderToStaticMarkup(<StepIndicators state={state} />);
    assert.equal((html.match(/<svg/g) ?? []).length, 1);
    assert.match(html, /aria-label="View details"/);
    assert.doesNotMatch(html, /aria-label="Completed|aria-label="Authorized/);
  }
});

test("chat entries omit generic headings and completed status but retain details buttons", () => {
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
    assert.match(html, /<button[^>]*aria-label="View details"/);
    assert.match(html, /aria-label="View details"/);
    assert.doesNotMatch(html, />completed|\/ Details|>Details</);
  }
  const tool = renderToStaticMarkup(<ToolStep step={step} actions={actions} />);
  assert.match(tool, />Ran<\/span>/);
  assert.match(tool, /<code>echo hello<\/code>/);
});
