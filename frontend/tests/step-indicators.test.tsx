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
    "completed",
    "failed",
    "streaming",
    "pending",
    "Needs approval",
    "Authorized",
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
    assert.equal(html.includes("animate-spin"), state === "streaming");
  }
});

test("details-only indicators omit the status icon", () => {
  const html = renderToStaticMarkup(<StepIndicators />);
  assert.equal((html.match(/<svg/g) ?? []).length, 1);
  assert.match(html, /aria-label="View details"/);
});

test("chat headings retain message labels without visible status or details text", () => {
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
    assert.match(html, /Assistant|Run Command|You/);
    assert.match(html, /aria-label="View details"/);
    assert.doesNotMatch(html, />completed|\/ Details|>Details</);
  }
});
