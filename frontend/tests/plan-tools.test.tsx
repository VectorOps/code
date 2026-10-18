import assert from "node:assert/strict";
import { test } from "vitest";
import { fireEvent, render, waitFor } from "@testing-library/react";
import type { Json } from "../src/models/json";
import type { ToolView } from "../src/components/tools/factory";
import { PlanDetails, PlanSummary } from "../src/components/tools/views";
import { ToolStep } from "../src/components/steps/ToolStep";
import { stepView } from "../src/models/steps";

const todos: Json[] = [
  { id: "a", title: "Task A", status: "pending" },
  { id: "b", title: "Task B", status: "in_progress" },
  { id: "c", title: "Task C", status: "completed" },
];

function tool(args: Json = { todos }, result?: Json): ToolView {
  return {
    name: "update_plan",
    arguments: args,
    output:
      result === undefined
        ? undefined
        : { content: [{ type: "text", text: JSON.stringify(result) }] },
    state: result === undefined ? "streaming" : "completed",
    failed: false,
    error: "",
    processCommand: undefined,
  };
}

test("plan summaries show every task with a consistently sized status icon", () => {
  const view = render(<PlanSummary tool={tool({}, { todos })} />);
  const items = view.getAllByRole("listitem");
  assert.deepEqual(
    items.map((item) => item.textContent),
    ["Task A", "Task B", "Task C"],
  );
  for (const status of ["pending", "in progress", "completed"]) {
    const icon = view.getByRole("img", { name: status });
    assert.equal(icon.tagName, "svg");
    assert.equal(icon.getAttribute("width"), "16");
    assert.equal(icon.getAttribute("height"), "16");
  }
  assert.equal(view.getByLabelText("pending").querySelector("path"), null);
  assert.equal(
    view.getByLabelText("in progress").querySelector("path")?.getAttribute("d"),
    "M12 7v5l3 2",
  );
  assert.equal(
    view.getByLabelText("completed").querySelector("path")?.getAttribute("d"),
    "m8 12 3 3 5-6",
  );
  assert.equal(view.getByRole("list", { name: "Plan" }).tagName, "UL");
});

test("plan summaries prefer the complete returned plan over a partial update", () => {
  const view = render(
    <PlanSummary
      tool={tool({ todos: [{ id: "c", status: "completed" }] }, { todos })}
    />,
  );
  assert.equal(view.getAllByRole("listitem").length, 3);
  assert.ok(view.getByText("Task C"));
});

test("plan details use the same status icons when output opens", () => {
  const view = render(<PlanDetails tool={tool({}, { todos })} />);
  assert.equal(view.queryByRole("list"), null);
  const disclosure = view.getByText("Result / output").closest("details");
  assert.ok(disclosure);
  disclosure.open = true;
  fireEvent(disclosure, new Event("toggle"));
  assert.deepEqual(
    view.getAllByRole("listitem").map((item) => item.textContent),
    ["Task A", "Task B", "Task C"],
  );
  assert.equal(view.getAllByRole("img").length, 3);
});

test("plan summaries fall back to task ids and pending status", () => {
  const view = render(
    <PlanSummary tool={tool({ todos: [{ id: "task-id" }] })} />,
  );
  assert.equal(view.getByRole("listitem").textContent, "task-id");
  assert.ok(view.getByLabelText("pending"));
});

test("long plan titles retain the full text in an ellipsis target and tooltip", () => {
  const title = "A long task description with several steps to perform ".repeat(
    20,
  );
  const view = render(
    <PlanSummary
      tool={tool({ todos: [{ id: "long", title, status: "pending" }] })}
    />,
  );
  const label = view.getByTitle(title.trim());
  assert.equal(label.getAttribute("title"), title);
  assert.equal(label.textContent, title);
  assert.ok(label.classList.contains("tool-plan-title"));
});

test("empty plans retain a compact fallback without stale requested tasks", () => {
  const view = render(<PlanSummary tool={tool({ todos }, { todos: [] })} />);
  assert.ok(view.getByText("Plan update"));
  assert.equal(view.queryByRole("list"), null);
  assert.equal(view.queryByText("Task A"), null);
});

test.each([
  "pending",
  "streaming",
  "completed",
  "failed",
  "cancelled",
  "Authorized",
])("argument-only plans are explicitly unconfirmed when %s", (state) => {
  const view = render(
    <PlanSummary tool={{ ...tool(), state, failed: state === "failed" }} />,
  );
  assert.ok(view.getByText("Requested updates (not confirmed)"));
  assert.ok(view.getByRole("list", { name: "Requested updates" }));
  assert.equal(view.queryByRole("list", { name: "Plan" }), null);
  assert.equal(view.getAllByRole("listitem").length, 3);
});

test("a returned plan replaces the unconfirmed preview", () => {
  const args = { todos: [{ id: "c", status: "completed" }] };
  const view = render(<PlanSummary tool={tool(args)} />);
  assert.equal(view.getByRole("listitem").textContent, "c");
  assert.ok(view.getByText("Requested updates (not confirmed)"));
  view.rerender(<PlanSummary tool={tool(args, { todos })} />);
  assert.equal(view.getAllByRole("listitem").length, 3);
  assert.ok(view.getByRole("list", { name: "Plan" }));
  assert.equal(view.queryByText("Requested updates (not confirmed)"), null);
});

test.each<Json>([{}, { todos: null }, { todos: "invalid" }])(
  "a result without a valid todo list keeps requests explicitly unconfirmed: %j",
  (result) => {
    const view = render(<PlanSummary tool={tool({ todos }, result)} />);
    assert.ok(view.getByText("Requested updates (not confirmed)"));
    assert.equal(view.getAllByRole("listitem").length, 3);
    assert.equal(view.queryByRole("list", { name: "Plan" }), null);
  },
);

test("plan details do not present failed requested updates as results", () => {
  const view = render(
    <PlanDetails
      tool={{ ...tool({ todos }, { error: "Update rejected" }), failed: true }}
    />,
  );
  const disclosure = view.getByText("Result / output").closest("details");
  assert.ok(disclosure);
  disclosure.open = true;
  fireEvent(disclosure, new Event("toggle"));
  assert.equal(view.queryByRole("list"), null);
  assert.ok(view.getByText(/Update rejected/));
});

test.each(["failed", "is_error"])(
  "failed todo updates retain their failure indicator for %s outcomes",
  (failure) => {
    const step = stepView({
      position: 1,
      step: {
        id: "plan",
        workflow_id: "root",
        operation: {
          kind: "llm.tool",
          input: { call: { name: "update_plan", arguments: { todos } } },
        },
        outcome:
          failure === "failed"
            ? { kind: "failed", error: { message: "Update rejected" } }
            : {
                kind: "completed",
                result: {
                  is_error: true,
                  content: [{ type: "text", text: "Update rejected" }],
                },
              },
      },
    });
    const view = render(
      <ToolStep
        step={step}
        actions={{
          pending: [],
          enabled: true,
          inspect: () => {},
          reply: async () => true,
        }}
      />,
    );
    assert.ok(view.getByRole("img", { name: "Failed" }));
    assert.ok(view.getByText("Requested updates (not confirmed)"));
    assert.equal(view.queryByRole("list", { name: "Plan" }), null);
  },
);

test("todo approval shows an unconfirmed preview and working approval actions", async () => {
  const call = { name: "update_plan", arguments: { todos } };
  const step = stepView({
    position: 1,
    step: {
      id: "plan",
      workflow_id: "root",
      operation: { kind: "llm.tool", input: { call, gate_only: true } },
      outcome: { kind: "pending" },
    },
  });
  let response: Json | undefined;
  const view = render(
    <ToolStep
      step={step}
      actions={{
        pending: [
          {
            request_id: "plan",
            execution_id: "execution",
            workflow_id: "root",
            request_type: "tool.approval.v1",
            channel: "approval",
            payload: { call },
            response_type: "json",
          },
        ],
        enabled: true,
        inspect: () => {},
        reply: async (_, value) => {
          response = value;
          return true;
        },
      }}
    />,
  );
  assert.ok(view.getByText("Requested updates (not confirmed)"));
  assert.ok(view.getByRole("list", { name: "Requested updates" }));
  assert.ok(view.getByText("Allow"));
  assert.ok(view.getByRole("button", { name: "Deny" }));
  fireEvent.click(view.getByRole("button", { name: "Approve once" }));
  await waitFor(() => assert.ok(view.getByText("Decision acknowledged")));
  assert.deepEqual(response, {
    decision: "approve",
    scope: "call",
    properties: {},
  });
});

test.each(["completed", "streaming", "authorized"])(
  "the todo tool displays only its plan in the feed when %s",
  (state) => {
    const step = stepView({
      position: 1,
      step: {
        id: "plan",
        workflow_id: "root",
        operation: {
          kind: "llm.tool",
          input: {
            call: { name: "update_plan", arguments: { todos } },
            gate_only: state === "authorized",
          },
        },
        outcome:
          state === "streaming"
            ? { kind: "streaming", state: { todos } }
            : { kind: "completed", result: { todos } },
      },
    });
    const view = render(
      <ToolStep
        step={step}
        actions={{
          pending: [],
          enabled: true,
          inspect: () => {},
          reply: async () => true,
        }}
      />,
    );
    assert.equal(view.container.querySelector(".tool-activity-label"), null);
    assert.equal(view.container.querySelector(".entry-state"), null);
    assert.equal(
      view.getByRole("button", { name: "View details" }).tagName,
      "BUTTON",
    );
    assert.equal(
      view.container.querySelector(".tool-activity-row")?.children.length,
      1,
    );
    assert.equal(
      view.getByRole("list", {
        name: state === "authorized" ? "Requested updates" : "Plan",
      }).textContent,
      "Task ATask BTask C",
    );
    assert.equal(view.getAllByRole("listitem").length, 3);
    assert.equal(view.queryByText("Result / output"), null);
  },
);
