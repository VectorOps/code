import assert from "node:assert/strict";
import { test } from "vitest";
import { fireEvent, render } from "@testing-library/react";
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
  const view = render(<PlanSummary tool={tool()} />);
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
      "SECTION",
    );
    assert.equal(
      view.container.querySelector(".tool-activity-row")?.children.length,
      1,
    );
    assert.equal(
      view.getByRole("list", { name: "Plan" }).textContent,
      "Task ATask BTask C",
    );
    assert.equal(view.getAllByRole("listitem").length, 3);
    assert.equal(view.queryByText("Result / output"), null);
  },
);
