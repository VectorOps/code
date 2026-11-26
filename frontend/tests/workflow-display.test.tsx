import { expect, test, vi } from "vitest";
import { fireEvent, render, within } from "@testing-library/react";
import type { Json } from "../src/models/json";
import * as presentation from "../src/models/workflow-display";
import type {
  HistoryItem,
  InputRequest,
  WorkflowState,
} from "../src/models/runner";
import { workflowProjection } from "../src/models/workflow-tree";
import { stepView } from "../src/models/steps";
import { StepFeed } from "../src/components/StepFeed";
import { defaultStepRegistry } from "../src/components/steps/defaults";

function item(
  position: number,
  kind: string,
  workflow = "root",
  message = "Hello",
  description?: string,
): HistoryItem {
  return {
    position,
    step: {
      id: `step-${position}`,
      workflow_id: workflow,
      operation: {
        kind,
        ...(kind === "workflow.spawn"
          ? { child_workflow_id: "child", description }
          : {}),
        input: { call: { name: "exec", arguments: { command: "ls" } } },
      },
      outcome: {
        kind: "completed",
        result: { content: [{ type: "text", text: message }] },
      },
    },
  };
}

const root: WorkflowState = {
  id: "root",
  definition_ref: "test:root",
  name: "Root name",
  presentation: {
    short_name: "Parent",
    bgcolor: "#123ABC",
    options: ["prefix_messages", "llm_only"],
  },
};
const child: WorkflowState = {
  id: "child",
  definition_ref: "test:child",
  name: "Child name",
  presentation: { short_name: "Helper", bgcolor: "#abcdef", options: [] },
};

test("display policy matches runner inheritance and label fallback", () => {
  const parent = presentation.resolveDisplay(root, "Fallback");
  const display = presentation.resolveDisplay(child, "Fallback", parent);
  expect(display.label).toBe("Helper");
  expect(display.bgcolor).toBe("#abcdef");
  expect(display.options).toEqual(new Set(["prefix_messages", "llm_only"]));
  expect(
    presentation.showsStep(display, stepView(item(1, "llm.generate"))),
  ).toBe(true);
  expect(presentation.showsStep(display, stepView(item(1, "llm.tool")))).toBe(
    false,
  );
  expect(
    presentation.showsStep(display, stepView(item(1, "custom.activity"))),
  ).toBe(false);
  expect(
    presentation.resolveDisplay(
      { id: "plain", definition_ref: "test:plain" },
      "Fallback",
      parent,
    ).bgcolor,
  ).toBeUndefined();
  expect(
    presentation.resolveDisplay({ ...child, presentation: {} }, "Fallback")
      .label,
  ).toBe("Child name");
  expect(
    presentation.resolveDisplay(
      { id: "plain", definition_ref: "test:plain" },
      "Fallback",
    ).label,
  ).toBe("test:plain");
  expect(presentation.resolveDisplay(undefined, "Legacy label").label).toBe(
    "Legacy label",
  );
  expect(
    presentation.resolveDisplay(
      { ...child, presentation: { bgcolor: "red;display:none" } },
      "Fallback",
    ).bgcolor,
  ).toBeUndefined();
  const hiddenParent = presentation.resolveDisplay(
    { ...root, presentation: { options: ["hide_subworkflows"] } },
    "Fallback",
  );
  expect(hiddenParent.hidden).toBe(false);
  const hiddenChild = presentation.resolveDisplay(
    child,
    "Fallback",
    hiddenParent,
  );
  expect(hiddenChild.hidden).toBe(true);
  expect(
    presentation.showsStep(hiddenChild, stepView(item(1, "llm.generate"))),
  ).toBe(false);
  expect(
    presentation.resolveDisplay(undefined, "Grandchild", hiddenChild).hidden,
  ).toBe(true);
});

function feed(
  history: HistoryItem[],
  states: Record<string, WorkflowState>,
  pending: InputRequest[] = [],
  inspect = vi.fn(),
) {
  return render(
    <StepFeed
      history={history}
      active
      registry={defaultStepRegistry}
      projection={workflowProjection(history, pending, "root", states)}
      actions={{
        pending,
        enabled: true,
        inspect,
        reply: async () => true,
      }}
    />,
  );
}

test.each([
  { description: undefined, workflow: child, expected: "Helper" },
  { description: "", workflow: child, expected: "Helper" },
  { description: " \t ", workflow: child, expected: "Helper" },
  {
    description: undefined,
    workflow: { ...child, presentation: {} },
    expected: "Child name",
  },
  {
    description: " Review display commits ",
    workflow: child,
    expected: "Review display commits",
  },
  {
    description:
      "Review display commits and verify the subworkflow description override",
    workflow: { ...child, presentation: {} },
    expected:
      "Review display commits and verify the subworkflow description override",
  },
  {
    description: "Legacy task",
    workflow: undefined,
    expected: "Agent: Legacy task",
  },
])(
  "spawn description $description resolves to $expected",
  ({ description, workflow, expected }) => {
    const history = [
      item(1, "workflow.spawn", "root", "Hello", description),
      item(2, "llm.generate", "child"),
    ];
    const states: Record<string, WorkflowState> = workflow
      ? { root, child: workflow }
      : { root };
    const original = JSON.stringify({ history, states });
    const view = feed(history, states);
    const branch = view.container.querySelector(
      "details.workflow-branch",
    ) as HTMLDetailsElement;
    expect(branch.querySelector("summary > span")?.textContent).toBe(expected);
    branch.open = true;
    fireEvent(branch, new Event("toggle"));
    expect(
      view.container.querySelector(".workflow-message-prefix")?.textContent,
    ).toBe(expected);
    expect(JSON.stringify({ history, states })).toBe(original);
  },
);

test("feed prefixes and tints messages while enforcing runner visibility without a local override", () => {
  const history = [item(1, "llm.generate"), item(2, "llm.tool")];
  const original = JSON.stringify(history);
  const view = feed(history, { root });
  expect(view.getByText("Parent").className).toBe("workflow-message-prefix");
  expect(view.getByText("Hello")).toBeTruthy();
  expect(view.queryByText("ls")).toBeNull();
  expect(
    view.container
      .querySelector('[data-history-key="step-1"]')
      ?.getAttribute("style"),
  ).toContain("#123abc");
  expect(
    view.queryByRole("checkbox", { name: "Show all workflow activity" }),
  ).toBeNull();
  expect(JSON.stringify(history)).toBe(original);
});

test("hidden descendants keep approvals and workflow failures accessible while suppressing messages", () => {
  const history = [
    item(1, "workflow.spawn"),
    item(2, "llm.generate", "child", "Hidden reply"),
    item(3, "llm.tool", "child"),
  ];
  history.push({
    position: 4,
    step: {
      id: "failed",
      workflow_id: "child",
      operation: { kind: "workflow.complete" },
      outcome: {
        kind: "failed",
        error: { message: "Workflow failed visibly" },
      },
    },
  });
  const pending: InputRequest[] = [
    {
      request_id: "step-3",
      execution_id: "execution",
      workflow_id: "child",
      request_type: "tool.approval.v1",
      channel: "tools.approval",
      response_type: "json",
      payload: { call: { name: "exec", arguments: { command: "ls" } } },
    },
  ];
  const view = feed(
    history,
    {
      root: { ...root, presentation: { options: ["hide_subworkflows"] } },
      child,
    },
    pending,
  );
  expect(view.queryByText("Hidden reply")).toBeNull();
  expect(view.getByText("Workflow failed visibly")).toBeTruthy();
  expect(view.getByRole("button", { name: "Review" })).toBeTruthy();
  expect(view.getByText("ls")).toBeTruthy();
  expect(
    view.container.querySelector('[id="decision-execution-step-3"]'),
  ).toBeTruthy();
});

test("child message prefixes use the child's name and color under inherited options", () => {
  const view = feed(
    [
      item(1, "workflow.spawn"),
      item(2, "llm.generate", "child"),
      item(3, "llm.tool", "child"),
    ],
    { root, child },
  );
  const branch = view.container.querySelector(
    "details.workflow-branch",
  ) as HTMLDetailsElement;
  branch.open = true;
  fireEvent(branch, new Event("toggle"));
  expect(
    view.container.querySelector(".workflow-message-prefix")?.textContent,
  ).toBe("Helper");
  expect(
    view.container
      .querySelector('[data-history-key="step-2"]')
      ?.getAttribute("style"),
  ).toContain("#abcdef");
  expect(view.queryByText("ls")).toBeNull();
});

function launch(
  position: number,
  name: string,
  args: Record<string, Json>,
  completedAt?: string,
): HistoryItem {
  return {
    position,
    step: {
      id: `launch-${position}`,
      workflow_id: "root",
      operation: {
        kind: "llm.tool",
        input: {
          call: { name, arguments: { reference: "test:child", ...args } },
        },
      },
      ...(completedAt ? { completed_at: completedAt } : {}),
      outcome: completedAt
        ? { kind: "completed", result: { is_error: true, content: [] } }
        : { kind: "pending" },
    },
  };
}

function spawn(
  position: number,
  childId: string,
  description?: string,
  worktree?: string,
): HistoryItem {
  return {
    position,
    step: {
      id: `spawn-${position}`,
      workflow_id: "root",
      created_at: "2026-01-01T00:00:01Z",
      operation: {
        kind: "workflow.spawn",
        child_workflow_id: childId,
        description: description ?? null,
        worktree_id: worktree ?? null,
      },
      outcome: { kind: "completed" },
    },
  };
}

test("parallel agent and workflow launches render only as child rows with the correct details", () => {
  const inspect = vi.fn();
  const view = feed(
    [
      launch(1, "run_agent", {
        description: "Review projection",
        worktree_id: "refactor",
      }),
      launch(2, "subworkflow", { description: "Review storage" }),
      spawn(3, "storage", "Review storage"),
      spawn(4, "child", "Review projection", "refactor"),
    ],
    { child },
    [],
    inspect,
  );

  expect(view.container.querySelectorAll(".tool-step")).toHaveLength(0);
  expect(view.getAllByText("Running")).toHaveLength(2);
  expect(
    view.getByText("Review projection").querySelector(".worktree-label"),
  ).toBeTruthy();
  for (const [label, id] of [
    ["Agent: Review storage", "launch-2"],
    ["Review projection", "launch-1"],
  ]) {
    const branch = view.getByText(label).closest("details")!;
    const summary = branch.querySelector("summary")!;
    const button = within(summary).getByRole("button", {
      name: "View launch details",
    });
    expect(summary.lastElementChild).toBe(button);
    expect(button.textContent).toBe("");
    expect(branch.open).toBe(false);
    fireEvent.click(button);
    expect(inspect).toHaveBeenLastCalledWith(id);
    expect(branch.open).toBe(false);
  }
});

test("child input disambiguates identical launch labels without guessing when state is missing", () => {
  const input = { options: { a: 1, b: 2 }, task: "review" };
  const history = [
    launch(1, "run_agent", { description: "Review", input: { task: "other" } }),
    launch(2, "run_agent", { description: "Review", input }),
    spawn(3, "child", "Review"),
  ];
  const ambiguous = feed(history, {});
  expect(ambiguous.getAllByText("Run Agent")).toHaveLength(2);
  ambiguous.unmount();

  const view = feed(history, {
    child: { ...child, input: { task: "review", options: { b: 2, a: 1 } } },
  });
  expect(
    view
      .getByText("Run Agent")
      .closest(".feed-item")
      ?.getAttribute("data-history-key"),
  ).toBe("launch-1");
});

test("unspawned calls, approvals, and earlier failures keep their tool cards", () => {
  const args = {
    reference: "test:child",
    description: "Review completed work",
  };
  const pending: InputRequest[] = [
    {
      request_id: "launch-3",
      execution_id: "execution",
      workflow_id: "root",
      request_type: "tool.approval.v1",
      channel: "tools.approval",
      response_type: "json",
      payload: { call: { name: "run_agent", arguments: args } },
    },
  ];
  const view = feed(
    [
      launch(1, "run_agent", { description: "Still launching" }),
      launch(2, "run_agent", args, "2026-01-01T00:00:00Z"),
      launch(3, "run_agent", args),
      launch(4, "run_agent", args, "2026-01-01T00:00:02Z"),
      spawn(5, "child", args.description),
    ],
    { child },
    pending,
  );

  expect(view.getByText("Approve Run Agent")).toBeTruthy();
  expect(
    [...view.container.querySelectorAll('[data-history-key^="launch-"]')].map(
      (row) => row.getAttribute("data-history-key"),
    ),
  ).toEqual(["launch-1", "launch-2", "launch-3"]);
  expect(
    view.getByText(args.description).closest("summary")?.textContent,
  ).toContain("Failed");
});
