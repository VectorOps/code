import { expect, test, vi } from "vitest";
import { fireEvent, render } from "@testing-library/react";
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
): HistoryItem {
  return {
    position,
    step: {
      id: `step-${position}`,
      workflow_id: workflow,
      operation: {
        kind,
        ...(kind === "workflow.spawn" ? { child_workflow_id: "child" } : {}),
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
        inspect: vi.fn(),
        reply: async () => true,
      }}
    />,
  );
}

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
