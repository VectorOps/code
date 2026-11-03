import * as React from "react";
import * as testing from "@testing-library/react";
import { beforeEach, expect, test } from "vitest";
import { DebugProvider, DebugOnly } from "../src/components/Diagnostics";
import { Workspace } from "../src/screens/Workspace";
import { ToolDetails } from "../src/components/tools/ToolDetails";
import { defaultToolFormatters } from "../src/components/tools/defaults";
import { ToolFormatterFactory } from "../src/components/tools/factory";
import { StepRegistry } from "../src/components/steps/registry";
import { StepFeed } from "../src/components/StepFeed";
import { stepView, toolDescription } from "../src/models/steps";
import type { HistoryItem } from "../src/models/runner";
import type { Json } from "../src/models/json";
import { ApprovalActions } from "../src/components/ApprovalActions";
import { WorkflowStats } from "../src/components/WorkflowStats";

beforeEach(() => {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

const message: HistoryItem = {
  position: 1,
  step: {
    id: "internal-step",
    workflow_id: "root-workflow",
    operation: { kind: "llm.generate" },
    outcome: { kind: "completed", result: "Hello" },
  },
};

function tool(name: string, args: Json, result: Json): HistoryItem {
  return {
    position: 2,
    step: {
      id: "internal-call",
      workflow_id: "root-workflow",
      operation: {
        kind: "llm.tool",
        input: { call: { name, arguments: args } },
      },
      outcome: {
        kind: "completed",
        result: {
          tool_name: name,
          content: [{ type: "text", text: JSON.stringify(result) }],
        },
      },
    },
  };
}

const file = tool(
  "read_file",
  { path: "hello.txt" },
  {
    path: "hello.txt",
    content: "File content",
    next_offset: 500,
    content_sha256: "internal-hash",
    truncated: true,
  },
);

function workspaceProps(): React.ComponentProps<typeof Workspace> {
  return {
    sessions: [],
    selected: "session",
    onSelect: () => {},
    history: [message, file],
    pending: [],
    status: "Connected",
    endpoint: "ws://localhost",
    enabled: true,
    busy: false,
    onStop: () => {},
    onReply: async () => true,
    newSession: null,
    connection: <p>Manual connection fields</p>,
    error: "",
    dismiss: () => {},
    hasOlder: false,
  };
}

function openOutput(view: testing.RenderResult) {
  const disclosure = view.getByText("Result / output").closest("details")!;
  disclosure.open = true;
  testing.fireEvent(disclosure, new Event("toggle"));
}

test("production hides debug steps and message inspection but retains useful tool details", () => {
  const view = testing.render(<Workspace {...workspaceProps()} />);
  expect(view.queryByRole("button", { name: "Debug steps" })).toBeNull();
  expect(view.getAllByRole("button", { name: "View details" })).toHaveLength(1);
  testing.fireEvent.click(view.getByText("Hello"));
  expect(view.queryByLabelText("Work inspector")).toBeNull();
  testing.fireEvent.click(view.getByText("Read"));
  expect(view.getByLabelText("Work inspector")).toBeTruthy();
  expect(view.queryByText("Raw call and result")).toBeNull();
  openOutput(view);
  expect(view.getByText("File content")).toBeTruthy();
  expect(view.getByText(/Partial content/)).toBeTruthy();
  expect(view.queryByText("internal-hash")).toBeNull();
  expect(view.queryByText("next offset")).toBeNull();
});

test.each(["debug", "activity"])(
  "disabling debug closes the %s inspector without reopening it later",
  (kind) => {
    const props = workspaceProps();
    const view = testing.render(
      <DebugProvider value={true}>
        <Workspace {...props} />
      </DebugProvider>,
    );
    testing.fireEvent.click(
      kind === "debug"
        ? view.getByRole("button", { name: "Debug steps" })
        : view.getByText("Hello"),
    );
    expect(view.getByLabelText("Work inspector")).toBeTruthy();
    view.rerender(
      <DebugProvider value={false}>
        <Workspace {...props} />
      </DebugProvider>,
    );
    expect(view.queryByLabelText("Work inspector")).toBeNull();
    view.rerender(
      <DebugProvider value={true}>
        <Workspace {...props} />
      </DebugProvider>,
    );
    expect(view.queryByLabelText("Work inspector")).toBeNull();
  },
);

test("disabling debug keeps tool inspection but removes diagnostics immediately", () => {
  const view = testing.render(
    <DebugProvider value={true}>
      <Workspace {...workspaceProps()} />
    </DebugProvider>,
  );
  testing.fireEvent.click(view.getByText("Read"));
  openOutput(view);
  expect(view.getByText("internal-hash")).toBeTruthy();
  view.rerender(
    <DebugProvider value={false}>
      <Workspace {...workspaceProps()} />
    </DebugProvider>,
  );
  expect(view.getByLabelText("Work inspector")).toBeTruthy();
  expect(view.getByText("File content")).toBeTruthy();
  expect(view.queryByText("Raw call and result")).toBeNull();
  expect(view.queryByText("internal-hash")).toBeNull();
});

test("managed production hides connection editing while attachment retains it", () => {
  const props = { ...workspaceProps(), managedConnection: true };
  const view = testing.render(<Workspace {...props} />);
  expect(
    view.queryByRole("button", { name: "Connection settings" }),
  ).toBeNull();
  expect(
    view.getByRole("img", { name: "Connection status: Connected" }),
  ).toBeTruthy();
  view.rerender(
    <DebugProvider value={true}>
      <Workspace {...props} />
    </DebugProvider>,
  );
  testing.fireEvent.click(
    view.getByRole("button", { name: "Connection settings" }),
  );
  expect(view.getByText("Manual connection fields")).toBeTruthy();
  view.rerender(
    <DebugProvider value={false}>
      <Workspace {...props} />
    </DebugProvider>,
  );
  expect(view.queryByLabelText("Work inspector")).toBeNull();
  view.rerender(<Workspace {...props} managedConnection={false} />);
  expect(
    view.getByRole("button", { name: "Connection settings" }),
  ).toBeTruthy();
});

test("process results retain exit status, output and warnings without transport handles", () => {
  const item = tool(
    "exec",
    {
      command: "echo hello",
      worktree_id: "internal-worktree",
      yield_timeout_ms: 100,
    },
    {
      process_id: "internal-process",
      cursor: 500,
      state: "exited",
      returncode: 1,
      timed_out: true,
      dropped_bytes: 12,
      chunks: [{ stream: "stdout", data: "Task output", encoding: "utf-8" }],
    },
  );
  const view = testing.render(
    <ToolDetails
      tool={toolDescription(stepView(item))}
      factory={defaultToolFormatters}
    />,
  );
  openOutput(view);
  expect(view.getByText("Task output")).toBeTruthy();
  expect(view.getByText("returncode")).toBeTruthy();
  expect(view.getByText("Execution timed out.")).toBeTruthy();
  expect(view.getByText("12 bytes of output were dropped.")).toBeTruthy();
  expect(view.container.textContent).not.toMatch(
    /internal-process|internal-worktree|cursor|yield timeout/,
  );
});

test("workflow identity and approval matching remain production information", () => {
  const view = testing.render(
    <>
      <WorkflowStats
        session={{
          execution_id: "internal-execution",
          root_workflow_id: "useful-workflow",
          reference: "agent",
          status: "running",
          active: true,
          pending_inputs: 0,
        }}
      />
      <ApprovalActions
        enabled
        reply={async () => true}
        request={{
          request_id: "internal-request",
          execution_id: "internal-execution",
          workflow_id: "useful-workflow",
          channel: "approval",
          request_type: "tool.approval.v1",
          response_type: "json",
          payload: {
            call: { name: "read_file", arguments: { path: "hello.txt" } },
            auth_context: {
              properties: { repository: "important-repository" },
            },
          },
        }}
      />
    </>,
  );
  expect(view.getByText("useful-workflow")).toBeTruthy();
  expect(view.queryByText("internal-execution")).toBeNull();
  expect(view.getByRole("switch", { name: "Match repository" })).toBeTruthy();
  expect(view.getByText('"important-repository"')).toBeTruthy();
});

test("non-envelope process results and domain IDs remain task content", () => {
  const item = tool(
    "exec",
    { command: "legacy command" },
    { answer: "Useful result", customer_id: "customer-123" },
  );
  const view = testing.render(
    <ToolDetails
      tool={toolDescription(stepView(item))}
      factory={defaultToolFormatters}
    />,
  );
  openOutput(view);
  expect(view.container.textContent).toContain("Useful result");
  expect(view.container.textContent).toContain("customer-123");
});

test("custom step and tool formatters receive the shared debug mode", () => {
  const toolView = toolDescription(stepView(file));
  const factory = new ToolFormatterFactory(
    {},
    {},
    {
      width: 500,
      summary: ({ debug }) => (
        <span>{debug ? "Debug summary" : "Task summary"}</span>
      ),
      details: ({ debug }) => <p>{debug ? "Debug tool" : "Task tool"}</p>,
    },
  );
  const registry = new StepRegistry([
    {
      id: "custom",
      matches: () => true,
      component: ({ debug }) => <p>{debug ? "Debug step" : "Task step"}</p>,
    },
  ]);
  const contents = (
    <>
      <ToolDetails tool={toolView} factory={factory} />
      <StepFeed
        history={[message]}
        registry={registry}
        active={false}
        actions={{
          pending: [],
          enabled: true,
          inspect: () => {},
          reply: async () => true,
        }}
      />
    </>
  );
  const view = testing.render(
    <DebugProvider value={false}>{contents}</DebugProvider>,
  );
  expect(view.getByText("Task step")).toBeTruthy();
  expect(view.getByText("Task tool")).toBeTruthy();
  view.rerender(<DebugProvider value={true}>{contents}</DebugProvider>);
  expect(view.getByText("Debug step")).toBeTruthy();
  expect(view.getByText("Debug tool")).toBeTruthy();
});

test("disabled diagnostic renderers do not evaluate their content", () => {
  let evaluated = false;
  testing.render(
    <DebugOnly>
      {() => {
        evaluated = true;
        return <p>Raw metadata</p>;
      }}
    </DebugOnly>,
  );
  expect(evaluated).toBe(false);
});
