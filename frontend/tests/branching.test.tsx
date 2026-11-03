import { DebugProvider } from "../src/components/Diagnostics";
import * as testing from "@testing-library/react";
import * as vitest from "vitest";
import type { Json, Params } from "../src/models/json";
import type * as models from "../src/models/runner";
import { selectedHistory } from "../src/models/selected-history";
import { workflowProjection } from "../src/models/workflow-tree";
import * as steps from "../src/models/steps";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import { useRunner } from "../src/hooks/useRunner";
import { StepFeed } from "../src/components/StepFeed";
import { StepBranchControls } from "../src/components/StepBranchControls";
import { DebugSteps } from "../src/components/DebugSteps";
import { defaultStepRegistry } from "../src/components/steps/defaults";
import type { StepActions } from "../src/components/steps/registry";
import { Workspace } from "../src/screens/Workspace";

const execution: models.Execution = {
  id: "existing",
  root_workflow_id: "root",
  status: "completed",
  checkpoint: { heads: { root: "new" } },
};

function item(
  id: string,
  previous: string | null = null,
  workflow = "root",
  operation: Json = { kind: "llm.generate" },
  result: Json = id,
  position = 1,
): models.HistoryItem {
  return {
    position,
    step: {
      id,
      previous_step_id: previous,
      workflow_id: workflow,
      operation,
      outcome: { kind: "completed", result },
    },
  };
}

function input(
  id = "input",
  response: Json = "Original text",
  requestType?: string,
) {
  return item(
    id,
    null,
    "root",
    {
      kind: "mailbox.input",
      input: {
        channel: "user.message",
        response_type: requestType ? "json" : "text",
        ...(requestType ? { request_type: requestType } : {}),
      },
    },
    response,
  );
}

function actions(): StepActions {
  return {
    pending: [],
    enabled: true,
    rootWorkflowId: "root",
    inspect: vitest.vi.fn(),
    reply: vitest.vi.fn(async () => true),
    branch: vitest.vi.fn(async () => true),
  };
}

function toggle(element: HTMLDetailsElement) {
  testing.act(() => {
    element.open = true;
    testing.fireEvent(element, new Event("toggle"));
  });
}

vitest.test(
  "debug steps show newest first without reordering shared history",
  () => {
    const history = [2, 1, 3].map((position) =>
      item(`step-${position}`, null, "root", undefined, null, position),
    );
    const view = testing.render(
      <DebugProvider value={true}>
        <DebugSteps history={history} />
      </DebugProvider>,
    );
    vitest
      .expect(
        [...view.container.querySelectorAll("summary")].map(
          (summary) => summary.textContent?.split(".")[0],
        ),
      )
      .toEqual(["3", "2", "1"]);
    vitest.expect(history.map((entry) => entry.position)).toEqual([2, 1, 3]);
  },
);

vitest.test(
  "branch RPC sends both actions and keeps the existing execution",
  async () => {
    class Connection extends RunnerConnection {
      calls: { method: string; params: Params }[] = [];
      override async request<T>(
        method: string,
        params: Params = {},
      ): Promise<T> {
        this.calls.push({ method, params });
        return {
          execution_id: "existing",
          operation_id: "operation",
          status: "accepted",
        } as T;
      }
    }
    const connection = new Connection("ws://localhost", "test");
    const api = new RunnerApi(connection);
    const receipt = await api.branch(
      "existing",
      "input",
      { kind: "insert_input", response: "Edited" },
      "edit-key",
    );
    await api.branch("existing", "answer", { kind: "restart" }, "restart-key");
    vitest.expect(receipt).toEqual({
      execution_id: "existing",
      operation_id: "operation",
      status: "accepted",
    });
    vitest.expect(connection.calls).toEqual([
      {
        method: "executions.branch",
        params: {
          execution_id: "existing",
          step_id: "input",
          action: { kind: "insert_input", response: "Edited" },
          idempotency_key: "edit-key",
        },
      },
      {
        method: "executions.branch",
        params: {
          execution_id: "existing",
          step_id: "answer",
          action: { kind: "restart" },
          idempotency_key: "restart-key",
        },
      },
    ]);
  },
);

vitest.test(
  "selected history excludes abandoned descendants and orphaned children",
  () => {
    const first = input();
    const old = item("old", "input", "root", undefined, "Old answer", 2);
    const spawn = item(
      "spawn",
      "input",
      "root",
      { kind: "workflow.spawn", child_workflow_id: "child" },
      null,
      3,
    );
    const child = item(
      "child-first",
      null,
      "child",
      undefined,
      "Child answer",
      4,
    );
    const completion = item(
      "child-done",
      "child-first",
      "child",
      { kind: "workflow.complete" },
      null,
      5,
    );
    const join = item(
      "join",
      "spawn",
      "root",
      { kind: "workflow.join" },
      null,
      6,
    );
    join.step = {
      ...(join.step as Record<string, Json>),
      outcome: {
        kind: "joined",
        child_completions: [
          { spawn_step_id: "spawn", completion_step_id: "child-done" },
        ],
      },
    };
    const newerChild = item(
      "child-new",
      "child-first",
      "child",
      undefined,
      "Wrong child branch",
      7,
    );
    const orphan = item(
      "orphan",
      null,
      "orphan-workflow",
      undefined,
      "Orphan",
      8,
    );
    const next = item("new", "join", "root", undefined, "New answer", 9);
    const history = [
      first,
      old,
      spawn,
      child,
      completion,
      join,
      newerChild,
      orphan,
      next,
    ];
    const record = {
      ...execution,
      checkpoint: {
        heads: { root: "new", child: "child-new", "orphan-workflow": "orphan" },
      },
    };
    vitest
      .expect(selectedHistory(history, record))
      .toEqual([first, spawn, child, completion, join, next]);
    vitest.expect(selectedHistory(history, record)[0]).toBe(first);
    vitest
      .expect(
        selectedHistory(history, { ...execution, checkpoint: { heads: {} } }),
      )
      .toEqual([]);
    vitest.expect(history).toHaveLength(9);
  },
);

vitest.test(
  "active children use checkpoint heads and missing heads never select old branches",
  () => {
    const spawn = item("spawn", null, "root", {
      kind: "workflow.spawn",
      child_workflow_id: "child",
    });
    const child = item("child-first", null, "child");
    const history = [spawn, child, item("abandoned", null)];
    vitest
      .expect(
        selectedHistory(history, {
          ...execution,
          checkpoint: { heads: { root: "spawn", child: "child-first" } },
        }),
      )
      .toEqual([spawn, child]);
    vitest
      .expect(
        selectedHistory(history, {
          ...execution,
          checkpoint: { heads: { root: "not-loaded" } },
        }),
      )
      .toEqual([]);
  },
);

vitest.test(
  "partial history hides child activity until its selected spawn is loaded",
  () => {
    const spawn = item(
      "spawn",
      null,
      "root",
      { kind: "workflow.spawn", child_workflow_id: "child" },
      null,
      1,
    );
    const child = item(
      "child-answer",
      null,
      "child",
      undefined,
      "Child answer",
      2,
    );
    const latest = item(
      "latest",
      "spawn",
      "root",
      undefined,
      "Latest answer",
      3,
    );
    const record = {
      ...execution,
      checkpoint: { heads: { root: "latest", child: "child-answer" } },
    };
    const partial = selectedHistory([child, latest], record);
    vitest.expect(partial).toEqual([latest]);
    const callbacks = actions();
    const view = testing.render(
      <StepFeed
        history={partial}
        registry={defaultStepRegistry}
        actions={callbacks}
        active
      />,
    );
    vitest.expect(view.queryByText(/Agent:/)).toBeNull();
    const loaded = selectedHistory([spawn, child, latest], record);
    view.rerender(
      <StepFeed
        history={loaded}
        registry={defaultStepRegistry}
        actions={callbacks}
        active
      />,
    );
    toggle(view.getByText(/Agent:/).closest("details")!);
    vitest.expect(view.getByText("Child answer")).toBeTruthy();
    view.rerender(
      <StepFeed
        history={partial}
        registry={defaultStepRegistry}
        actions={callbacks}
        active
      />,
    );
    vitest.expect(view.queryByText("Child answer")).toBeNull();
  },
);

vitest.test(
  "orphan child input remains a decision, never a main composer request",
  () => {
    const root: models.InputRequest = {
      execution_id: "existing",
      request_id: "root-input",
      workflow_id: "root",
      request_type: "chat.message.v1",
      channel: "user.message",
      payload: null,
      response_type: "text",
    };
    const child = { ...root, request_id: "child-input", workflow_id: "child" };
    const projection = workflowProjection([], [root, child], "root");
    vitest.expect(projection.chat).toEqual([root]);
    vitest.expect(projection.decisions).toEqual([child]);
    vitest.expect(projection.roots.map((node) => node.id)).toEqual(["root"]);
  },
);

vitest.test(
  "editing chat text preserves images and uses structured input responses",
  async () => {
    const image = { type: "image", mime_type: "image/png", data: "AAAA" };
    const entry = input(
      "chat",
      { content: [{ type: "text", text: "Original text" }, image] },
      "chat.message.v1",
    );
    const callbacks = actions();
    const view = testing.render(
      <StepBranchControls step={steps.stepView(entry)} actions={callbacks} />,
    );
    testing.fireEvent.click(view.getByRole("button", { name: "Edit input" }));
    const editor = view.getByRole("textbox", { name: "Edit input" });
    vitest.expect(editor).toHaveProperty("value", "Original text");
    vitest.expect(document.activeElement).toBe(editor);
    testing.fireEvent.change(editor, { target: { value: "Edited text" } });
    testing.fireEvent.click(
      view.getByRole("button", { name: "Save and branch" }),
    );
    await testing.waitFor(() =>
      vitest.expect(callbacks.branch).toHaveBeenCalledWith("chat", {
        kind: "insert_input",
        response: { content: [{ type: "text", text: "Edited text" }, image] },
      }),
    );
    await testing.waitFor(() =>
      vitest.expect(view.queryByRole("textbox")).toBeNull(),
    );
    vitest.expect(entry.step).toMatchObject({
      outcome: {
        result: { content: [{ type: "text", text: "Original text" }, image] },
      },
    });
  },
);

vitest.test(
  "input editor supports text and JSON, validates JSON and retains failed drafts",
  async () => {
    const callbacks = actions();
    vitest.vi.mocked(callbacks.branch!).mockResolvedValueOnce(false);
    const text = testing.render(
      <StepBranchControls step={steps.stepView(input())} actions={callbacks} />,
    );
    testing.fireEvent.click(text.getByRole("button", { name: "Edit input" }));
    testing.fireEvent.change(text.getByRole("textbox"), {
      target: { value: "Edited" },
    });
    testing.fireEvent.click(
      text.getByRole("button", { name: "Save and branch" }),
    );
    await testing.waitFor(() =>
      vitest.expect(callbacks.branch).toHaveBeenCalledWith("input", {
        kind: "insert_input",
        response: "Edited",
      }),
    );
    vitest.expect(text.getByRole("textbox")).toHaveProperty("value", "Edited");
    text.unmount();
    const structured = testing.render(
      <StepBranchControls
        step={steps.stepView(input("json", { text: "old" }, "custom.input"))}
        actions={callbacks}
      />,
    );
    testing.fireEvent.click(
      structured.getByRole("button", { name: "Edit input" }),
    );
    testing.fireEvent.change(structured.getByRole("textbox"), {
      target: { value: "{" },
    });
    testing.fireEvent.click(
      structured.getByRole("button", { name: "Save and branch" }),
    );
    await testing.waitFor(() =>
      vitest.expect(structured.getByRole("alert")).toBeTruthy(),
    );
    vitest.expect(callbacks.branch).toHaveBeenCalledTimes(1);
    testing.fireEvent.change(structured.getByRole("textbox"), {
      target: { value: '{"text":"new"}' },
    });
    testing.fireEvent.click(
      structured.getByRole("button", { name: "Save and branch" }),
    );
    await testing.waitFor(() =>
      vitest.expect(callbacks.branch).toHaveBeenCalledWith("json", {
        kind: "insert_input",
        response: { text: "new" },
      }),
    );
  },
);

vitest.test.each(["user", "assistant", "tool"])(
  "%s branch icons live in the card's right-aligned controls without opening details",
  async (kind) => {
    const entry =
      kind === "user"
        ? input()
        : kind === "tool"
          ? item(
              "tool",
              null,
              "root",
              {
                kind: "llm.tool",
                input: {
                  call: { name: "read_file", arguments: { path: "file.txt" } },
                },
              },
              { content: [{ type: "text", text: "{}" }] },
            )
          : item("assistant");
    const callbacks = actions();
    const view = testing.render(
      <DebugProvider value={true}>
        <StepFeed
          history={[entry]}
          registry={defaultStepRegistry}
          actions={callbacks}
          active={false}
        />
      </DebugProvider>,
    );
    const card = view
      .getByRole("button", { name: "View details" })
      .closest(".step-card")!;
    const controls = card.querySelector(
      kind === "tool"
        ? ".tool-activity-row > .tool-step-controls"
        : ".assistant-controls",
    )!;
    vitest
      .expect(view.queryByRole("button", { name: "Restart this step" }))
      .toBeNull();
    vitest
      .expect(view.container.querySelector(".step-branch-controls"))
      .toBeNull();
    if (kind === "user") {
      const edit = view.getByRole("button", { name: "Edit input" });
      vitest.expect(controls.contains(edit)).toBe(true);
      testing.fireEvent.click(edit);
      const editor = view.getByRole("textbox", { name: "Edit input" });
      vitest.expect(card.contains(editor)).toBe(true);
      testing.fireEvent.change(editor, { target: { value: "Edited" } });
      testing.fireEvent.click(
        view.getByText(/Save this input on a new branch/),
      );
      vitest.expect(callbacks.inspect).not.toHaveBeenCalled();
      testing.fireEvent.click(view.getByRole("button", { name: "Cancel" }));
    }
    vitest.expect(callbacks.inspect).not.toHaveBeenCalled();
    testing.fireEvent.click(card);
    vitest
      .expect(callbacks.inspect)
      .toHaveBeenCalledExactlyOnceWith(steps.stepView(entry).id);
  },
);

vitest.test(
  "hidden tool cards do not leave orphaned branch buttons in the feed",
  () => {
    const entry = item(
      "agent-call",
      null,
      "root",
      {
        kind: "llm.tool",
        input: {
          call: {
            name: "run_agent",
            arguments: { reference: "child:workflow" },
          },
        },
      },
      { content: [{ type: "text", text: "{}" }] },
    );
    const view = testing.render(
      <StepFeed
        history={[entry]}
        registry={defaultStepRegistry}
        actions={actions()}
        active={false}
      />,
    );
    vitest
      .expect(view.queryByRole("button", { name: "Restart this step" }))
      .toBeNull();
    vitest.expect(view.container.querySelector(".step-card")).toBeNull();
  },
);

vitest.test(
  "branch actions are compact icons with accessible names and explanatory tooltips",
  () => {
    const view = testing.render(
      <StepBranchControls step={steps.stepView(input())} actions={actions()} />,
    );
    const edit = view.getByRole("button", { name: "Edit input" });
    for (const button of [edit]) {
      vitest.expect(button.textContent).toBe("");
      vitest.expect(button.classList.contains("branch-icon")).toBe(true);
      vitest
        .expect(button.querySelector("svg")?.getAttribute("aria-hidden"))
        .toBe("true");
      vitest
        .expect(button.querySelector("svg")?.getAttribute("width"))
        .toBe("16");
    }
    vitest
      .expect(view.queryByRole("button", { name: "Restart this step" }))
      .toBeNull();
    vitest.expect(edit.title).toBe("Edit input on a new branch");
  },
);

vitest.test.each([false, true])(
  "failed operation retry uses its own ID, with a following completion: %s",
  async (hasCompletion) => {
    const successful = item("successful");
    const failed = item(
      "failed-operation",
      "successful",
      "root",
      { kind: "llm.generate" },
      null,
      2,
    );
    failed.step = {
      ...(failed.step as Record<string, Json>),
      outcome: {
        kind: "failed",
        error: { type: "ProviderError", message: "Request failed" },
      },
    };
    const completion = item(
      "completion",
      "failed-operation",
      "root",
      { kind: "workflow.complete" },
      null,
      3,
    );
    const history = hasCompletion
      ? [successful, failed, completion]
      : [successful, failed];
    const callbacks = actions();
    const view = testing.render(
      <StepBranchControls
        step={steps.stepView(failed, history)}
        actions={callbacks}
      />,
    );
    const retry = view.getByRole("button", { name: "Retry this step" });
    vitest
      .expect(retry.title)
      .toBe("Retry this step and continue on a new branch");
    testing.fireEvent.click(retry);
    vitest
      .expect(view.getByText(/Earlier recorded results are reused/))
      .toBeTruthy();
    vitest.expect(callbacks.branch).not.toHaveBeenCalled();
    testing.fireEvent.click(view.getByRole("button", { name: "Retry step" }));
    await testing.waitFor(() =>
      vitest
        .expect(callbacks.branch)
        .toHaveBeenCalledExactlyOnceWith("failed-operation", {
          kind: "restart",
        }),
    );
  },
);

vitest.test("failed workflow stop exposes a retry control", () => {
  const stopped = item(
    "failed-stop",
    null,
    "root",
    { kind: "workflow.complete" },
    null,
  );
  stopped.step = {
    ...(stopped.step as Record<string, Json>),
    outcome: {
      kind: "failed",
      error: { type: "WorkflowError", message: "Workflow stopped" },
    },
  };
  const view = testing.render(
    <StepBranchControls step={steps.stepView(stopped)} actions={actions()} />,
  );
  vitest
    .expect(view.getByRole("button", { name: "Retry this step" }))
    .toBeTruthy();
});

vitest.test(
  "retry requires confirmation, supports cancellation and disables disconnected controls",
  async () => {
    const callbacks = actions();
    const failed = item("answer");
    failed.step = {
      ...(failed.step as Record<string, Json>),
      outcome: { kind: "failed", error: { type: "Error", message: "failed" } },
    };
    const props = { step: steps.stepView(failed), actions: callbacks };
    const view = testing.render(<StepBranchControls {...props} />);
    testing.fireEvent.click(
      view.getByRole("button", { name: "Retry this step" }),
    );
    vitest.expect(callbacks.branch).not.toHaveBeenCalled();
    vitest
      .expect(view.getByText(/External changes are not undone/))
      .toBeTruthy();
    testing.fireEvent.click(view.getByRole("button", { name: "Cancel" }));
    vitest.expect(callbacks.branch).not.toHaveBeenCalled();
    view.rerender(
      <StepBranchControls
        {...props}
        actions={{ ...callbacks, enabled: false }}
      />,
    );
    vitest
      .expect(view.getByRole("button", { name: "Retry this step" }))
      .toHaveProperty("disabled", true);
    view.rerender(<StepBranchControls {...props} />);
    testing.fireEvent.click(
      view.getByRole("button", { name: "Retry this step" }),
    );
    testing.fireEvent.click(view.getByRole("button", { name: "Retry step" }));
    await testing.waitFor(() =>
      vitest
        .expect(callbacks.branch)
        .toHaveBeenCalledWith("answer", { kind: "restart" }),
    );
  },
);

vitest.test(
  "feed controls only target the root; root spawn and hidden debug steps can restart",
  () => {
    const callbacks = actions();
    const history = [
      input(),
      item("spawn", "input", "root", {
        kind: "workflow.spawn",
        child_workflow_id: "child",
        description: "Child",
      }),
      item("child-answer", null, "child"),
      item("delivery", "spawn", "root", { kind: "workflow.resume" }),
    ];
    const view = testing.render(
      <StepFeed
        history={history}
        actions={callbacks}
        registry={defaultStepRegistry}
        active={false}
      />,
    );
    vitest
      .expect(view.queryByRole("button", { name: "Restart this step" }))
      .toBeNull();
    toggle(view.getByText("Agent: Child").closest("details")!);
    vitest
      .expect(view.queryByRole("button", { name: "Restart this step" }))
      .toBeNull();
    vitest.expect(view.getByText("child-answer")).toBeTruthy();
    vitest
      .expect(view.getAllByRole("button", { name: "Edit input" }))
      .toHaveLength(1);
    view.unmount();
    const debug = testing.render(
      <DebugProvider value={true}>
        <DebugSteps
          history={[item("internal", null, "root", { kind: "workflow.join" })]}
          actions={callbacks}
        />
      </DebugProvider>,
    );
    toggle(debug.getByText("1. workflow.join / completed").closest("details")!);
    vitest
      .expect(debug.queryByRole("button", { name: "Restart this step" }))
      .toBeNull();
  },
);

function mockRunner() {
  vitest.vi
    .spyOn(RunnerApi.prototype, "workflowState")
    .mockImplementation(async (_execution, id) => ({
      id,
      definition_ref: `test:${id}`,
    }));
  vitest.vi
    .spyOn(RunnerConnection.prototype, "connect")
    .mockImplementation(function (this: RunnerConnection) {
      this.onStatus("Connected");
      this.onReady();
    });
  vitest.vi.spyOn(RunnerApi.prototype, "subscribe").mockResolvedValue({});
  vitest.vi.spyOn(RunnerApi.prototype, "sessions").mockResolvedValue({
    executions: [
      {
        execution_id: "existing",
        reference: "chat:workflow",
        status: "cancelled",
        active: false,
        pending_inputs: 0,
      },
    ],
    next_after: null,
  });
  vitest.vi.spyOn(RunnerApi.prototype, "workflows").mockResolvedValue([]);
  vitest.vi.spyOn(RunnerApi.prototype, "pending").mockResolvedValue([]);
  vitest.vi.spyOn(RunnerApi.prototype, "inboxes").mockResolvedValue([]);
  vitest.vi
    .spyOn(RunnerApi.prototype, "history")
    .mockResolvedValue({ items: [], next_cursor: null });
  return vitest.vi
    .spyOn(RunnerApi.prototype, "execution")
    .mockResolvedValue(execution);
}

vitest.test(
  "branch retries reuse keys, auto-resume via RPC only, and apply pushed heads without reads",
  async () => {
    const read = mockRunner();
    const branch = vitest.vi
      .spyOn(RunnerApi.prototype, "branch")
      .mockResolvedValue({
        execution_id: "existing",
        operation_id: "operation",
        status: "accepted",
      })
      .mockRejectedValueOnce(new Error("Connection lost"));
    const start = vitest.vi.spyOn(RunnerApi.prototype, "start");
    const stop = vitest.vi.spyOn(RunnerApi.prototype, "stop");
    const resume = vitest.vi.spyOn(RunnerApi.prototype, "resume");
    const view = testing.renderHook(() =>
      useRunner({ endpoint: "ws://localhost", token: "test" }),
    );
    await testing.waitFor(() =>
      vitest.expect(view.result.current.sessions).toHaveLength(1),
    );
    testing.act(() => view.result.current.select("existing"));
    await testing.waitFor(() =>
      vitest.expect(view.result.current.execution).toEqual(execution),
    );
    await testing.act(async () =>
      vitest
        .expect(await view.result.current.branch("answer", { kind: "restart" }))
        .toBe(false),
    );
    await testing.act(async () =>
      vitest
        .expect(await view.result.current.branch("answer", { kind: "restart" }))
        .toBe(true),
    );
    vitest.expect(branch.mock.calls[1]).toEqual(branch.mock.calls[0]);
    vitest.expect(view.result.current.selected).toBe("existing");
    vitest
      .expect(view.result.current.branching?.operation_id)
      .toBe("operation");
    vitest.expect(view.result.current.enabled).toBe(true);
    testing.act(() =>
      view.result.current.api!.connection.onEvent({
        kind: "execution.branch_applied",
        execution_id: "other",
        operation_id: "operation",
      }),
    );
    vitest
      .expect(view.result.current.branching?.operation_id)
      .toBe("operation");
    const updated = {
      ...execution,
      checkpoint: { heads: { root: "restarted" } },
    };
    const synthetic = {
      id: "synthetic",
      revision: 1,
      workflow_id: "root",
      operation: { kind: "mailbox.input" },
      outcome: { kind: "completed", result: "Edited input" },
    };
    testing.act(() => {
      const connection = view.result.current.api!.connection;
      connection.onEvent({
        kind: "step.completed",
        execution_id: "existing",
        position: 1,
        step: synthetic,
      });
      connection.onEvent({
        kind: "execution.branch_applied",
        execution_id: "existing",
        operation_id: "operation",
        input_step_id: "synthetic",
      });
      connection.onEvent({
        kind: "execution.state_updated",
        execution_id: "existing",
        execution: updated,
      });
    });
    vitest.expect(view.result.current.execution).toEqual(updated);
    vitest
      .expect(view.result.current.history.items)
      .toEqual([{ position: 1, step: synthetic }]);
    vitest.expect(view.result.current.branching).toBeUndefined();
    vitest.expect(read).toHaveBeenCalledTimes(1);
    vitest.expect(RunnerApi.prototype.history).toHaveBeenCalledTimes(1);
    vitest.expect(view.result.current.enabled).toBe(true);
    vitest.expect(start).not.toHaveBeenCalled();
    vitest.expect(stop).not.toHaveBeenCalled();
    vitest.expect(resume).not.toHaveBeenCalled();
    await testing.act(async () => {
      await view.result.current.branch("answer", { kind: "restart" });
    });
    vitest.expect(branch.mock.calls[2][3]).not.toBe(branch.mock.calls[1][3]);
    view.unmount();
  },
);

vitest.test.each(["child", "delivery"])(
  "unsupported %s targets never expose branch controls",
  (target) => {
    const entry =
      target === "child"
        ? item("child-input", null, "child", {
            kind: "mailbox.input",
            input: { channel: "user.message" },
          })
        : item("delivery", null, "root", { kind: "workflow.resume" });
    const view = testing.render(
      <StepBranchControls step={steps.stepView(entry)} actions={actions()} />,
    );
    vitest.expect(view.queryByRole("button")).toBeNull();
  },
);

vitest.test(
  "duplicate submissions are disabled while awaiting branch acceptance",
  async () => {
    const callbacks = actions();
    let finish: (value: boolean) => void = () => {};
    vitest.vi.mocked(callbacks.branch!).mockReturnValue(
      new Promise<boolean>((resolve) => {
        finish = resolve;
      }),
    );
    const view = testing.render(
      <StepBranchControls step={steps.stepView(input())} actions={callbacks} />,
    );
    testing.fireEvent.click(view.getByRole("button", { name: "Edit input" }));
    testing.fireEvent.click(
      view.getByRole("button", { name: "Save and branch" }),
    );
    const submit = view.getByRole("button", { name: "Submitting…" });
    vitest.expect(submit).toHaveProperty("disabled", true);
    testing.fireEvent.click(submit);
    vitest.expect(callbacks.branch).toHaveBeenCalledTimes(1);
    await testing.act(async () => {
      finish(true);
    });
  },
);

vitest.test(
  "workspace renders selected branches while Debug steps retains abandoned history",
  async () => {
    vitest.vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    );
    const first = input();
    const old = item(
      "old",
      "input",
      "root",
      undefined,
      "Abandoned response",
      2,
    );
    const next = item(
      "new",
      "input",
      "root",
      undefined,
      "Selected response",
      3,
    );
    const branch = vitest.vi.fn(async () => true);
    const view = testing.render(
      <DebugProvider value={true}>
        <Workspace
          sessions={[
            {
              execution_id: "existing",
              root_workflow_id: "root",
              reference: "chat:workflow",
              status: "cancelled",
              active: false,
              pending_inputs: 0,
            },
          ]}
          selected="existing"
          onSelect={() => {}}
          history={[first, old, next]}
          execution={execution}
          pending={[]}
          status="Connected"
          endpoint="ws://localhost"
          enabled
          busy={false}
          onStop={() => {}}
          onReply={async () => true}
          onBranch={branch}
          newSession={null}
          connection={null}
          error=""
          dismiss={() => {}}
          hasOlder={false}
        />
      </DebugProvider>,
    );
    vitest.expect(view.getByText("Selected response")).toBeTruthy();
    vitest.expect(view.queryByText("Abandoned response")).toBeNull();
    vitest
      .expect(view.queryByRole("button", { name: "Restart this step" }))
      .toBeNull();
    testing.fireEvent.click(view.getByRole("button", { name: "Debug steps" }));
    const details = view
      .getByText("2. llm.generate / completed")
      .closest("details")!;
    toggle(details);
    const archived = testing.within(details);
    vitest.expect(archived.getByText(/Abandoned response/)).toBeTruthy();
    vitest
      .expect(archived.queryByRole("button", { name: "Restart this step" }))
      .toBeNull();
    vitest.expect(branch).not.toHaveBeenCalled();
    view.unmount();
    vitest.vi.unstubAllGlobals();
  },
);

vitest.test(
  "inspecting a retried tool uses selected history when abandoned calls reuse IDs",
  () => {
    vitest.vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        disconnect() {}
      },
    );
    const generation = (
      id: string,
      previous: string | null,
      position: number,
      command: string,
    ) =>
      item(
        id,
        previous,
        "root",
        { kind: "llm.generate" },
        {
          content: [
            {
              type: "tool_call",
              id: "call",
              name: "exec",
              arguments: { command },
            },
          ],
        },
        position,
      );
    const tool = (id: string, previous: string, position: number) =>
      item(
        id,
        previous,
        "root",
        { kind: "llm.tool" },
        {
          tool_name: "exec",
          tool_call_id: "call",
          content: [{ type: "text", text: "{}" }],
        },
        position,
      );
    const history = [
      generation("first", null, 1, "echo correct-branch"),
      tool("old-tool", "first", 2),
      generation("abandoned", "old-tool", 3, "echo abandoned-branch"),
      tool("retried-tool", "first", 4),
    ];
    const view = testing.render(
      <Workspace
        sessions={[]}
        selected="existing"
        onSelect={() => {}}
        history={history}
        execution={{
          ...execution,
          checkpoint: { heads: { root: "retried-tool" } },
        }}
        pending={[]}
        status="Connected"
        endpoint="ws://localhost"
        enabled
        busy={false}
        onStop={() => {}}
        onReply={async () => true}
        newSession={null}
        connection={null}
        error=""
        dismiss={() => {}}
        hasOlder={false}
      />,
    );
    testing.fireEvent.click(view.getByText("echo correct-branch"));
    const inspector = view.getByLabelText("Work inspector");
    vitest
      .expect(inspector.querySelector(".tool-code"))
      .toHaveProperty("textContent", "echo correct-branch");
    vitest.expect(inspector.textContent).not.toContain("echo abandoned-branch");
    view.unmount();
    vitest.vi.unstubAllGlobals();
  },
);

vitest.test(
  "grouped successful tool calls do not expose restart controls",
  async () => {
    const callbacks = actions();
    const history = [1, 2].map((position) =>
      item(
        `tool-${position}`,
        null,
        "root",
        {
          kind: "llm.tool",
          input: {
            call: {
              name: "read_file",
              arguments: { path: `file-${position}` },
            },
          },
        },
        { content: [{ type: "text", text: "{}" }] },
        position,
      ),
    );
    const view = testing.render(
      <StepFeed
        history={history}
        registry={defaultStepRegistry}
        actions={callbacks}
        active={false}
      />,
    );
    const details = view.container.querySelector(
      ".tool-group details",
    ) as HTMLDetailsElement;
    toggle(details);
    vitest
      .expect(view.queryByRole("button", { name: "Restart this step" }))
      .toBeNull();
    vitest.expect(callbacks.branch).not.toHaveBeenCalled();
  },
);

vitest.test(
  "branch errors arriving before the receipt are surfaced without leaving a pending transition",
  async () => {
    mockRunner();
    const view = testing.renderHook(() =>
      useRunner({ endpoint: "ws://localhost", token: "test" }),
    );
    await testing.waitFor(() =>
      vitest.expect(view.result.current.sessions).toHaveLength(1),
    );
    testing.act(() => view.result.current.select("existing"));
    vitest.vi
      .spyOn(RunnerApi.prototype, "branch")
      .mockImplementation(async () => {
        view.result.current.api!.connection.onEvent({
          kind: "execution.branch_failed",
          execution_id: "existing",
          operation_id: "failed",
          error: "Prefix changed",
        });
        return {
          execution_id: "existing",
          operation_id: "failed",
          status: "accepted",
        };
      });
    await testing.act(async () => {
      await view.result.current.branch("answer", { kind: "restart" });
    });
    vitest.expect(view.result.current.error).toBe("Prefix changed");
    vitest.expect(view.result.current.branching).toBeUndefined();
    vitest.expect(view.result.current.enabled).toBe(true);
    view.unmount();
  },
);

vitest.test(
  "reconnection rereads checkpoint heads after missed branch events",
  async () => {
    const read = mockRunner();
    const view = testing.renderHook(() =>
      useRunner({ endpoint: "ws://localhost", token: "test" }),
    );
    await testing.waitFor(() =>
      vitest.expect(view.result.current.sessions).toHaveLength(1),
    );
    testing.act(() => view.result.current.select("existing"));
    await testing.waitFor(() =>
      vitest.expect(view.result.current.execution).toEqual(execution),
    );
    const updated = {
      ...execution,
      checkpoint: { heads: { root: "other-branch" } },
    };
    read.mockResolvedValue(updated);
    testing.act(() => {
      const connection = view.result.current.api!.connection;
      connection.onStatus("Disconnected");
      connection.onStatus("Connected");
      connection.onReady();
    });
    await testing.waitFor(() =>
      vitest.expect(view.result.current.execution).toEqual(updated),
    );
    view.unmount();
  },
);
