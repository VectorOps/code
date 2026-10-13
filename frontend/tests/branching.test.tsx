import * as testing from "@testing-library/react";
import * as vitest from "vitest";
import type { Json, Params } from "../src/models/json";
import type * as models from "../src/models/runner";
import { selectedHistory } from "../src/models/selected-history";
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
      <StepFeed
        history={[entry]}
        registry={defaultStepRegistry}
        actions={callbacks}
        active={false}
      />,
    );
    const card = view.getByRole("button", { name: "View details" });
    const controls = card.querySelector(
      kind === "tool"
        ? ".tool-activity-row > .tool-step-controls"
        : ".assistant-controls",
    )!;
    const restart = view.getByRole("button", { name: "Restart this step" });
    vitest.expect(restart.closest(".step-card")).toBe(card);
    vitest.expect(controls.contains(restart)).toBe(true);
    vitest.expect(restart.classList.contains("step-details-button")).toBe(true);
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
    testing.fireEvent.click(
      view.getByRole("button", { name: "Restart this step" }),
    );
    const editor = card.querySelector(".branch-editor")!;
    vitest.expect(card.lastElementChild).toBe(editor);
    testing.fireEvent.click(
      view.getByText(/Earlier recorded results are reused/),
    );
    vitest.expect(callbacks.inspect).not.toHaveBeenCalled();
    testing.fireEvent.click(view.getByRole("button", { name: "Restart step" }));
    await testing.waitFor(() =>
      vitest
        .expect(callbacks.branch)
        .toHaveBeenCalledWith(steps.stepView(entry).id, { kind: "restart" }),
    );
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
    const restart = view.getByRole("button", { name: "Restart this step" });
    for (const button of [edit, restart]) {
      vitest.expect(button.textContent).toBe("");
      vitest.expect(button.classList.contains("branch-icon")).toBe(true);
      vitest
        .expect(button.querySelector("svg")?.getAttribute("aria-hidden"))
        .toBe("true");
      vitest
        .expect(button.querySelector("svg")?.getAttribute("width"))
        .toBe("16");
    }
    vitest.expect(edit.title).toBe("Edit input on a new branch");
    vitest
      .expect(restart.title)
      .toBe("Restart this step and continue on a new branch");
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

vitest.test(
  "restart requires confirmation, supports cancellation and disables disconnected controls",
  async () => {
    const callbacks = actions();
    const props = { step: steps.stepView(item("answer")), actions: callbacks };
    const view = testing.render(<StepBranchControls {...props} />);
    testing.fireEvent.click(
      view.getByRole("button", { name: "Restart this step" }),
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
      .expect(view.getByRole("button", { name: "Restart this step" }))
      .toHaveProperty("disabled", true);
    view.rerender(<StepBranchControls {...props} />);
    testing.fireEvent.click(
      view.getByRole("button", { name: "Restart this step" }),
    );
    testing.fireEvent.click(view.getByRole("button", { name: "Restart step" }));
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
      .expect(view.getAllByRole("button", { name: "Restart this step" }))
      .toHaveLength(1);
    toggle(view.getByText("Agent: Child").closest("details")!);
    vitest
      .expect(view.getAllByRole("button", { name: "Restart this step" }))
      .toHaveLength(2);
    vitest.expect(view.getByText("child-answer")).toBeTruthy();
    vitest
      .expect(view.getAllByRole("button", { name: "Edit input" }))
      .toHaveLength(1);
    view.unmount();
    const debug = testing.render(
      <DebugSteps
        history={[item("internal", null, "root", { kind: "workflow.join" })]}
        actions={callbacks}
      />,
    );
    toggle(debug.getByText("1. workflow.join / completed").closest("details")!);
    vitest
      .expect(debug.getByRole("button", { name: "Restart this step" }))
      .toBeTruthy();
  },
);

function mockRunner() {
  vitest.vi
    .spyOn(RunnerConnection.prototype, "connect")
    .mockImplementation(function (this: RunnerConnection) {
      this.onStatus("Connected");
      this.onReady();
    });
  vitest.vi.spyOn(RunnerApi.prototype, "subscribe").mockResolvedValue({});
  vitest.vi.spyOn(RunnerApi.prototype, "sessions").mockResolvedValue([
    {
      execution_id: "existing",
      reference: "chat:workflow",
      status: "cancelled",
      active: false,
      pending_inputs: 0,
    },
  ]);
  vitest.vi.spyOn(RunnerApi.prototype, "workflows").mockResolvedValue([]);
  vitest.vi.spyOn(RunnerApi.prototype, "pending").mockResolvedValue([]);
  vitest.vi
    .spyOn(RunnerApi.prototype, "history")
    .mockResolvedValue({ items: [], next_cursor: null });
  return vitest.vi
    .spyOn(RunnerApi.prototype, "execution")
    .mockResolvedValue(execution);
}

vitest.test(
  "branch retries reuse keys, auto-resume via RPC only, and refresh selected heads on application",
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
    read.mockResolvedValue(updated);
    testing.act(() =>
      view.result.current.api!.connection.onEvent({
        kind: "execution.branch_applied",
        execution_id: "existing",
        operation_id: "operation",
        input_step_id: "synthetic",
      }),
    );
    await testing.waitFor(() =>
      vitest.expect(view.result.current.execution).toEqual(updated),
    );
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
      />,
    );
    vitest.expect(view.getByText("Selected response")).toBeTruthy();
    vitest.expect(view.queryByText("Abandoned response")).toBeNull();
    vitest
      .expect(view.getAllByRole("button", { name: "Restart this step" }))
      .toHaveLength(2);
    testing.fireEvent.click(
      view.getByRole("button", { name: "Debug steps (3)" }),
    );
    const details = view
      .getByText("2. llm.generate / completed")
      .closest("details")!;
    toggle(details);
    const archived = testing.within(details);
    vitest.expect(archived.getByText(/Abandoned response/)).toBeTruthy();
    testing.fireEvent.click(
      archived.getByRole("button", { name: "Restart this step" }),
    );
    testing.fireEvent.click(
      archived.getByRole("button", { name: "Restart step" }),
    );
    await testing.waitFor(() =>
      vitest.expect(branch).toHaveBeenCalledWith("old", { kind: "restart" }),
    );
    view.unmount();
    vitest.vi.unstubAllGlobals();
  },
);

vitest.test(
  "grouped tool calls each expose their own root restart point",
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
    const controls = view.getAllByRole("button", { name: "Restart this step" });
    vitest.expect(controls).toHaveLength(2);
    testing.fireEvent.click(controls[1]);
    testing.fireEvent.click(view.getByRole("button", { name: "Restart step" }));
    await testing.waitFor(() =>
      vitest
        .expect(callbacks.branch)
        .toHaveBeenCalledWith("tool-2", { kind: "restart" }),
    );
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
