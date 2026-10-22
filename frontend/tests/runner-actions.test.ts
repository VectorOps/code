import * as testing from "@testing-library/react";
import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import { useRunner } from "../src/hooks/useRunner";
import type { Params } from "../src/models/json";
import type { Session } from "../src/models/runner";

vitest.afterEach(() => vitest.vi.restoreAllMocks());
vitest.beforeEach(() => {
  vitest.vi
    .spyOn(RunnerApi.prototype, "workflowState")
    .mockImplementation(async (_execution, id) => ({
      id,
      definition_ref: `test:${id}`,
    }));
});

vitest.test(
  "deletion uses the execution RPC, retains failures, and clears only the deleted selection",
  async () => {
    const session: Session = {
      execution_id: "existing",
      reference: null,
      status: "completed",
      active: false,
      pending_inputs: 0,
    };
    vitest.vi
      .spyOn(RunnerConnection.prototype, "connect")
      .mockImplementation(function (this: RunnerConnection) {
        this.onStatus("Connected");
        this.onReady();
      });
    vitest.vi.spyOn(RunnerApi.prototype, "subscribe").mockResolvedValue({});
    vitest.vi.spyOn(RunnerApi.prototype, "sessions").mockResolvedValue({
      executions: [session, { ...session, execution_id: "other" }],
      next_after: null,
    });
    vitest.vi.spyOn(RunnerApi.prototype, "workflows").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "pending").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "inboxes").mockResolvedValue([]);
    vitest.vi
      .spyOn(RunnerApi.prototype, "history")
      .mockResolvedValue({ items: [], next_cursor: null });
    vitest.vi
      .spyOn(RunnerApi.prototype, "execution")
      .mockImplementation(async (id) => ({
        id,
        root_workflow_id: "root",
        status: "completed",
        checkpoint: { heads: {} },
      }));
    const request = vitest.vi.spyOn(RunnerConnection.prototype, "request");
    const view = testing.renderHook(() =>
      useRunner({ endpoint: "ws://localhost", token: "test" }),
    );
    await testing.waitFor(() =>
      vitest.expect(view.result.current.sessions).toHaveLength(2),
    );
    testing.act(() => view.result.current.select("existing"));
    request.mockRejectedValueOnce(new Error("Deletion failed"));
    await testing.act(async () => {
      vitest
        .expect(await view.result.current.deleteExecution("existing"))
        .toBe(false);
    });
    vitest.expect(view.result.current.selected).toBe("existing");
    vitest.expect(view.result.current.sessions).toHaveLength(2);
    vitest.expect(view.result.current.error).toContain("Deletion failed");
    let finish = (_value: unknown) => {};
    request.mockReturnValueOnce(new Promise((resolve) => (finish = resolve)));
    let deleting: Promise<boolean>;
    testing.act(() => {
      deleting = view.result.current.deleteExecution("existing");
    });
    vitest.expect(view.result.current.busy).toBe(true);
    testing.act(() => view.result.current.select("other"));
    await testing.act(async () => {
      finish({ execution_id: "existing", deleted: true });
      await deleting;
    });
    vitest.expect(request).toHaveBeenLastCalledWith("executions.delete", {
      execution_id: "existing",
    });
    vitest.expect(view.result.current.selected).toBe("other");
    vitest
      .expect(view.result.current.sessions.map((item) => item.execution_id))
      .toEqual(["other"]);
    await testing.waitFor(() =>
      vitest.expect(view.result.current.execution?.id).toBe("other"),
    );
    testing.act(() =>
      view.result.current.api!.connection.onEvent({
        kind: "execution.deleted",
        execution_id: "other",
        deleted: true,
      }),
    );
    vitest.expect(view.result.current.selected).toBe("");
    vitest.expect(view.result.current.sessions).toEqual([]);
    vitest.expect(view.result.current.execution).toBeUndefined();
    vitest.expect(view.result.current.history.items).toEqual([]);
    vitest.expect(view.result.current.pending).toEqual([]);
    vitest.expect(view.result.current.steering).toBeUndefined();
    vitest.expect(view.result.current.busy).toBe(false);
    view.unmount();
  },
);

vitest.test(
  "steering removal uses the inbox RPC and live queue updates",
  async () => {
    vitest.vi
      .spyOn(RunnerConnection.prototype, "connect")
      .mockImplementation(function (this: RunnerConnection) {
        this.onStatus("Connected");
        this.onReady();
      });
    vitest.vi.spyOn(RunnerApi.prototype, "subscribe").mockResolvedValue({});
    vitest.vi.spyOn(RunnerApi.prototype, "sessions").mockResolvedValue({
      executions: [],
      next_after: null,
    });
    vitest.vi.spyOn(RunnerApi.prototype, "workflows").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "pending").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "history").mockResolvedValue({
      items: [],
      next_cursor: null,
    });
    vitest.vi.spyOn(RunnerApi.prototype, "execution").mockResolvedValue({
      id: "existing",
      root_workflow_id: "root",
      status: "running",
      checkpoint: { heads: {} },
    });
    const address = {
      execution_id: "existing",
      workflow_id: "root",
      conversation_id: "steer",
    };
    const message = {
      id: "queued",
      created_at: "2026-01-01",
      payload: "Use PostgreSQL",
    };
    vitest.vi
      .spyOn(RunnerApi.prototype, "inboxes")
      .mockResolvedValue([{ address, status: "open", queued_messages: 1 }]);
    vitest.vi
      .spyOn(RunnerApi.prototype, "inboxMessages")
      .mockResolvedValue([message]);
    const request = vitest.vi.spyOn(RunnerConnection.prototype, "request");
    const view = testing.renderHook(() =>
      useRunner({ endpoint: "ws://localhost", token: "test" }),
    );
    await testing.act(async () => {
      vitest
        .expect(await view.result.current.removeSteering(message.id))
        .toBe(false);
    });
    vitest.expect(request).not.toHaveBeenCalled();
    testing.act(() => view.result.current.select("existing"));
    await testing.waitFor(() =>
      vitest.expect(view.result.current.steering?.messages).toEqual([message]),
    );
    request.mockRejectedValueOnce(new Error("Removal failed"));
    await testing.act(async () => {
      vitest
        .expect(await view.result.current.removeSteering(message.id))
        .toBe(false);
    });
    vitest.expect(view.result.current.error).toContain("Removal failed");
    vitest.expect(view.result.current.steering?.messages).toEqual([message]);
    vitest.expect(view.result.current.busy).toBe(false);
    request.mockImplementationOnce(async () => {
      view.result.current.api!.connection.onEvent({
        kind: "mailbox.removed",
        execution_id: address.execution_id,
        address,
        status: "open",
        queued_messages: 0,
        message_id: message.id,
        reason: "host_remove",
      });
      return message;
    });
    await testing.act(async () => {
      vitest
        .expect(await view.result.current.removeSteering(message.id))
        .toBe(true);
    });
    vitest.expect(request).toHaveBeenLastCalledWith("mailbox.inboxes.remove", {
      ...address,
      message_id: message.id,
    });
    vitest
      .expect(view.result.current.steering)
      .toEqual({ address, messages: [], count: 0 });
    vitest.expect(view.result.current.error).toBe("");
    request.mockResolvedValueOnce(null);
    await testing.act(async () => {
      vitest
        .expect(await view.result.current.removeSteering(message.id))
        .toBe(true);
    });
    vitest.expect(view.result.current.steering?.count).toBe(0);
    vitest.expect(view.result.current.error).toBe("");
    view.unmount();
  },
);

vitest.test(
  "launch retry identity includes the worktree selection",
  async () => {
    vitest.vi
      .spyOn(RunnerConnection.prototype, "connect")
      .mockImplementation(() => {});
    const start = vitest.vi
      .spyOn(RunnerApi.prototype, "start")
      .mockRejectedValue(new Error("Connection lost"));
    const view = testing.renderHook(() =>
      useRunner({ endpoint: "ws://localhost", token: "test" }),
    );
    await testing.act(async () => {
      await view.result.current.start("agent", {}, "tree-a");
    });
    await testing.act(async () => {
      await view.result.current.start("agent", {}, "tree-a");
    });
    vitest.expect(start.mock.calls[1]).toEqual(start.mock.calls[0]);
    await testing.act(async () => {
      await view.result.current.start("agent", {}, "tree-b");
    });
    vitest.expect(start.mock.calls[2][3]).toBe("tree-b");
    vitest.expect(start.mock.calls[2][2]).not.toBe(start.mock.calls[0][2]);
    await testing.act(async () => {
      await view.result.current.start("agent", {});
    });
    vitest.expect(start.mock.calls[3][3]).toBeUndefined();
    vitest.expect(start.mock.calls[3][2]).not.toBe(start.mock.calls[2][2]);
  },
);

vitest.test(
  "loads older sessions on demand and refreshes an older selection after reconnect",
  async () => {
    const first: Session = {
      execution_id: "first",
      reference: null,
      name: "Recent",
      status: "completed",
      active: false,
      pending_inputs: 0,
    };
    const older = { ...first, execution_id: "older", name: "Older" };
    const refreshed = { ...older, name: "Refreshed older conversation" };
    vitest.vi
      .spyOn(RunnerConnection.prototype, "connect")
      .mockImplementation(function (this: RunnerConnection) {
        this.onStatus("Connected");
        this.onReady();
      });
    vitest.vi.spyOn(RunnerApi.prototype, "subscribe").mockResolvedValue({});
    const sessions = vitest.vi
      .spyOn(RunnerApi.prototype, "sessions")
      .mockResolvedValueOnce({ executions: [first], next_after: "first" })
      .mockResolvedValueOnce({ executions: [older], next_after: null })
      .mockResolvedValue({ executions: [first], next_after: "first" });
    const summary = vitest.vi
      .spyOn(RunnerApi.prototype, "session")
      .mockResolvedValue(refreshed);
    const workflows = vitest.vi
      .spyOn(RunnerApi.prototype, "workflows")
      .mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "pending").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "inboxes").mockResolvedValue([]);
    vitest.vi
      .spyOn(RunnerApi.prototype, "history")
      .mockResolvedValue({ items: [], next_cursor: null });
    vitest.vi.spyOn(RunnerApi.prototype, "execution").mockResolvedValue({
      id: "older",
      root_workflow_id: "root",
      status: "completed",
      checkpoint: { heads: {} },
    });
    const view = testing.renderHook(() =>
      useRunner({ endpoint: "ws://localhost", token: "test" }),
    );
    await testing.waitFor(() =>
      vitest.expect(view.result.current.sessions).toEqual([first]),
    );
    vitest.expect(sessions).toHaveBeenCalledTimes(1);
    vitest.expect(workflows).toHaveBeenCalledTimes(1);
    vitest.expect(view.result.current.hasMoreSessions).toBe(true);
    await testing.act(async () => {
      vitest.expect(await view.result.current.loadMoreSessions()).toBe(true);
    });
    vitest.expect(sessions).toHaveBeenLastCalledWith("first");
    vitest.expect(view.result.current.hasMoreSessions).toBe(false);
    testing.act(() => view.result.current.select("older"));
    await testing.act(async () => {
      const connection = view.result.current.api!.connection;
      connection.onStatus("Disconnected");
      connection.onStatus("Connected");
      connection.onReady();
    });
    await testing.waitFor(() =>
      vitest.expect(view.result.current.sessions).toContainEqual(refreshed),
    );
    vitest.expect(view.result.current.selected).toBe("older");
    vitest.expect(sessions).toHaveBeenCalledTimes(3);
    vitest.expect(summary).toHaveBeenCalledExactlyOnceWith("older");
    testing.act(() =>
      view.result.current.api!.connection.onEvent({
        kind: "step.started",
        execution_id: "background",
      }),
    );
    const activity = view.result.current.activitySessions;
    testing.act(() =>
      view.result.current.api!.connection.onEvent({
        kind: "step.started",
        execution_id: "background",
      }),
    );
    vitest.expect(view.result.current.activitySessions).toBe(activity);
    view.unmount();
  },
);

vitest.test(
  "applies summaries from pushes and reconciles on reconnect without idle polling",
  async () => {
    vitest.vi.useFakeTimers();
    let session: Session = {
      execution_id: "existing",
      reference: "chat:workflow",
      status: "running",
      active: true,
      pending_inputs: 0,
    };
    vitest.vi
      .spyOn(RunnerConnection.prototype, "connect")
      .mockImplementation(function (this: RunnerConnection) {
        this.onStatus("Connected");
        this.onReady();
      });
    const subscribe = vitest.vi
      .spyOn(RunnerApi.prototype, "subscribe")
      .mockResolvedValue({});
    const sessions = vitest.vi
      .spyOn(RunnerApi.prototype, "sessions")
      .mockImplementation(async () => ({
        executions: [session],
        next_after: null,
      }));
    vitest.vi
      .spyOn(RunnerApi.prototype, "session")
      .mockImplementation(async () => session);
    const workflows = vitest.vi
      .spyOn(RunnerApi.prototype, "workflows")
      .mockResolvedValue([]);
    const pending = vitest.vi
      .spyOn(RunnerApi.prototype, "pending")
      .mockResolvedValue([]);
    const execution = vitest.vi
      .spyOn(RunnerApi.prototype, "execution")
      .mockImplementation(async () => ({
        id: "existing",
        root_workflow_id: "root",
        status: session.status,
        checkpoint: { heads: {} },
      }));
    const history = vitest.vi
      .spyOn(RunnerApi.prototype, "history")
      .mockResolvedValue({ items: [], next_cursor: null });
    const inboxes = vitest.vi
      .spyOn(RunnerApi.prototype, "inboxes")
      .mockResolvedValue([]);
    const stop = vitest.vi
      .spyOn(RunnerApi.prototype, "stop")
      .mockResolvedValue({});
    try {
      const view = testing.renderHook(() =>
        useRunner({ endpoint: "ws://localhost", token: "test" }),
      );
      await testing.act(async () => {});
      testing.act(() => view.result.current.select("existing"));
      await testing.act(() => vitest.vi.advanceTimersByTimeAsync(150));
      vitest.expect(view.result.current.execution?.status).toBe("running");
      await testing.act(() => vitest.vi.advanceTimersByTimeAsync(30000));
      for (const read of [
        subscribe,
        sessions,
        workflows,
        pending,
        execution,
        history,
        inboxes,
      ])
        vitest.expect(read).toHaveBeenCalledTimes(1);

      session = {
        ...session,
        name: "Generated title",
        usage: {
          totals: {
            input_tokens: 100,
            output_tokens: 20,
            cost_estimate_usd: "0.01",
            estimated_calls: 1,
            incomplete_calls: 0,
          },
          root_context: null,
        },
      };
      testing.act(() => {
        view.result.current.api!.connection.onEvent({
          kind: "workflow.updated",
          execution_id: "existing",
          workflow: {
            id: "root",
            definition_ref: "chat:workflow",
            name: "Generated title",
            presentation: {
              short_name: "Chat",
              bgcolor: "#123456",
              options: ["prefix_messages"],
            },
          },
        });
        view.result.current.api!.connection.onEvent({
          kind: "execution.updated",
          execution_id: session.execution_id,
          summary: session,
        });
      });
      vitest.expect(view.result.current.sessions[0]).toEqual(session);
      vitest
        .expect(view.result.current.workflowStates.root.presentation)
        .toEqual({
          short_name: "Chat",
          bgcolor: "#123456",
          options: ["prefix_messages"],
        });
      await testing.act(() => vitest.vi.advanceTimersByTimeAsync(200));
      for (const read of [
        sessions,
        workflows,
        execution,
        history,
        pending,
        inboxes,
      ])
        vitest.expect(read).toHaveBeenCalledTimes(1);

      session = { ...session, status: "paused", pending_inputs: 1 };
      testing.act(() => {
        view.result.current.api!.connection.onEvent({
          kind: "execution.updated",
          execution_id: session.execution_id,
          summary: session,
        });
      });
      await testing.act(() => vitest.vi.advanceTimersByTimeAsync(149));
      vitest.expect(sessions).toHaveBeenCalledTimes(1);
      await testing.act(() => vitest.vi.advanceTimersByTimeAsync(1));
      vitest.expect(sessions).toHaveBeenCalledTimes(1);
      for (const read of [execution, history, pending, inboxes])
        vitest.expect(read).toHaveBeenCalledTimes(1);
      vitest.expect(view.result.current.sessions[0].pending_inputs).toBe(1);
      vitest.expect(view.result.current.execution?.status).toBe("paused");

      const step = {
        id: "tool-step",
        revision: 1,
        workflow_id: "root",
        operation: { kind: "llm.tool" },
        outcome: { kind: "completed", result: { tool_name: "read_file" } },
      };
      const approval = {
        execution_id: "existing",
        request_id: "approval",
        status: "pending",
        channel: "tools.approval",
        payload: {},
        response_type: "json",
      };
      testing.act(() => {
        const connection = view.result.current.api!.connection;
        connection.onEvent({
          kind: "step.completed",
          execution_id: "existing",
          position: 1,
          step,
        });
        connection.onEvent({
          kind: "execution.state_updated",
          execution_id: "existing",
          execution: {
            id: "existing",
            root_workflow_id: "root",
            status: "paused",
            checkpoint: { heads: { root: "tool-step" } },
          },
        });
        connection.onEvent({
          kind: "mailbox.requested",
          execution_id: "existing",
          request: approval,
        });
      });
      vitest
        .expect(view.result.current.history.items)
        .toEqual([{ position: 1, step }]);
      vitest
        .expect(view.result.current.execution?.checkpoint.heads)
        .toEqual({ root: "tool-step" });
      vitest.expect(view.result.current.pending).toEqual([approval]);
      await testing.act(async () => {
        vitest.expect(await view.result.current.stop()).toBe(true);
      });
      vitest.expect(stop).toHaveBeenCalledWith("existing");
      await testing.act(() => vitest.vi.advanceTimersByTimeAsync(1000));
      for (const read of [sessions, execution, history, pending, inboxes])
        vitest.expect(read).toHaveBeenCalledTimes(1);

      session = {
        ...session,
        status: "completed",
        active: false,
        pending_inputs: 0,
      };
      await testing.act(async () => {
        const connection = view.result.current.api!.connection;
        connection.onStatus("Disconnected");
        connection.onStatus("Connected");
        connection.onReady();
      });
      await testing.act(() => vitest.vi.advanceTimersByTimeAsync(150));
      vitest.expect(subscribe).toHaveBeenCalledTimes(2);
      vitest.expect(workflows).toHaveBeenCalledTimes(2);
      vitest.expect(sessions).toHaveBeenCalledTimes(2);
      vitest.expect(history).toHaveBeenCalledTimes(2);
      vitest.expect(view.result.current.sessions[0].status).toBe("completed");
      vitest.expect(view.result.current.execution?.status).toBe("completed");
    } finally {
      testing.cleanup();
      vitest.vi.useRealTimers();
    }
  },
);

vitest.test(
  "pushed status survives an older selected-execution response",
  async () => {
    vitest.vi.useFakeTimers();
    const session: Session = {
      execution_id: "existing",
      reference: "chat:workflow",
      status: "running",
      active: true,
      pending_inputs: 0,
    };
    vitest.vi
      .spyOn(RunnerConnection.prototype, "connect")
      .mockImplementation(function (this: RunnerConnection) {
        this.onStatus("Connected");
        this.onReady();
      });
    vitest.vi.spyOn(RunnerApi.prototype, "subscribe").mockResolvedValue({});
    vitest.vi
      .spyOn(RunnerApi.prototype, "sessions")
      .mockResolvedValue({ executions: [session], next_after: null });
    vitest.vi.spyOn(RunnerApi.prototype, "workflows").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "pending").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "inboxes").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "history").mockResolvedValue({
      items: [],
      next_cursor: null,
    });
    const record = {
      id: "existing",
      root_workflow_id: "root",
      status: "running",
      checkpoint: { heads: {} },
    };
    let finish = (_value: typeof record) => {};
    vitest.vi
      .spyOn(RunnerApi.prototype, "execution")
      .mockReturnValue(new Promise((resolve) => (finish = resolve)));
    try {
      const view = testing.renderHook(() =>
        useRunner({ endpoint: "ws://localhost", token: "test" }),
      );
      await testing.act(async () => {});
      testing.act(() => view.result.current.select("existing"));
      await testing.act(() => vitest.vi.advanceTimersByTimeAsync(150));
      await testing.act(async () => {
        view.result.current.api!.connection.onEvent({
          kind: "execution.updated",
          execution_id: session.execution_id,
          summary: { ...session, status: "completed", active: false },
        });
        finish(record);
      });
      vitest.expect(view.result.current.execution?.status).toBe("completed");
      vitest.expect(record.status).toBe("running");
      vitest.expect(RunnerApi.prototype.sessions).toHaveBeenCalledTimes(1);
    } finally {
      testing.cleanup();
      vitest.vi.useRealTimers();
    }
  },
);

vitest.test(
  "resume uses the existing execution and the runner resume wire contract",
  async () => {
    class Connection extends RunnerConnection {
      calls: { method: string; params: Params }[] = [];
      override async request<T>(
        method: string,
        params: Params = {},
      ): Promise<T> {
        this.calls.push({ method, params });
        return { execution_id: "existing", status: "starting" } as T;
      }
    }
    const connection = new Connection("ws://localhost", "test");
    const accepted = await new RunnerApi(connection).resume(
      "existing",
      "retry-key",
    );
    vitest
      .expect(accepted)
      .toEqual({ execution_id: "existing", status: "starting" });
    vitest.expect(connection.calls).toEqual([
      {
        method: "executions.resume",
        params: { execution_id: "existing", idempotency_key: "retry-key" },
      },
    ]);
  },
);

vitest.test.each([
  ["worktree is not ready: deleted", "worktree is not ready: deleted"],
  [null, "Execution failed."],
])(
  "accepted resume surfaces background failure %s despite a cancelled snapshot",
  async (error, expected) => {
    const session: Session = {
      execution_id: "existing",
      reference: "builtin:agent",
      status: "cancelled",
      active: false,
      pending_inputs: 0,
    };
    let connection: RunnerConnection;
    vitest.vi
      .spyOn(RunnerConnection.prototype, "connect")
      .mockImplementation(function (this: RunnerConnection) {
        connection = this;
        this.onStatus("Connected");
        this.onReady();
      });
    vitest.vi.spyOn(RunnerApi.prototype, "subscribe").mockResolvedValue({});
    vitest.vi.spyOn(RunnerApi.prototype, "sessions").mockResolvedValue({
      executions: [session],
      next_after: null,
    });
    vitest.vi.spyOn(RunnerApi.prototype, "workflows").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "pending").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "execution").mockResolvedValue({
      id: "existing",
      root_workflow_id: "root",
      status: "cancelled",
      checkpoint: { heads: {} },
    });
    vitest.vi
      .spyOn(RunnerApi.prototype, "history")
      .mockResolvedValue({ items: [], next_cursor: null });
    vitest.vi.spyOn(RunnerApi.prototype, "resume").mockResolvedValue({
      execution_id: "existing",
      status: "starting",
    });
    const view = testing.renderHook(() =>
      useRunner({ endpoint: "ws://localhost", token: "test" }),
    );
    await testing.waitFor(() =>
      vitest.expect(view.result.current.sessions).toHaveLength(1),
    );
    testing.act(() => view.result.current.select("existing"));
    await testing.act(async () => {
      vitest.expect(await view.result.current.resume()).toBe(true);
    });
    testing.act(() => {
      connection.onEvent({
        kind: "execution.failed",
        execution_id: "other",
        error: "Unrelated failure",
      });
    });
    vitest.expect(view.result.current.error).toBe("");
    testing.act(() => {
      connection.onEvent({
        kind: "execution.failed",
        execution_id: "existing",
        error,
      });
      connection.onEvent({
        kind: "execution.updated",
        execution_id: "existing",
        summary: session,
      });
    });
    vitest.expect(view.result.current.error).toBe(expected);
    vitest.expect(view.result.current.sessions[0]).toMatchObject({
      status: "cancelled",
      active: false,
    });
    testing.act(() => view.result.current.dismiss());
    vitest.expect(view.result.current.error).toBe("");
    view.unmount();
  },
);

vitest.test(
  "resume retains retry keys, uses pushed state and stays in the selected conversation",
  async () => {
    let session: Session = {
      execution_id: "existing",
      reference: "chat:workflow",
      status: "cancelled",
      active: false,
      pending_inputs: 0,
    };
    let connection: RunnerConnection;
    vitest.vi
      .spyOn(RunnerConnection.prototype, "connect")
      .mockImplementation(function (this: RunnerConnection) {
        connection = this;
        this.onStatus("Connected");
        this.onReady();
      });
    vitest.vi.spyOn(RunnerApi.prototype, "subscribe").mockResolvedValue({});
    vitest.vi
      .spyOn(RunnerApi.prototype, "sessions")
      .mockImplementation(async () => ({
        executions: [session],
        next_after: null,
      }));
    vitest.vi.spyOn(RunnerApi.prototype, "workflows").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "pending").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "execution").mockResolvedValue({
      id: "existing",
      root_workflow_id: "root",
      status: "cancelled",
      checkpoint: { heads: {} },
    });
    vitest.vi
      .spyOn(RunnerApi.prototype, "history")
      .mockResolvedValue({ items: [], next_cursor: null });
    const start = vitest.vi.spyOn(RunnerApi.prototype, "start");
    const resume = vitest.vi
      .spyOn(RunnerApi.prototype, "resume")
      .mockImplementation(async () => {
        session = { ...session, status: "starting", active: true };
        connection.onEvent({
          kind: "execution.updated",
          execution_id: session.execution_id,
          summary: session,
        });
        return { execution_id: session.execution_id, status: "starting" };
      })
      .mockRejectedValueOnce(new Error("Connection lost"));
    vitest.vi
      .spyOn(RunnerApi.prototype, "stop")
      .mockImplementation(async () => {
        session = { ...session, status: "cancelled", active: false };
        connection.onEvent({
          kind: "execution.updated",
          execution_id: session.execution_id,
          summary: session,
        });
        return {};
      });
    const view = testing.renderHook(() =>
      useRunner({ endpoint: "ws://localhost", token: "test" }),
    );
    await testing.waitFor(() =>
      vitest.expect(view.result.current.sessions).toHaveLength(1),
    );
    testing.act(() => view.result.current.select("existing"));
    await testing.act(async () => {
      vitest.expect(await view.result.current.resume()).toBe(false);
    });
    vitest.expect(view.result.current.error).toBe("Error: Connection lost");
    await testing.act(async () => {
      vitest.expect(await view.result.current.resume()).toBe(true);
    });
    vitest.expect(resume.mock.calls[1]).toEqual(resume.mock.calls[0]);
    vitest.expect(view.result.current.selected).toBe("existing");
    vitest
      .expect(view.result.current.sessions[0])
      .toMatchObject({ status: "starting", active: true });
    await testing.act(async () => {
      await view.result.current.stop();
    });
    vitest
      .expect(view.result.current.sessions[0])
      .toMatchObject({ status: "cancelled", active: false });
    await testing.act(async () => {
      await view.result.current.resume();
    });
    vitest.expect(resume.mock.calls[2][0]).toBe("existing");
    vitest.expect(resume.mock.calls[2][1]).not.toBe(resume.mock.calls[1][1]);
    vitest.expect(start).not.toHaveBeenCalled();
    vitest.expect(RunnerApi.prototype.sessions).toHaveBeenCalledTimes(1);
    view.unmount();
  },
);
