import * as testing from "@testing-library/react";
import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import { useRunner } from "../src/hooks/useRunner";
import type { Params } from "../src/models/json";
import type { Session } from "../src/models/runner";

vitest.afterEach(() => vitest.vi.restoreAllMocks());

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
      .mockImplementation(async () => [session]);
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
          },
        });
        view.result.current.api!.connection.onEvent({
          kind: "execution.updated",
          execution_id: session.execution_id,
          summary: session,
        });
      });
      vitest.expect(view.result.current.sessions[0]).toEqual(session);
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
      vitest.expect(execution).toHaveBeenCalledTimes(2);
      vitest.expect(history).toHaveBeenCalledTimes(2);
      vitest.expect(view.result.current.sessions[0].pending_inputs).toBe(1);
      vitest.expect(view.result.current.execution?.status).toBe("paused");

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
      vitest.expect(history).toHaveBeenCalledTimes(3);
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
      .mockResolvedValue([session]);
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
      .mockImplementation(async () => [session]);
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
