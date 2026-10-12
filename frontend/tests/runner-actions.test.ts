import * as testing from "@testing-library/react";
import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import { useRunner } from "../src/hooks/useRunner";
import type { Params } from "../src/models/json";
import type { Session } from "../src/models/runner";

vitest.afterEach(() => vitest.vi.restoreAllMocks());

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
  "resume retains retry keys, refreshes state and stays in the selected conversation",
  async () => {
    let session: Session = {
      execution_id: "existing",
      reference: "chat:workflow",
      status: "cancelled",
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
    vitest.vi
      .spyOn(RunnerApi.prototype, "sessions")
      .mockImplementation(async () => [session]);
    vitest.vi.spyOn(RunnerApi.prototype, "workflows").mockResolvedValue([]);
    vitest.vi.spyOn(RunnerApi.prototype, "pending").mockResolvedValue([]);
    vitest.vi
      .spyOn(RunnerApi.prototype, "history")
      .mockResolvedValue({ items: [], next_cursor: null });
    const start = vitest.vi.spyOn(RunnerApi.prototype, "start");
    const resume = vitest.vi
      .spyOn(RunnerApi.prototype, "resume")
      .mockImplementation(async () => {
        session = { ...session, status: "starting", active: true };
        return { execution_id: session.execution_id, status: "starting" };
      })
      .mockRejectedValueOnce(new Error("Connection lost"));
    vitest.vi
      .spyOn(RunnerApi.prototype, "stop")
      .mockImplementation(async () => {
        session = { ...session, status: "cancelled", active: false };
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
    view.unmount();
  },
);
