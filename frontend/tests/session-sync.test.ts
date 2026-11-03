import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import {
  SessionSync,
  type SessionList,
} from "../src/communication/session-sync";
import type { Session, SessionPage } from "../src/models/runner";
import type { Params } from "../src/models/json";

const original: Session = {
  execution_id: "existing",
  reference: "chat:workflow",
  name: "Old title",
  status: "running",
  active: true,
  pending_inputs: 0,
};

function page(
  executions: Session[],
  next_after: string | null = null,
): SessionPage {
  return { executions, next_after };
}

function setup() {
  const api = new RunnerApi(new RunnerConnection("ws://localhost", "test"));
  const subscribe = vitest.vi.spyOn(api, "subscribe").mockResolvedValue({});
  const sessions = vitest.vi
    .spyOn(api, "sessions")
    .mockResolvedValue(page([original]));
  const publish = vitest.vi.fn<(view: SessionList) => void>();
  const sync = new SessionSync(api, publish);
  const push = (summary: Session) =>
    sync.apply({
      kind: "execution.updated",
      execution_id: summary.execution_id,
      summary,
    });
  return { sync, subscribe, sessions, publish, push };
}

vitest.afterEach(() => vitest.vi.restoreAllMocks());

vitest.test(
  "preserves worktree associations in snapshots and live updates",
  async () => {
    const { sync, sessions, publish, push } = setup();
    const summary = { ...original, worktree_id: "tree-1" };
    sessions.mockResolvedValue(page([summary]));
    await sync.synchronize(() => true);
    vitest.expect(sync.get("existing")?.worktree_id).toBe("tree-1");
    const updated = { ...summary, pending_inputs: 1 };
    push(updated);
    vitest.expect(publish.mock.lastCall![0].sessions).toEqual([updated]);
  },
);

vitest.test(
  "preserves generated descriptions from snapshots and live updates",
  async () => {
    const { sync, sessions, publish, push } = setup();
    const summary = {
      ...original,
      name: "Coding agent",
      description: "Investigate session titles",
    };
    sessions.mockResolvedValue(page([summary]));
    await sync.synchronize(() => true);
    vitest.expect(sync.get("existing")).toEqual(summary);
    const updated = { ...summary, description: "Fix session descriptions" };
    push(updated);
    vitest.expect(publish.mock.lastCall![0].sessions).toEqual([updated]);
    vitest.expect(sync.get("existing")?.name).toBe("Coding agent");
  },
);

vitest.test.each(["synchronize", "loadMore"] as const)(
  "deletion survives stale %s responses and later updates",
  async (method) => {
    const { sync, sessions, publish, push } = setup();
    sessions.mockResolvedValueOnce(page([original], "existing"));
    await sync.synchronize(() => true);
    let finish = (_value: SessionPage) => {};
    sessions.mockReturnValueOnce(new Promise((resolve) => (finish = resolve)));
    const loading = sync[method](() => true);
    await Promise.resolve();
    sync.remove("existing");
    vitest.expect(sync.get("existing")).toBeUndefined();
    finish(page([original]));
    await loading;
    push(original);
    vitest.expect(publish.mock.lastCall![0].sessions).toEqual([]);
  },
);

vitest.test(
  "subscribes before fetching and overlays pushes received during either request",
  async () => {
    const { sync, subscribe, sessions, publish, push } = setup();
    let subscribed = (_value: object) => {};
    let fetched = (_value: SessionPage) => {};
    subscribe.mockReturnValue(new Promise((resolve) => (subscribed = resolve)));
    sessions.mockReturnValue(new Promise((resolve) => (fetched = resolve)));
    const loading = sync.synchronize(() => true);
    vitest.expect(subscribe).toHaveBeenCalledTimes(1);
    vitest.expect(sessions).not.toHaveBeenCalled();
    const titled = { ...original, name: "Pushed title" };
    push(titled);
    subscribed({});
    await Promise.resolve();
    vitest.expect(sessions).toHaveBeenCalledTimes(1);
    const created = { ...original, execution_id: "new", name: "New run" };
    push(created);
    fetched(page([original]));
    await vitest.expect(loading).resolves.toBe(true);
    vitest
      .expect(publish.mock.lastCall![0].sessions)
      .toEqual([created, titled]);
    vitest.expect(sync.get("existing")).toBe(titled);
  },
);

vitest.test(
  "prepends new executions and replaces complete summaries in place without requests",
  async () => {
    const { sync, sessions, publish, push } = setup();
    const second = { ...original, execution_id: "second" };
    sessions.mockResolvedValue(page([original, second]));
    await sync.synchronize(() => true);
    const withUsage: Session = {
      ...original,
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
    push(withUsage);
    const replaced = {
      ...original,
      name: null,
      usage: null,
      status: "completed",
      active: false,
    };
    vitest.expect(push(replaced)?.previous).toBe(withUsage);
    const created = { ...original, execution_id: "third" };
    push(created);
    vitest
      .expect(publish.mock.lastCall![0].sessions)
      .toEqual([created, replaced, second]);
    const updatedSecond = { ...second, name: "Updated title" };
    push(updatedSecond);
    vitest
      .expect(publish.mock.lastCall![0].sessions)
      .toEqual([created, replaced, updatedSecond]);
    const newest = { ...original, execution_id: "fourth" };
    push(newest);
    vitest
      .expect(publish.mock.lastCall![0].sessions)
      .toEqual([newest, created, replaced, updatedSecond]);
    vitest.expect(sessions).toHaveBeenCalledTimes(1);
  },
);

vitest.test(
  "reconnect discards stale in-flight responses and reconciles removed executions",
  async () => {
    const { sync, sessions, publish, push } = setup();
    await sync.synchronize(() => true);
    let finish = (_value: SessionPage) => {};
    sessions.mockReturnValueOnce(new Promise((resolve) => (finish = resolve)));
    const stale = sync.synchronize(() => true);
    await Promise.resolve();
    sync.invalidate();
    const current = { ...original, execution_id: "current" };
    sessions.mockResolvedValue(page([current]));
    await sync.synchronize(() => true);
    const updated = { ...current, name: "Latest title" };
    push(updated);
    finish(page([original]));
    await vitest.expect(stale).resolves.toBe(false);
    vitest.expect(publish.mock.lastCall![0].sessions).toEqual([updated]);
  },
);

vitest.test(
  "loads only one initial page and follows the cursor on demand",
  async () => {
    const { sync, sessions, publish } = setup();
    sessions.mockResolvedValueOnce(page([original], "existing"));
    await sync.synchronize(() => true);
    vitest.expect(sessions).toHaveBeenCalledExactlyOnceWith();
    vitest.expect(publish.mock.lastCall![0]).toEqual({
      sessions: [original],
      hasMore: true,
      loading: false,
    });
    const second = { ...original, execution_id: "second" };
    sessions.mockResolvedValueOnce(page([original, second]));
    await vitest.expect(sync.loadMore(() => true)).resolves.toBe(true);
    vitest.expect(sessions).toHaveBeenLastCalledWith("existing");
    vitest.expect(publish.mock.lastCall![0]).toEqual({
      sessions: [original, second],
      hasMore: false,
      loading: false,
    });
    await vitest.expect(sync.loadMore(() => true)).resolves.toBe(false);
    vitest.expect(sessions).toHaveBeenCalledTimes(2);
  },
);

vitest.test(
  "guards duplicate page loads and overlays pushes during the request",
  async () => {
    const { sync, sessions, publish, push } = setup();
    sessions.mockResolvedValueOnce(page([original], "existing"));
    await sync.synchronize(() => true);
    let finish = (_value: SessionPage) => {};
    sessions.mockReturnValueOnce(new Promise((resolve) => (finish = resolve)));
    const loading = sync.loadMore(() => true);
    const before = publish.mock.calls.at(-2)![0].sessions;
    vitest.expect(publish.mock.lastCall![0].sessions).toBe(before);
    vitest.expect(publish.mock.lastCall![0].loading).toBe(true);
    await vitest.expect(sync.loadMore(() => true)).resolves.toBe(false);
    const second = { ...original, execution_id: "second" };
    const updated = { ...second, name: "Pushed title" };
    push(updated);
    const created = { ...original, execution_id: "new" };
    push(created);
    finish(page([original, second]));
    await loading;
    vitest
      .expect(publish.mock.lastCall![0].sessions)
      .toEqual([created, updated, original]);
    vitest.expect(sessions).toHaveBeenCalledTimes(2);
  },
);

vitest.test(
  "reconnect invalidates an older page response and its cursor",
  async () => {
    const { sync, sessions, publish } = setup();
    sessions.mockResolvedValueOnce(page([original], "existing"));
    await sync.synchronize(() => true);
    let finish = (_value: SessionPage) => {};
    sessions.mockReturnValueOnce(new Promise((resolve) => (finish = resolve)));
    const loading = sync.loadMore(() => true);
    const current = { ...original, execution_id: "current" };
    sessions.mockResolvedValueOnce(page([current], "current"));
    await sync.synchronize(() => true);
    finish(page([{ ...original, execution_id: "obsolete" }]));
    await vitest.expect(loading).resolves.toBe(false);
    vitest.expect(publish.mock.lastCall![0]).toEqual({
      sessions: [current],
      hasMore: true,
      loading: false,
    });
    await sync.loadMore(() => true);
    vitest.expect(sessions).toHaveBeenLastCalledWith("current");
  },
);

vitest.test(
  "a failed page can be retried without losing sessions or the cursor",
  async () => {
    const { sync, sessions, publish } = setup();
    sessions.mockResolvedValueOnce(page([original], "existing"));
    await sync.synchronize(() => true);
    sessions.mockRejectedValueOnce(new Error("Unavailable"));
    await vitest
      .expect(sync.loadMore(() => true))
      .rejects.toThrow("Unavailable");
    vitest.expect(publish.mock.lastCall![0]).toEqual({
      sessions: [original],
      hasMore: true,
      loading: false,
    });
    await sync.loadMore(() => true);
    vitest.expect(sessions).toHaveBeenLastCalledWith("existing");
  },
);

vitest.test(
  "disconnect clears published loading state and ignores a stale page failure",
  async () => {
    const { sync, sessions, publish } = setup();
    sessions.mockResolvedValueOnce(page([original], "existing"));
    await sync.synchronize(() => true);
    let reject = (_reason: Error) => {};
    sessions.mockReturnValueOnce(
      new Promise((_resolve, fail) => (reject = fail)),
    );
    const loading = sync.loadMore(() => true);
    sync.invalidate();
    vitest
      .expect(publish.mock.lastCall![0])
      .toEqual({ sessions: [original], hasMore: false, loading: false });
    reject(new Error("Disconnected"));
    await vitest.expect(loading).resolves.toBe(false);
  },
);

vitest.test(
  "retains and refreshes a selected session outside the first page",
  async () => {
    const { sync, sessions, publish } = setup();
    await sync.synchronize(() => true);
    const current = { ...original, execution_id: "current" };
    sessions.mockResolvedValueOnce(page([current], "current"));
    await sync.synchronize(() => true, "existing");
    vitest.expect(sync.get("existing")).toBe(original);
    vitest.expect(publish.mock.lastCall![0].sessions).toContain(original);
    const updated = { ...original, name: "Refreshed title" };
    vitest.vi.spyOn(RunnerApi.prototype, "session").mockResolvedValue(updated);
    await sync.refresh("existing", () => true);
    vitest.expect(sync.get("existing")).toEqual(updated);
    vitest
      .expect(RunnerApi.prototype.session)
      .toHaveBeenCalledExactlyOnceWith("existing");
  },
);

vitest.test(
  "a pushed summary wins over an older selected-session refresh",
  async () => {
    const { sync, push } = setup();
    await sync.synchronize(() => true);
    let finish = (_value: Session) => {};
    vitest.vi
      .spyOn(RunnerApi.prototype, "session")
      .mockReturnValue(new Promise((resolve) => (finish = resolve)));
    const refreshing = sync.refresh("existing", () => true);
    const updated = { ...original, name: "Live title" };
    push(updated);
    finish(original);
    await refreshing;
    vitest.expect(sync.get("existing")).toBe(updated);
  },
);

vitest.test(
  "session requests expose one wire page without following next_after",
  async () => {
    const connection = new RunnerConnection("ws://localhost", "test");
    const request = vitest.vi
      .spyOn(connection, "request")
      .mockResolvedValue(page([original], "existing"));
    const api = new RunnerApi(connection);
    await vitest
      .expect(api.sessions())
      .resolves.toEqual(page([original], "existing"));
    vitest
      .expect(request)
      .toHaveBeenCalledExactlyOnceWith("executions.summaries", {
        after: null,
        limit: 100,
      });
    await api.sessions("existing");
    vitest.expect(request).toHaveBeenLastCalledWith("executions.summaries", {
      after: "existing",
      limit: 100,
    });
    request.mockResolvedValue(original);
    await api.session("existing");
    vitest.expect(request).toHaveBeenLastCalledWith("executions.summary", {
      execution_id: "existing",
    });
  },
);

vitest.test(
  "ignores unrelated workflow snapshots and malformed or mismatched summaries",
  () => {
    const { sync, sessions, publish } = setup();
    const events: Params[] = [
      {
        kind: "workflow.updated",
        execution_id: "existing",
        workflow: {
          id: "root",
          definition_ref: "chat:workflow",
          name: "Runtime title",
        },
      },
      { kind: "execution.updated", execution_id: "other", summary: original },
      { kind: "execution.updated", execution_id: "existing", summary: {} },
    ];
    for (const event of events)
      vitest.expect(sync.apply(event)).toBeUndefined();
    vitest.expect(publish).not.toHaveBeenCalled();
    vitest.expect(sessions).not.toHaveBeenCalled();
  },
);
