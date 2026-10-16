import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import { SessionSync } from "../src/communication/session-sync";
import type { Session } from "../src/models/runner";
import type { Params } from "../src/models/json";

const original: Session = {
  execution_id: "existing",
  reference: "chat:workflow",
  name: "Old title",
  status: "running",
  active: true,
  pending_inputs: 0,
};

function setup() {
  const api = new RunnerApi(new RunnerConnection("ws://localhost", "test"));
  const subscribe = vitest.vi.spyOn(api, "subscribe").mockResolvedValue({});
  const sessions = vitest.vi
    .spyOn(api, "sessions")
    .mockResolvedValue([original]);
  const publish = vitest.vi.fn<(sessions: Session[]) => void>();
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
  "subscribes before fetching and overlays pushes received during either request",
  async () => {
    const { sync, subscribe, sessions, publish, push } = setup();
    let subscribed = (_value: object) => {};
    let fetched = (_value: Session[]) => {};
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
    fetched([original]);
    await vitest.expect(loading).resolves.toBe(true);
    vitest.expect(publish.mock.lastCall![0]).toEqual([titled, created]);
    vitest.expect(sync.get("existing")).toBe(titled);
  },
);

vitest.test(
  "replaces complete summaries in place and appends new executions without requests",
  async () => {
    const { sync, sessions, publish, push } = setup();
    const second = { ...original, execution_id: "second" };
    sessions.mockResolvedValue([original, second]);
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
      .expect(publish.mock.lastCall![0])
      .toEqual([replaced, second, created]);
    vitest.expect(sessions).toHaveBeenCalledTimes(1);
  },
);

vitest.test(
  "reconnect discards stale in-flight responses and reconciles removed executions",
  async () => {
    const { sync, sessions, publish, push } = setup();
    await sync.synchronize(() => true);
    let finish = (_value: Session[]) => {};
    sessions.mockReturnValueOnce(new Promise((resolve) => (finish = resolve)));
    const stale = sync.synchronize(() => true);
    await Promise.resolve();
    sync.invalidate();
    const current = { ...original, execution_id: "current" };
    sessions.mockResolvedValue([current]);
    await sync.synchronize(() => true);
    const updated = { ...current, name: "Latest title" };
    push(updated);
    finish([original]);
    await vitest.expect(stale).resolves.toBe(false);
    vitest.expect(publish.mock.lastCall![0]).toEqual([updated]);
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
