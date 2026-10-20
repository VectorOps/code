import assert from "node:assert/strict";
import { test } from "vitest";
import { HistorySync } from "../src/communication/history-sync";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import { stepView } from "../src/models/steps";
import type { HistoryItem } from "../src/models/runner";
import type { Params } from "../src/models/json";

function item(
  position: number,
  state = "completed",
  revision = 1,
): HistoryItem {
  return {
    position,
    step: {
      id: `step-${position}`,
      revision,
      workflow_id: "root",
      operation: { kind: "llm.generate" },
      outcome: { kind: state, result: {} },
    },
  };
}

class Connection extends RunnerConnection {
  items: HistoryItem[] = [];
  queries: Params[] = [];
  fail = false;

  constructor() {
    super("ws://localhost", "test");
  }

  override async request<T>(_method: string, params: Params = {}): Promise<T> {
    this.queries.push(params);
    if (this.fail) throw new Error("Disconnected");
    const before = params.before_step_id
      ? this.items.find(
          (entry) => (entry.step as Params).id === params.before_step_id,
        )?.position
      : undefined;
    const items = [...this.items]
      .filter((entry) => before === undefined || entry.position < before)
      .sort((a, b) => b.position - a.position);
    const offset = Number(params.cursor ?? 0);
    const limit = Number(params.limit);
    return {
      items: items.slice(offset, offset + limit).reverse(),
      next_cursor:
        items.length > offset + limit ? String(offset + limit) : null,
    } as T;
  }
}

test("history bootstraps a bounded window and merges positioned pushes without reads", async () => {
  const connection = new Connection();
  connection.items = Array.from({ length: 1200 }, (_, index) =>
    item(index + 1),
  );
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  const initial = await sync.refresh(() => true);
  assert.equal(initial.items.length, 1000);
  assert.equal(initial.items[0].position, 201);
  assert.equal(initial.hasOlder, true);
  assert.equal(connection.queries.length, 2);
  assert.equal(connection.queries[0].limit, 500);
  assert.equal(connection.queries[1].limit, 500);
  assert.equal(connection.queries[1].cursor, "500");
  connection.queries = [];
  const pushed = item(1201);
  const updated = sync.apply({
    kind: "step.completed",
    execution_id: "execution",
    ...pushed,
  })!;
  assert.equal(updated.items.length, 1001);
  assert.equal(updated.items[0], initial.items[0]);
  assert.equal(connection.queries.length, 0);
  assert.equal(
    sync.apply({
      kind: "step.completed",
      execution_id: "execution",
      ...pushed,
    }),
    updated,
  );
});

test("initial loading requests only the remaining window and stops when inactive", async () => {
  const connection = new Connection();
  connection.items = Array.from({ length: 1200 }, (_, index) =>
    item(index + 1),
  );
  const sync = new HistorySync(new RunnerApi(connection), "execution", {
    initial: 750,
    maximum: 2000,
    target: 750,
    older: 200,
  });
  const loaded = await sync.refresh(() => true);
  assert.deepEqual(
    connection.queries.map((query) => query.limit),
    [500, 250],
  );
  assert.equal(loaded.items.length, 750);
  assert.equal(loaded.items[0].position, 451);
  assert.equal(loaded.items.at(-1)?.position, 1200);
  connection.queries = [];
  await sync.refresh(() => false);
  assert.equal(connection.queries.length, 1);
  assert.equal(sync.snapshot, loaded);
});

test("a failed second initial page does not publish a partial window", async () => {
  class FailingConnection extends Connection {
    override async request<T>(method: string, params: Params = {}): Promise<T> {
      if (params.cursor) throw new Error("Disconnected");
      return super.request<T>(method, params);
    }
  }
  const connection = new FailingConnection();
  connection.items = Array.from({ length: 1200 }, (_, index) =>
    item(index + 1),
  );
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  await assert.rejects(sync.refresh(() => true));
  assert.equal(sync.snapshot.items.length, 0);
});

test("reconnect refreshes only the tail, preserving unchanged identities", async () => {
  const connection = new Connection();
  connection.items = Array.from({ length: 1100 }, (_, index) =>
    item(index + 1),
  );
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  const initial = await sync.refresh(() => true);
  connection.items[100] = item(101, "failed", 2);
  connection.fail = true;
  await assert.rejects(sync.refresh(() => true));
  connection.fail = false;
  const updated = await sync.refresh(() => true);
  assert.notEqual(updated.items[0], initial.items[0]);
  assert.equal(updated.items[1], initial.items[1]);
  connection.items[101] = item(102, "failed", 2);
  connection.items.push(item(1101));
  const reconciled = await sync.refresh(() => true);
  assert.notEqual(reconciled.items[0], updated.items[1]);
  assert.equal(reconciled.items[1], updated.items[2]);
  assert.equal(reconciled.items.length, 1000);
});

test("inactive loads are discarded and step revisions reject stale replacement snapshots", async () => {
  const connection = new Connection();
  connection.items = [item(1, "streaming")];
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  assert.equal((await sync.refresh(() => false)).items.length, 0);
  const initial = await sync.refresh(() => true);
  connection.queries = [];
  const event = {
    kind: "step.completed",
    execution_id: "execution",
    ...item(1, "completed", 3),
  };
  const updated = sync.apply(event)!;
  assert.notEqual(updated.items[0], initial.items[0]);
  assert.equal(sync.apply({ ...event, ...item(1, "streaming", 2) }), updated);
  assert.equal(sync.apply(event), updated);
  assert.equal(sync.apply({ ...event, execution_id: "other" }), undefined);
  assert.equal(sync.apply({ ...event, position: null }), undefined);
  assert.equal(connection.queries.length, 0);
  assert.equal(await sync.refresh(() => true), updated);
});

test("pushes received during history backfill take precedence over its response", async () => {
  const connection = new Connection();
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  let finish = (_value: unknown) => {};
  connection.request = <T>() =>
    new Promise<T>((resolve) => (finish = (value) => resolve(value as T)));
  const loading = sync.refresh(() => true);
  sync.apply({
    kind: "step.created",
    execution_id: "execution",
    ...item(1, "streaming"),
  });
  sync.apply({
    kind: "step.updated",
    execution_id: "execution",
    ...item(1, "streaming", 2),
  });
  finish({ items: [item(1)], next_cursor: null });
  assert.equal((await loading).items[0].step, sync.snapshot.items[0].step);
  assert.equal((sync.snapshot.items[0].step as Params).revision, 2);
});

test("unknown updates and evicted additions are ignored, but new terminal steps are admitted", async () => {
  const connection = new Connection();
  connection.items = Array.from({ length: 1500 }, (_, index) =>
    item(index + 1),
  );
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  const initial = await sync.refresh(() => true);
  for (const entry of [
    item(1, "failed", 2),
    item(1501, "streaming", 2),
    item(1),
  ])
    assert.equal(
      sync.apply({
        kind: "step.completed",
        execution_id: "execution",
        ...entry,
      }),
      initial,
    );
  for (const position of [1502, 1501])
    sync.apply({
      kind: "step.failed",
      execution_id: "execution",
      ...item(position, "failed"),
    });
  assert.deepEqual(
    sync.snapshot.items.slice(-2).map((entry) => entry.position),
    [1501, 1502],
  );
});

test("trimming respects the viewport, review overrides, and configurable thresholds", async () => {
  const connection = new Connection();
  connection.items = Array.from({ length: 20 }, (_, index) => item(index + 1));
  const sync = new HistorySync(new RunnerApi(connection), "execution", {
    initial: 4,
    maximum: 6,
    target: 3,
    older: 2,
  });
  await sync.refresh(() => true);
  for (let position = 21; position <= 24; position++)
    sync.apply({
      kind: "step.completed",
      execution_id: "execution",
      ...item(position),
    });
  sync.follow(false);
  const reviewing = sync.snapshot;
  assert.equal(sync.trim(Infinity), reviewing);
  sync.follow(true);
  assert.deepEqual(
    sync.trim(20).items.map((entry) => entry.position),
    [20, 21, 22, 23, 24],
  );
  assert.deepEqual(
    sync.trim(Infinity).items.map((entry) => entry.position),
    [22, 23, 24],
  );
  const trimmed = sync.snapshot;
  assert.equal(
    sync.apply({
      kind: "step.created",
      execution_id: "execution",
      ...item(21),
    }),
    trimmed,
  );
  assert.equal(trimmed.hasOlder, true);
});

test("older pages use the oldest retained ID and remain until review ends", async () => {
  const connection = new Connection();
  connection.items = Array.from({ length: 1400 }, (_, index) =>
    item(index + 1),
  );
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  await sync.refresh(() => true);
  await sync.loadOlder(() => true);
  assert.equal(sync.snapshot.items.length, 1200);
  assert.deepEqual(connection.queries[2], {
    execution_id: "execution",
    selection: "tail",
    limit: 200,
    before_step_id: "step-401",
  });
  assert.equal(sync.trim(Infinity).items.length, 1200);
  sync.follow(true);
  assert.equal(sync.trim(Infinity).items.length, 1000);
  await sync.loadOlder(() => true);
  assert.equal(connection.queries[3].before_step_id, "step-401");
  assert.equal(sync.snapshot.items.length, 1200);
});

test("older requests are single-flight and invalidated responses cannot repopulate history", async () => {
  const connection = new Connection();
  connection.items = Array.from({ length: 1200 }, (_, index) =>
    item(index + 1),
  );
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  await sync.refresh(() => true);
  let finish = (_value: unknown) => {};
  let reads = 0;
  connection.request = <T>() => {
    reads++;
    return new Promise<T>(
      (resolve) => (finish = (value) => resolve(value as T)),
    );
  };
  const loading = sync.loadOlder(() => true);
  assert.equal(sync.snapshot.loadingOlder, true);
  await sync.loadOlder(() => true);
  assert.equal(reads, 1);
  sync.invalidate();
  finish({ items: [item(1)], next_cursor: null });
  await loading;
  assert.equal(sync.snapshot.items.length, 1000);
  assert.equal(sync.snapshot.loadingOlder, false);
});

test("failed older loads retain their boundary and can be retried", async () => {
  const connection = new Connection();
  connection.items = Array.from({ length: 1200 }, (_, index) =>
    item(index + 1),
  );
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  const initial = await sync.refresh(() => true);
  connection.fail = true;
  await assert.rejects(sync.loadOlder(() => true));
  assert.equal(sync.snapshot.loadingOlder, false);
  assert.equal(sync.snapshot.items, initial.items);
  assert.equal(sync.snapshot.hasOlder, true);
  connection.fail = false;
  await sync.loadOlder(() => true);
  assert.equal(sync.snapshot.items.length, 1200);
  assert.equal(sync.snapshot.hasOlder, false);
});

test("long live histories remain bounded and changed limits apply without reloading", () => {
  const connection = new Connection();
  const sync = new HistorySync(new RunnerApi(connection), "execution", {
    initial: 20,
    maximum: 40,
    target: 20,
    older: 10,
  });
  for (let position = 1; position <= 10000; position++) {
    sync.apply({
      kind: "step.completed",
      execution_id: "execution",
      ...item(position),
    });
    assert.ok(sync.trim(Infinity).items.length <= 40);
  }
  sync.configure({ initial: 10, maximum: 10, target: 5, older: 5 });
  assert.equal(sync.trim(Infinity).items.length, 5);
  assert.equal(sync.snapshot.items.at(-1)?.position, 10000);
  assert.equal(connection.queries.length, 0);
});

test("tool lookup respects workflow, name, position, and nearest generation", () => {
  const generation = (position: number, workflow: string, name: string) => ({
    position,
    step: {
      id: `generation-${position}`,
      workflow_id: workflow,
      operation: { kind: "llm.generate" },
      outcome: {
        kind: "completed",
        result: {
          content: [
            { type: "tool_call", id: "call", name, arguments: position },
          ],
        },
      },
    },
  });
  const tool: HistoryItem = {
    position: 5,
    step: {
      id: "tool",
      workflow_id: "root",
      operation: { kind: "llm.tool" },
      outcome: {
        kind: "completed",
        result: { tool_call_id: "call", tool_name: "read" },
      },
    },
  };
  const history = [
    generation(6, "root", "read"),
    tool,
    generation(4, "other", "read"),
    generation(3, "root", "write"),
    generation(2, "root", "read"),
    generation(1, "root", "read"),
  ];
  const view = stepView(tool, history);
  assert.deepEqual(view.toolCall, {
    type: "tool_call",
    id: "call",
    name: "read",
    arguments: 2,
  });
  assert.equal(stepView(tool, [...history]), view);
});
