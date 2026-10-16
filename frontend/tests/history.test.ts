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
    const ids = params.step_ids as string[] | undefined;
    const items = [...this.items]
      .filter((entry) => !ids || ids.includes(`step-${entry.position}`))
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

test("history bootstraps once and merges positioned pushes without reads", async () => {
  const connection = new Connection();
  connection.items = Array.from({ length: 1200 }, (_, index) =>
    item(index + 1),
  );
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  const initial = await sync.refresh(() => true);
  assert.equal(initial.items.length, 1200);
  assert.equal(connection.queries.length, 3);
  connection.queries = [];
  const pushed = item(1201);
  const updated = sync.apply({
    kind: "step.completed",
    execution_id: "execution",
    ...pushed,
  })!;
  assert.equal(updated.items.length, 1201);
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

test("reconnect backfills missed revisions and new steps, preserving unchanged identities", async () => {
  const connection = new Connection();
  connection.items = Array.from({ length: 600 }, (_, index) => item(index + 1));
  const sync = new HistorySync(new RunnerApi(connection), "execution");
  const initial = await sync.refresh(() => true);
  connection.items[0] = item(1, "failed", 2);
  connection.fail = true;
  await assert.rejects(sync.refresh(() => true));
  connection.fail = false;
  const updated = await sync.refresh(() => true);
  assert.notEqual(updated.items[0], initial.items[0]);
  assert.equal(updated.items[1], initial.items[1]);
  connection.items[1] = item(2, "failed", 2);
  connection.items.push(item(601));
  const reconciled = await sync.refresh(() => true);
  assert.notEqual(reconciled.items[1], updated.items[1]);
  assert.equal(reconciled.items[2], updated.items[2]);
  assert.equal(reconciled.items.length, 601);
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
    kind: "step.updated",
    execution_id: "execution",
    ...item(1, "streaming", 2),
  });
  finish({ items: [item(1)], next_cursor: null });
  assert.equal((await loading).items[0].step, sync.snapshot.items[0].step);
  assert.equal((sync.snapshot.items[0].step as Params).revision, 2);
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
