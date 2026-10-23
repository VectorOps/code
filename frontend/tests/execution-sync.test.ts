import { expect, test, vi } from "vitest";
import {
  ExecutionSync,
  type ExecutionView,
} from "../src/communication/execution-sync";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import type * as models from "../src/models/runner";
import { stepView, toolDescription } from "../src/models/steps";

const execution: models.Execution = {
  id: "execution",
  root_workflow_id: "root",
  status: "running",
  checkpoint: { heads: {} },
};
const address: models.InboxAddress = {
  execution_id: "execution",
  workflow_id: "root",
  conversation_id: "steer",
};
const request = {
  execution_id: "execution",
  request_id: "approval",
  workflow_id: "root",
  request_type: "tool.approval.v1",
  channel: "tools.approval",
  payload: { call: { name: "read_file", arguments: { path: "README.md" } } },
  response_type: "json" as const,
  status: "pending",
};
const message: models.InboxMessage = {
  id: "message",
  created_at: "2026-10-16T12:00:00Z",
  payload: { content: [{ type: "text", text: "Change direction" }] },
};

function setup() {
  const api = new RunnerApi(new RunnerConnection("ws://localhost", "test"));
  const reads = {
    execution: vi.spyOn(api, "execution").mockResolvedValue(execution),
    workflow: vi
      .spyOn(api, "workflowState")
      .mockImplementation(async (_execution, id) => ({
        id,
        definition_ref: `test:${id}`,
        presentation: { short_name: "Agent", options: ["llm_only"] },
      })),
    history: vi
      .spyOn(api, "history")
      .mockResolvedValue({ items: [], next_cursor: null }),
    pending: vi.spyOn(api, "pending").mockResolvedValue([]),
    inboxes: vi.spyOn(api, "inboxes").mockResolvedValue([]),
    messages: vi.spyOn(api, "inboxMessages").mockResolvedValue([]),
  };
  let view: ExecutionView = {
    history: { items: [], hasOlder: false, loadingOlder: false },
    pending: [],
  };
  const sync = new ExecutionSync(api, "execution", (value) => (view = value));
  const apply = (event: Parameters<ExecutionSync["apply"]>[0]) =>
    sync.apply({ execution_id: "execution", ...event });
  return { sync, reads, apply, view: () => view };
}

test("live tool, checkpoint, approval and steering pushes update the view without reads", async () => {
  const { sync, reads, apply, view } = setup();
  await sync.synchronize(() => true);
  expect(view().workflowStates?.root.presentation?.options).toEqual([
    "llm_only",
  ]);
  for (const read of Object.values(reads)) read.mockClear();
  apply({
    kind: "workflow.updated",
    workflow: {
      id: "root",
      definition_ref: "test:root",
      presentation: { short_name: "Updated", bgcolor: "#123456", options: [] },
    },
  });
  expect(view().workflowStates?.root.presentation?.short_name).toBe("Updated");

  apply({ kind: "mailbox.requested", request });
  expect(view().pending).toEqual([request]);
  apply({
    kind: "step.completed",
    position: 2,
    step: {
      id: "tool",
      revision: 1,
      workflow_id: "root",
      previous_step_id: "generation",
      operation: { kind: "llm.tool", input: {} },
      outcome: {
        kind: "completed",
        result: { tool_call_id: "call", tool_name: "read_file", content: [] },
      },
    },
  });
  apply({
    kind: "step.completed",
    position: 1,
    step: {
      id: "generation",
      revision: 1,
      workflow_id: "root",
      operation: { kind: "llm.generate" },
      outcome: {
        kind: "completed",
        result: {
          content: [
            {
              type: "tool_call",
              id: "call",
              name: "read_file",
              arguments: { path: "README.md" },
            },
          ],
        },
      },
    },
  });
  const history = view().history.items;
  expect(history.map((item) => item.position)).toEqual([1, 2]);
  expect(toolDescription(stepView(history[1], history))).toMatchObject({
    name: "read_file",
    arguments: { path: "README.md" },
    state: "completed",
  });
  const branched = {
    ...execution,
    checkpoint: { heads: { root: "tool", child: "child-step" } },
  };
  apply({ kind: "execution.state_updated", execution: branched });
  expect(view().execution).toEqual(branched);
  apply({ kind: "execution.branch_applied", input_step_id: "tool" });
  apply({
    kind: "mailbox.answered",
    request: { ...request, status: "answered" },
  });
  expect(view().pending).toEqual([]);
  apply({ kind: "mailbox.requested", request });
  apply({ kind: "mailbox.cancelled", request_ids: ["approval"] });
  expect(view().pending).toEqual([]);
  apply({
    kind: "mailbox.state_updated",
    address,
    status: "open",
    queued_messages: 0,
  });
  expect(view().steering).toEqual({ address, messages: [], count: 0 });
  apply({
    kind: "mailbox.enqueued",
    address,
    status: "open",
    queued_messages: 1,
    message_id: message.id,
    message,
  });
  expect(view().steering).toEqual({ address, messages: [message], count: 1 });
  apply({
    kind: "mailbox.removed",
    address,
    status: "open",
    queued_messages: 0,
    message_id: message.id,
  });
  expect(view().steering?.messages).toEqual([]);
  apply({
    kind: "mailbox.state_updated",
    address,
    status: "closed",
    queued_messages: 0,
  });
  expect(view().steering).toBeUndefined();
  const previous = view();
  apply({
    kind: "execution.state_updated",
    execution_id: "other",
    execution: { ...execution, id: "other" },
  });
  expect(view()).toBe(previous);
  for (const read of Object.values(reads)) expect(read).not.toHaveBeenCalled();
});

test("pushes during bootstrap win over older execution, input, history and inbox responses", async () => {
  const { sync, reads, apply, view } = setup();
  reads.pending.mockResolvedValue([request]);
  reads.inboxes.mockResolvedValue([
    { address, status: "open", queued_messages: 1 },
  ]);
  let finish = (_messages: models.InboxMessage[]) => {};
  reads.messages.mockReturnValue(new Promise((resolve) => (finish = resolve)));
  const loading = sync.synchronize(() => true);
  await vi.waitFor(() => expect(reads.messages).toHaveBeenCalledTimes(1));
  apply({
    kind: "workflow.updated",
    workflow: {
      id: "root",
      definition_ref: "test:root",
      presentation: { options: ["prefix_messages"] },
    },
  });
  const updated = {
    ...execution,
    status: "paused",
    checkpoint: { heads: { root: "stream" } },
  };
  apply({ kind: "execution.state_updated", execution: updated });
  apply({ kind: "mailbox.cancelled", request_ids: [request.request_id] });
  apply({
    kind: "mailbox.removed",
    address,
    status: "open",
    queued_messages: 0,
    message_id: "stale",
  });
  apply({
    kind: "mailbox.enqueued",
    address,
    status: "open",
    queued_messages: 1,
    message_id: message.id,
    message,
  });
  const step = {
    id: "stream",
    workflow_id: "root",
    operation: { kind: "llm.generate" },
  };
  apply({
    kind: "step.created",
    position: 1,
    step: { ...step, revision: 1, outcome: { kind: "pending" } },
  });
  apply({
    kind: "step.updated",
    position: 1,
    step: {
      ...step,
      revision: 2,
      outcome: { kind: "streaming", state: { old: "state" } },
    },
  });
  apply({
    kind: "step.updated",
    position: 1,
    step: {
      ...step,
      revision: 3,
      outcome: { kind: "streaming", state: { replacement: "state" } },
    },
  });
  finish([{ ...message, id: "stale" }]);
  await loading;
  expect(view().workflowStates?.root.presentation?.options).toEqual([
    "prefix_messages",
  ]);
  expect(view().execution).toEqual(updated);
  expect(view().pending).toEqual([]);
  expect(view().steering).toEqual({ address, messages: [message], count: 1 });
  expect(view().history.items).toEqual([
    {
      position: 1,
      step: {
        ...step,
        revision: 3,
        outcome: { kind: "streaming", state: { replacement: "state" } },
      },
    },
  ]);
});

test("disconnect discards in-flight bootstrap responses and reconnect reloads the baseline", async () => {
  const { sync, reads, apply, view } = setup();
  let finish = (_execution: models.Execution) => {};
  reads.execution.mockReturnValueOnce(
    new Promise((resolve) => (finish = resolve)),
  );
  const stale = sync.synchronize(() => true);
  apply({ kind: "execution.state_updated", execution });
  sync.invalidate();
  const latest = {
    ...execution,
    status: "cancelled",
    checkpoint: { heads: { root: "new" } },
  };
  reads.execution.mockResolvedValue(latest);
  await sync.synchronize(() => true);
  finish(execution);
  await stale;
  expect(view().execution).toEqual(latest);
  expect(reads.execution).toHaveBeenCalledTimes(2);
  for (const read of [reads.history, reads.pending])
    expect(read).toHaveBeenCalledTimes(1);
  expect(reads.inboxes).not.toHaveBeenCalled();
});

test("saved presentation is loaded for checkpoint, history, spawn and pending workflows", async () => {
  const { sync, reads, view } = setup();
  reads.execution.mockResolvedValue({
    ...execution,
    checkpoint: { heads: { checkpoint: "head" } },
  });
  reads.history.mockResolvedValue({
    items: [
      {
        position: 1,
        step: {
          id: "spawn",
          workflow_id: "history",
          operation: { kind: "workflow.spawn", child_workflow_id: "spawned" },
        },
      },
    ],
    next_cursor: null,
  });
  reads.pending.mockResolvedValue([{ ...request, workflow_id: "pending" }]);
  await sync.synchronize(() => true);
  expect(Object.keys(view().workflowStates ?? {}).sort()).toEqual([
    "checkpoint",
    "history",
    "pending",
    "root",
    "spawned",
  ]);
  expect(reads.workflow).toHaveBeenCalledTimes(5);
  expect(reads.workflow).toHaveBeenCalledWith("execution", "root");
});

test("loading older history fetches only missing workflow metadata", async () => {
  const { sync, reads, view } = setup();
  sync.configureHistory({ initial: 1, older: 1, target: 1, maximum: 2 });
  reads.history.mockResolvedValueOnce({
    items: [
      {
        position: 2,
        step: {
          id: "latest",
          workflow_id: "root",
          operation: { kind: "llm.generate" },
        },
      },
    ],
    next_cursor: "older",
  });
  await sync.synchronize(() => true);
  reads.workflow.mockClear();
  reads.history.mockResolvedValueOnce({
    items: [
      {
        position: 1,
        step: {
          id: "old",
          workflow_id: "old-child",
          operation: { kind: "llm.generate" },
        },
      },
    ],
    next_cursor: null,
  });
  await sync.loadOlder(() => true);
  expect(view().workflowStates?.["old-child"].presentation?.options).toEqual([
    "llm_only",
  ]);
  expect(reads.workflow).toHaveBeenCalledExactlyOnceWith(
    "execution",
    "old-child",
  );
});

test("workflow metadata loads backwards in sequential batches below the RPC cap", async () => {
  const { sync, reads, view } = setup();
  const ids = [
    "root",
    ...Array.from({ length: 464 }, (_, index) => `workflow-${index}`),
  ];
  reads.execution.mockResolvedValue({
    ...execution,
    checkpoint: {
      heads: Object.fromEntries(ids.map((id) => [id, `${id}-head`])),
    },
  });
  let outstanding = 0;
  let peak = 0;
  const releases: (() => void)[] = [];
  reads.workflow.mockImplementation(
    (_execution, id) =>
      new Promise((resolve) => {
        outstanding++;
        peak = Math.max(peak, outstanding);
        releases.push(() => {
          outstanding--;
          resolve({ id, definition_ref: `test:${id}` });
        });
      }),
  );
  const loading = sync.synchronize(() => true);
  const newestFirst = [...ids].reverse();
  for (let offset = 0; offset < ids.length; offset += 32) {
    const end = Math.min(offset + 32, ids.length);
    await vi.waitFor(() => expect(reads.workflow).toHaveBeenCalledTimes(end));
    expect(reads.workflow.mock.calls.map((call) => call[1])).toEqual(
      newestFirst.slice(0, end),
    );
    expect(outstanding).toBe(end - offset);
    const batch = releases.splice(0);
    for (const release of batch.slice(1)) release();
    await Promise.resolve();
    expect(reads.workflow).toHaveBeenCalledTimes(end);
    batch[0]();
  }
  await loading;
  expect(peak).toBe(32);
  expect(outstanding).toBe(0);
  expect(Object.keys(view().workflowStates ?? {})).toEqual(newestFirst);
});

test("newest history references take priority and workflow IDs are deduplicated", async () => {
  const { sync, reads } = setup();
  reads.execution.mockResolvedValue({
    ...execution,
    checkpoint: { heads: { root: "head", older: "old", newer: "new" } },
  });
  reads.history.mockResolvedValue({
    items: [
      {
        position: 1,
        step: { id: "old", workflow_id: "older" },
      },
      {
        position: 2,
        step: {
          id: "new",
          workflow_id: "newer",
          operation: { kind: "workflow.spawn", child_workflow_id: "child" },
        },
      },
    ],
    next_cursor: null,
  });
  reads.pending.mockResolvedValue([
    { ...request, workflow_id: "newer" },
    { ...request, workflow_id: "pending" },
  ]);
  await sync.synchronize(() => true);
  expect(reads.workflow.mock.calls.map((call) => call[1])).toEqual([
    "child",
    "newer",
    "older",
    "pending",
    "root",
  ]);
});

test.each(["invalidated", "inactive"])(
  "%s workflow metadata loading does not schedule another batch",
  async (reason) => {
    const { sync, reads, view } = setup();
    reads.execution.mockResolvedValue({
      ...execution,
      checkpoint: {
        heads: Object.fromEntries(
          Array.from({ length: 65 }, (_, index) => [
            `workflow-${index}`,
            "head",
          ]),
        ),
      },
    });
    let finish = () => {};
    const batch = new Promise<void>((resolve) => (finish = resolve));
    reads.workflow.mockImplementation(async (_execution, id) => {
      await batch;
      return { id, definition_ref: `test:${id}` };
    });
    let active = true;
    const loading = sync.synchronize(() => active);
    await vi.waitFor(() => expect(reads.workflow).toHaveBeenCalledTimes(32));
    if (reason === "invalidated") sync.invalidate();
    else active = false;
    finish();
    await loading;
    expect(reads.workflow).toHaveBeenCalledTimes(32);
    expect(view().execution).toBeUndefined();
    expect(view().workflowStates).toBeUndefined();
  },
);

test("workflow state uses the persisted workflow RPC", async () => {
  const connection = new RunnerConnection("ws://localhost", "test");
  const request = vi
    .spyOn(connection, "request")
    .mockResolvedValue({ id: "child", definition_ref: "test:child" });
  const api = new RunnerApi(connection);
  expect(await api.workflowState("execution", "child")).toEqual({
    id: "child",
    definition_ref: "test:child",
  });
  expect(request).toHaveBeenCalledWith("workflows.state", {
    execution_id: "execution",
    workflow_id: "child",
  });
});
