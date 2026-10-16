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
    history: vi
      .spyOn(api, "history")
      .mockResolvedValue({ items: [], next_cursor: null }),
    pending: vi.spyOn(api, "pending").mockResolvedValue([]),
    inboxes: vi.spyOn(api, "inboxes").mockResolvedValue([]),
    messages: vi.spyOn(api, "inboxMessages").mockResolvedValue([]),
  };
  let view: ExecutionView = {
    history: { items: [], next_cursor: null },
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
  for (const read of Object.values(reads)) read.mockClear();

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
