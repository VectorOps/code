import { object, text, type Json } from "./json";
import { messageText } from "./activity";
import type { HistoryItem, InputRequest, WorkflowState } from "./runner";
import { resolveDisplay } from "./workflow-display";
import { isChatRequest } from "./runner";
import { stepView, toolDescription, type StepView } from "./steps";

export type WorkflowNode = {
  id: string;
  label: string;
  workflow?: WorkflowState;
  parent?: WorkflowNode;
  spawn?: StepView;
  entries: (StepView | WorkflowNode)[];
  requests: InputRequest[];
  status: string;
};

type WorkflowInputs = { chat: InputRequest[]; decisions: InputRequest[] };
const inputViews = new WeakMap<InputRequest[], WorkflowInputs>();

export function workflowTree(
  history: HistoryItem[],
  pending: InputRequest[],
  states: Record<string, WorkflowState> = {},
) {
  const nodes = new Map<string, WorkflowNode>();
  const calls = new Map<string, { name: string; description: string }[]>();
  function node(id: string): WorkflowNode {
    let value = nodes.get(id);
    if (!value) {
      value = {
        id,
        workflow: states[id],
        label: "Workflow",
        entries: [],
        requests: [],
        status: "Unfinished",
      };
      nodes.set(id, value);
    }
    return value;
  }
  for (const item of [...history].sort((a, b) => a.position - b.position)) {
    const step = stepView(item, history);
    const owner = node(step.workflow);
    if (step.kind === "mailbox.drain" && step.state === "completed") {
      const messages = step.outcome.result;
      if (Array.isArray(messages))
        messages.forEach((message, index) => {
          const payload = object(message).content
            ? message
            : object(message).payload;
          if (!object(payload).content) return;
          owner.entries.push({
            ...step,
            id: `${step.id}:steering:${index}`,
            kind: "mailbox.input",
            name: "request_input",
            output: payload as Json,
            message: messageText(payload as Json),
            input: {
              request_type: "chat.message.v1",
              channel: "user.message",
              response_type: "text",
              steering: true,
              source_step_id: step.id,
            },
          });
        });
    }
    if (step.kind === "llm.tool") {
      const tool = toolDescription(step);
      if (["run_agent", "subworkflow"].includes(tool.name) && !tool.failed) {
        const args = object(tool.arguments);
        const name = text(args.reference).trim();
        if (name) {
          const pendingCalls = calls.get(owner.id) ?? [];
          pendingCalls.push({
            name,
            description: text(args.description).trim(),
          });
          calls.set(owner.id, pendingCalls);
        }
      }
    }
    const childId = text(step.operation.child_workflow_id);
    if (step.kind === "workflow.spawn" && childId) {
      const child = node(childId);
      let ancestor: WorkflowNode | undefined = owner;
      while (ancestor && ancestor !== child) ancestor = ancestor.parent;
      if (!ancestor && !child.parent) {
        const description = text(step.operation.description).trim();
        const pendingCalls = calls.get(owner.id) ?? [];
        const matches = pendingCalls.filter(
          (call) => call.description === description,
        );
        const call =
          matches.length === 1
            ? matches[0]
            : pendingCalls.length === 1
              ? pendingCalls[0]
              : undefined;
        child.label = `Agent: ${description || call?.name || "Unnamed"}`;
        if (call)
          calls.set(
            owner.id,
            pendingCalls.filter((candidate) => candidate !== call),
          );
        child.parent = owner;
        child.spawn = step;
        owner.entries.push(child);
        continue;
      }
    }
    if (step.kind !== "mailbox.drain") owner.entries.push(step);
    if (step.kind === "workflow.complete")
      owner.status = step.state === "failed" ? "Failed" : "Completed";
  }
  for (const request of pending)
    node(request.workflow_id || "").requests.push(request);
  for (const value of nodes.values())
    if (value.workflow)
      value.label = resolveDisplay(
        value.workflow,
        value.label,
        undefined,
        text(value.spawn?.operation.description),
      ).label;
  return [...nodes.values()].filter((value) => !value.parent);
}

export function workflowInputs(
  history: HistoryItem[],
  pending: InputRequest[],
) {
  const { chat, decisions } = workflowProjection(history, pending);
  return { chat, decisions };
}

export function workflowProjection(
  history: HistoryItem[],
  pending: InputRequest[],
  rootWorkflowId?: string,
  states?: Record<string, WorkflowState>,
) {
  const tree = workflowTree(history, pending, states);
  const roots = rootWorkflowId
    ? tree.filter((root) => root.id === rootWorkflowId)
    : tree;
  const chat = rootWorkflowId
    ? pending.filter(
        (request) =>
          request.workflow_id === rootWorkflowId && isChatRequest(request),
      )
    : roots.flatMap((root) => root.requests.filter(isChatRequest));
  const composerRequests = new Set(chat);
  for (const root of roots)
    root.requests = root.requests.filter(
      (request) => !composerRequests.has(request),
    );
  const decisions = pending.filter((request) => !composerRequests.has(request));
  const previous = inputViews.get(pending);
  const inputs = {
    chat: previous && sameRequests(previous.chat, chat) ? previous.chat : chat,
    decisions:
      previous && sameRequests(previous.decisions, decisions)
        ? previous.decisions
        : decisions,
  };
  inputViews.set(pending, inputs);
  return {
    roots,
    ...inputs,
  };
}

function sameRequests(previous: InputRequest[], next: InputRequest[]) {
  return (
    previous.length === next.length &&
    previous.every((request, index) => request === next[index])
  );
}

export type WorkflowProjection = ReturnType<typeof workflowProjection>;

export function workflowRequests(node: WorkflowNode): InputRequest[] {
  return [
    ...node.requests,
    ...node.entries.flatMap((entry) =>
      "entries" in entry ? workflowRequests(entry) : [],
    ),
  ];
}

export function workflowPath(node: WorkflowNode): string {
  const label = node.label;
  return node.parent ? `${workflowPath(node.parent)} / ${label}` : label;
}

export function workflowWorktree(
  node: WorkflowNode,
  rootWorktreeId?: string,
): string | undefined {
  return (
    text(node.spawn?.operation.worktree_id) ||
    (node.parent
      ? workflowWorktree(node.parent, rootWorktreeId)
      : rootWorktreeId)
  );
}

export function workflowFailuresRecovered(node: WorkflowNode): boolean {
  return node.status === "Completed" && workflowRequests(node).length === 0;
}
