import { object, text } from "./json";
import type { HistoryItem, InputRequest } from "./runner";
import { isChatRequest } from "./runner";
import { stepView, toolDescription, type StepView } from "./steps";

export type WorkflowNode = {
  id: string;
  label: string;
  parent?: WorkflowNode;
  entries: (StepView | WorkflowNode)[];
  requests: InputRequest[];
  status: string;
};

export function workflowTree(history: HistoryItem[], pending: InputRequest[]) {
  const nodes = new Map<string, WorkflowNode>();
  const calls = new Map<string, { name: string; description: string }[]>();
  function node(id: string): WorkflowNode {
    let value = nodes.get(id);
    if (!value) {
      value = {
        id,
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
        owner.entries.push(child);
        continue;
      }
    }
    owner.entries.push(step);
    if (step.kind === "workflow.complete")
      owner.status = step.state === "failed" ? "Failed" : "Completed";
  }
  for (const request of pending)
    node(request.workflow_id || "").requests.push(request);
  return [...nodes.values()].filter((value) => !value.parent);
}

export function workflowInputs(
  history: HistoryItem[],
  pending: InputRequest[],
) {
  const roots = workflowTree(history, pending);
  const chat = roots.flatMap((root) => root.requests.filter(isChatRequest));
  const composerRequests = new Set(chat);
  return {
    chat,
    decisions: pending.filter((request) => !composerRequests.has(request)),
  };
}

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
