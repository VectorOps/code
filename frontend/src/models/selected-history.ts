import { object, text } from "./json";
import type { Execution, HistoryItem } from "./runner";

export function selectedHistory(
  history: HistoryItem[],
  execution: Execution,
): HistoryItem[] {
  const byId = new Map(
    history.map((item) => [text(object(item.step).id), item]),
  );
  const selected = new Set<string>();
  const visited = new Set<string>();

  function collect(workflow: string, head: string | undefined) {
    if (visited.has(workflow)) return;
    visited.add(workflow);
    const path: HistoryItem[] = [];
    while (head && !selected.has(head)) {
      const item = byId.get(head);
      if (!item) break;
      const step = object(item.step);
      if (step.workflow_id !== workflow) break;
      selected.add(head);
      path.push(item);
      head = text(step.previous_step_id);
    }
    const completions = new Map<string, string>();
    for (const item of path) {
      const children = object(object(item.step).outcome).child_completions;
      for (const value of Array.isArray(children) ? children : []) {
        const child = object(value);
        completions.set(
          text(child.spawn_step_id),
          text(child.completion_step_id),
        );
      }
    }
    for (const item of path) {
      const step = object(item.step);
      const operation = object(step.operation);
      if (operation.kind !== "workflow.spawn") continue;
      const child = text(operation.child_workflow_id);
      if (child)
        collect(
          child,
          completions.get(text(step.id)) ?? execution.checkpoint.heads[child],
        );
    }
  }

  collect(
    execution.root_workflow_id,
    execution.checkpoint.heads[execution.root_workflow_id],
  );
  return history.filter((item) => selected.has(text(object(item.step).id)));
}

export function branchSuccessors(
  history: HistoryItem[],
  execution: Execution | undefined,
): ReadonlyMap<string, string> {
  const successors = new Map<string, string>();
  if (!execution) return successors;
  const steps = selectedHistory(history, execution)
    .map((item) => object(item.step))
    .filter((step) => step.workflow_id === execution.root_workflow_id);
  const ids = new Set(steps.map((step) => text(step.id)));
  for (const step of steps) {
    const previous = text(step.previous_step_id);
    if (ids.has(previous) && object(step.operation).kind !== "workflow.resume")
      successors.set(previous, text(step.id));
  }
  return successors;
}
