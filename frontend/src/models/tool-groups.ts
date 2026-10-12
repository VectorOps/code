import type { InputRequest } from "./runner";
import { approvalFor, toolDescription, type StepView } from "./steps";
import type { WorkflowNode } from "./workflow-tree";

export type ToolGrouping = Readonly<Record<string, boolean | string>>;

export const defaultToolGrouping: ToolGrouping = {
  list_files: true,
  read_file: true,
  read_files: true,
  exec: true,
  poll_process: "exec",
};

export type ToolGroup = {
  id: string;
  name: string;
  steps: StepView[];
};

type Entry = StepView | WorkflowNode;

export function groupTools(
  entries: readonly Entry[],
  pending: InputRequest[],
  configuration: ToolGrouping = defaultToolGrouping,
): (Entry | ToolGroup)[] {
  const result: (Entry | ToolGroup)[] = [];
  let run: ToolGroup | undefined;
  function flush() {
    if (run) result.push(run.steps.length > 1 ? run : run.steps[0]);
    run = undefined;
  }
  for (const entry of entries) {
    const name = "entries" in entry ? undefined : toolDescription(entry).name;
    const policy = name === undefined ? undefined : configuration[name];
    const groupName = policy === true ? name : policy || undefined;
    const eligible =
      !("entries" in entry) &&
      entry.kind === "llm.tool" &&
      name !== undefined &&
      name !== "apply_patch" &&
      groupName !== undefined &&
      entry.input.gate_only !== true &&
      !approvalFor(entry, pending) &&
      !pending.some(
        (request) =>
          request.request_id === entry.id &&
          request.workflow_id === entry.workflow,
      );
    if (!eligible || "entries" in entry || !groupName) {
      flush();
      result.push(entry);
      continue;
    }
    if (run?.name !== groupName || run.steps[0].workflow !== entry.workflow) {
      flush();
      run = { id: `tool-group-${entry.id}`, name: groupName, steps: [] };
    }
    run.steps.push(entry);
  }
  flush();
  return result;
}
