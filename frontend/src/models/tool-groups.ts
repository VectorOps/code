import type { InputRequest } from "./runner";
import { approvalFor, toolDescription, type StepView } from "./steps";
import type { WorkflowNode } from "./workflow-tree";

export type ToolGrouping = Readonly<Record<string, boolean>>;

export const defaultToolGrouping: ToolGrouping = {
  list_files: true,
  read_file: true,
  read_files: true,
  exec: true,
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
    const eligible =
      !("entries" in entry) &&
      entry.kind === "llm.tool" &&
      name !== undefined &&
      name !== "apply_patch" &&
      configuration[name] === true &&
      entry.input.gate_only !== true &&
      !approvalFor(entry, pending) &&
      !pending.some(
        (request) =>
          request.request_id === entry.id &&
          request.workflow_id === entry.workflow,
      );
    if (!eligible || "entries" in entry || !name) {
      flush();
      result.push(entry);
      continue;
    }
    if (run?.name !== name || run.steps[0].workflow !== entry.workflow) {
      flush();
      run = { id: `tool-group-${entry.id}`, name, steps: [] };
    }
    run.steps.push(entry);
  }
  flush();
  return result;
}
