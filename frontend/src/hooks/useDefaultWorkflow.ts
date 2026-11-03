import { useDesktopSettings } from "../components/DesktopSettings";
import type { Workflow } from "../models/runner";

export function resolveDefaultWorkflow(
  workflows: Workflow[],
  reference: string,
) {
  const exact = workflows.find(
    (item) => item.descriptor.reference === reference,
  );
  if (exact || reference !== "agent") return exact;
  const builtin = workflows.find(
    (item) => item.descriptor.reference === "builtin:agent",
  );
  if (builtin) return builtin;
  const named = workflows.filter(
    (item) => item.descriptor.name.toLowerCase() === "agent",
  );
  return named.length === 1 ? named[0] : undefined;
}

export function useDefaultWorkflow() {
  const { value, save } = useDesktopSettings();
  function change(next: string) {
    save({ default_workflow: next });
  }
  return { reference: value.default_workflow, change };
}
