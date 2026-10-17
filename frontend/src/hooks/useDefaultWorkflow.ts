import { useState } from "react";
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
  const [reference, setReference] = useState(() => {
    try {
      return localStorage.getItem("runner.defaultWorkflow") || "agent";
    } catch {
      return "agent";
    }
  });
  function change(next: string) {
    setReference(next);
    try {
      localStorage.setItem("runner.defaultWorkflow", next);
    } catch {}
  }
  return { reference, change };
}
