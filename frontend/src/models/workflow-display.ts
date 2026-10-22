import type { WorkflowState } from "./runner";
import type { StepView } from "./steps";
import { isUserResponse } from "./steps";

export type PresentationOption =
  "prefix_messages" | "llm_only" | "hide_subworkflows";

export type WorkflowPresentation = {
  short_name?: string | null;
  bgcolor?: string | null;
  options?: PresentationOption[];
};

export type WorkflowDisplay = {
  label: string;
  bgcolor?: string;
  options: ReadonlySet<PresentationOption>;
  hidden: boolean;
};

export function resolveDisplay(
  workflow: WorkflowState | undefined,
  fallback: string,
  parent?: WorkflowDisplay,
  labelOverride?: string,
): WorkflowDisplay {
  const own = workflow?.presentation;
  const options = new Set([
    ...(parent?.options ?? []),
    ...(own?.options ?? []),
  ]);
  const hidden = Boolean(
    parent?.hidden || parent?.options.has("hide_subworkflows"),
  );
  const defaultLabel = (
    own?.short_name ||
    workflow?.name ||
    workflow?.definition_ref ||
    fallback
  )
    .trim()
    .replace(/\s+/g, " ")
    .slice(0, 40);
  return {
    label: labelOverride?.trim() || defaultLabel,
    bgcolor:
      own?.bgcolor && /^#[0-9a-f]{6}$/i.test(own.bgcolor)
        ? own.bgcolor.toLowerCase()
        : undefined,
    options,
    hidden,
  };
}

export function displayKind(
  step: StepView,
): "message" | "tool" | "activity" | "status" {
  if (step.kind === "workflow.complete" && step.state === "failed")
    return "status";
  if (
    step.kind === "llm.generate" ||
    step.kind === "echo" ||
    isUserResponse(step)
  )
    return "message";
  return step.kind === "llm.tool" ? "tool" : "activity";
}

export function showsStep(display: WorkflowDisplay, step: StepView): boolean {
  const kind = displayKind(step);
  if (kind === "status") return true;
  return (
    !display.hidden && (kind === "message" || !display.options.has("llm_only"))
  );
}
