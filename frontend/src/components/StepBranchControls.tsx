import * as React from "react";
import type { StepView } from "../models/steps";
import type { StepActions } from "./steps/registry";
import { FocusRegion } from "../focus/FocusProvider";

export function useStepBranchControls(step: StepView, actions: StepActions) {
  const [mode, setMode] = React.useState<"restart" | "after" | null>(null);
  const [error, setError] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const confirm = React.useRef<HTMLButtonElement>(null);
  const trigger = React.useRef<HTMLButtonElement | null>(null);
  const lastMode = React.useRef<"restart" | "after">("after");
  const restoreFocus = React.useRef(false);
  React.useEffect(() => {
    if (mode) confirm.current?.focus();
    else if (restoreFocus.current) {
      trigger.current?.focus({ preventScroll: true });
      restoreFocus.current = false;
    }
  }, [mode]);
  if (
    !actions.branch ||
    !actions.rootWorkflowId ||
    step.workflow !== actions.rootWorkflowId ||
    step.kind === "workflow.resume"
  )
    return { icons: null, editor: null };
  const editable =
    actions.editInput &&
    step.kind === "mailbox.input" &&
    step.state === "completed";
  const successor = actions.branchSuccessors?.get(step.id);
  const branchAfter =
    actions.allowBranchAfter !== false &&
    Boolean(successor) &&
    ["completed", "joined"].includes(step.state) &&
    !["workflow.spawn", "workflow.complete"].includes(step.kind);
  const failed = step.state === "failed" || actions.workflowFailed === true;
  const restartLabel = failed ? "Retry this step" : "Restart this step";
  const unavailable = mode === "after" && !successor;
  const disabled = !actions.enabled || submitting || unavailable;

  function open(next: "restart" | "after") {
    lastMode.current = next;
    setMode(next);
    setError("");
  }

  function cancel() {
    restoreFocus.current = true;
    setMode(null);
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const target = mode === "after" ? successor : step.id;
    if (disabled || !mode || !actions.branch || !target) return;
    setSubmitting(true);
    setError("");
    try {
      if (await actions.branch(target, { kind: "restart" })) setMode(null);
      else
        setError(
          "Branch was not confirmed. Review the workflow error before retrying.",
        );
    } catch (reason) {
      setError(String(reason));
    } finally {
      setSubmitting(false);
    }
  }

  const editor = mode ? (
    <FocusRegion
      mode="protected"
      className="branch-editor"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === "Escape" && !submitting) {
          event.preventDefault();
          cancel();
        }
      }}
    >
      <form onSubmit={submit}>
        <p>
          {mode === "after"
            ? "Continue after this step on a new branch. This step and earlier results are preserved; execution continues from the next recorded step."
            : "Re-run this step and any following steps on a new branch. Earlier recorded results are reused."}{" "}
          Previous history is kept. External changes are not undone.
        </p>
        {error && <p role="alert">{error}</p>}
        {unavailable && (
          <p role="alert">
            This branch point is no longer available. Cancel and choose another
            step.
          </p>
        )}
        <div className="branch-buttons">
          <button ref={confirm} type="submit" disabled={disabled}>
            {submitting
              ? "Submitting…"
              : mode === "after"
                ? "Create branch"
                : failed
                  ? "Retry step"
                  : "Restart step"}
          </button>
          <button
            type="button"
            className="quiet-button"
            disabled={submitting}
            onClick={cancel}
          >
            Cancel
          </button>
        </div>
      </form>
    </FocusRegion>
  ) : null;
  const icons = !mode ? (
    <span className="branch-icons">
      {editable && (
        <button
          type="button"
          className="step-details-button branch-icon"
          aria-label="Edit input"
          title="Edit input on a new branch"
          disabled={disabled}
          onClick={(event) => {
            event.stopPropagation();
            actions.editInput?.(step);
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="m16 4 4 4M4 16l-1 5 5-1L20 8a2.8 2.8 0 0 0-4-4Z" />
          </svg>
        </button>
      )}
      {branchAfter && (
        <button
          ref={lastMode.current === "after" ? trigger : undefined}
          type="button"
          className="step-details-button branch-icon branch-after-icon"
          aria-label="Branch after this step"
          title="Branch after this step"
          disabled={disabled}
          onClick={(event) => {
            event.stopPropagation();
            open("after");
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="6" cy="5" r="2" />
            <circle cx="6" cy="19" r="2" />
            <circle cx="18" cy="5" r="2" />
            <path d="M6 7v10M6 13h6a6 6 0 0 0 6-6" />
          </svg>
        </button>
      )}
      {failed && (
        <button
          type="button"
          className="step-details-button branch-icon"
          aria-label={restartLabel}
          title={`${restartLabel} and continue on a new branch`}
          ref={lastMode.current === "restart" ? trigger : undefined}
          disabled={disabled}
          onClick={(event) => {
            event.stopPropagation();
            open("restart");
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M3 11a9 9 0 0 1 15.4-6.4L21 7M21 3v4h-4M21 13a9 9 0 0 1-15.4 6.4L3 17M3 21v-4h4" />
          </svg>
        </button>
      )}
    </span>
  ) : null;
  return { icons, editor };
}

export function StepBranchControls(props: {
  step: StepView;
  actions: StepActions;
}) {
  const { icons, editor } = useStepBranchControls(props.step, props.actions);
  if (!icons && !editor) return null;
  return (
    <div className="step-branch-controls">
      {icons}
      {editor}
    </div>
  );
}
