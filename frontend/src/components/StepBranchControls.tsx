import * as React from "react";
import type { BranchAction } from "../models/runner";
import * as steps from "../models/steps";
import type { StepActions } from "./steps/registry";
import { FocusRegion } from "../focus/FocusProvider";

export function useStepBranchControls(
  step: steps.StepView,
  actions: StepActions,
) {
  const [mode, setMode] = React.useState<"edit" | "restart" | null>(null);
  const [value, setValue] = React.useState("");
  const [error, setError] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const textarea = React.useRef<HTMLTextAreaElement>(null);
  React.useEffect(() => {
    if (mode === "edit") textarea.current?.focus();
  }, [mode]);
  if (
    !actions.branch ||
    !actions.rootWorkflowId ||
    step.workflow !== actions.rootWorkflowId ||
    step.kind === "workflow.resume"
  )
    return { icons: null, editor: null };
  const editable = step.kind === "mailbox.input" && step.state === "completed";
  const failed = step.state === "failed";
  const restartLabel = failed ? "Retry this step" : "Restart this step";
  const disabled = !actions.enabled || submitting;

  function open(next: "edit" | "restart") {
    setMode(next);
    setValue(editable ? steps.inputEditValue(step) : "");
    setError("");
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (disabled || !mode || !actions.branch) return;
    setSubmitting(true);
    setError("");
    try {
      const action: BranchAction =
        mode === "edit"
          ? {
              kind: "insert_input",
              response: steps.editedInputResponse(step, value),
            }
          : { kind: "restart" };
      if (await actions.branch(step.id, action)) setMode(null);
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
      onKeyDown={(event) => event.stopPropagation()}
    >
      <form onSubmit={submit}>
        {mode === "edit" && (
          <label>
            Edit input
            <textarea
              ref={textarea}
              value={value}
              disabled={submitting}
              onChange={(event) => setValue(event.target.value)}
              rows={4}
            />
          </label>
        )}
        <p>
          {mode === "edit"
            ? "Save this input on a new branch and continue automatically. Existing attachments are kept."
            : "Re-run this step and any following steps on a new branch. Earlier recorded results are reused."}{" "}
          Previous history is kept. External changes are not undone.
        </p>
        {error && <p role="alert">{error}</p>}
        <div className="branch-buttons">
          <button type="submit" disabled={disabled}>
            {submitting
              ? "Submitting…"
              : mode === "edit"
                ? "Save and branch"
                : failed
                  ? "Retry step"
                  : "Restart step"}
          </button>
          <button
            type="button"
            className="quiet-button"
            disabled={submitting}
            onClick={() => setMode(null)}
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
            open("edit");
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
      {failed && (
        <button
          type="button"
          className="step-details-button branch-icon"
          aria-label={restartLabel}
          title={`${restartLabel} and continue on a new branch`}
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
  step: steps.StepView;
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
