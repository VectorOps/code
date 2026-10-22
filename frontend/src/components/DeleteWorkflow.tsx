import * as React from "react";

export function DeleteWorkflow({
  executionId,
  name,
  enabled,
  blocked,
  remove,
  cancel,
}: {
  executionId: string;
  name: string;
  enabled: boolean;
  blocked: boolean;
  remove: (executionId: string) => Promise<boolean>;
  cancel: () => void;
}) {
  const [submitting, setSubmitting] = React.useState(false);
  const [error, setError] = React.useState("");
  const inFlight = React.useRef(false);

  async function confirm() {
    if (!enabled || blocked || inFlight.current) return;
    inFlight.current = true;
    setSubmitting(true);
    setError("");
    try {
      if (await remove(executionId)) cancel();
      else
        setError("Workflow was not deleted. Review the error and try again.");
    } catch (reason) {
      setError(String(reason));
    } finally {
      inFlight.current = false;
      setSubmitting(false);
    }
  }

  return (
    <div role="group" aria-label="Confirm workflow deletion">
      <p>Delete “{name}”?</p>
      <p>
        <code>{executionId}</code>
      </p>
      <p>
        This permanently deletes this workflow run and its conversation and
        activity history. This cannot be undone. The workflow definition and
        project files are not deleted.
      </p>
      {blocked && (
        <p>
          Stop the workflow and wait for it to finish stopping before deleting
          it.
        </p>
      )}
      {error && <p role="alert">{error}</p>}
      <div className="header-actions">
        <button
          type="button"
          className="quiet-button"
          disabled={submitting}
          onClick={cancel}
        >
          Cancel
        </button>
        <button
          type="button"
          className="border-red-700 bg-red-700 text-white hover:bg-red-800"
          disabled={!enabled || blocked || submitting}
          onClick={() => void confirm()}
        >
          {submitting ? "Deleting…" : "Permanently delete workflow"}
        </button>
      </div>
    </div>
  );
}
