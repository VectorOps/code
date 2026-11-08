import * as React from "react";
import type { Session, WorkflowDiskUsage } from "../models/runner";
import { formatBytes } from "../models/disk-usage";
import { DebugOnly } from "./Diagnostics";

export function WorkflowStats({
  session,
  workflowId = session.root_workflow_id,
  loadDiskUsage,
}: {
  session: Session;
  workflowId?: string | null;
  loadDiskUsage?: (executionId: string) => Promise<WorkflowDiskUsage>;
}) {
  const totals = session.usage?.totals;
  const [size, setSize] = React.useState<number>();
  const [error, setError] = React.useState("");
  React.useEffect(() => {
    let active = true;
    setSize(undefined);
    setError("");
    if (loadDiskUsage) {
      void loadDiskUsage(session.execution_id).then(
        (usage) => {
          if (active) setSize(usage.size_bytes);
        },
        (reason: unknown) => {
          if (active)
            setError(
              `Unable to load workflow size. ${reason instanceof Error ? reason.message : "Request failed."}`,
            );
        },
      );
    }
    return () => {
      active = false;
    };
  }, [loadDiskUsage, session.execution_id]);
  return (
    <>
      <dl>
        <dt>Workflow</dt>
        <dd>
          <code>{workflowId || "Not available"}</code>
        </dd>
        <DebugOnly>
          {() => (
            <>
              <dt>Execution</dt>
              <dd>{session.execution_id}</dd>
            </>
          )}
        </DebugOnly>
        <dt>Status</dt>
        <dd>{session.status}</dd>
        <dt>Workflow size</dt>
        <dd
          title={
            size === undefined ? undefined : `${size.toLocaleString()} bytes`
          }
        >
          {!loadDiskUsage
            ? "Runner disconnected."
            : error
              ? "Unavailable"
              : size === undefined
                ? "Loading…"
                : formatBytes(size)}
        </dd>
        <dt>Input tokens</dt>
        <dd>{totals?.input_tokens.toLocaleString() ?? "—"}</dd>
        <dt>Output tokens</dt>
        <dd>{totals?.output_tokens.toLocaleString() ?? "—"}</dd>
        <dt>Estimated cost (USD)</dt>
        <dd>{totals ? `$${totals.cost_estimate_usd}` : "—"}</dd>
      </dl>
      {error && <p role="alert">{error}</p>}
      <p>
        Workflow size includes child workflows and excludes shared Runner files.
      </p>
      {!totals && <p>No usage reported yet.</p>}
      {totals && (
        <p>
          Usage includes child workflows and retained branches.
          {totals.incomplete_calls > 0 &&
            " Some calls have incomplete usage or pricing; cost may be understated."}
        </p>
      )}
    </>
  );
}
