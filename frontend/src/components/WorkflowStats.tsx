import type { Session } from "../models/runner";
import { DebugOnly } from "./Diagnostics";

export function WorkflowStats({
  session,
  workflowId = session.root_workflow_id,
}: {
  session: Session;
  workflowId?: string | null;
}) {
  const totals = session.usage?.totals;
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
        <dt>Input tokens</dt>
        <dd>{totals?.input_tokens.toLocaleString() ?? "—"}</dd>
        <dt>Output tokens</dt>
        <dd>{totals?.output_tokens.toLocaleString() ?? "—"}</dd>
        <dt>Estimated cost (USD)</dt>
        <dd>{totals ? `$${totals.cost_estimate_usd}` : "—"}</dd>
      </dl>
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
