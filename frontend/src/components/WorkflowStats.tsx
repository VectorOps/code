import type { Session } from "../models/runner";

export function WorkflowStats({ session }: { session: Session }) {
  const totals = session.usage?.totals;
  return (
    <>
      <dl>
        <dt>Execution</dt>
        <dd>{session.execution_id}</dd>
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
