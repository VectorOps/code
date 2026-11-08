import * as React from "react";
import type { RunnerApi } from "../communication/runner-api";
import { SettingsIcon } from "../components/SettingsIcon";
import { DeleteWorkflow } from "../components/DeleteWorkflow";
import { formatBytes } from "../models/disk-usage";
import { workflowDisplayName } from "../models/runner";
import type { DiskUsageReport, Session } from "../models/runner";

export function DiskUsage({
  api,
  onDelete,
}: {
  api: RunnerApi;
  onDelete?: (executionId: string) => Promise<boolean>;
}) {
  const [report, setReport] = React.useState<DiskUsageReport>();
  const [sessions, setSessions] = React.useState<Record<string, Session>>({});
  const [deleting, setDeleting] = React.useState("");
  const [error, setError] = React.useState("");
  const [busy, setBusy] = React.useState(true);
  const [revision, refresh] = React.useReducer((value) => value + 1, 0);
  const confirmation = React.useRef<HTMLTableCellElement>(null);
  const deleteTrigger = React.useRef<HTMLButtonElement>(null);

  React.useEffect(() => {
    if (deleting) confirmation.current?.querySelector("button")?.focus();
    else deleteTrigger.current?.focus();
  }, [deleting]);

  React.useEffect(() => {
    let active = true;
    setBusy(true);
    setError("");
    setReport(undefined);
    setSessions({});
    setDeleting("");
    async function load() {
      try {
        const next = await api.diskUsage();
        if (!active) return;
        setReport(next);
        const summaries = await Promise.all(
          next.top_workflows.map(async ({ execution_id }) => {
            try {
              const session = await api.session(execution_id);
              return [execution_id, session] as const;
            } catch {
              return null;
            }
          }),
        );
        if (active)
          setSessions(
            Object.fromEntries(summaries.filter((entry) => entry !== null)),
          );
      } catch (reason) {
        if (active)
          setError(
            `Unable to load disk usage. ${reason instanceof Error ? reason.message : "Request failed."}`,
          );
      } finally {
        if (active) setBusy(false);
      }
    }
    void load();
    return () => {
      active = false;
    };
  }, [api, revision]);

  function name(executionId: string) {
    const session = sessions[executionId];
    return session
      ? workflowDisplayName(
          session.reference,
          [],
          session.name,
          session.description,
        )
      : executionId;
  }

  return (
    <>
      <div className="settings-toolbar">
        <p>Live file-size estimates for this runner.</p>
        <button
          className="settings-icon-button"
          title="Refresh disk usage"
          aria-label="Refresh disk usage"
          disabled={busy || !!deleting}
          onClick={refresh}
        >
          <SettingsIcon name="refresh" />
        </button>
      </div>
      {busy && <p role="status">Loading disk usage…</p>}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {report && (
        <>
          <dl>
            <dt>Total disk usage</dt>
            <dd title={`${report.total_bytes.toLocaleString()} bytes`}>
              {formatBytes(report.total_bytes)}
            </dd>
            <dt>Workflows</dt>
            <dd>{report.workflow_count.toLocaleString()}</dd>
          </dl>
          <p>
            Total includes shared Runner files. Workflow sizes include child
            workflows. Sizes are logical file bytes, not allocated disk blocks.
          </p>
          <h4>Top 10 workflows by size</h4>
          {report.top_workflows.length ? (
            <table className="disk-usage-table">
              <thead>
                <tr>
                  <th scope="col">Workflow</th>
                  <th scope="col">Status</th>
                  <th scope="col">Size</th>
                  <th scope="col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {report.top_workflows.map((workflow) => (
                  <React.Fragment key={workflow.execution_id}>
                    <tr>
                      <td>
                        {sessions[workflow.execution_id] && (
                          <div>{name(workflow.execution_id)}</div>
                        )}
                        <code>{workflow.execution_id}</code>
                      </td>
                      <td>
                        {sessions[workflow.execution_id]?.status ??
                          (busy ? "Loading…" : "Unavailable")}
                      </td>
                      <td
                        title={`${workflow.size_bytes.toLocaleString()} bytes`}
                      >
                        {formatBytes(workflow.size_bytes)}
                      </td>
                      <td>
                        <button
                          className="settings-icon-button"
                          title={
                            sessions[workflow.execution_id]
                              ? `Delete ${name(workflow.execution_id)}`
                              : "Workflow status unavailable"
                          }
                          aria-label={`Delete ${name(workflow.execution_id)}`}
                          disabled={
                            busy ||
                            !!deleting ||
                            !onDelete ||
                            !sessions[workflow.execution_id]
                          }
                          onClick={(event) => {
                            deleteTrigger.current = event.currentTarget;
                            setDeleting(workflow.execution_id);
                          }}
                        >
                          <SettingsIcon name="delete" />
                        </button>
                      </td>
                    </tr>
                    {deleting === workflow.execution_id && onDelete && (
                      <tr>
                        <td colSpan={4} ref={confirmation}>
                          <DeleteWorkflow
                            executionId={deleting}
                            name={name(deleting)}
                            enabled={!busy}
                            blocked={[
                              "running",
                              "starting",
                              "stopping",
                            ].includes(sessions[deleting]?.status)}
                            remove={async (executionId) => {
                              const deleted = await onDelete(executionId);
                              if (deleted) refresh();
                              return deleted;
                            }}
                            cancel={() => setDeleting("")}
                          />
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          ) : (
            <p>No workflows stored yet.</p>
          )}
        </>
      )}
    </>
  );
}
