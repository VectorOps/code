import * as React from "react";
import { useWorktreeList, useWorktrees } from "../components/Worktrees";
import { WorktreeCreate } from "../components/WorktreeCreate";
import type { Worktree } from "../models/worktrees";

export function WorktreeManager() {
  const { api, records, loading, error, refresh } = useWorktreeList();
  const [creating, setCreating] = React.useState(false);
  const [showDeleted, setShowDeleted] = React.useState(false);
  const visible = records.filter(
    (record) => showDeleted || record.status !== "deleted",
  );
  return (
    <>
      <p>
        Manage isolated Git working directories for workflows and subagents.
      </p>
      <div className="worktree-actions">
        <button
          disabled={!api || loading || creating}
          onClick={() => setCreating(true)}
        >
          New worktree
        </button>
        <button
          className="quiet-button"
          disabled={!api || loading || creating}
          onClick={() => void refresh()}
        >
          Refresh worktrees
        </button>
      </div>
      <label className="worktree-checkbox">
        <input
          type="checkbox"
          checked={showDeleted}
          onChange={(event) => setShowDeleted(event.target.checked)}
        />
        Show deleted worktrees
      </label>
      {error && <p role="alert">Unable to load worktrees. {error}</p>}
      {loading && <p role="status">Loading worktrees…</p>}
      {creating && (
        <WorktreeCreate
          cancel={() => setCreating(false)}
          created={() => setCreating(false)}
        />
      )}
      {!loading && !error && !visible.length && (
        <p>No worktrees yet. Create one here or when launching a workflow.</p>
      )}
      {visible.map((record) => (
        <WorktreeCard key={record.id} record={record} />
      ))}
    </>
  );
}

function WorktreeCard({ record }: { record: Worktree }) {
  const { api, remember, loading } = useWorktrees();
  const [editing, setEditing] = React.useState(false);
  const [name, setName] = React.useState(record.name);
  const [description, setDescription] = React.useState(record.description);
  const [deleting, setDeleting] = React.useState(false);
  const [force, setForce] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const disabled = busy || loading || !api;
  async function apply(operation: () => Promise<Worktree>) {
    setBusy(true);
    setError("");
    try {
      remember(await operation());
      setEditing(false);
      setDeleting(false);
      setForce(false);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setBusy(false);
    }
  }
  return (
    <article className="worktree-card" aria-label={record.name}>
      <h4>{record.name}</h4>
      <p>
        {record.status}
        {record.inspection
          ? ` · ${record.inspection.dirty ? "Uncommitted changes" : "Clean"}`
          : ""}{" "}
        · {record.inspection?.branch || record.branch}
      </p>
      <code className="worktree-path">{record.path}</code>
      {record.description && <p>{record.description}</p>}
      <details>
        <summary>Worktree details</summary>
        <p className="worktree-path">Repository: {record.repository.root}</p>
        <p className="worktree-path">Base commit: {record.base_commit}</p>
        <p>
          Created by: {record.origin.kind === "workflow" ? "Workflow" : "User"}
        </p>
        <p>Created: {record.created_at}</p>
        {record.last_used_at && <p>Last used: {record.last_used_at}</p>}
        <p className="worktree-path">ID: {record.id}</p>
      </details>
      {record.error && <p role="alert">{record.error}</p>}
      {error && <p role="alert">{error}</p>}
      {record.status !== "deleted" && !editing && !deleting && (
        <div className="worktree-actions">
          <button
            className="quiet-button"
            disabled={disabled}
            onClick={() => {
              setName(record.name);
              setDescription(record.description);
              setEditing(true);
            }}
          >
            Edit
          </button>
          <button
            className="quiet-button"
            disabled={
              disabled ||
              record.status === "creating" ||
              record.status === "deleting"
            }
            onClick={() => setDeleting(true)}
          >
            Delete
          </button>
        </div>
      )}
      {editing && (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (api)
              void apply(() => api.update(record.id, name.trim(), description));
          }}
        >
          <fieldset disabled={disabled}>
            <legend>Edit worktree</legend>
            <label>
              Worktree name
              <input
                required
                maxLength={200}
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <label>
              Description
              <textarea
                maxLength={4000}
                value={description}
                onChange={(event) => setDescription(event.target.value)}
              />
            </label>
            <div className="worktree-actions">
              <button disabled={!name.trim()}>Save worktree</button>
              <button
                type="button"
                className="quiet-button"
                onClick={() => setEditing(false)}
              >
                Cancel
              </button>
            </div>
          </fieldset>
        </form>
      )}
      {deleting && (
        <div
          className="worktree-confirmation"
          role="group"
          aria-label={`Delete ${record.name}`}
        >
          <p>
            Delete “{record.name}” and its working directory? Worktrees in use
            cannot be deleted.
          </p>
          <label className="worktree-checkbox">
            <input
              type="checkbox"
              disabled={disabled}
              checked={force}
              onChange={(event) => setForce(event.target.checked)}
            />
            Force deletion (discard uncommitted changes)
          </label>
          {force && (
            <p role="alert">
              Uncommitted changes in this worktree will be permanently lost.
              Check for orphaned processes before removing abandoned usage
              leases.
            </p>
          )}
          <div className="worktree-actions">
            <button
              disabled={disabled}
              onClick={() => {
                if (api) void apply(() => api.delete(record.id, force));
              }}
            >
              Confirm deletion
            </button>
            <button
              className="quiet-button"
              disabled={busy}
              onClick={() => {
                setDeleting(false);
                setForce(false);
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </article>
  );
}
