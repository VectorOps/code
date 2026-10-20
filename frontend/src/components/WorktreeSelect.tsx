import * as React from "react";
import { useWorktreeList } from "./Worktrees";
import { WorktreeCreate } from "./WorktreeCreate";

export function WorktreeSelect({
  value,
  change,
  enabled,
  creating,
  setCreating,
}: {
  value: string;
  change: (id: string) => void;
  enabled: boolean;
  creating: boolean;
  setCreating: (value: boolean) => void;
}) {
  const { api, records, loading, error, refresh } = useWorktreeList();
  const selected = records.find((item) => item.id === value);
  return (
    <section aria-label="Workflow worktree">
      <label>
        Worktree
        <select
          value={value}
          disabled={!enabled || !api || loading || creating}
          onChange={(event) => change(event.target.value)}
        >
          <option value="">Default project directory</option>
          {value && selected?.status !== "ready" && (
            <option value={value} disabled>
              {selected?.name || value} ({selected?.status || "unavailable"})
            </option>
          )}
          {records
            .filter((item) => item.status === "ready")
            .map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} — {item.branch}
              </option>
            ))}
        </select>
      </label>
      {selected && <small className="worktree-path">{selected.path}</small>}
      {loading && <p role="status">Loading worktrees…</p>}
      {error && <p role="alert">Unable to load worktrees. {error}</p>}
      <div className="worktree-actions">
        <button
          type="button"
          className="quiet-button"
          disabled={!enabled || !api || creating || loading}
          onClick={() => setCreating(true)}
        >
          New worktree
        </button>
        <button
          type="button"
          className="quiet-button"
          disabled={!enabled || !api || creating || loading}
          onClick={() => void refresh()}
        >
          Refresh worktrees
        </button>
      </div>
      {creating && (
        <WorktreeCreate
          cancel={() => setCreating(false)}
          created={(record) => {
            change(record.id);
            setCreating(false);
          }}
        />
      )}
    </section>
  );
}
