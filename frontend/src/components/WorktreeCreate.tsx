import * as React from "react";
import { useWorktrees } from "./Worktrees";
import type { Worktree } from "../models/worktrees";

export function WorktreeCreate({
  created,
  cancel,
}: {
  created: (record: Worktree) => void;
  cancel: () => void;
}) {
  const { api, remember } = useWorktrees();
  const [name, setName] = React.useState("");
  const [description, setDescription] = React.useState("");
  const [branch, setBranch] = React.useState("");
  const [base, setBase] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const retry = React.useRef({ signature: "", key: "" });
  const working = React.useRef(false);
  const onCreated = React.useRef(created);
  onCreated.current = created;
  const generation = React.useRef(0);
  React.useLayoutEffect(() => {
    retry.current = { signature: "", key: "" };
    working.current = false;
    setBusy(false);
    setError("");
    return () => {
      generation.current++;
    };
  }, [api]);
  async function create() {
    if (!api || working.current || !name.trim()) return;
    const current = generation.current;
    working.current = true;
    setBusy(true);
    setError("");
    const input = {
      name: name.trim(),
      description,
      ...(branch.trim() ? { branch: branch.trim() } : {}),
      base_ref: base.trim() || "HEAD",
    };
    const signature = JSON.stringify(input);
    if (retry.current.signature !== signature)
      retry.current = { signature, key: crypto.randomUUID() };
    try {
      const record = await api.create(input, retry.current.key);
      remember(record);
      if (current !== generation.current) return;
      retry.current = { signature: "", key: "" };
      onCreated.current(record);
    } catch (reason) {
      if (current === generation.current) setError(String(reason));
    } finally {
      if (current === generation.current) {
        working.current = false;
        setBusy(false);
      }
    }
  }
  return (
    <fieldset className="worktree-create" disabled={busy || !api}>
      <legend>New worktree</legend>
      <label>
        Worktree name
        <input
          value={name}
          maxLength={200}
          onChange={(event) => setName(event.target.value)}
        />
      </label>
      <label>
        Description
        <textarea
          value={description}
          maxLength={4000}
          rows={2}
          onChange={(event) => setDescription(event.target.value)}
        />
      </label>
      <label>
        Branch (optional)
        <input
          value={branch}
          maxLength={1024}
          placeholder="Generate a branch"
          onChange={(event) => setBranch(event.target.value)}
        />
      </label>
      <label>
        Base ref
        <input
          value={base}
          maxLength={1024}
          placeholder="HEAD"
          onChange={(event) => setBase(event.target.value)}
        />
      </label>
      {error && <p role="alert">{error}</p>}
      <div className="worktree-actions">
        <button
          type="button"
          disabled={!name.trim()}
          onClick={() => void create()}
        >
          {busy ? "Creating…" : "Create worktree"}
        </button>
        <button type="button" className="quiet-button" onClick={cancel}>
          Cancel
        </button>
      </div>
    </fieldset>
  );
}
