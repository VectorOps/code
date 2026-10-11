import { useState } from "react";
import type { Session } from "../models/runner";

export function SessionSidebar({
  sessions,
  selected,
  status,
  select,
  collapse,
  create,
  configure,
}: {
  sessions: Session[];
  selected: string;
  status: string;
  select: (id: string) => void;
  collapse: () => void;
  create: () => void;
  configure: () => void;
}) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const visible = sessions.filter(
    (session) =>
      `${session.reference} ${session.execution_id}`
        .toLowerCase()
        .includes(search.toLowerCase()) &&
      (filter !== "running" || session.active) &&
      (filter !== "attention" || session.pending_inputs > 0),
  );
  return (
    <aside className="session-sidebar" aria-label="Session navigation">
      <div className="sidebar-heading">
        <strong>Runner</strong>
        <button
          className="quiet-button"
          onClick={collapse}
          aria-label="Collapse navigation"
        >
          Hide
        </button>
      </div>
      <button onClick={create}>New conversation</button>
      <label className="search-label">
        Find a session
        <input
          type="search"
          placeholder="Workflow or execution ID"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </label>
      <div className="session-filters" aria-label="Filter sessions">
        {["all", "running", "attention"].map((value) => (
          <button
            key={value}
            aria-pressed={filter === value}
            onClick={() => setFilter(value)}
          >
            {value === "attention" ? "Needs input" : value}
          </button>
        ))}
      </div>
      <nav className="session-list" aria-label="Conversations">
        {visible.map((session) => (
          <button
            key={session.execution_id}
            aria-current={
              selected === session.execution_id ? "page" : undefined
            }
            onClick={() => select(session.execution_id)}
          >
            <span>{session.reference || "Workflow"}</span>
            <small>
              {session.pending_inputs
                ? `${session.pending_inputs} pending`
                : session.status}{" "}
              / {session.execution_id.slice(0, 8)}
            </small>
          </button>
        ))}
      </nav>
      {sessions.length === 0 && <p>No conversations yet.</p>}
      <div className="sidebar-footer">
        <span role="status">{status}</span>
        <button className="quiet-button" onClick={configure}>
          Connection settings
        </button>
      </div>
    </aside>
  );
}
