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
  collapsed = false,
}: {
  collapsed?: boolean;
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
    <aside
      className={`session-sidebar${collapsed ? " session-sidebar-collapsed" : ""}`}
      aria-label="Session navigation"
    >
      <div className="sidebar-heading">
        {!collapsed && <strong>Runner</strong>}
        <button
          className="quiet-button"
          onClick={collapse}
          aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
          title={collapsed ? "Expand navigation" : "Collapse navigation"}
          aria-expanded={!collapsed}
        >
          {collapsed ? (
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <rect x="3" y="4" width="18" height="16" rx="2" />
              <path d="M9 4v16m4-12 4 4-4 4" />
            </svg>
          ) : (
            "Collapse"
          )}
        </button>
      </div>
      <button
        onClick={create}
        aria-label="New conversation"
        title={collapsed ? "New conversation" : undefined}
      >
        {collapsed ? "+" : "New conversation"}
      </button>
      {!collapsed && (
        <>
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
        </>
      )}
      <div className="sidebar-footer">
        {!collapsed && <span role="status">{status}</span>}
        <button
          className="quiet-button"
          onClick={configure}
          aria-label="Connection settings"
          title={collapsed ? `Connection settings: ${status}` : undefined}
        >
          {collapsed ? (
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <path d="M8 3v5m8-5v5M6 8h12v3a6 6 0 0 1-12 0Zm6 9v4" />
            </svg>
          ) : (
            "Connection settings"
          )}
        </button>
      </div>
    </aside>
  );
}
