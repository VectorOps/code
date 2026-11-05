import * as React from "react";
import * as models from "../models/runner";
import { StatusIcon } from "./StatusIcon";
import { WorktreeLabel } from "./Worktrees";

const pageSize = 100;
const noActivity = new Set<string>();

type SessionEntry = { session: models.Session; name: string; search: string };

const SessionRow = React.memo(function SessionRow({
  session,
  name,
  selected,
  activity,
  collapsed,
  select,
}: {
  session: models.Session;
  name: string;
  selected: boolean;
  activity: boolean;
  collapsed: boolean;
  select: (id: string) => void;
}) {
  const status = session.pending_inputs > 0 ? "Needs input" : session.status;
  const label = `${name} — ${status}${activity ? " — New activity" : ""}`;
  return (
    <button
      aria-current={selected ? "page" : undefined}
      aria-label={collapsed ? label : undefined}
      title={collapsed ? label : undefined}
      onClick={() => select(session.execution_id)}
    >
      <span className="session-list-heading">
        <StatusIcon
          status={status}
          label={collapsed ? name : "Workflow status"}
        />
        {!collapsed && (
          <span className="session-list-title" title={name}>
            {name}
          </span>
        )}
        {activity && (
          <span
            className="session-list-activity"
            aria-label="New activity"
            title="New activity"
          >
            ●
          </span>
        )}
      </span>
      {!collapsed && session.worktree_id && (
        <WorktreeLabel id={session.worktree_id} />
      )}
    </button>
  );
});

const SessionRows = React.memo(function SessionRows({
  entries,
  selected,
  activity,
  collapsed,
  select,
}: {
  entries: SessionEntry[];
  selected: string;
  activity: Set<string>;
  collapsed: boolean;
  select: (id: string) => void;
}) {
  return entries.map(({ session, name }) => (
    <SessionRow
      key={session.execution_id}
      session={session}
      name={name}
      selected={selected === session.execution_id}
      activity={activity.has(session.execution_id)}
      collapsed={collapsed}
      select={select}
    />
  ));
});

export const SessionSidebar = React.memo(function SessionSidebar({
  sessions,
  activitySessions = noActivity,
  hasMore = false,
  loading = false,
  loadMore,
  workflows,
  selected,
  status,
  select,
  collapse,
  create,
  advanced,
  enabled = true,
  configure,
  collapsed = false,
}: {
  collapsed?: boolean;
  sessions: models.Session[];
  hasMore?: boolean;
  loading?: boolean;
  loadMore?: () => Promise<boolean>;
  activitySessions?: Set<string>;
  workflows: models.Workflow[];
  selected: string;
  status: string;
  select: (id: string) => void;
  collapse: () => void;
  create: () => void;
  advanced?: () => void;
  enabled?: boolean;
  configure?: () => void;
}) {
  const [search, setSearch] = React.useState("");
  const [filter, setFilter] = React.useState("all");
  const [position, setPosition] = React.useState<{
    after: string | null;
    first: string | null;
  }>({ after: null, first: null });
  const list = React.useRef<HTMLElement>(null);
  const query = React.useRef("");
  query.current = JSON.stringify([search, filter]);
  const names = React.useMemo(
    () =>
      new Map(
        workflows.map(({ descriptor }) => [
          descriptor.reference,
          descriptor.name,
        ]),
      ),
    [workflows],
  );
  const entries = React.useMemo(
    () =>
      sessions.map((session) => {
        const name = models.workflowDisplayName(
          session.reference,
          names,
          session.name,
          session.description,
        );
        return {
          session,
          name,
          search:
            `${name} ${session.reference} ${session.execution_id}`.toLowerCase(),
        };
      }),
    [sessions, names],
  );
  const visible = React.useMemo(() => {
    const normalized = search.toLowerCase();
    return entries.filter(
      ({ session, search: value }) =>
        (filter !== "running" || session.active) &&
        (filter !== "attention" || session.pending_inputs > 0) &&
        value.includes(normalized),
    );
  }, [entries, search, filter]);
  const start = React.useMemo(() => {
    if (position.after === null) return 0;
    const boundary = visible.findIndex(
      ({ session }) => session.execution_id === position.after,
    );
    if (boundary >= 0 && boundary + 1 < visible.length) return boundary + 1;
    return Math.max(
      0,
      visible.findIndex(
        ({ session }) => session.execution_id === position.first,
      ),
    );
  }, [visible, position]);
  const rows = React.useMemo(
    () => visible.slice(start, start + pageSize),
    [visible, start],
  );
  const moreLoaded = start + pageSize < visible.length;
  React.useEffect(() => {
    if (list.current) list.current.scrollTop = 0;
  }, [position, search, filter]);
  async function older() {
    const next = {
      after: rows.at(-1)?.session.execution_id ?? null,
      first: rows[0]?.session.execution_id ?? null,
    };
    if (moreLoaded) setPosition(next);
    else {
      const current = query.current;
      if ((await loadMore?.()) && query.current === current) setPosition(next);
    }
  }
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
      <div className="new-conversation-actions">
        <button
          onClick={create}
          disabled={!enabled}
          aria-label="New conversation"
          aria-keyshortcuts="Meta+N Control+N"
          title="New conversation (Cmd/Ctrl+N)"
        >
          {collapsed ? "+" : "New conversation"}
        </button>
        <button
          onClick={advanced ?? create}
          aria-label="Advanced new conversation"
          aria-keyshortcuts="Meta+Shift+N Control+Shift+N"
          title="Advanced: choose workflow and inputs (Cmd/Ctrl+Shift+N)"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            aria-hidden="true"
          >
            <path d="M4 7h16M4 17h16" />
            <circle cx="9" cy="7" r="3" />
            <circle cx="15" cy="17" r="3" />
          </svg>
        </button>
      </div>
      {!collapsed && (
        <>
          <label className="search-label">
            Find a session
            <input
              type="search"
              placeholder="Workflow or execution ID"
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setPosition({ after: null, first: null });
              }}
            />
          </label>
          <div className="session-filters" aria-label="Filter sessions">
            {["all", "running", "attention"].map((value) => (
              <button
                key={value}
                aria-pressed={filter === value}
                onClick={() => {
                  setFilter(value);
                  setPosition({ after: null, first: null });
                }}
              >
                {value === "attention" ? "Needs input" : value}
              </button>
            ))}
          </div>
        </>
      )}
      <nav
        ref={list}
        className="session-list"
        aria-label="Conversations"
        aria-busy={loading}
      >
        <SessionRows
          entries={rows}
          selected={selected}
          activity={activitySessions}
          collapsed={collapsed}
          select={select}
        />
      </nav>
      {(visible.length > pageSize || hasMore || loading) && (
        <div
          className="session-pagination"
          role="group"
          aria-label="Conversation pages"
        >
          <span
            className={collapsed ? "session-pagination-description" : undefined}
            aria-live="polite"
          >
            {visible.length
              ? `${start + 1}–${start + rows.length} of ${visible.length} loaded`
              : "No loaded matches"}
          </span>
          <div>
            <button
              className="quiet-button"
              disabled={start === 0}
              onClick={() =>
                setPosition({
                  after:
                    visible[Math.max(0, start - pageSize) - 1]?.session
                      .execution_id ?? null,
                  first: null,
                })
              }
              aria-label="Newer conversations"
              title="Newer conversations"
            >
              {collapsed ? "↑" : "Newer"}
            </button>
            <button
              className="quiet-button"
              disabled={
                !moreLoaded &&
                (!hasMore || loading || !loadMore || status !== "Connected")
              }
              onClick={() => void older()}
              aria-label={
                moreLoaded ? "Older conversations" : "Load older conversations"
              }
              title={
                moreLoaded ? "Older conversations" : "Load older conversations"
              }
            >
              {collapsed
                ? loading
                  ? "…"
                  : "↓"
                : moreLoaded
                  ? "Older"
                  : loading
                    ? "Loading…"
                    : "Load older"}
            </button>
          </div>
        </div>
      )}
      {!collapsed && (
        <>
          {hasMore && (
            <p className="session-search-scope">
              Search and filters cover loaded conversations. Load older
              conversations to include more.
            </p>
          )}
          {!loading && sessions.length === 0 && <p>No conversations yet.</p>}
          {!loading && sessions.length > 0 && visible.length === 0 && (
            <p>No matching loaded conversations.</p>
          )}
        </>
      )}
      <div className="sidebar-footer">
        <StatusIcon status={status} label="Connection status" />
        {!collapsed && <span className="connection-label">Backend</span>}
        {configure && (
          <button
            className="quiet-button icon-button"
            onClick={configure}
            aria-label="Connection settings"
            title={`Connection settings: ${status}`}
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <path d="M8 3v5m8-5v5M6 8h12v3a6 6 0 0 1-12 0Zm6 9v4" />
            </svg>
          </button>
        )}
      </div>
    </aside>
  );
});
