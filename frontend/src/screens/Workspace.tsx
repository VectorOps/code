import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Json } from "../models/json";
import type { Session, InputRequest, HistoryItem } from "../models/runner";
import { inputResponse } from "../models/runner";
import { workflowInputs } from "../models/workflow-tree";
import { activity } from "../models/activity";
import { StepFeed } from "../components/StepFeed";
import { DebugSteps } from "../components/DebugSteps";
import { defaultStepRegistry } from "../components/steps/defaults";
import type { StepRegistry } from "../components/steps/registry";
import { ActivityDetails } from "../components/ActivityDetails";
import { Decision } from "../components/Decision";
import { SessionSidebar } from "../components/SessionSidebar";
import { Composer } from "../components/Composer";
import { defaultToolFormatters } from "../components/tools/defaults";
import type { ToolFormatterFactory } from "../components/tools/factory";
import { approvalFor, stepView, toolDescription } from "../models/steps";
type Inspector =
  | { kind: "debug" }
  | { kind: "activity"; id: string }
  | { kind: "decisions" }
  | { kind: "session" }
  | { kind: "new" }
  | { kind: "connection" };

export function Workspace({
  sessions,
  selected,
  onSelect,
  history,
  pending,
  status,
  endpoint,
  enabled,
  busy,
  onStop,
  onReply,
  newSession,
  connection,
  error,
  dismiss,
  hasOlder,
  stepRegistry = defaultStepRegistry,
  toolFormatters = defaultToolFormatters,
}: {
  toolFormatters?: ToolFormatterFactory;
  stepRegistry?: StepRegistry;
  sessions: Session[];
  selected: string;
  onSelect: (id: string) => void;
  history: HistoryItem[];
  pending: InputRequest[];
  status: string;
  endpoint: string;
  enabled: boolean;
  busy: boolean;
  onStop: () => void;
  onReply: (
    request: InputRequest,
    response: Json,
    key: string,
  ) => Promise<boolean>;
  newSession: ReactNode;
  connection: ReactNode;
  error: string;
  dismiss: () => void;
  hasOlder: boolean;
}) {
  const [leftOpen, setLeftOpen] = useState(true);
  const [inspector, setInspector] = useState<Inspector | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [newActivity, setNewActivity] = useState(false);
  const scroll = useRef<HTMLDivElement>(null);
  const positions = useRef<Record<string, number>>({});
  const following = useRef(true);
  const inspectorRef = useRef<HTMLElement>(null);
  const restoreFocus = useRef<HTMLElement | null>(null);
  const retry = useRef({ signature: "", key: "" });
  const current = sessions.find((session) => session.execution_id === selected);
  const scope = `${endpoint}/${selected}`;
  const draft = drafts[scope] || "";
  const { chat: chatRequests, decisions } = useMemo(
    () => workflowInputs(history, pending),
    [history, pending],
  );
  const chat = chatRequests.length === 1 ? chatRequests[0] : undefined;
  const inspectedItem =
    inspector?.kind === "activity"
      ? history.find((value) => activity(value).id === inspector.id)
      : undefined;
  const inspectedStep = inspectedItem
    ? stepView(inspectedItem, history)
    : undefined;
  const inspectedTool =
    inspectedStep?.kind === "llm.tool"
      ? toolDescription(inspectedStep, approvalFor(inspectedStep, pending))
      : undefined;
  const inspectorWidth = inspectedTool
    ? toolFormatters.resolve(inspectedTool).width
    : 420;

  function open(next: Inspector) {
    restoreFocus.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    setInspector(next);
  }
  function close() {
    setInspector(null);
    restoreFocus.current?.focus();
  }
  useEffect(() => {
    if (inspector) inspectorRef.current?.focus();
  }, [inspector?.kind]);
  useEffect(() => {
    setInspector(null);
    setNewActivity(false);
    following.current = true;
  }, [scope]);
  useEffect(() => {
    const area = scroll.current;
    if (!area) return;
    area.scrollTop = positions.current[scope] ?? area.scrollHeight;
    following.current =
      area.scrollHeight - area.scrollTop - area.clientHeight < 80;
    return () => {
      positions.current[scope] = area.scrollTop;
    };
  }, [scope]);
  useEffect(() => {
    const area = scroll.current;
    if (!area) return;
    if (following.current) area.scrollTop = area.scrollHeight;
    else setNewActivity(true);
  }, [history]);

  async function send() {
    if (!chat || !draft.trim() || !enabled) return;
    const signature = JSON.stringify([scope, chat.request_id, draft]);
    if (retry.current.signature !== signature)
      retry.current = { signature, key: crypto.randomUUID() };
    const response = inputResponse(chat, draft);
    if (await onReply(chat, response, retry.current.key)) {
      setDrafts((values) => ({
        ...values,
        [scope]: values[scope] === draft ? "" : values[scope],
      }));
    }
  }

  return (
    <main className="workspace-shell">
      {leftOpen && (
        <SessionSidebar
          sessions={sessions}
          selected={selected}
          status={status}
          select={onSelect}
          collapse={() => setLeftOpen(false)}
          create={() => open({ kind: "new" })}
          configure={() => open({ kind: "connection" })}
        />
      )}
      <section className="chat-workspace" aria-label="Conversation">
        <header className="session-header">
          <div>
            {!leftOpen && (
              <button
                className="quiet-button"
                onClick={() => setLeftOpen(true)}
              >
                Show navigation
              </button>
            )}
            <h1>{current?.reference || "Your workspace"}</h1>
            <span className="session-subtitle">
              {current
                ? `${current.status} / ${current.execution_id.slice(0, 8)}`
                : "Start a conversation to begin"}
            </span>
          </div>
          <div className="header-actions">
            <button
              className="quiet-button"
              disabled={!selected}
              onClick={() => open({ kind: "decisions" })}
            >
              Pending ({decisions.length})
            </button>
            <button
              className="quiet-button"
              disabled={!selected}
              onClick={() => open({ kind: "session" })}
            >
              Session
            </button>
            <button disabled={!enabled || !current?.active} onClick={onStop}>
              Stop
            </button>
          </div>
        </header>
        {error && (
          <div role="alert" className="workspace-error">
            <span>{error}</span>
            <button className="quiet-button" onClick={dismiss}>
              Dismiss
            </button>
          </div>
        )}
        <div className="feed-controls">
          <span>{selected ? "Conversation and activity" : "Welcome"}</span>
          {selected && (
            <button
              className="quiet-button"
              onClick={() => open({ kind: "debug" })}
            >
              Debug steps ({history.length})
            </button>
          )}
        </div>
        <div
          className="conversation-feed"
          ref={scroll}
          onScroll={() => {
            const area = scroll.current!;
            positions.current[scope] = area.scrollTop;
            following.current =
              area.scrollHeight - area.scrollTop - area.clientHeight < 80;
            if (following.current) setNewActivity(false);
          }}
        >
          {!selected ? (
            <div className="welcome-state">
              <h2>What would you like to work on?</h2>
              <p>
                Choose a workflow to start, or return to a conversation in the
                sidebar.
              </p>
              <button onClick={() => open({ kind: "new" })}>
                New conversation
              </button>
            </div>
          ) : (
            <div className="feed-content">
              {hasOlder && (
                <p className="history-boundary">
                  Showing the latest 100 persisted steps. Earlier history is not
                  shown in this view.
                </p>
              )}
              <StepFeed
                key={scope}
                history={history}
                active={Boolean(current?.active)}
                registry={stepRegistry}
                actions={{
                  tools: toolFormatters,
                  pending: decisions,
                  enabled,
                  reply: onReply,
                  inspect: (id) => open({ kind: "activity", id }),
                }}
              />
            </div>
          )}
        </div>
        {newActivity && (
          <button
            className="new-activity"
            onClick={() => {
              following.current = true;
              if (scroll.current)
                scroll.current.scrollTop = scroll.current.scrollHeight;
              setNewActivity(false);
            }}
          >
            New activity / Jump to latest
          </button>
        )}
        <Composer
          selected={Boolean(selected)}
          pendingCount={decisions.length}
          ready={Boolean(chat)}
          enabled={enabled}
          busy={busy}
          value={draft}
          change={(value) =>
            setDrafts((values) => ({ ...values, [scope]: value }))
          }
          send={send}
          review={() => open({ kind: "decisions" })}
        />
      </section>
      {inspector && (
        <button
          className="fixed inset-0 z-[19] cursor-default rounded-none border-0 p-0"
          style={{ background: "transparent" }}
          aria-label="Dismiss inspector"
          tabIndex={-1}
          onPointerDown={(event) => event.preventDefault()}
          onClick={(event) => {
            event.stopPropagation();
            close();
          }}
        />
      )}
      {inspector && (
        <aside
          className="work-inspector"
          style={{ width: `min(${inspectorWidth}px, 100vw)` }}
          ref={inspectorRef}
          tabIndex={-1}
          aria-label="Work inspector"
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              event.stopPropagation();
              close();
            }
          }}
        >
          <div className="inspector-heading">
            <h2>
              {
                {
                  activity: "Activity details",
                  debug: "Debug steps",
                  decisions: "Pending decisions",
                  session: "Session details",
                  new: "New conversation",
                  connection: "Connection",
                }[inspector.kind]
              }
            </h2>
            <button
              className="quiet-button"
              onClick={close}
              aria-label="Close inspector"
            >
              Close
            </button>
          </div>
          <div className="inspector-content">
            {inspector.kind === "debug" && <DebugSteps history={history} />}
            {inspector.kind === "new" && newSession}
            {inspector.kind === "connection" && connection}
            {inspector.kind === "session" && (
              <>
                <h3>{current?.reference}</h3>
                <dl>
                  <dt>Execution</dt>
                  <dd>{selected}</dd>
                  <dt>Backend</dt>
                  <dd>{endpoint}</dd>
                  <dt>Execution state</dt>
                  <dd>{current?.status}</dd>
                  <dt>Connection</dt>
                  <dd>{status}</dd>
                </dl>
                <p>
                  Child workflow activities are grouped beneath their spawn.
                  Expand a workflow to inspect its activity and pending input.
                </p>
                <p>
                  Stopping does not undo completed edits. Closing this panel
                  does not stop execution.
                </p>
              </>
            )}
            {inspector.kind === "activity" && (
              <ActivityDetails
                item={inspectedItem}
                history={history}
                pending={pending}
                tools={toolFormatters}
              />
            )}
            {inspector.kind === "decisions" &&
              (decisions.length ? (
                decisions.map((request) => (
                  <Decision
                    key={`${endpoint}/${request.execution_id}/${request.request_id}`}
                    request={request}
                    disabled={!enabled}
                    value={drafts[`${scope}/${request.request_id}`] || ""}
                    change={(value) =>
                      setDrafts((values) => ({
                        ...values,
                        [`${scope}/${request.request_id}`]: value,
                      }))
                    }
                    reply={(response, key) => onReply(request, response, key)}
                  />
                ))
              ) : (
                <p>
                  No pending decisions. This view updates as the workflow runs.
                </p>
              ))}
          </div>
        </aside>
      )}
    </main>
  );
}
