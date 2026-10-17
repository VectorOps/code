import * as React from "react";
import type { Json } from "../models/json";
import * as models from "../models/runner";
import { workflowProjection } from "../models/workflow-tree";
import { selectedHistory } from "../models/selected-history";
import { activity, messageText } from "../models/activity";
import { StepFeed } from "../components/StepFeed";
import { DebugSteps } from "../components/DebugSteps";
import { defaultStepRegistry } from "../components/steps/defaults";
import type { StepRegistry } from "../components/steps/registry";
import { ActivityDetails } from "../components/ActivityDetails";
import { Decision } from "../components/Decision";
import { SessionSidebar } from "../components/SessionSidebar";
import { Composer } from "../components/Composer";
import { useConversationScroll } from "../hooks/useConversationScroll";
import { useImageAttachments } from "../hooks/useImageAttachments";
import * as chatInput from "../models/chat-input";
import type { PromptHistoryApi } from "../communication/prompt-history";
import type { FileSearchApi } from "../communication/file-search";
import { defaultToolFormatters } from "../components/tools/defaults";
import type { ToolFormatterFactory } from "../components/tools/factory";
import { approvalFor, stepView, toolDescription } from "../models/steps";
import {
  FocusProvider,
  useFocusBlocker,
  useFocusRegion,
} from "../focus/FocusProvider";
type Inspector =
  | { kind: "debug" }
  | { kind: "activity"; id: string }
  | { kind: "decisions" }
  | { kind: "steering" }
  | { kind: "session" }
  | { kind: "new" }
  | { kind: "settings" }
  | { kind: "connection" };

export function Workspace(
  props: React.ComponentProps<typeof WorkspaceContent>,
) {
  return (
    <FocusProvider scope={`${props.endpoint}/${props.selected}`}>
      <WorkspaceContent {...props} />
    </FocusProvider>
  );
}

function WorkspaceContent({
  sessions,
  activitySessions = new Set<string>(),
  workflows = [],
  selected,
  onSelect,
  onCreate,
  history,
  execution,
  branching = false,
  pending,
  steering,
  status,
  endpoint,
  enabled,
  busy,
  onStop,
  onResume,
  onReply,
  onSteer,
  onBranch,
  newSession,
  connection,
  settings,
  error,
  dismiss,
  hasOlder,
  stepRegistry = defaultStepRegistry,
  toolFormatters = defaultToolFormatters,
  promptHistory,
  fileSearch,
}: {
  toolFormatters?: ToolFormatterFactory;
  stepRegistry?: StepRegistry;
  sessions: models.Session[];
  activitySessions?: Set<string>;
  workflows?: models.Workflow[];
  selected: string;
  onSelect: (id: string) => void;
  onCreate?: () => Promise<boolean>;
  history: models.HistoryItem[];
  execution?: models.Execution;
  branching?: boolean;
  pending: models.InputRequest[];
  steering?: {
    address: models.InboxAddress;
    messages: models.InboxMessage[];
    count: number;
  };
  status: string;
  endpoint: string;
  enabled: boolean;
  busy: boolean;
  onStop: () => void | Promise<void>;
  onResume?: () => Promise<void>;
  onReply: (
    request: models.InputRequest,
    response: Json,
    key: string,
  ) => Promise<boolean>;
  onSteer?: (value: string) => Promise<boolean>;
  onBranch?: (id: string, action: models.BranchAction) => Promise<boolean>;
  newSession: React.ReactNode;
  connection: React.ReactNode;
  settings?: React.ReactNode;
  error: string;
  dismiss: () => void;
  hasOlder: boolean;
  promptHistory?: PromptHistoryApi;
  fileSearch?: FileSearchApi;
}) {
  const [leftOpen, setLeftOpen] = React.useState(true);
  const [inspector, setInspector] = React.useState<Inspector | null>(null);
  const [drafts, setDrafts] = React.useState<Record<string, string>>({});
  const [historyError, setHistoryError] = React.useState("");
  const inspectorRef = React.useRef<HTMLElement>(null);
  const workspaceRef = useFocusRegion<HTMLElement>("workspace");
  useFocusBlocker(inspector !== null, true);
  const retry = React.useRef({ signature: "", key: "" });
  const current = sessions.find((session) => session.execution_id === selected);
  const currentName = models.workflowDisplayName(
    current?.reference ?? null,
    workflows,
    current?.name,
  );
  const scope = `${endpoint}/${selected}`;
  const draft = drafts[scope] || "";
  const attachments = useImageAttachments(scope);
  const conversationHistory = React.useMemo(
    () => (execution ? selectedHistory(history, execution) : history),
    [history, execution],
  );
  const projection = React.useMemo(
    () => workflowProjection(conversationHistory, pending),
    [conversationHistory, pending],
  );
  const { chat: chatRequests, decisions } = projection;
  const feedRevision = React.useMemo(
    () => ({ history: conversationHistory, pending }),
    [conversationHistory, pending],
  );
  const feed = useConversationScroll(scope, feedRevision);
  const chat = chatRequests.length === 1 ? chatRequests[0] : undefined;
  const canSteer = chatRequests.length === 0 && Boolean(steering);
  const messageError = chatInput.messageError(
    draft,
    attachments.images,
    chat ? (chat.request_type ?? "text") : undefined,
  );
  const steeringError =
    canSteer && attachments.images.length
      ? "Steering messages support text only. Remove the images to send."
      : messageError;
  const inspectedItem = React.useMemo(
    () =>
      inspector?.kind === "activity"
        ? history.find((value) => activity(value).id === inspector.id)
        : undefined,
    [history, inspector],
  );
  const inspectionHistory =
    inspectedItem && conversationHistory.includes(inspectedItem)
      ? conversationHistory
      : history;
  const inspectedStep = inspectedItem
    ? stepView(inspectedItem, inspectionHistory)
    : undefined;
  const inspectedTool =
    inspectedStep?.kind === "llm.tool"
      ? toolDescription(inspectedStep, approvalFor(inspectedStep, pending))
      : undefined;
  const inspectorWidth =
    inspector?.kind === "settings"
      ? 850
      : inspectedTool
        ? toolFormatters.resolve(inspectedTool).width
        : 420;

  const open = React.useCallback((next: Inspector) => {
    setInspector(next);
  }, []);
  const inspect = React.useCallback(
    (id: string) => open({ kind: "activity", id }),
    [open],
  );
  const actions = React.useMemo(
    () => ({
      tools: toolFormatters,
      pending: decisions,
      enabled,
      reply: onReply,
      inspect,
      rootWorkflowId:
        execution?.root_workflow_id ?? current?.root_workflow_id ?? undefined,
      branch: onBranch,
    }),
    [
      toolFormatters,
      decisions,
      enabled,
      onReply,
      inspect,
      onBranch,
      execution?.root_workflow_id,
      current?.root_workflow_id,
    ],
  );
  function close() {
    setInspector(null);
  }
  React.useEffect(() => {
    if (inspector) inspectorRef.current?.focus();
  }, [inspector?.kind]);
  React.useEffect(() => {
    setInspector(null);
  }, [scope]);

  const create = React.useCallback(() => {
    if (!enabled) return;
    if (onCreate) {
      setInspector(null);
      void onCreate();
    } else open({ kind: "new" });
  }, [enabled, onCreate, open]);

  React.useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.altKey ||
        !(event.metaKey || event.ctrlKey) ||
        event.key.toLowerCase() !== "n" ||
        event.isComposing ||
        event.keyCode === 229
      )
        return;
      event.preventDefault();
      if (event.repeat) return;
      if (event.shiftKey) open({ kind: "new" });
      else create();
    };
    document.addEventListener("keydown", keydown);
    return () => document.removeEventListener("keydown", keydown);
  }, [create, open]);

  async function send() {
    if (
      (!chat && !canSteer) ||
      (!draft.trim() && !attachments.images.length) ||
      !enabled ||
      steeringError
    )
      return;
    const images = attachments.images.flatMap((item) =>
      item.state === "ready" ? [item.image] : [],
    );
    if (!chat) {
      const historyKey = crypto.randomUUID();
      if (onSteer && (await onSteer(draft))) {
        setDrafts((values) => ({
          ...values,
          [scope]: values[scope] === draft ? "" : values[scope],
        }));
        remember(draft, historyKey);
      }
      return;
    }
    const response = models.inputResponse(chat, draft, images);
    const submittedDraft = draft;
    const signature = JSON.stringify([scope, chat.request_id, response]);
    if (retry.current.signature !== signature)
      retry.current = { signature, key: crypto.randomUUID() };
    const key = retry.current.key;
    if (await onReply(chat, response, key)) {
      attachments.remove(
        attachments.images.map((image) => image.id),
        scope,
      );
      setDrafts((values) => ({
        ...values,
        [scope]: values[scope] === submittedDraft ? "" : values[scope],
      }));
      remember(submittedDraft, key);
    }
  }

  function remember(text: string, key: string) {
    if (!promptHistory || !text.trim()) return;
    void promptHistory.append(text, selected, key).then(
      () => setHistoryError(""),
      () =>
        setHistoryError(
          "Message sent, but prompt history could not be saved on the host.",
        ),
    );
  }

  return (
    <main className="workspace-shell" ref={workspaceRef}>
      <SessionSidebar
        collapsed={!leftOpen}
        sessions={sessions}
        activitySessions={activitySessions}
        workflows={workflows}
        selected={selected}
        status={status}
        select={onSelect}
        collapse={() => setLeftOpen((value) => !value)}
        create={create}
        advanced={() => open({ kind: "new" })}
        enabled={enabled}
        configure={() => open({ kind: "connection" })}
      />
      <section className="chat-workspace" aria-label="Conversation">
        <header className="session-header">
          <div>
            <h1>{current ? currentName : "Your workspace"}</h1>
            <span className="session-subtitle">
              {current
                ? `${current.status} / ${current.execution_id.slice(0, 8)}`
                : "Start a conversation to begin"}
            </span>
          </div>
          <div className="header-actions">
            <button
              className="quiet-button"
              aria-label="Settings"
              title="Settings"
              onClick={() => open({ kind: "settings" })}
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
                <path d="m9 3-.5 2-2 1-2-.5-2 3 1.5 1.5v3L2.5 15l2 3 2-.5 2 1 .5 2h4l.5-2 2-1 2 .5 2-3-1.5-2v-3L20 8.5l-2-3-2 .5-2-1-.5-2Z" />
                <circle cx="11.5" cy="12" r="3" />
              </svg>
            </button>
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
          ref={feed.viewport}
          role="region"
          aria-label="Conversation history"
        >
          {!selected ? (
            <div className="welcome-state" ref={feed.content}>
              <h2>What would you like to work on?</h2>
              <p>
                Start with your default workflow, or return to a conversation in
                the sidebar.
              </p>
              <button disabled={!enabled} onClick={create}>
                New conversation
              </button>
              <button
                className="quiet-button"
                onClick={() => open({ kind: "new" })}
              >
                Advanced
              </button>
            </div>
          ) : (
            <div className="feed-content" ref={feed.content}>
              {hasOlder && (
                <p className="history-boundary">
                  Showing the latest 100 persisted steps. Earlier history is not
                  shown in this view.
                </p>
              )}
              <StepFeed
                key={scope}
                history={conversationHistory}
                active={Boolean(current?.active)}
                registry={stepRegistry}
                actions={actions}
                projection={projection}
                onReview={feed.pauseFollowing}
              />
              {branching && (
                <p role="status">
                  Branch accepted. The workflow will continue automatically.
                </p>
              )}
              {current?.active &&
                current.status === "running" &&
                current.pending_inputs === 0 &&
                pending.length === 0 && (
                  <p className="running-indicator" role="status">
                    Running...
                  </p>
                )}
            </div>
          )}
        </div>
        {feed.newActivity && (
          <button className="new-activity" onClick={feed.jumpToLatest}>
            New activity / Jump to latest
          </button>
        )}
        <Composer
          selected={Boolean(selected)}
          promptHistory={promptHistory}
          fileSearch={fileSearch}
          executionId={selected}
          scope={scope}
          session={current}
          history={history}
          pendingCount={decisions.length}
          steeringCount={steering?.count ?? 0}
          reviewSteering={() => open({ kind: "steering" })}
          ready={Boolean(chat || canSteer)}
          enabled={enabled}
          busy={busy}
          value={draft}
          images={attachments.images}
          attach={attachments.add}
          removeImage={(id) => attachments.remove([id])}
          attachmentError={attachments.error}
          messageError={steeringError}
          change={(value) =>
            setDrafts((values) => ({ ...values, [scope]: value }))
          }
          send={send}
          stop={onStop}
          resume={onResume}
          review={() => open({ kind: "decisions" })}
        />
        {historyError && (
          <p role="alert" className="prompt-history-error">
            {historyError}
          </p>
        )}
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
                  steering: "Queued steering messages",
                  session: "Session details",
                  new: "Advanced new conversation",
                  connection: "Connection",
                  settings: "Settings",
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
            {inspector.kind === "debug" && (
              <DebugSteps history={history} actions={actions} />
            )}
            {inspector.kind === "new" && newSession}
            {inspector.kind === "connection" && connection}
            {inspector.kind === "settings" && settings}
            {inspector.kind === "session" && (
              <>
                <h3>{currentName}</h3>
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
                history={inspectionHistory}
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
            {inspector.kind === "steering" &&
              (steering?.messages.length ? (
                steering.messages.map((message) => (
                  <article className="steering-message" key={message.id}>
                    <p>{messageText(message.payload)}</p>
                    <small>{message.created_at}</small>
                  </article>
                ))
              ) : (
                <p>No queued steering messages.</p>
              ))}
          </div>
        </aside>
      )}
    </main>
  );
}
