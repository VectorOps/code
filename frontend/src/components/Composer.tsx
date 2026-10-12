import { useRef, useState } from "react";
import { flushSync } from "react-dom";
import type { Session } from "../models/runner";
import { useComposerTarget, useFocusCoordinator } from "../focus/FocusProvider";

type Activity = "send" | "stop" | "resume" | "review" | "idle";
type WorkflowState = Pick<Session, "status" | "active" | "pending_inputs">;

function workflowActivity(
  selected: boolean,
  ready: boolean,
  pendingCount: number,
  session?: WorkflowState,
): Activity {
  if (!selected) return "idle";
  if (session?.status === "stopping") return "stop";
  if (ready) return "send";
  if (session?.status === "paused") return pendingCount > 0 ? "review" : "send";
  if (session?.active) return "stop";
  if (pendingCount > 0) return "review";
  if (session?.status === "cancelled" && session.pending_inputs === 0)
    return "resume";
  return "idle";
}

const activityLabels: Record<Activity, string> = {
  send: "Send message",
  stop: "Stop workflow",
  resume: "Start workflow",
  review: "Review pending input",
  idle: "No workflow action available",
};

const progressLabels: Record<Activity, string> = {
  send: "Sending message",
  stop: "Stopping workflow",
  resume: "Starting workflow",
  review: "Opening pending input",
  idle: "No workflow action available",
};

export function Composer({
  selected,
  session,
  pendingCount,
  ready,
  enabled,
  busy,
  value,
  change,
  send,
  stop,
  resume,
  review,
}: {
  selected: boolean;
  session?: WorkflowState;
  pendingCount: number;
  ready: boolean;
  enabled: boolean;
  busy: boolean;
  value: string;
  change: (value: string) => void;
  send: () => Promise<void>;
  stop?: () => void | Promise<void>;
  resume?: () => Promise<void>;
  review: () => void;
}) {
  const inFlight = useRef(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const form = useRef<HTMLFormElement>(null);
  const coordinator = useFocusCoordinator();
  useComposerTarget(textarea, {
    isEditable: () =>
      Boolean(
        textarea.current &&
        !textarea.current.disabled &&
        !textarea.current.readOnly,
      ),
    focus: () => textarea.current?.focus({ preventScroll: true }),
    beginTyping: (text) => {
      flushSync(() => change(value + text));
      const element = textarea.current;
      if (!element) return;
      element.focus({ preventScroll: true });
      element.setSelectionRange(element.value.length, element.value.length);
    },
  });
  const [operating, setOperating] = useState<Activity | null>(null);
  const activity = workflowActivity(selected, ready, pendingCount, session);
  const callback =
    activity === "send"
      ? send
      : activity === "stop"
        ? stop
        : activity === "resume"
          ? resume
          : activity === "review"
            ? review
            : undefined;
  const canAct =
    enabled &&
    !busy &&
    !operating &&
    Boolean(callback) &&
    session?.status !== "stopping" &&
    (activity !== "send" || (ready && Boolean(value.trim())));
  const displayedActivity = operating ?? activity;
  const label = operating
    ? progressLabels[operating]
    : session?.status === "stopping"
      ? progressLabels.stop
      : activityLabels[activity];

  async function submit() {
    if (!canAct || !callback || inFlight.current) return;
    const recover =
      activity === "send" && form.current
        ? coordinator?.captureComposerRecovery(form.current)
        : undefined;
    inFlight.current = true;
    setOperating(activity);
    try {
      await callback();
      recover?.();
    } finally {
      inFlight.current = false;
      setOperating(null);
    }
  }

  return (
    <footer className="composer-area">
      {pendingCount > 0 && (
        <button className="pending-banner" onClick={review}>
          {pendingCount} pending {pendingCount === 1 ? "response" : "responses"}{" "}
          / Review
        </button>
      )}
      <form
        ref={form}
        onSubmit={(event) => {
          event.preventDefault();
          if (activity === "send") void submit();
        }}
      >
        <label htmlFor="message-draft" className="sr-only">
          Message draft
        </label>
        <textarea
          ref={textarea}
          id="message-draft"
          rows={3}
          placeholder={
            selected
              ? "Write an instruction..."
              : "Select a conversation to write a message"
          }
          disabled={!selected}
          value={value}
          onChange={(event) => change(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              (event.ctrlKey || event.metaKey) &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              if (activity === "send") void submit();
            }
          }}
        />
        <div className="composer-actions">
          <span>
            {activity === "send" && ready
              ? "Ready for your message. Ctrl/Cmd+Enter to send."
              : activity === "stop"
                ? "Draft freely while the workflow runs."
                : activity === "resume"
                  ? "Start the stopped workflow to continue. Your draft will be kept."
                  : activity === "review"
                    ? "Review the pending input to continue."
                    : selected
                      ? "Draft freely. Sending is available when the workflow asks for input."
                      : "One conversation per top-level workflow."}
          </span>
          <button
            className="composer-submit"
            type={activity === "send" ? "submit" : "button"}
            onClick={activity === "send" ? undefined : () => void submit()}
            disabled={!canAct}
            aria-label={label}
            title={label}
            aria-busy={
              Boolean(operating) || busy || session?.status === "stopping"
            }
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="currentColor"
              aria-hidden="true"
            >
              {displayedActivity === "stop" ? (
                <rect x="6" y="6" width="12" height="12" rx="1" />
              ) : (
                <path d="M8 5v14l11-7z" />
              )}
            </svg>
          </button>
        </div>
      </form>
    </footer>
  );
}
