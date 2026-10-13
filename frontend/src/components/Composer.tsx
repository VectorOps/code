import { useRef, useState } from "react";
import { flushSync } from "react-dom";
import type { Session } from "../models/runner";
import * as chatInput from "../models/chat-input";
import { ImageThumbnail } from "./ImageThumbnail";
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
  steeringCount = 0,
  reviewSteering,
  ready,
  enabled,
  busy,
  value,
  images = [],
  attach,
  removeImage,
  attachmentError = "",
  messageError = "",
  change,
  send,
  stop,
  resume,
  review,
}: {
  selected: boolean;
  session?: WorkflowState;
  pendingCount: number;
  steeringCount?: number;
  reviewSteering?: () => void;
  ready: boolean;
  enabled: boolean;
  busy: boolean;
  value: string;
  images?: readonly chatInput.ImageAttachment[];
  attach?: (files: File[]) => void;
  removeImage?: (id: string) => void;
  attachmentError?: string;
  messageError?: string;
  change: (value: string) => void;
  send: () => Promise<void>;
  stop?: () => void | Promise<void>;
  resume?: () => Promise<void>;
  review: () => void;
}) {
  const inFlight = useRef(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const form = useRef<HTMLFormElement>(null);
  const picker = useRef<HTMLInputElement>(null);
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
  const inputError = messageError || chatInput.messageError(value, images);
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
    (activity !== "send" ||
      (ready && Boolean(value.trim() || images.length) && !inputError));
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
        onPaste={(event) => {
          if (!selected || !attach) return;
          const files = Array.from(event.clipboardData.files);
          if (!files.length) {
            for (const item of Array.from(event.clipboardData.items)) {
              if (item.kind !== "file") continue;
              const file = item.getAsFile();
              if (file) files.push(file);
            }
          }
          if (!files.length) return;
          if (!event.clipboardData.getData("text/plain"))
            event.preventDefault();
          attach(files);
        }}
        onSubmit={(event) => {
          event.preventDefault();
          if (activity === "send") void submit();
        }}
      >
        {images.length > 0 && (
          <ul className="composer-images" aria-label="Attached images">
            {images.map((item) => (
              <li key={item.id}>
                {item.state === "ready" && (
                  <ImageThumbnail
                    src={chatInput.imageSource(item.image)}
                    name={item.name}
                  />
                )}
                {item.state !== "ready" && <span>{item.name}</span>}
                {item.state === "preparing" && (
                  <span role="status">Preparing…</span>
                )}
                {item.state === "failed" && (
                  <span role="alert">{item.error}</span>
                )}
                <button
                  type="button"
                  className="quiet-button composer-image-remove"
                  aria-label={`Remove ${item.name}`}
                  title={`Remove ${item.name}`}
                  onClick={() => removeImage?.(item.id)}
                >
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    aria-hidden="true"
                  >
                    <path d="m6 6 12 12M6 18 18 6" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        )}
        {(attachmentError || inputError) && (
          <p
            className="composer-image-error"
            role={
              images.some((item) => item.state === "preparing")
                ? "status"
                : "alert"
            }
          >
            {attachmentError || inputError}
          </p>
        )}
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
          {attach && (
            <>
              <input
                ref={picker}
                className="hidden"
                type="file"
                multiple
                accept={chatInput.imageAccept}
                aria-label="Choose images"
                disabled={!selected}
                onChange={(event) => {
                  const files = Array.from(event.currentTarget.files ?? []);
                  event.currentTarget.value = "";
                  if (files.length) attach(files);
                }}
              />
              <button
                type="button"
                className="quiet-button composer-attach"
                disabled={!selected}
                aria-label="Attach images"
                title="Attach images (PNG, JPEG, GIF, WebP; up to 768 KiB each)"
                onClick={() => picker.current?.click()}
              >
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.7"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                </svg>
              </button>
            </>
          )}
          <span role={operating ? "status" : undefined}>
            {operating
              ? `${progressLabels[operating]}…`
              : activity === "send" && ready
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
          {steeringCount > 0 && reviewSteering && (
            <button
              type="button"
              className="steering-queue-button"
              aria-label={`${steeringCount} queued steering messages`}
              onClick={reviewSteering}
            >
              Steering · {steeringCount}
            </button>
          )}
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
