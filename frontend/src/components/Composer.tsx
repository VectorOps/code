import { useRef, useState } from "react";
import type { HistoryItem, Session } from "../models/runner";
import * as chatInput from "../models/chat-input";
import { ImageThumbnail } from "./ImageThumbnail";
import { ComposerUsage } from "./ComposerUsage";
import { PromptTextarea } from "./PromptTextarea";
import type { PromptHistoryApi } from "../communication/prompt-history";
import type { FileSearchApi } from "../communication/file-search";
import { useFocusCoordinator } from "../focus/FocusProvider";

type Activity = "send" | "stop" | "resume" | "review" | "idle";
type WorkflowState = Pick<
  Session,
  "status" | "active" | "pending_inputs" | "usage"
>;

function workflowActivity(
  selected: boolean,
  ready: boolean,
  sendable: boolean,
  pendingCount: number,
  session?: WorkflowState,
): Activity {
  if (!selected) return "idle";
  if (session?.status === "stopping") return "stop";
  if (ready && (sendable || !session?.active || session.pending_inputs > 0))
    return "send";
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
  history = [],
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
  promptHistory,
  fileSearch,
  executionId,
  scope,
}: {
  selected: boolean;
  session?: WorkflowState;
  history?: HistoryItem[];
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
  promptHistory?: PromptHistoryApi;
  fileSearch?: FileSearchApi;
  executionId?: string;
  scope?: string;
}) {
  const inFlight = useRef(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const form = useRef<HTMLFormElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  const coordinator = useFocusCoordinator();
  const [operating, setOperating] = useState<Activity | null>(null);
  const inputError = messageError || chatInput.messageError(value, images);
  const activity = workflowActivity(
    selected,
    ready,
    Boolean(value.trim() || images.length) && !inputError,
    pendingCount,
    session,
  );
  const callbacks = { send, stop, resume, review, idle: undefined };
  const available =
    selected &&
    enabled &&
    !busy &&
    !operating &&
    session?.status !== "stopping";
  function canAct(next: Activity) {
    return (
      available &&
      Boolean(callbacks[next]) &&
      (next === "send"
        ? ready && Boolean(value.trim() || images.length) && !inputError
        : next === "stop"
          ? Boolean(session?.active)
          : next === activity)
    );
  }
  const displayedActivity = operating ?? activity;
  const label = operating
    ? progressLabels[operating]
    : session?.status === "stopping"
      ? progressLabels.stop
      : activityLabels[activity];

  async function submit(next: Activity = activity) {
    const callback = callbacks[next];
    if (!canAct(next) || !callback || inFlight.current) return;
    const recover =
      next === "send" && form.current
        ? coordinator?.captureComposerRecovery(form.current)
        : undefined;
    inFlight.current = true;
    setOperating(next);
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
        <PromptTextarea
          inputRef={textarea}
          attach={attach}
          selected={selected}
          value={value}
          change={change}
          historyApi={promptHistory}
          fileSearch={fileSearch}
          executionId={executionId}
          scope={scope}
          submit={() => {
            if (activity === "send") void submit();
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
          <span
            className="composer-help"
            role={operating ? "status" : undefined}
          >
            {operating
              ? `${progressLabels[operating]}…`
              : activity === "send" && ready
                ? `Enter to send · Modifier+Enter for newline · Ctrl+R for history${fileSearch ? " · @ for files" : ""}`
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
          {selected && session && (
            <ComposerUsage session={session} history={history} />
          )}
          <button
            className="composer-submit"
            type={activity === "send" ? "submit" : "button"}
            onClick={activity === "send" ? undefined : () => void submit()}
            disabled={!canAct(activity)}
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
