import { useRef, useState } from "react";

export function Composer({
  selected,
  pendingCount,
  ready,
  enabled,
  busy,
  value,
  change,
  send,
  review,
}: {
  selected: boolean;
  pendingCount: number;
  ready: boolean;
  enabled: boolean;
  busy: boolean;
  value: string;
  change: (value: string) => void;
  send: () => Promise<void>;
  review: () => void;
}) {
  const inFlight = useRef(false);
  const [sending, setSending] = useState(false);
  const canSend =
    selected && enabled && ready && !busy && !sending && Boolean(value.trim());

  async function submit() {
    if (!canSend || inFlight.current) return;
    inFlight.current = true;
    setSending(true);
    try {
      await send();
    } finally {
      inFlight.current = false;
      setSending(false);
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
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
        <label htmlFor="message-draft" className="sr-only">
          Message draft
        </label>
        <textarea
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
              void submit();
            }
          }}
        />
        <div className="composer-actions">
          <span>
            {ready
              ? "Ready for your message. Ctrl/Cmd+Enter to send."
              : selected
                ? "Draft freely. Sending is available when the workflow asks for input."
                : "One conversation per top-level workflow."}
          </span>
          <button disabled={!canSend}>
            {busy || sending ? "Sending..." : "Send"}
          </button>
        </div>
      </form>
    </footer>
  );
}
