import { useState } from "react";
import * as history from "../models/history-limits";

const labels: Record<keyof history.HistoryLimits, string> = {
  initial: "Initial history steps",
  maximum: "History retention ceiling",
  target: "History trim target",
  older: "Earlier steps per request",
};

export function HistorySettings({
  limits,
  change,
}: {
  limits: history.HistoryLimits;
  change: (limits: history.HistoryLimits) => void;
}) {
  const [draft, setDraft] = useState(limits);
  const [error, setError] = useState("");
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        try {
          change(history.historyLimits(draft));
          setError("");
        } catch (reason) {
          setError(String(reason));
        }
      }}
    >
      <h4>Conversation history</h4>
      <p>
        Chat and debug share this window of raw steps; some steps are hidden
        from chat. Saved on this device. Trimming pauses while reviewing history
        and resumes at the latest activity.
      </p>
      {(Object.keys(labels) as (keyof history.HistoryLimits)[]).map((key) => (
        <label key={key}>
          {labels[key]}
          <input
            type="number"
            min={1}
            step={1}
            max={key === "older" ? 500 : undefined}
            value={draft[key] || ""}
            onChange={(event) =>
              setDraft({ ...draft, [key]: Number(event.target.value) })
            }
          />
        </label>
      ))}
      <button type="submit">Save history settings</button>
      <button
        type="button"
        onClick={() => {
          setDraft(history.defaultHistoryLimits);
          change(history.defaultHistoryLimits);
          setError("");
        }}
      >
        Reset history settings
      </button>
      {error && <p role="alert">{error}</p>}
    </form>
  );
}
