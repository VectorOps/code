import type { HistoryItem } from "../models/runner";
import { activity } from "../models/activity";

export function DebugSteps({ history }: { history: HistoryItem[] }) {
  return (
    <>
      <p>
        Raw steps in the loaded history window, including those hidden from
        chat.
      </p>
      {history.map((item) => {
        const step = activity(item);
        return (
          <details key={step.id}>
            <summary>
              {item.position}. {step.kind} / {step.state}
            </summary>
            <pre>{JSON.stringify(item.step, null, 2)}</pre>
          </details>
        );
      })}
    </>
  );
}
