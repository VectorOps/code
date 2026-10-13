import type { HistoryItem } from "../models/runner";
import { activity } from "../models/activity";
import { LazyDetails } from "./LazyDetails";
import { memo } from "react";
import { stepView } from "../models/steps";
import type { StepActions } from "./steps/registry";
import { StepBranchControls } from "./StepBranchControls";

export const DebugSteps = memo(function DebugSteps({
  history,
  actions,
}: {
  history: HistoryItem[];
  actions?: StepActions;
}) {
  return (
    <>
      <p>
        Raw steps in the loaded history window, including those hidden from
        chat.
      </p>
      {history.map((item) => {
        const step = activity(item);
        return (
          <LazyDetails
            key={step.id}
            summary={`${item.position}. ${step.kind} / ${step.state}`}
          >
            {() => (
              <>
                {actions && (
                  <StepBranchControls
                    step={stepView(item, history)}
                    actions={actions}
                  />
                )}
                <pre>{JSON.stringify(item.step, null, 2)}</pre>
              </>
            )}
          </LazyDetails>
        );
      })}
    </>
  );
});
