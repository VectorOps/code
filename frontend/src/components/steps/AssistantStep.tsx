import { memo } from "react";
import { MarkdownBody } from "../MarkdownBody";
import { object, text } from "../../models/json";
import type { StepProps } from "./registry";
import { StepIndicators } from "./StepIndicators";
import { StepCard } from "./StepCard";
import { useStepBranchControls } from "../StepBranchControls";

export const AssistantStep = memo(function AssistantStep({
  step,
  actions,
}: StepProps) {
  const streaming = step.state === "streaming" || step.state === "pending";
  const branch = useStepBranchControls(step, actions);
  return (
    <StepCard
      className="assistant-step"
      busy={streaming}
      inspect={() => actions.inspect(step.id)}
    >
      <div className="assistant-controls">
        {streaming && <span>{step.message ? "Writing" : "Thinking"}</span>}
        <StepIndicators state={step.state} />
        {branch.icons}
      </div>
      <MarkdownBody message={step.message} />
      {step.state === "failed" && (
        <p className="failure-text">
          {text(object(step.outcome.error).message) ||
            "The agent could not finish its response."}
        </p>
      )}
      {branch.editor}
    </StepCard>
  );
});
