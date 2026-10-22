import { memo } from "react";
import { object, text } from "../../models/json";
import { MarkdownBody } from "../MarkdownBody";
import { CodeBlock } from "../CodeBlock";
import { useStepBranchControls } from "../StepBranchControls";
import { StepCard } from "./StepCard";
import { StepIndicators } from "./StepIndicators";
import type { StepProps } from "./registry";

export const EchoStep = memo(function EchoStep({ step, actions }: StepProps) {
  const branch = useStepBranchControls(step, actions);
  const value = step.state === "completed" ? step.output : step.input.value;
  return (
    <StepCard
      className="assistant-step echo-step"
      busy={step.state === "pending"}
      inspect={() => actions.inspect(step.id)}
    >
      <div className="assistant-controls">
        <StepIndicators state={step.state} />
        {branch.icons}
      </div>
      {typeof value === "string" ? (
        <MarkdownBody message={value} />
      ) : value !== undefined ? (
        <CodeBlock>{JSON.stringify(value, null, 2)}</CodeBlock>
      ) : null}
      {step.state === "failed" && (
        <p className="failure-text">
          {text(object(step.outcome.error).message) ||
            "The echo step could not finish."}
        </p>
      )}
      {branch.editor}
    </StepCard>
  );
});
