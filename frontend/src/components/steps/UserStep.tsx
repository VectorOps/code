import type { StepProps } from "./registry";
import { StepIndicators } from "./StepIndicators";
import { memo } from "react";

export const UserStep = memo(function UserStep({ step, actions }: StepProps) {
  return (
    <section className="feed-entry user-step">
      <div className="assistant-controls">
        <StepIndicators inspect={() => actions.inspect(step.id)} />
      </div>
      <div className="message-text">
        {step.message || "Message with attachments"}
      </div>
    </section>
  );
});
