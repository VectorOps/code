import type { StepProps } from "./registry";
import { StepIndicators } from "./StepIndicators";

export function UserStep({ step, actions }: StepProps) {
  return (
    <section className="feed-entry user-step">
      <button
        className="activity-heading"
        onClick={() => actions.inspect(step.id)}
      >
        <span>You</span>
        <StepIndicators />
      </button>
      <div className="message-text">
        {step.message || "Message with attachments"}
      </div>
    </section>
  );
}
