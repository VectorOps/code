import type { StepProps } from "./registry";

export function UserStep({ step, actions }: StepProps) {
  return (
    <section className="feed-entry user-step">
      <button
        className="activity-heading"
        onClick={() => actions.inspect(step.id)}
      >
        <span>You</span>
        <span className="entry-state">Details</span>
      </button>
      <div className="message-text">
        {step.message || "Message with attachments"}
      </div>
    </section>
  );
}
