import { approvalFor, toolDescription } from "../../models/steps";
import { ApprovalActions } from "../ApprovalActions";
import type { StepProps } from "./registry";
import { defaultToolFormatters } from "../tools/defaults";
import "../tools/tools.css";

export function ToolStep({ step, actions }: StepProps) {
  const approval = approvalFor(step, actions.pending);
  const tool = toolDescription(step, approval);
  const factory = actions.tools ?? defaultToolFormatters;
  const formatter = factory.resolve(tool);
  if (!approval && formatter.visible?.(tool) === false) return null;
  const Summary = formatter.summary;
  return (
    <section className="feed-entry tool-step">
      <button
        className="activity-heading"
        onClick={() => actions.inspect(step.id)}
      >
        <span>{factory.name(tool.name)}</span>
        <span className="entry-state">
          {approval ? "Needs approval" : tool.failed ? "Failed" : tool.state} /
          Details
        </span>
      </button>
      <Summary tool={tool} />
      {approval && (
        <ApprovalActions
          key={approval.request_id}
          request={approval}
          enabled={actions.enabled}
          reply={actions.reply}
        />
      )}
    </section>
  );
}
