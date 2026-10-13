import { approvalFor, toolDescription } from "../../models/steps";
import { ApprovalActions } from "../ApprovalActions";
import type { StepProps } from "./registry";
import { defaultToolFormatters } from "../tools/defaults";
import { StepIndicators } from "./StepIndicators";
import { StepCard } from "./StepCard";
import "../tools/tools.css";
import { memo } from "react";
import { useStepBranchControls } from "../StepBranchControls";

export const ToolStep = memo(function ToolStep({ step, actions }: StepProps) {
  const approval = approvalFor(step, actions.pending);
  const tool = toolDescription(step, approval);
  const factory = actions.tools ?? defaultToolFormatters;
  const formatter = factory.resolve(tool);
  const branch = useStepBranchControls(step, actions);
  if (!approval && formatter.visible?.(tool) === false) return null;
  const Summary = formatter.summary;
  return (
    <StepCard
      className={`tool-step${approval ? " tool-approval" : ""}`}
      inspect={() => actions.inspect(step.id)}
      busy={
        !approval && (tool.state === "pending" || tool.state === "streaming")
      }
    >
      <div className="tool-activity-row">
        <span className="tool-activity-label">
          {approval
            ? `Approve ${factory.name(tool.name)}`
            : factory.activity(tool)}
        </span>
        <Summary tool={tool} />
        {(!formatter.hideIndicators ||
          approval ||
          tool.failed ||
          branch.icons) && (
          <span className="tool-step-controls inline-flex shrink-0 items-center gap-2">
            {(!formatter.hideIndicators || approval || tool.failed) && (
              <StepIndicators
                state={
                  approval
                    ? "Needs approval"
                    : tool.failed
                      ? "Failed"
                      : tool.state
                }
              />
            )}
            {branch.icons}
          </span>
        )}
      </div>
      {approval && (
        <ApprovalActions
          key={approval.request_id}
          request={approval}
          enabled={actions.enabled}
          reply={actions.reply}
        />
      )}
      {branch.editor}
    </StepCard>
  );
});
