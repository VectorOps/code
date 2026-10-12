import { useMemo } from "react";
import type { ToolGroup as Group } from "../models/tool-groups";
import { toolDescription } from "../models/steps";
import type { StepActions, StepRegistry } from "./steps/registry";
import { defaultToolFormatters } from "./tools/defaults";
import { useFeedExpansion } from "./FeedState";

export function ToolGroup({
  group,
  registry,
  actions,
  failuresRecovered = false,
}: {
  group: Group;
  registry: StepRegistry;
  actions: StepActions;
  failuresRecovered?: boolean;
}) {
  const [expanded, setExpanded] = useFeedExpansion(group.id);
  const factory = actions.tools ?? defaultToolFormatters;
  const tools = useMemo(
    () => group.steps.map((step) => toolDescription(step)),
    [group],
  );
  const failures = tools.filter((tool) => tool.failed);
  const running = tools.filter((tool) =>
    ["pending", "streaming"].includes(tool.state),
  );
  const latest = running.at(-1) ?? tools.at(-1);
  const Summary = latest ? factory.resolve(latest).summary : undefined;
  return (
    <div className="tool-group feed-item">
      <details
        open={expanded}
        onToggle={(event) => setExpanded(event.currentTarget.open)}
      >
        <summary>
          <span className="tool-group-heading">
            <span className="tool-activity-label">
              {factory.name(group.name)} · {tools.length} calls
            </span>
            {latest && Summary && <Summary tool={latest} />}
            {running.length > 0 && (
              <span className="tool-group-running" role="status">
                <span className="workflow-spinner" aria-hidden="true" />
                Running
              </span>
            )}
            {failures.length > 0 && (
              <span
                className="tool-failure-marker"
                title={
                  failuresRecovered
                    ? "Workflow completed after these failures"
                    : "Review failed calls"
                }
              >
                {failures.length} failed
              </span>
            )}
          </span>
        </summary>
        {expanded && (
          <div className="tool-group-calls">
            {group.steps.map((step) => {
              const Component = registry.resolve(step);
              return Component ? (
                <div className="tool-group-call" key={step.id}>
                  <Component
                    step={step}
                    actions={actions}
                    failuresRecovered={failuresRecovered}
                  />
                </div>
              ) : null;
            })}
          </div>
        )}
      </details>
    </div>
  );
}
