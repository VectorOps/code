import { useState } from "react";
import type { ToolGroup as Group } from "../models/tool-groups";
import { toolDescription } from "../models/steps";
import type { StepActions, StepRegistry } from "./steps/registry";
import { defaultToolFormatters } from "./tools/defaults";

export function ToolGroup({
  group,
  registry,
  actions,
}: {
  group: Group;
  registry: StepRegistry;
  actions: StepActions;
}) {
  const [expanded, setExpanded] = useState(false);
  const factory = actions.tools ?? defaultToolFormatters;
  const tools = group.steps.map((step) => toolDescription(step));
  const mixed = new Set(tools.map((tool) => tool.name)).size > 1;
  const previewStart = Math.max(0, tools.length - 2);
  const states = new Map<string, number>();
  for (const tool of tools) {
    const state = tool.failed ? "Failed" : tool.state;
    states.set(state, (states.get(state) ?? 0) + 1);
  }
  return (
    <details
      className="tool-group"
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>
        <span>
          {factory.name(group.name)} ({tools.length} calls)
        </span>
        <span className="entry-state">
          {Array.from(states, ([state, count]) => `${count} ${state}`).join(
            " / ",
          )}
        </span>
        {!expanded && (
          <div className="tool-group-preview">
            {previewStart > 0 && (
              <div className="tool-group-overflow">
                ... {previewStart} earlier{" "}
                {previewStart === 1 ? "call" : "calls"}
              </div>
            )}
            {tools.slice(previewStart).map((tool, index) => {
              const Summary = factory.resolve(tool).summary;
              return (
                <div
                  className="tool-group-preview-call"
                  key={group.steps[previewStart + index].id}
                >
                  {mixed && (
                    <span className="entry-state">
                      {factory.name(tool.name)}
                    </span>
                  )}
                  <Summary tool={tool} />
                </div>
              );
            })}
          </div>
        )}
      </summary>
      {expanded && (
        <div className="tool-group-calls">
          {group.steps.map((step) => {
            const Component = registry.resolve(step);
            return Component ? (
              <div className="tool-group-call" key={step.id}>
                <Component step={step} actions={actions} />
              </div>
            ) : null;
          })}
        </div>
      )}
    </details>
  );
}
