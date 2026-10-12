import { activity } from "../models/activity";
import type { HistoryItem, InputRequest } from "../models/runner";
import { approvalFor, stepView, toolDescription } from "../models/steps";
import { defaultToolFormatters } from "./tools/defaults";
import type { ToolFormatterFactory } from "./tools/factory";
import { ToolDetails } from "./tools/ToolDetails";
import { LazyDetails } from "./LazyDetails";
import { memo } from "react";

export const ActivityDetails = memo(function ActivityDetails({
  item,
  pending = [],
  history = [],
  tools = defaultToolFormatters,
}: {
  item?: HistoryItem;
  pending?: InputRequest[];
  history?: HistoryItem[];
  tools?: ToolFormatterFactory;
}) {
  if (!item) return <p>This activity is outside the current history window.</p>;
  const entry = activity(item);
  if (entry.kind === "llm.tool") {
    const step = stepView(item, history);
    return (
      <ToolDetails
        tool={toolDescription(step, approvalFor(step, pending))}
        factory={tools}
      />
    );
  }
  return (
    <>
      <h3>{entry.name || entry.kind}</h3>
      <dl>
        <dt>State</dt>
        <dd>{entry.state}</dd>
        <dt>Workflow</dt>
        <dd>{entry.workflow}</dd>
        <dt>Step</dt>
        <dd>{entry.id}</dd>
      </dl>
      <h3>Action and arguments</h3>
      <pre>{JSON.stringify(entry.operation, null, 2)}</pre>
      <h3>Result</h3>
      <pre>{JSON.stringify(entry.outcome, null, 2)}</pre>
      <LazyDetails summary="Complete record">
        {() => <pre>{JSON.stringify(item.step, null, 2)}</pre>}
      </LazyDetails>
    </>
  );
});
