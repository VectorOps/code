import type { ToolFormatterFactory, ToolView } from "./factory";
import { ToolError } from "./views";
import { LazyDetails } from "../LazyDetails";
import "./tools.css";
import { memo } from "react";

export const ToolDetails = memo(function ToolDetails({
  tool,
  factory,
}: {
  tool: ToolView;
  factory: ToolFormatterFactory;
}) {
  const Details = factory.resolve(tool).details;
  return (
    <div className="tool-details">
      <h3>{factory.name(tool.name)}</h3>
      <p className="entry-state">{tool.failed ? "Failed" : tool.state}</p>
      {tool.authorizationRuleId && (
        <p>
          Authorization rule: <code>{tool.authorizationRuleId}</code>
          <br />
          <small>
            The referenced rule may have been deleted since this call.
          </small>
        </p>
      )}
      {tool.failed && <ToolError tool={tool} />}
      <Details tool={tool} />
      <LazyDetails summary="Raw call and result">
        {() => <pre>{JSON.stringify(tool, null, 2)}</pre>}
      </LazyDetails>
    </div>
  );
});
