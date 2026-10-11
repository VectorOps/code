import type { ToolFormatterFactory, ToolView } from "./factory";
import { ToolError } from "./views";
import "./tools.css";

export function ToolDetails({
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
      {tool.failed && <ToolError tool={tool} />}
      <Details tool={tool} />
      <details>
        <summary>Raw call and result</summary>
        <pre>{JSON.stringify(tool, null, 2)}</pre>
      </details>
    </div>
  );
}
