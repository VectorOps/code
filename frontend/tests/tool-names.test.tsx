import { expect, test } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { defaultToolFormatters } from "../src/components/tools/defaults";
import { ToolDetails } from "../src/components/tools/ToolDetails";
import { ToolStep } from "../src/components/steps/ToolStep";
import { stepView, toolDescription } from "../src/models/steps";

test("pending legacy tools use the recorded tool name without guessing arguments", () => {
  const step = stepView({
    position: 1,
    step: {
      id: "pending",
      workflow_id: "root",
      operation: { kind: "llm.tool", input: { tool_name: "exec" } },
      outcome: { kind: "pending" },
    },
  });
  const tool = toolDescription(step);
  expect(tool.name).toBe("exec");
  expect(tool.arguments).toBeUndefined();
});

test.each([
  ["mcp_github_search_issues_0123456789abcdef0123", "MCP Github Search Issues"],
  ["mcp_discover_0123456789abcdef01234567", "MCP Discover"],
  ["mcp_prompt_0123456789abcdef0123456789abcdef", "MCP Prompt"],
  ["mcp_resource_0123456789abcdef0123456789abcdef", "MCP Resource"],
  ["mcp_github_search_issues", "MCP Github Search Issues"],
  ["mcp_github_lookup_deadbeef", "MCP Github Lookup Deadbeef"],
  ["custom_tool_0123456789abcdef0123", "Custom Tool 0123456789abcdef0123"],
  [
    "mcp_github_lookup_zzzzzzzzzzzzzzzzzzzz",
    "MCP Github Lookup Zzzzzzzzzzzzzzzzzzzz",
  ],
  ["read_file", "Read File"],
])("formats %s without exposing generated MCP digests", (name, expected) => {
  expect(defaultToolFormatters.name(name)).toBe(expected);
});

test("MCP call labels and details hide digests without changing tool identity", () => {
  const name = "mcp_github_search_issues_0123456789abcdef0123";
  const step = stepView({
    position: 1,
    step: {
      id: "call-1",
      workflow_id: "root",
      operation: {
        kind: "llm.tool",
        input: { call: { name, arguments: { query: "open bugs" } } },
      },
      outcome: { kind: "completed", result: { tool_name: name } },
    },
  });
  const tool = toolDescription(step);
  const call = renderToStaticMarkup(
    <ToolStep
      step={step}
      actions={{
        pending: [],
        enabled: true,
        inspect: () => {},
        reply: async () => true,
      }}
    />,
  );
  const details = renderToStaticMarkup(
    <ToolDetails tool={tool} factory={defaultToolFormatters} />,
  );
  for (const markup of [call, details]) {
    expect(markup).toContain("MCP Github Search Issues");
    expect(markup).not.toContain("0123456789abcdef0123");
    expect(markup).not.toContain("Mcp");
  }
  expect(defaultToolFormatters.activity({ ...tool, state: "Authorized" })).toBe(
    "Authorized MCP Github Search Issues",
  );
  expect(defaultToolFormatters.activity({ ...tool, state: "cancelled" })).toBe(
    "MCP Github Search Issues cancelled",
  );
  expect(tool.name).toBe(name);
});
