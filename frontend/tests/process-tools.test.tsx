import { DebugProvider } from "../src/components/Diagnostics";
import assert from "node:assert/strict";
import { test } from "vitest";
import { fireEvent, render } from "@testing-library/react";
import { renderToStaticMarkup } from "react-dom/server";
import type { Json } from "../src/models/json";
import type { HistoryItem } from "../src/models/runner";
import { stepView, toolDescription } from "../src/models/steps";
import { ProcessDetails, ProcessSummary } from "../src/components/tools/views";
import { ToolGroup } from "../src/components/ToolGroup";
import { StepRegistry } from "../src/components/steps/registry";
import { groupTools } from "../src/models/tool-groups";

function tool(
  name: string,
  position: number,
  args: Json,
  result: Json = {},
  workflow = "root",
): HistoryItem {
  return {
    position,
    step: {
      id: `tool-${position}`,
      workflow_id: workflow,
      operation: {
        kind: "llm.tool",
        input: { call: { name, arguments: args } },
      },
      outcome: {
        kind: "completed",
        result: {
          tool_name: name,
          content: [{ type: "text", text: JSON.stringify(result) }],
        },
      },
    },
  };
}

test("poll summaries and details show the matching exec command", () => {
  const command = { kind: "argv", argv: ["npm", "run", "build"] };
  const exec = tool("exec", 1, { command }, { process_id: "process-a" });
  const unrelated = tool(
    "exec",
    2,
    { command: "other command" },
    { process_id: "process-b" },
  );
  const poll = tool(
    "poll_process",
    3,
    { process_id: "process-a" },
    { chunks: [] },
  );
  const history = [poll, unrelated, exec];
  const description = toolDescription(stepView(poll, history));
  assert.deepEqual(description.processCommand, command);
  const summary = renderToStaticMarkup(<ProcessSummary tool={description} />);
  assert.match(summary, /npm run build/);
  assert.doesNotMatch(summary, /process-a|other command/);
  const details = renderToStaticMarkup(<ProcessDetails tool={description} />);
  assert.match(details, /npm run build/);
  assert.doesNotMatch(details, /process-a/);
  assert.match(
    renderToStaticMarkup(
      <DebugProvider value={true}>
        <ProcessDetails tool={description} />
      </DebugProvider>,
    ),
    /process-a/,
  );
  assert.doesNotMatch(details, /No new output/);
  const view = render(<ProcessDetails tool={description} />);
  const disclosure = view.getByText("Result / output").closest("details");
  assert.ok(disclosure);
  disclosure.open = true;
  fireEvent(disclosure, new Event("toggle"));
  assert.ok(view.getByText("No new output."));
});

test("process commands resolve from presentation and generated tool calls", () => {
  const command = { kind: "shell", script: "echo hello" };
  const generation: HistoryItem = {
    position: 1,
    step: {
      id: "generation",
      workflow_id: "root",
      operation: { kind: "llm.generate" },
      outcome: {
        kind: "completed",
        result: {
          content: [
            {
              type: "tool_call",
              id: "call",
              name: "exec",
              arguments: { command },
            },
          ],
        },
      },
    },
  };
  const exec: HistoryItem = {
    position: 2,
    step: {
      id: "exec",
      workflow_id: "root",
      operation: { kind: "llm.tool" },
      outcome: {
        kind: "completed",
        result: {
          tool_call_id: "call",
          tool_name: "exec",
          process_id: "process-a",
        },
      },
    },
  };
  const presented: HistoryItem = {
    position: 2,
    step: {
      ...(exec.step as Record<string, Json>),
      presentation: {
        call: { name: "exec", arguments: { command } },
        result: { process_id: "process-a" },
      },
    },
  };
  for (const origin of [exec, presented]) {
    const poll = tool("poll_process", 3, { process_id: "process-a" });
    const description = toolDescription(
      stepView(poll, [generation, origin, poll]),
    );
    assert.deepEqual(description.processCommand, command);
    assert.match(
      renderToStaticMarkup(<ProcessSummary tool={description} />),
      /echo hello/,
    );
  }
});

test("unmatched, earlier and other-workflow polls do not borrow commands", () => {
  const exec = tool(
    "exec",
    2,
    { command: "echo hello" },
    { process_id: "process-a" },
  );
  for (const poll of [
    tool("poll_process", 3, { process_id: "unknown" }),
    tool("poll_process", 1, { process_id: "process-a" }),
    tool("poll_process", 3, { process_id: "process-a" }, {}, "other"),
  ]) {
    const description = toolDescription(stepView(poll, [exec, poll]));
    assert.equal(description.processCommand, undefined);
    const summary = renderToStaticMarkup(<ProcessSummary tool={description} />);
    assert.doesNotMatch(summary, /echo hello/);
    assert.match(summary, /Process details/);
    assert.match(
      renderToStaticMarkup(
        <DebugProvider value={true}>
          <ProcessSummary tool={description} />
        </DebugProvider>,
      ),
      /unknown|process-a/,
    );
  }
});

test("process context refreshes when exec history becomes available", () => {
  const exec = tool(
    "exec",
    1,
    { command: "echo hello" },
    { process_id: "process-a" },
  );
  const poll = tool("poll_process", 2, { process_id: "process-a" });
  const before = stepView(poll, [poll]);
  assert.equal(before.processCommand, undefined);
  const history = [exec, poll];
  const after = stepView(poll, history);
  assert.equal(after.processCommand, "echo hello");
  assert.notEqual(before, after);
  assert.equal(stepView(poll, history), after);
});

test("collapsed groups show one command preview and the call count", () => {
  const exec = tool(
    "exec",
    1,
    { command: "echo hello" },
    { process_id: "process-a" },
  );
  const poll = tool("poll_process", 2, { process_id: "process-a" });
  const history = [exec, poll];
  const group = groupTools(
    history.map((item) => stepView(item, history)),
    [],
  )[0];
  assert.ok("steps" in group);
  const html = renderToStaticMarkup(
    <ToolGroup
      group={group}
      registry={new StepRegistry([])}
      actions={{
        pending: [],
        enabled: true,
        inspect: () => {},
        reply: async () => true,
      }}
    />,
  );
  assert.match(html, /Run Command/);
  assert.match(html, /2 calls/);
  assert.equal((html.match(/title="echo hello"/g) ?? []).length, 1);
  assert.doesNotMatch(html, /completed|tool-group-preview/);
  assert.doesNotMatch(html, /process-a/);
});

test("streaming polls and other process actions retain command context", () => {
  const exec = tool(
    "exec",
    1,
    { command: "echo hello" },
    { process_id: "process-a" },
  );
  for (const name of ["poll_process", "write_stdin", "kill_process"]) {
    const call = tool(name, 2, { process_id: "process-a" });
    const streaming: HistoryItem = {
      ...call,
      step: {
        ...(call.step as Record<string, Json>),
        outcome: { kind: "streaming", state: { tool_name: name } },
      },
    };
    assert.equal(
      stepView(streaming, [exec, streaming]).processCommand,
      "echo hello",
    );
  }
});
