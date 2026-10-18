import assert from "node:assert/strict";
import { test } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { render } from "@testing-library/react";
import { StepFeed } from "../src/components/StepFeed";
import { AssistantStep } from "../src/components/steps/AssistantStep";
import { UserStep } from "../src/components/steps/UserStep";
import { ToolStep } from "../src/components/steps/ToolStep";
import { defaultStepRegistry } from "../src/components/steps/defaults";
import { ToolDetails } from "../src/components/tools/ToolDetails";
import { defaultToolFormatters } from "../src/components/tools/defaults";
import type { StepActions } from "../src/components/steps/registry";
import type { HistoryItem, InputRequest } from "../src/models/runner";
import { stepView, toolDescription } from "../src/models/steps";

const actions: StepActions = {
  pending: [],
  enabled: true,
  inspect: () => {},
  reply: async () => true,
};

function call(position: number, state = "completed"): HistoryItem {
  return {
    position,
    step: {
      id: `call-${position}`,
      workflow_id: "root",
      operation: {
        kind: "llm.tool",
        input: {
          call: { name: "exec", arguments: { command: "uname -a" } },
        },
      },
      outcome: {
        kind: state,
        error:
          state === "failed" ? { message: "Command was not found." } : null,
      },
    },
  };
}

function feed(history: HistoryItem[], pending: InputRequest[] = []) {
  return renderToStaticMarkup(
    <StepFeed
      history={history}
      registry={defaultStepRegistry}
      actions={{ ...actions, pending }}
      active
    />,
  );
}

test("sent user images remain visible alongside the message text", () => {
  const step = stepView({
    position: 1,
    step: {
      id: "user-message",
      workflow_id: "root",
      operation: {
        kind: "mailbox.input",
        input: { request_type: "chat.message.v1" },
      },
      outcome: {
        kind: "completed",
        result: {
          content: [
            { type: "text", text: "What is in this image?" },
            { type: "image", mime_type: "image/png", data: "cG5n" },
          ],
        },
      },
    },
  });
  const view = render(<UserStep step={step} actions={actions} />);
  assert.ok(view.getByText("What is in this image?"));
  assert.equal(
    view.getByAltText("Attached image 1").getAttribute("src"),
    "data:image/png;base64,cG5n",
  );
});

test("running tools use a compact activity label, spinner and command bubble", () => {
  for (const state of ["pending", "streaming"]) {
    const html = renderToStaticMarkup(
      <ToolStep step={stepView(call(1, state))} actions={actions} />,
    );
    assert.match(html, />Running<\/span>/);
    assert.match(html, /animate-spin/);
    assert.match(html, /<code>uname -a<\/code>/);
    assert.match(html, /aria-busy="true"/);
    assert.doesNotMatch(html, /activity-heading/);
  }
});

test("running groups retain a single preview and short running status", () => {
  const html = feed([call(1), call(2, "pending")]);
  assert.match(html, /2 calls/);
  assert.match(html, /role="status"/);
  assert.match(html, /workflow-spinner/);
  assert.equal((html.match(/<code>uname -a<\/code>/g) ?? []).length, 1);
  assert.doesNotMatch(html, /Completed|completed/);
});

test("grouped failures retain their warning without rendering error details", () => {
  const html = feed([call(1, "failed"), call(2)]);
  assert.match(html, /1 failed/);
  assert.match(html, /tool-failure-marker/);
  assert.doesNotMatch(html, /Command was not found\.|failure-text/);
});

test("successful workflow completion reduces historical failures to group markers", () => {
  const html = feed([
    call(1, "failed"),
    call(2),
    {
      position: 3,
      step: {
        id: "complete",
        workflow_id: "root",
        operation: { kind: "workflow.complete" },
        outcome: { kind: "completed" },
      },
    },
  ]);
  assert.match(html, /1 failed/);
  assert.match(html, /Workflow completed after these failures/);
  assert.doesNotMatch(html, /Command was not found\.|tool-failure"/);
});

test("standalone failures keep their warning and clickable card without error text", () => {
  const html = feed([call(1, "failed")]);
  assert.match(html, /<code>uname -a<\/code>/);
  assert.doesNotMatch(html, /Command was not found\.|failure-text/);
  assert.match(html, /aria-label="Failed"/);
  assert.match(
    html,
    /<button[^>]*class="step-card-details"[^>]*>View details<\/button>/,
  );
});

test("failed workflows retain tool failure warnings without error text", () => {
  const html = feed([
    call(1, "failed"),
    call(2),
    {
      position: 3,
      step: {
        id: "complete",
        workflow_id: "root",
        operation: { kind: "workflow.complete" },
        outcome: { kind: "failed" },
      },
    },
  ]);
  assert.match(html, /1 failed/);
  assert.doesNotMatch(html, /Command was not found\./);
  assert.doesNotMatch(html, /Workflow completed after these failures/);
});

test("tool-reported errors retain only their warning even with a completed step", () => {
  const html = feed([
    {
      position: 1,
      step: {
        id: "error-result",
        workflow_id: "root",
        operation: {
          kind: "llm.tool",
          input: { call: { name: "exec", arguments: { command: "uname -a" } } },
        },
        outcome: {
          kind: "completed",
          result: {
            is_error: true,
            content: [{ type: "text", text: "Permission denied." }],
          },
        },
      },
    },
  ]);
  assert.match(html, /aria-label="Failed"/);
  assert.doesNotMatch(html, /Permission denied\./);
  assert.doesNotMatch(html, /aria-label="Completed"/);
});

test("approvals stay outside groups with prominent actions", () => {
  const pending: InputRequest[] = [
    {
      request_id: "call-2",
      execution_id: "execution",
      workflow_id: "root",
      request_type: "tool.approval.v1",
      channel: "tool.approval",
      response_type: "json",
      payload: { call: { name: "exec", arguments: { command: "uname -a" } } },
    },
  ];
  const html = feed([call(1), call(2, "pending")], pending);
  assert.match(html, /tool-step tool-approval/);
  assert.match(html, />Approve Run Command<\/span>/);
  assert.match(html, /Approve|Deny/);
  assert.match(html, /Needs your input \(1\)/);
  assert.doesNotMatch(html, /2 calls/);
});

test("pending questions keep their card alongside compact failure warnings", () => {
  const html = feed(
    [
      call(1, "failed"),
      call(2),
      {
        position: 3,
        step: {
          id: "complete",
          workflow_id: "root",
          operation: { kind: "workflow.complete" },
          outcome: { kind: "completed" },
        },
      },
    ],
    [
      {
        request_id: "question",
        execution_id: "execution",
        workflow_id: "root",
        channel: "agent.question",
        response_type: "text",
        payload: { question: "Which command should I try next?" },
      },
    ],
  );
  assert.match(html, /decision-card/);
  assert.match(html, /Send response/);
  assert.match(html, /1 failed/);
  assert.doesNotMatch(html, /Command was not found\./);
  assert.doesNotMatch(html, /Workflow completed after these failures/);
});

test("tool errors remain available in the debug details view", () => {
  const html = renderToStaticMarkup(
    <ToolDetails
      tool={toolDescription(stepView(call(1, "failed")))}
      factory={defaultToolFormatters}
    />,
  );
  assert.match(html, /aria-label="Tool error"/);
  assert.match(html, /Command was not found\./);
  assert.match(html, /Raw call and result/);
});

test("patch calls keep diff content out of the collapsed row", () => {
  const step = stepView({
    position: 1,
    step: {
      id: "patch",
      workflow_id: "root",
      operation: {
        kind: "llm.tool",
        input: {
          call: {
            name: "apply_patch",
            arguments: {
              text: "*** Update File: example.py\n@@\n-old\n+new",
            },
          },
        },
      },
      outcome: { kind: "completed" },
    },
  });
  const html = renderToStaticMarkup(<ToolStep step={step} actions={actions} />);
  assert.match(html, /example.py/);
  assert.doesNotMatch(html, /<pre|tool-diff|\+new/);
});

test("streaming agent text has no repeated assistant header", () => {
  const step = stepView({
    position: 1,
    step: {
      id: "assistant",
      workflow_id: "root",
      operation: { kind: "llm.generate" },
      outcome: { kind: "streaming", state: "Here is the answer." },
    },
  });
  const html = renderToStaticMarkup(
    <AssistantStep step={step} actions={actions} />,
  );
  assert.match(html, /Here is the answer\./);
  assert.match(html, /Writing/);
  assert.match(html, /aria-busy="true"/);
  assert.doesNotMatch(html, />Assistant<|activity-heading/);
});

test("empty agent output does not mount a chat row or status-only placeholder", () => {
  for (const state of ["pending", "streaming", "completed", "cancelled"]) {
    for (const message of ["", " \n\t "]) {
      const history: HistoryItem[] = [
        {
          position: 1,
          step: {
            id: "assistant",
            workflow_id: "root",
            operation: { kind: "llm.generate" },
            outcome: { kind: state, state: message, result: message },
          },
        },
      ];
      assert.equal(
        defaultStepRegistry.resolve(stepView(history[0], history)),
        undefined,
      );
      assert.equal(feed(history), "");
    }
  }
});

test("an empty pending response appears only once streamed text arrives", () => {
  const entry = (state: string, message: string): HistoryItem => ({
    position: 1,
    step: {
      id: "assistant",
      workflow_id: "root",
      operation: { kind: "llm.generate" },
      outcome: { kind: state, state: message, result: message },
    },
  });
  const view = render(
    <StepFeed
      history={[entry("pending", "")]}
      registry={defaultStepRegistry}
      actions={actions}
      active
    />,
  );
  assert.equal(view.container.querySelector(".assistant-step"), null);
  assert.equal(view.container.querySelector(".feed-item"), null);
  view.rerender(
    <StepFeed
      history={[entry("streaming", "Visible response")]}
      registry={defaultStepRegistry}
      actions={actions}
      active
    />,
  );
  assert.ok(view.getByText("Visible response"));
  assert.ok(view.getByText("Writing"));
  view.rerender(
    <StepFeed
      history={[entry("completed", " \n ")]}
      registry={defaultStepRegistry}
      actions={actions}
      active
    />,
  );
  assert.equal(view.container.querySelector(".assistant-step"), null);
});

test("agent failures remain visible even when no response text was produced", () => {
  const html = feed([
    {
      position: 1,
      step: {
        id: "assistant",
        workflow_id: "root",
        operation: { kind: "llm.generate" },
        outcome: {
          kind: "failed",
          error: { message: "Unable to generate a response." },
        },
      },
    },
  ]);
  assert.match(html, /Unable to generate a response\./);
  assert.match(html, /aria-label="Failed"/);
});
