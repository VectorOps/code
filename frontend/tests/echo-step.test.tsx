import assert from "node:assert/strict";
import { fireEvent, render } from "@testing-library/react";
import { renderToStaticMarkup } from "react-dom/server";
import { test } from "vitest";
import { StepFeed } from "../src/components/StepFeed";
import { AssistantStep } from "../src/components/steps/AssistantStep";
import { defaultStepRegistry } from "../src/components/steps/defaults";
import type { StepActions } from "../src/components/steps/registry";
import { object, type Json } from "../src/models/json";
import type { HistoryItem } from "../src/models/runner";
import { stepView } from "../src/models/steps";

const actions: StepActions = {
  pending: [],
  enabled: true,
  inspect: () => {},
  reply: async () => true,
};

function echo(value: Json, state = "completed"): HistoryItem {
  return {
    position: 1,
    step: {
      id: "echo-message",
      workflow_id: "root",
      operation: { kind: "echo", input: { value } },
      outcome: {
        kind: state,
        ...(state === "completed" ? { result: value } : {}),
      },
    },
  };
}

function feed(item: HistoryItem, handlers = actions) {
  return (
    <StepFeed
      history={[item]}
      registry={defaultStepRegistry}
      actions={handlers}
      active
    />
  );
}

test("echo text appears in the main chat as an inspectable message", () => {
  const inspected: string[] = [];
  const view = render(
    feed(echo("You're right"), {
      ...actions,
      inspect: (id) => inspected.push(id),
    }),
  );
  const message = view.getByText("You're right");
  assert.ok(message.closest(".echo-step"));
  assert.equal(message.tagName, "P");
  fireEvent.click(message);
  assert.deepEqual(inspected, ["echo-message"]);
  fireEvent.click(view.getByRole("button", { name: "View details" }));
  assert.deepEqual(inspected, ["echo-message", "echo-message"]);
});

test("completed echoes use their recorded result rather than input", () => {
  const item = echo("Original input");
  item.step = {
    ...object(item.step),
    outcome: { kind: "completed", result: "Recorded output" },
  };
  const html = renderToStaticMarkup(feed(item));
  assert.match(html, /Recorded output/);
  assert.doesNotMatch(html, /Original input/);
});

test("echo JSON values retain structure, null, and falsy scalars", () => {
  for (const value of [
    { answer: "piano", count: 2 },
    ["hint", 1],
    null,
    false,
    0,
  ]) {
    const view = render(feed(echo(value)));
    assert.equal(
      view.container.querySelector(".echo-step pre")?.textContent,
      JSON.stringify(value, null, 2),
    );
    view.unmount();
  }
});

test("echo Markdown preserves soft line breaks and skips raw HTML", () => {
  const value = "First line\nSecond line\n\n<script>alert('unsafe')</script>";
  const view = render(feed(echo(value)));
  assert.equal(
    view.container.querySelector(".markdown-content p")?.textContent,
    "First line\nSecond line",
  );
  assert.equal(view.container.querySelector("script"), null);
  assert.ok(!view.container.textContent?.includes("alert('unsafe')"));
});

test("echo uses the same Markdown, GFM, and code rendering as LLM messages", () => {
  const message = [
    "# Result",
    "**Bold** and ~~removed~~ with `inline code`.",
    "| Name | Value |\n| --- | --- |\n| Answer | Piano |",
    "```python\ndef example():\n    return 1\n```",
    "[Details](https://example.com) ![Example](https://example.com/image.png)",
  ].join("\n\n");
  const view = render(
    <>
      {feed(echo(message))}
      <AssistantStep
        step={stepView({
          position: 2,
          step: {
            id: "assistant-message",
            workflow_id: "root",
            operation: { kind: "llm.generate" },
            outcome: { kind: "completed", result: message },
          },
        })}
        actions={actions}
      />
    </>,
  );
  const bodies = view.container.querySelectorAll(".markdown-content");
  assert.equal(bodies.length, 2);
  assert.equal(bodies[0].innerHTML, bodies[1].innerHTML);
  const body = bodies[0];
  assert.equal(body.querySelector("h1")?.textContent, "Result");
  assert.equal(body.querySelector("strong")?.textContent, "Bold");
  assert.equal(body.querySelector("del")?.textContent, "removed");
  assert.ok(body.querySelector("table"));
  assert.ok(body.querySelector("pre code .hljs-keyword"));
  const link = body.querySelector("a");
  assert.equal(link?.getAttribute("href"), "https://example.com");
  assert.equal(link?.getAttribute("target"), "_blank");
  assert.equal(link?.getAttribute("rel"), "noopener noreferrer");
  assert.ok(body.textContent?.includes("[Image: Example]"));
  assert.equal(body.querySelector("img"), null);
});

test("pending and cancelled echoes retain their value and status", () => {
  for (const state of ["pending", "cancelled"]) {
    const html = renderToStaticMarkup(feed(echo("Message", state)));
    assert.match(html, /Message/);
    assert.match(
      html,
      new RegExp(`aria-label="${state[0].toUpperCase()}${state.slice(1)}"`),
    );
    assert.match(html, new RegExp(`aria-busy="${state === "pending"}"`));
  }
});

test("failed echoes show their error and fallback without requiring output", () => {
  for (const message of ["Unable to echo.", ""]) {
    const item = echo("Message", "failed");
    item.step = {
      ...object(item.step),
      outcome: { kind: "failed", error: { message } },
    };
    const html = renderToStaticMarkup(feed(item));
    assert.match(html, /aria-label="Failed"/);
    assert.ok(html.includes(message || "The echo step could not finish."));
  }
});
