import * as React from "react";
import * as testing from "@testing-library/react";
import * as vitest from "vitest";
import { vi } from "vitest";
import ReactMarkdown from "react-markdown";
import hljs from "highlight.js/lib/common";
import type { HistoryItem, InputRequest } from "../src/models/runner";
import type { Json } from "../src/models/json";
import { stepView, toolDescription, toolPayload } from "../src/models/steps";
import { workflowProjection } from "../src/models/workflow-tree";
import { StepFeed } from "../src/components/StepFeed";
import { defaultStepRegistry } from "../src/components/steps/defaults";
import type { StepActions } from "../src/components/steps/registry";
import { Workspace } from "../src/screens/Workspace";
import { ToolDetails } from "../src/components/tools/ToolDetails";
import { defaultToolFormatters } from "../src/components/tools/defaults";
import { FileSummary, GenericSummary } from "../src/components/tools/views";
import { DebugSteps } from "../src/components/DebugSteps";
import { ActivityDetails } from "../src/components/ActivityDetails";
import { HighlightedCode } from "../src/components/HighlightedCode";
import {
  FeedStateProvider,
  useDecisionDraft,
} from "../src/components/FeedState";

vi.mock("react-markdown", async () => {
  const actual =
    await vi.importActual<typeof import("react-markdown")>("react-markdown");
  return { ...actual, default: vi.fn(actual.default) };
});

let originalScroll: PropertyDescriptor | undefined;

vitest.beforeEach(() => {
  vitest.vi.clearAllMocks();
  originalScroll = Object.getOwnPropertyDescriptor(
    Element.prototype,
    "scrollIntoView",
  );
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    configurable: true,
    value: vitest.vi.fn(),
  });
  vitest.vi.stubGlobal(
    "ResizeObserver",
    class implements ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
});

vitest.afterEach(() => {
  if (originalScroll)
    Object.defineProperty(Element.prototype, "scrollIntoView", originalScroll);
  else Reflect.deleteProperty(Element.prototype, "scrollIntoView");
  vitest.vi.restoreAllMocks();
  vitest.vi.unstubAllGlobals();
});

function item(
  position: number,
  kind: string,
  workflow = "root",
  operation: Record<string, Json> = {},
  result: Json = `Message ${position}`,
): HistoryItem {
  return {
    position,
    step: {
      id: `step-${position}`,
      workflow_id: workflow,
      operation: { kind, ...operation },
      outcome: { kind: "completed", result },
    },
  };
}

function tool(position: number, name = "custom", args: Json = {}) {
  return item(
    position,
    "llm.tool",
    "root",
    {
      input: { call: { name, arguments: args } },
    },
    { content: [{ type: "text", text: '{"deferred":true}' }] },
  );
}

function actions(pending: InputRequest[] = []): StepActions {
  return {
    pending,
    enabled: true,
    inspect: vitest.vi.fn(),
    reply: vitest.vi.fn(async () => true),
  };
}

function nestedHistory() {
  return [
    item(1, "workflow.spawn", "root", {
      child_workflow_id: "child",
      description: "Child",
    }),
    item(2, "workflow.spawn", "child", {
      child_workflow_id: "grandchild",
      description: "Grandchild",
    }),
    item(3, "llm.generate", "grandchild"),
  ];
}

function disclosure(view: testing.RenderResult, label: string) {
  const element = view.getByText(label).closest("details");
  if (!element) throw new Error(`Missing disclosure: ${label}`);
  return element;
}

function toggle(element: HTMLDetailsElement, open: boolean) {
  testing.act(() => {
    element.open = open;
    testing.fireEvent(element, new Event("toggle"));
  });
}

vitest.test(
  "tool output and raw serialization wait for opening and unmount on closing",
  () => {
    const description = toolDescription(stepView(tool(1)));
    const highlight = vitest.vi.spyOn(hljs, "highlight");
    const serialize = vitest.vi.spyOn(JSON, "stringify");
    const view = testing.render(
      <ToolDetails tool={description} factory={defaultToolFormatters} />,
    );
    vitest.expect(highlight).not.toHaveBeenCalled();
    vitest
      .expect(serialize.mock.calls.some(([value]) => value === description))
      .toBe(false);
    toggle(disclosure(view, "Result / output"), true);
    vitest.expect(highlight).toHaveBeenCalledTimes(1);
    vitest.expect(view.container.querySelector(".hljs")).not.toBeNull();
    toggle(disclosure(view, "Result / output"), false);
    vitest.expect(view.container.querySelector(".hljs")).toBeNull();
    toggle(disclosure(view, "Raw call and result"), true);
    vitest.expect(serialize).toHaveBeenCalledWith(description, null, 2);
    toggle(disclosure(view, "Raw call and result"), false);
    vitest.expect(view.container.querySelector("pre")).toBeNull();
  },
);

vitest.test(
  "debug records and complete activity records serialize only when opened",
  () => {
    const history = [item(1, "llm.generate")];
    const serialize = vitest.vi.spyOn(JSON, "stringify");
    const debug = testing.render(<DebugSteps history={history} />);
    vitest
      .expect(serialize.mock.calls.some(([value]) => value === history[0].step))
      .toBe(false);
    toggle(disclosure(debug, "1. llm.generate / completed"), true);
    vitest.expect(serialize).toHaveBeenCalledWith(history[0].step, null, 2);
    toggle(disclosure(debug, "1. llm.generate / completed"), false);
    vitest.expect(debug.container.querySelector("pre")).toBeNull();
    debug.unmount();
    serialize.mockClear();
    const details = testing.render(<ActivityDetails item={history[0]} />);
    vitest
      .expect(serialize.mock.calls.some(([value]) => value === history[0].step))
      .toBe(false);
    toggle(disclosure(details, "Complete record"), true);
    vitest.expect(serialize).toHaveBeenCalledWith(history[0].step, null, 2);
  },
);

vitest.test(
  "specialized tool formatters defer payload parsing until output opens",
  () => {
    const cases: { name: string; args: Json; result: Json }[] = [
      {
        name: "read_file",
        args: { path: "file.txt" },
        result: { content: "deferred-value" },
      },
      {
        name: "list_files",
        args: {},
        result: {
          entries: [{ path: "deferred-value", kind: "file", size: 10 }],
        },
      },
      {
        name: "exec",
        args: { command: "echo output" },
        result: {
          chunks: [
            { stream: "stdout", encoding: "utf-8", data: "deferred-value" },
          ],
        },
      },
      {
        name: "list_processes",
        args: {},
        result: [{ process_id: "deferred-value" }],
      },
      {
        name: "update_plan",
        args: {},
        result: {
          todos: [{ id: "todo", title: "deferred-value", status: "pending" }],
        },
      },
      {
        name: "web_fetch",
        args: { url: "https://example.com" },
        result: { content: "deferred-value" },
      },
    ];
    const parse = vitest.vi.spyOn(JSON, "parse");
    for (const entry of cases) {
      const raw = JSON.stringify(entry.result);
      const call = item(
        1,
        "llm.tool",
        "root",
        { input: { call: { name: entry.name, arguments: entry.args } } },
        { content: [{ type: "text", text: raw }] },
      );
      const description = toolDescription(stepView(call));
      parse.mockClear();
      const view = testing.render(
        <ToolDetails tool={description} factory={defaultToolFormatters} />,
      );
      vitest.expect(parse).not.toHaveBeenCalledWith(raw);
      vitest.expect(view.queryByText("deferred-value")).toBeNull();
      toggle(disclosure(view, "Result / output"), true);
      vitest.expect(parse).toHaveBeenCalledWith(raw);
      vitest
        .expect(view.getAllByText("deferred-value").length)
        .toBeGreaterThan(0);
      view.unmount();
    }
  },
);

vitest.test(
  "workflow bodies stop rendering after collapse and retain nested expansion",
  () => {
    const history = nestedHistory();
    const callbacks = actions();
    const view = testing.render(
      <StepFeed
        history={history}
        registry={defaultStepRegistry}
        actions={callbacks}
        active
      />,
    );
    vitest.expect(view.queryByText("Message 3")).toBeNull();
    toggle(disclosure(view, "Agent: Child"), true);
    toggle(disclosure(view, "Agent: Grandchild"), true);
    vitest.expect(view.getByText("Message 3")).toBeTruthy();
    const renders = vitest.vi.mocked(ReactMarkdown).mock.calls.length;
    toggle(disclosure(view, "Agent: Child"), false);
    vitest.expect(view.queryByText("Message 3")).toBeNull();
    view.rerender(
      <StepFeed
        history={[...history, item(4, "llm.generate", "grandchild")]}
        registry={defaultStepRegistry}
        actions={callbacks}
        active
      />,
    );
    vitest.expect(ReactMarkdown).toHaveBeenCalledTimes(renders);
    toggle(disclosure(view, "Agent: Child"), true);
    vitest.expect(disclosure(view, "Agent: Grandchild").open).toBe(true);
    vitest.expect(view.getByText("Message 4")).toBeTruthy();
  },
);

vitest.test(
  "pending input auto-opens ancestors and Review restores a collapsed draft and focus",
  () => {
    const history = nestedHistory();
    const request: InputRequest = {
      execution_id: "execution",
      request_id: "question",
      workflow_id: "grandchild",
      channel: "agent.question",
      response_type: "text",
      payload: { question: "Which command?" },
    };
    const callbacks = actions();
    const view = testing.render(
      <StepFeed
        history={history}
        registry={defaultStepRegistry}
        actions={callbacks}
        active
      />,
    );
    view.rerender(
      <StepFeed
        history={history}
        registry={defaultStepRegistry}
        actions={{ ...callbacks, pending: [request] }}
        active
      />,
    );
    vitest.expect(disclosure(view, "Agent: Child").open).toBe(true);
    vitest.expect(disclosure(view, "Agent: Grandchild").open).toBe(true);
    testing.fireEvent.change(view.getByRole("textbox"), {
      target: { value: "Saved response" },
    });
    toggle(disclosure(view, "Agent: Child"), false);
    vitest.expect(view.queryByRole("textbox")).toBeNull();
    const scroll = vitest.vi.mocked(Element.prototype.scrollIntoView);
    testing.fireEvent.click(view.getByRole("button", { name: "Review" }));
    vitest
      .expect(view.getByRole("textbox"))
      .toHaveProperty("value", "Saved response");
    vitest
      .expect(document.activeElement)
      .toBe(view.getByLabelText("Input required: agent.question"));
    vitest
      .expect(scroll)
      .toHaveBeenCalledWith({ block: "center", behavior: "smooth" });
  },
);

vitest.test(
  "decision retries retain their idempotency key across collapsing",
  async () => {
    const request: InputRequest = {
      execution_id: "execution",
      request_id: "question",
      workflow_id: "grandchild",
      channel: "agent.question",
      response_type: "text",
      payload: {},
    };
    const callbacks = actions([request]);
    const reply = vitest.vi.fn(async () => false);
    const view = testing.render(
      <StepFeed
        history={nestedHistory()}
        registry={defaultStepRegistry}
        actions={{ ...callbacks, reply }}
        active
      />,
    );
    testing.fireEvent.change(view.getByRole("textbox"), {
      target: { value: "Try again" },
    });
    await testing.act(async () =>
      testing.fireEvent.click(
        view.getByRole("button", { name: "Send response" }),
      ),
    );
    toggle(disclosure(view, "Agent: Child"), false);
    toggle(disclosure(view, "Agent: Child"), true);
    await testing.act(async () =>
      testing.fireEvent.click(
        view.getByRole("button", { name: "Send response" }),
      ),
    );
    vitest.expect(reply).toHaveBeenCalledTimes(2);
    vitest.expect(reply.mock.calls[1]).toEqual(reply.mock.calls[0]);
  },
);

vitest.test(
  "a decision reply finishing after remount updates the retained draft",
  async () => {
    const request: InputRequest = {
      execution_id: "execution",
      request_id: "question",
      workflow_id: "grandchild",
      channel: "agent.question",
      response_type: "text",
      payload: {},
    };
    let finish: (result: boolean) => void = () => {};
    const reply = vitest.vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          finish = resolve;
        }),
    );
    const view = testing.render(
      <StepFeed
        history={nestedHistory()}
        registry={defaultStepRegistry}
        actions={{ ...actions([request]), reply }}
        active
      />,
    );
    testing.fireEvent.change(view.getByRole("textbox"), {
      target: { value: "Response" },
    });
    testing.fireEvent.click(
      view.getByRole("button", { name: "Send response" }),
    );
    toggle(disclosure(view, "Agent: Child"), false);
    toggle(disclosure(view, "Agent: Child"), true);
    await testing.act(async () => {
      finish(true);
    });
    vitest.expect(view.getByRole("textbox")).toHaveProperty("value", "");
  },
);

vitest.test(
  "an approval in flight survives collapse and updates the remounted controls",
  async () => {
    const call = { name: "exec", arguments: { command: "echo hello" } };
    const request: InputRequest = {
      execution_id: "execution",
      request_id: "step-3",
      workflow_id: "grandchild",
      request_type: "tool.approval.v1",
      channel: "tool.approval",
      response_type: "json",
      payload: { call },
    };
    const history = [
      ...nestedHistory().slice(0, 2),
      item(3, "llm.tool", "grandchild", { input: { call } }),
    ];
    let finish: (result: boolean) => void = () => {};
    const reply = vitest.vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          finish = resolve;
        }),
    );
    const view = testing.render(
      <StepFeed
        history={history}
        registry={defaultStepRegistry}
        actions={{ ...actions([request]), reply }}
        active
      />,
    );
    testing.fireEvent.click(view.getByRole("button", { name: "Approve once" }));
    toggle(disclosure(view, "Agent: Child"), false);
    toggle(disclosure(view, "Agent: Child"), true);
    vitest
      .expect(view.getByRole("button", { name: "Approve once" }))
      .toHaveProperty("disabled", true);
    await testing.act(async () => {
      finish(true);
    });
    vitest.expect(view.getByText("Decision acknowledged")).toBeTruthy();
    vitest.expect(reply).toHaveBeenCalledTimes(1);
  },
);

vitest.test("collapsed tool groups never render their calls", () => {
  const history = [
    tool(1, "exec", { command: "first command" }),
    tool(2, "exec", { command: "second command" }),
  ];
  const callbacks = actions();
  const view = testing.render(
    <StepFeed
      history={history}
      registry={defaultStepRegistry}
      actions={callbacks}
      active
    />,
  );
  const group = view.getByText(/2 calls/).closest("details");
  if (!group) throw new Error("Missing tool group");
  vitest.expect(view.container.querySelector(".tool-group-calls")).toBeNull();
  toggle(group, true);
  vitest
    .expect(view.container.querySelectorAll(".tool-group-call"))
    .toHaveLength(2);
  toggle(group, false);
  view.rerender(
    <StepFeed
      history={[...history, tool(3, "exec", { command: "third command" })]}
      registry={defaultStepRegistry}
      actions={callbacks}
      active
    />,
  );
  vitest.expect(view.container.querySelector(".tool-group-calls")).toBeNull();
});

vitest.test(
  "composer, navigation, inspector and enabled-state changes do not reparse markdown",
  () => {
    const props: React.ComponentProps<typeof Workspace> = {
      sessions: [],
      selected: "session",
      onSelect: vitest.vi.fn(),
      history: [item(1, "llm.generate")],
      pending: [],
      status: "Connected",
      endpoint: "ws://localhost",
      enabled: true,
      busy: false,
      onStop: vitest.vi.fn(),
      onReply: vitest.vi.fn(async () => true),
      newSession: null,
      connection: null,
      error: "",
      dismiss: vitest.vi.fn(),
      hasOlder: false,
    };
    const resolve = vitest.vi.spyOn(defaultStepRegistry, "resolve");
    const view = testing.render(<Workspace {...props} />);
    const renders = vitest.vi.mocked(ReactMarkdown).mock.calls.length;
    resolve.mockClear();
    testing.fireEvent.change(view.getByLabelText("Message draft"), {
      target: { value: "Typing" },
    });
    testing.fireEvent.click(
      view.getByRole("button", { name: "Collapse navigation" }),
    );
    testing.fireEvent.click(view.getByRole("button", { name: "View details" }));
    vitest.expect(resolve).not.toHaveBeenCalled();
    view.rerender(<Workspace {...props} enabled={false} busy />);
    vitest.expect(ReactMarkdown).toHaveBeenCalledTimes(renders);
  },
);

vitest.test("updating one step does not reparse unchanged messages", () => {
  const first = item(1, "llm.generate");
  const second = item(2, "llm.generate");
  const callbacks = actions();
  const view = testing.render(
    <StepFeed
      history={[first, second]}
      registry={defaultStepRegistry}
      actions={callbacks}
      active
    />,
  );
  vitest.vi.mocked(ReactMarkdown).mockClear();
  view.rerender(
    <StepFeed
      history={[first, item(2, "llm.generate", "root", {}, "Updated message")]}
      registry={defaultStepRegistry}
      actions={callbacks}
      active
    />,
  );
  vitest.expect(ReactMarkdown).toHaveBeenCalledTimes(1);
  vitest
    .expect(vitest.vi.mocked(ReactMarkdown).mock.calls[0][0].children)
    .toBe("Updated message");
});

vitest.test(
  "streamed code stays plain until completion and large blocks stay escaped",
  () => {
    const completed = item(
      1,
      "llm.generate",
      "root",
      {},
      '```python\nprint("<script>")\n```',
    );
    const streaming: HistoryItem = {
      ...completed,
      step: {
        ...(completed.step as Record<string, Json>),
        outcome: {
          kind: "streaming",
          state: '```python\nprint("<script>")\n```',
        },
      },
    };
    const highlight = vitest.vi.spyOn(hljs, "highlight");
    const callbacks = actions();
    const view = testing.render(
      <StepFeed
        history={[streaming]}
        registry={defaultStepRegistry}
        actions={callbacks}
        active
      />,
    );
    vitest.expect(highlight).not.toHaveBeenCalled();
    view.rerender(
      <StepFeed
        history={[completed]}
        registry={defaultStepRegistry}
        actions={callbacks}
        active
      />,
    );
    vitest.expect(highlight).toHaveBeenCalledTimes(1);
    view.unmount();
    highlight.mockClear();
    const large = testing.render(
      <HighlightedCode language="python">
        {"<script>".repeat(3000)}
      </HighlightedCode>,
    );
    vitest.expect(highlight).not.toHaveBeenCalled();
    vitest.expect(large.container.querySelector("script")).toBeNull();
  },
);

vitest.test(
  "unchanged code blocks are not highlighted again when surrounding markdown changes",
  () => {
    const message = '```python\nprint("hello")\n```';
    const callbacks = actions();
    const highlight = vitest.vi.spyOn(hljs, "highlight");
    const view = testing.render(
      <StepFeed
        history={[item(1, "llm.generate", "root", {}, message)]}
        registry={defaultStepRegistry}
        actions={callbacks}
        active
      />,
    );
    view.rerender(
      <StepFeed
        history={[
          item(1, "llm.generate", "root", {}, `${message}\n\nMore text`),
        ]}
        registry={defaultStepRegistry}
        actions={callbacks}
        active
      />,
    );
    vitest.expect(highlight).toHaveBeenCalledTimes(1);
  },
);

vitest.test(
  "file previews avoid output parsing and generic previews stay bounded",
  () => {
    const file = toolDescription(
      stepView(tool(1, "read_file", { path: "large.txt" })),
    );
    const parse = vitest.vi.spyOn(JSON, "parse");
    const fileView = testing.render(<FileSummary tool={file} />);
    vitest.expect(parse).not.toHaveBeenCalledWith('{"deferred":true}');
    fileView.unmount();
    const generic = toolDescription(
      stepView(
        tool(2, "custom", {
          data: "x".repeat(100_000),
          nested: { secret: "large" },
        }),
      ),
    );
    const serialize = vitest.vi.spyOn(JSON, "stringify");
    const view = testing.render(<GenericSummary tool={generic} />);
    vitest.expect(serialize).not.toHaveBeenCalled();
    vitest.expect(view.container.textContent?.length).toBeLessThan(150);
    vitest
      .expect(
        view.container.querySelector(".tool-summary")?.getAttribute("title")
          ?.length,
      )
      .toBeLessThanOrEqual(600);
  },
);

vitest.test(
  "tool descriptions and parsed payloads preserve unchanged identity",
  () => {
    const step = stepView(tool(1));
    const description = toolDescription(step);
    vitest.expect(toolDescription(step)).toBe(description);
    vitest
      .expect(toolPayload(description.output))
      .toBe(toolPayload(description.output));
  },
);

vitest.test(
  "one workflow projection separates composer requests from inline decisions",
  () => {
    const root: InputRequest = {
      execution_id: "execution",
      request_id: "chat",
      workflow_id: "root",
      request_type: "chat.message.v1",
      channel: "user.message",
      response_type: "text",
      payload: {},
    };
    const child = {
      ...root,
      request_id: "child-chat",
      workflow_id: "grandchild",
    };
    const pending = [root, child];
    const projection = workflowProjection(nestedHistory(), pending);
    vitest.expect(projection.chat).toEqual([root]);
    vitest.expect(projection.decisions).toEqual([child]);
    vitest.expect(projection.roots[0].requests).toEqual([]);
    const updated = workflowProjection(
      [...nestedHistory(), item(4, "llm.generate")],
      pending,
    );
    vitest.expect(updated.decisions).toBe(projection.decisions);
    const view = testing.render(
      <StepFeed
        history={[]}
        registry={defaultStepRegistry}
        actions={actions(projection.decisions)}
        projection={projection}
        active
      />,
    );
    vitest.expect(view.getAllByRole("textbox")).toHaveLength(1);
    vitest.expect(view.getByText("Needs your input (1)")).toBeTruthy();
  },
);

function RetainedDraft({ request }: { request: InputRequest }) {
  const [value, change] = useDecisionDraft(request);
  return (
    <input
      aria-label="Retained draft"
      value={value}
      onChange={(event) => change(event.target.value)}
    />
  );
}

vitest.test(
  "retained drafts are isolated by request and pruned when requests resolve",
  () => {
    const first: InputRequest = {
      execution_id: "execution",
      request_id: "first",
      channel: "question",
      response_type: "text",
      payload: {},
    };
    const second = { ...first, request_id: "second" };
    const pending = [first, second];
    const view = testing.render(
      <FeedStateProvider requests={pending}>
        <RetainedDraft request={first} />
      </FeedStateProvider>,
    );
    testing.fireEvent.change(view.getByLabelText("Retained draft"), {
      target: { value: "First draft" },
    });
    view.rerender(
      <FeedStateProvider requests={pending}>
        <RetainedDraft request={second} />
      </FeedStateProvider>,
    );
    vitest
      .expect(view.getByLabelText("Retained draft"))
      .toHaveProperty("value", "");
    view.rerender(
      <FeedStateProvider requests={pending}>
        <RetainedDraft request={first} />
      </FeedStateProvider>,
    );
    vitest
      .expect(view.getByLabelText("Retained draft"))
      .toHaveProperty("value", "First draft");
    view.rerender(<FeedStateProvider requests={[]} children={null} />);
    view.rerender(
      <FeedStateProvider requests={[first]}>
        <RetainedDraft request={first} />
      </FeedStateProvider>,
    );
    vitest
      .expect(view.getByLabelText("Retained draft"))
      .toHaveProperty("value", "");
  },
);
