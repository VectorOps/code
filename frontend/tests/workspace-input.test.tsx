import { StrictMode, useState, type ComponentProps } from "react";
import { act, render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { Workspace } from "../src/screens/Workspace";
import type { InputRequest } from "../src/models/runner";
import { createPortal } from "react-dom";
import { useFocusBlocker, useFocusRegion } from "../src/focus/FocusProvider";
import { defaultStepRegistry } from "../src/components/steps/defaults";

vitest.beforeEach(() => {
  vitest.vi.stubGlobal(
    "ResizeObserver",
    class implements ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
});

vitest.afterEach(() => vitest.vi.unstubAllGlobals());

vitest.test(
  "one composer control stops, resumes and sends without losing the draft",
  async () => {
    const props = workspaceProps();
    const onResume = vitest.vi.fn(async () => {});
    const running = {
      execution_id: "session-a",
      reference: "chat:workflow",
      status: "running",
      active: true,
      pending_inputs: 0,
    };
    const view = render(
      <Workspace
        {...props}
        sessions={[running]}
        pending={[]}
        onResume={onResume}
      />,
    );
    const user = userEvent.setup();
    await user.type(view.getByLabelText("Message draft"), "Saved instruction");
    const stop = view.getByRole("button", { name: "Stop workflow" });
    vitest.expect(stop.closest(".composer-area")).not.toBeNull();
    vitest.expect(view.queryByRole("button", { name: /^Stop$/ })).toBeNull();
    await user.click(stop);
    vitest.expect(props.onStop).toHaveBeenCalledOnce();
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "Saved instruction");
    view.rerender(
      <Workspace
        {...props}
        sessions={[{ ...running, status: "cancelled", active: false }]}
        pending={[]}
        onResume={onResume}
      />,
    );
    await user.click(view.getByRole("button", { name: "Start workflow" }));
    vitest.expect(onResume).toHaveBeenCalledOnce();
    vitest.expect(props.onReply).not.toHaveBeenCalled();
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "Saved instruction");
    view.rerender(
      <Workspace
        {...props}
        sessions={[
          { ...running, status: "paused", active: false, pending_inputs: 1 },
        ]}
        onResume={onResume}
      />,
    );
    await user.click(view.getByRole("button", { name: "Send message" }));
    vitest
      .expect(props.onReply)
      .toHaveBeenCalledWith(
        request,
        { content: [{ type: "text", text: "Saved instruction" }] },
        vitest.expect.any(String),
      );
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "");
  },
);

const request: InputRequest = {
  request_id: "message-request",
  execution_id: "session-a",
  workflow_id: "root",
  request_type: "chat.message.v1",
  channel: "user.message",
  response_type: "text",
  payload: {},
};

function workspaceProps(): ComponentProps<typeof Workspace> {
  return {
    sessions: [],
    selected: "session-a",
    onSelect: vitest.vi.fn(),
    history: [],
    pending: [request],
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
}

vitest.test.each(["", "Change direction"])(
  "a running steering workflow uses the same control for draft %j and stop",
  async (value) => {
    const props = workspaceProps();
    const onSteer = vitest.vi.fn(async () => true);
    const view = render(
      <Workspace
        {...props}
        sessions={[
          {
            execution_id: "session-a",
            reference: "chat:workflow",
            status: "running",
            active: true,
            pending_inputs: 0,
          },
        ]}
        pending={[]}
        steering={{
          address: {
            execution_id: "session-a",
            workflow_id: "root",
            conversation_id: "steer",
          },
          messages: [],
          count: 0,
        }}
        onSteer={onSteer}
      />,
    );
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft");
    if (value) await user.type(draft, value);
    if (value) {
      vitest
        .expect(view.queryByRole("button", { name: "Stop workflow" }))
        .toBeNull();
      await user.click(view.getByRole("button", { name: "Send message" }));
      vitest.expect(onSteer).toHaveBeenCalledWith(value);
      vitest.expect(draft).toHaveProperty("value", "");
    }
    const stop = view.getByRole("button", { name: "Stop workflow" });
    vitest.expect(stop).toHaveProperty("disabled", false);
    await user.click(stop);
    vitest.expect(props.onStop).toHaveBeenCalledOnce();
    vitest.expect(onSteer).toHaveBeenCalledTimes(value ? 1 : 0);
    vitest.expect(props.onReply).not.toHaveBeenCalled();
    vitest.expect(draft).toHaveProperty("value", "");
  },
);

vitest.test(
  "image-only messages preserve attachments on a failed reply and reuse the retry key",
  async () => {
    const props = workspaceProps();
    const onReply = vitest.vi.fn(async () => true).mockResolvedValueOnce(false);
    const view = render(<Workspace {...props} onReply={onReply} />);
    const user = userEvent.setup();
    await user.upload(
      view.getByLabelText("Choose images"),
      new File(["png"], "shot.png", { type: "image/png" }),
    );
    await waitFor(() =>
      vitest.expect(view.getByAltText("shot.png")).toBeTruthy(),
    );
    await user.click(view.getByRole("button", { name: "Send message" }));
    vitest
      .expect(onReply)
      .toHaveBeenCalledWith(
        request,
        { content: [{ type: "image", mime_type: "image/png", data: "cG5n" }] },
        vitest.expect.any(String),
      );
    vitest.expect(view.getByAltText("shot.png")).toBeTruthy();
    await user.click(view.getByRole("button", { name: "Send message" }));
    vitest.expect(onReply.mock.calls[1]).toEqual(onReply.mock.calls[0]);
    vitest.expect(view.queryByAltText("shot.png")).toBeNull();
  },
);

vitest.test(
  "a successful reply removes only submitted images and preserves additions during send",
  async () => {
    let finish: (result: boolean) => void = () => {};
    const props = workspaceProps();
    const onReply = vitest.vi.fn<ComponentProps<typeof Workspace>["onReply"]>(
      () =>
        new Promise<boolean>((resolve) => {
          finish = resolve;
        }),
    );
    const view = render(<Workspace {...props} onReply={onReply} />);
    const user = userEvent.setup();
    await user.upload(
      view.getByLabelText("Choose images"),
      new File(["png"], "first.png", { type: "image/png" }),
    );
    await waitFor(() =>
      vitest.expect(view.getByAltText("first.png")).toBeTruthy(),
    );
    await user.type(view.getByLabelText("Message draft"), "Look at this");
    await user.click(view.getByRole("button", { name: "Send message" }));
    await user.upload(
      view.getByLabelText("Choose images"),
      new File(["png"], "second.png", { type: "image/png" }),
    );
    await waitFor(() =>
      vitest.expect(view.getByAltText("second.png")).toBeTruthy(),
    );
    await act(async () => finish(true));
    vitest.expect(view.queryByAltText("first.png")).toBeNull();
    vitest.expect(view.getByAltText("second.png")).toBeTruthy();
    vitest.expect(onReply.mock.calls[0][1]).toEqual({
      content: [
        { type: "text", text: "Look at this" },
        { type: "image", mime_type: "image/png", data: "cG5n" },
      ],
    });
  },
);

vitest.test(
  "images are retained per session and text-only workflows cannot send them",
  async () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const user = userEvent.setup();
    await user.upload(
      view.getByLabelText("Choose images"),
      new File(["png"], "shot.png", { type: "image/png" }),
    );
    await waitFor(() =>
      vitest.expect(view.getByAltText("shot.png")).toBeTruthy(),
    );
    view.rerender(<Workspace {...props} selected="session-b" pending={[]} />);
    vitest.expect(view.queryByAltText("shot.png")).toBeNull();
    view.rerender(
      <Workspace
        {...props}
        pending={[{ ...request, request_type: undefined }]}
      />,
    );
    vitest.expect(view.getByAltText("shot.png")).toBeTruthy();
    vitest
      .expect(view.getByRole("button", { name: "Send message" }))
      .toHaveProperty("disabled", true);
    vitest.expect(view.getByText(/accepts text only/)).toBeTruthy();
    vitest.expect(props.onReply).not.toHaveBeenCalled();
    await user.click(view.getByRole("button", { name: "Remove shot.png" }));
    vitest.expect(view.queryByAltText("shot.png")).toBeNull();
  },
);

vitest.test("running status stays last in the feed during streaming", () => {
  const props = workspaceProps();
  const running = {
    execution_id: props.selected,
    reference: "chat:workflow",
    status: "running",
    active: true,
    pending_inputs: 0,
  };
  const view = render(
    <Workspace {...props} sessions={[running]} pending={[]} />,
  );
  const indicator = view.getByText("Running...");
  vitest.expect(indicator.getAttribute("role")).toBe("status");
  vitest.expect(indicator.className).toBe("running-indicator");
  vitest
    .expect(view.container.querySelector(".feed-content")?.lastElementChild)
    .toBe(indicator);

  view.rerender(
    <Workspace
      {...props}
      sessions={[running]}
      enabled={false}
      busy
      pending={[]}
      history={[
        {
          position: 1,
          step: {
            id: "assistant",
            workflow_id: "root",
            operation: { kind: "llm.generate" },
            outcome: { kind: "streaming", state: "Working on it" },
          },
        },
      ]}
    />,
  );
  vitest.expect(view.getByText("Working on it")).toBeTruthy();
  vitest.expect(view.getAllByText("Running...")).toHaveLength(1);
  vitest
    .expect(view.container.querySelector(".feed-content")?.lastElementChild)
    .toBe(indicator);
});

vitest.test(
  "running status follows only the selected active session and disappears when it stops",
  () => {
    const props = workspaceProps();
    const running = {
      execution_id: props.selected,
      reference: "chat:workflow",
      status: "running",
      active: true,
      pending_inputs: 0,
    };
    const view = render(
      <Workspace {...props} sessions={[running]} pending={[]} />,
    );
    vitest.expect(view.getByText("Running...")).toBeTruthy();
    for (const status of ["paused", "cancelled", "completed", "failed"]) {
      view.rerender(
        <Workspace
          {...props}
          sessions={[{ ...running, status, active: false }]}
          pending={[]}
        />,
      );
      vitest.expect(view.queryByText("Running...")).toBeNull();
    }
    view.rerender(<Workspace {...props} sessions={[running]} pending={[]} />);
    vitest.expect(view.getByText("Running...")).toBeTruthy();
    for (const selected of ["session-b", ""]) {
      view.rerender(
        <Workspace
          {...props}
          sessions={[running]}
          selected={selected}
          pending={[]}
        />,
      );
      vitest.expect(view.queryByText("Running...")).toBeNull();
    }
  },
);

vitest.test(
  "running status hides for input requests or blocked session summaries and returns on resume",
  () => {
    const props = workspaceProps();
    const running = {
      execution_id: props.selected,
      reference: "chat:workflow",
      status: "running",
      active: true,
      pending_inputs: 0,
    };
    const view = render(
      <Workspace {...props} sessions={[running]} pending={[]} />,
    );
    vitest.expect(view.getByText("Running...")).toBeTruthy();
    for (const pending of [
      request,
      { ...request, request_type: undefined, channel: "agent.question" },
      {
        ...request,
        workflow_id: "child",
        request_type: "tool.approval.v1",
        channel: "tool.approval",
      },
    ]) {
      view.rerender(
        <Workspace {...props} sessions={[running]} pending={[pending]} />,
      );
      vitest.expect(view.queryByText("Running...")).toBeNull();
    }
    for (const session of [
      { ...running, pending_inputs: 1 },
      { ...running, status: "paused" },
      { ...running, status: "stopping" },
    ]) {
      view.rerender(<Workspace {...props} sessions={[session]} pending={[]} />);
      vitest.expect(view.queryByText("Running...")).toBeNull();
    }
    view.rerender(<Workspace {...props} sessions={[running]} pending={[]} />);
    vitest.expect(view.getByText("Running...")).toBeTruthy();
  },
);

vitest.test(
  "drafting does not send until exactly one root chat request is ready",
  async () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} pending={[]} />);
    const user = userEvent.setup();
    await user.type(view.getByLabelText("Message draft"), "Hello");
    await user.keyboard("{Control>}{Enter}{/Control}");
    vitest.expect(props.onReply).not.toHaveBeenCalled();
    view.rerender(
      <Workspace
        {...props}
        pending={[request, { ...request, request_id: "other" }]}
      />,
    );
    await user.keyboard("{Control>}{Enter}{/Control}");
    vitest.expect(props.onReply).not.toHaveBeenCalled();
    view.rerender(<Workspace {...props} />);
    await user.keyboard("{Control>}{Enter}{/Control}");
    vitest.expect(props.onReply).toHaveBeenCalledWith(
      request,
      {
        content: [{ type: "text", text: "Hello" }],
      },
      vitest.expect.any(String),
    );
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "");
  },
);

vitest.test(
  "an unsuccessful reply preserves the draft and its retry key",
  async () => {
    const onReply = vitest.vi.fn(async () => true).mockResolvedValueOnce(false);
    const view = render(<Workspace {...workspaceProps()} onReply={onReply} />);
    const user = userEvent.setup();
    await user.type(view.getByLabelText("Message draft"), "Try again");
    await user.keyboard("{Control>}{Enter}{/Control}");
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "Try again");
    await user.keyboard("{Control>}{Enter}{/Control}");
    vitest.expect(onReply).toHaveBeenCalledTimes(2);
    vitest.expect(onReply.mock.calls[1]).toEqual(onReply.mock.calls[0]);
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "");
  },
);

vitest.test(
  "editing a draft during a reply does not lose the new text",
  async () => {
    let finish: (result: boolean) => void = () => {};
    const onReply = vitest.vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          finish = resolve;
        }),
    );
    const view = render(<Workspace {...workspaceProps()} onReply={onReply} />);
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft");
    await user.type(draft, "First");
    await user.keyboard("{Control>}{Enter}{/Control}");
    vitest.expect(draft).toHaveProperty("value", "First");
    vitest.expect(view.getByText("Sending message…")).toBeTruthy();
    await user.type(draft, " more");
    await act(async () => finish(true));
    vitest.expect(onReply).toHaveBeenCalledWith(
      request,
      {
        content: [{ type: "text", text: "First" }],
      },
      vitest.expect.any(String),
    );
    vitest.expect(draft).toHaveProperty("value", "First more");
  },
);

vitest.test(
  "an old session reply does not clear the selected session's draft",
  async () => {
    let finish: (result: boolean) => void = () => {};
    const props = workspaceProps();
    const onReply = vitest.vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          finish = resolve;
        }),
    );
    const view = render(<Workspace {...props} onReply={onReply} />);
    const user = userEvent.setup();
    await user.type(view.getByLabelText("Message draft"), "Session A");
    await user.keyboard("{Control>}{Enter}{/Control}");
    view.rerender(
      <Workspace
        {...props}
        selected="session-b"
        onReply={onReply}
        pending={[]}
      />,
    );
    await user.type(view.getByLabelText("Message draft"), "Session B");
    await act(async () => finish(true));
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "Session B");
    view.rerender(<Workspace {...props} onReply={onReply} />);
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "");
  },
);

vitest.test(
  "passive typing appends once, preserves the draft, and does not scroll",
  async () => {
    const view = render(
      <StrictMode>
        <Workspace {...workspaceProps()} pending={[]} />
      </StrictMode>,
    );
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft") as HTMLTextAreaElement;
    const viewport = view.getByRole("region", { name: "Conversation history" });
    await user.type(draft, "Saved");
    draft.setSelectionRange(0, 5);
    viewport.scrollTop = 37;
    const focus = vitest.vi.spyOn(draft, "focus");
    await user.click(view.getByText(/No messages yet/));
    vitest.expect(document.activeElement).not.toBe(draft);
    await user.keyboard(" Hello");
    vitest.expect(document.activeElement).toBe(draft);
    vitest.expect(draft.value).toBe("Saved Hello");
    vitest.expect(draft.selectionStart).toBe(draft.value.length);
    vitest.expect(focus).toHaveBeenCalledWith({ preventScroll: true });
    vitest.expect(viewport.scrollTop).toBe(37);
  },
);

vitest.test(
  "search keeps typing and closing a slideout immediately allows chat entry",
  async () => {
    const view = render(
      <Workspace
        {...workspaceProps()}
        settings={<input aria-label="Setting value" />}
      />,
    );
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft");
    await user.type(view.getByLabelText("Find a session"), "search");
    vitest.expect(draft).toHaveProperty("value", "");
    const opener = view.getByRole("button", { name: "Settings" });
    await user.click(opener);
    await user.keyboard("ignored");
    vitest.expect(draft).toHaveProperty("value", "");
    await user.type(view.getByLabelText("Setting value"), "value");
    await user.keyboard("{Escape}");
    vitest.expect(view.queryByLabelText("Work inspector")).toBeNull();
    vitest.expect(document.activeElement).toBe(opener);
    await user.keyboard("message");
    vitest.expect(draft).toHaveProperty("value", "message");
  },
);

vitest.test(
  "inline Review focus and response input are protected",
  async () => {
    const question: InputRequest = {
      ...request,
      request_id: "question",
      request_type: undefined,
      channel: "agent.question",
    };
    const view = render(
      <Workspace {...workspaceProps()} pending={[question]} />,
    );
    const user = userEvent.setup();
    mockScrollIntoView();
    await user.click(view.getByRole("button", { name: /^Review$/ }));
    const target = view.getByLabelText("Input required: agent.question");
    vitest.expect(document.activeElement).toBe(target);
    await user.keyboard("ignored");
    const draft = view.getByLabelText("Message draft");
    vitest.expect(draft).toHaveProperty("value", "");
    await user.type(view.getByRole("textbox", { name: /Response/ }), "answer");
    vitest.expect(draft).toHaveProperty("value", "");
  },
);

function mockScrollIntoView() {
  const original = Object.getOwnPropertyDescriptor(
    Element.prototype,
    "scrollIntoView",
  );
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    configurable: true,
    value: vitest.vi.fn(),
  });
  vitest.onTestFinished(() => {
    if (original)
      Object.defineProperty(Element.prototype, "scrollIntoView", original);
    else Reflect.deleteProperty(Element.prototype, "scrollIntoView");
  });
}

function PortalBlocker() {
  const [open, setOpen] = useState(true);
  useFocusBlocker(open);
  return open
    ? createPortal(
        <div aria-label="Custom popup">
          <button onClick={() => setOpen(false)}>Close popup</button>
        </div>,
        document.body,
      )
    : null;
}

function ProtectedWidget() {
  const ref = useFocusRegion<HTMLDivElement>("protected");
  return (
    <div ref={ref} tabIndex={0}>
      Custom widget
    </div>
  );
}

vitest.test(
  "custom renderers can protect non-native keyboard surfaces",
  async () => {
    const registry = defaultStepRegistry.extend({
      id: "protected-widget",
      matches: () => true,
      component: ProtectedWidget,
    });
    const view = render(
      <Workspace
        {...workspaceProps()}
        stepRegistry={registry}
        history={[
          {
            position: 1,
            step: {
              id: "step-1",
              workflow_id: "root",
              operation: { kind: "llm.generate" },
              outcome: { kind: "completed", result: "Message" },
            },
          },
        ]}
      />,
    );
    const user = userEvent.setup();
    await user.click(view.getByText("Custom widget"));
    await user.keyboard("ignored");
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "");
    await user.click(
      view.getByRole("region", { name: "Conversation history" }),
    );
    await user.keyboard("allowed");
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "allowed");
  },
);

vitest.test(
  "nested portal blockers prevent routing until the last popup closes",
  async () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} settings={<PortalBlocker />} />);
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft");
    await user.click(view.getByRole("button", { name: "Settings" }));
    await user.click(
      view.getByRole("region", { name: "Conversation history" }),
    );
    await user.keyboard("blocked");
    vitest.expect(draft).toHaveProperty("value", "");
    await user.click(view.getByRole("button", { name: "Close popup" }));
    vitest.expect(view.queryByLabelText("Custom popup")).toBeNull();
    await user.keyboard("still blocked");
    vitest.expect(draft).toHaveProperty("value", "");
    await user.click(view.getByRole("button", { name: "Close inspector" }));
    await user.keyboard("allowed");
    vitest.expect(draft).toHaveProperty("value", "allowed");
  },
);

vitest.test(
  "post-send recovery never overrides newer focus or a session switch",
  async () => {
    let finish: (value: boolean) => void = () => {};
    const props = workspaceProps();
    const onReply = vitest.vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          finish = resolve;
        }),
    );
    const view = render(<Workspace {...props} onReply={onReply} />);
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft");
    await user.type(draft, "First");
    await user.click(view.getByRole("button", { name: "Send message" }));
    const search = view.getByLabelText("Find a session");
    await user.click(search);
    await act(async () => finish(true));
    vitest.expect(document.activeElement).toBe(search);
    await user.type(draft, "Second");
    await user.click(view.getByRole("button", { name: "Send message" }));
    view.rerender(
      <Workspace {...props} selected="session-b" onReply={onReply} />,
    );
    await act(async () => finish(true));
    vitest.expect(document.activeElement).not.toBe(draft);
  },
);

vitest.test(
  "sending by mouse returns focus when the interaction has not changed",
  async () => {
    const view = render(<Workspace {...workspaceProps()} />);
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft");
    await user.type(draft, "Hello");
    await user.click(view.getByRole("button", { name: "Send message" }));
    vitest.expect(document.activeElement).toBe(draft);
  },
);

vitest.test(
  "outside dismissal of step details allows typing without a transcript click",
  async () => {
    const view = render(
      <Workspace
        {...workspaceProps()}
        history={[
          {
            position: 1,
            step: {
              id: "step-1",
              workflow_id: "root",
              operation: { kind: "llm.generate" },
              outcome: { kind: "completed", result: "Message" },
            },
          },
        ]}
      />,
    );
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft");
    const opener = view.getByRole("button", { name: "View details" });
    vitest.expect(opener.tagName).toBe("BUTTON");
    vitest.expect(opener.classList.contains("step-card-details")).toBe(true);
    await user.click(view.getByText("Message"));
    vitest.expect(view.getByLabelText("Work inspector")).toBeTruthy();
    await user.keyboard("blocked");
    vitest.expect(draft).toHaveProperty("value", "");
    await user.click(view.getByRole("button", { name: "Dismiss inspector" }));
    vitest.expect(view.queryByLabelText("Work inspector")).toBeNull();
    vitest.expect(document.activeElement).toBe(opener);
    await user.keyboard("Hello");
    vitest.expect(document.activeElement).toBe(draft);
    vitest.expect(draft).toHaveProperty("value", "Hello");
  },
);

vitest.test(
  "sidebar navigation and its background allow typing while Space still activates buttons",
  async () => {
    const view = render(<Workspace {...workspaceProps()} />);
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft");
    await user.click(view.getByRole("button", { name: "Collapse navigation" }));
    await user.keyboard(" ");
    vitest
      .expect(view.getByRole("button", { name: "Collapse navigation" }))
      .toBeTruthy();
    vitest.expect(draft).toHaveProperty("value", "");
    await user.keyboard("Hi");
    vitest.expect(draft).toHaveProperty("value", "Hi");
    await user.click(view.getByLabelText("Session navigation"));
    await user.keyboard(" there");
    vitest.expect(draft).toHaveProperty("value", "Hi there");
  },
);

vitest.test(
  "selecting a session allows immediate typing into that session's retained draft",
  async () => {
    const props = workspaceProps();
    const sessions = ["a", "b"].map((name) => ({
      execution_id: `session-${name}`,
      reference: `chat:${name}`,
      status: "running",
      active: true,
      pending_inputs: 0,
    }));
    const view = render(
      <Workspace {...props} sessions={sessions} pending={[]} />,
    );
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft");
    await user.type(draft, "Session A");
    await user.click(view.getByRole("button", { name: /chat:b/ }));
    vitest.expect(props.onSelect).toHaveBeenCalledWith("session-b");
    view.rerender(
      <Workspace
        {...props}
        sessions={sessions}
        selected="session-b"
        pending={[]}
      />,
    );
    await user.keyboard("Session B");
    vitest.expect(draft).toHaveProperty("value", "Session B");
    view.rerender(<Workspace {...props} sessions={sessions} pending={[]} />);
    vitest.expect(draft).toHaveProperty("value", "Session A");
  },
);

vitest.test(
  "registered tool approval targets and their buttons retain keyboard ownership",
  async () => {
    mockScrollIntoView();
    const call = { name: "exec", arguments: { command: "echo hello" } };
    const approval: InputRequest = {
      ...request,
      request_id: "step-1",
      request_type: "tool.approval.v1",
      channel: "tool.approval",
      response_type: "json",
      payload: { call },
    };
    const view = render(
      <Workspace
        {...workspaceProps()}
        pending={[approval]}
        history={[
          {
            position: 1,
            step: {
              id: "step-1",
              workflow_id: "root",
              operation: { kind: "llm.tool", input: { call } },
              outcome: {
                kind: "completed",
                result: { content: [{ type: "text", text: "Result" }] },
              },
            },
          },
        ]}
      />,
    );
    const user = userEvent.setup();
    const draft = view.getByLabelText("Message draft");
    await user.click(view.getByRole("button", { name: /^Review$/ }));
    vitest.expect(document.activeElement?.id).toBe("decision-session-a-step-1");
    await user.keyboard("ignored");
    vitest.expect(draft).toHaveProperty("value", "");
    const approve = view.getByRole("button", { name: "Approve once" });
    act(() => approve.focus());
    await user.keyboard("ignored");
    vitest.expect(document.activeElement).toBe(approve);
    vitest.expect(draft).toHaveProperty("value", "");
  },
);
