import type { ComponentProps } from "react";
import { act, render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { Workspace } from "../src/screens/Workspace";
import type { InputRequest } from "../src/models/runner";

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
