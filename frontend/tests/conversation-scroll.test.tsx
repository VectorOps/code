import type { ComponentProps } from "react";
import { act, fireEvent, render } from "@testing-library/react";
import * as vitest from "vitest";
import { Workspace } from "../src/screens/Workspace";
import type { HistoryItem } from "../src/models/runner";

class ObservedResize implements ResizeObserver {
  static instances: ObservedResize[] = [];
  targets = new Set<Element>();

  constructor(private callback: ResizeObserverCallback) {
    ObservedResize.instances.push(this);
  }

  observe(target: Element) {
    this.targets.add(target);
  }

  unobserve(target: Element) {
    this.targets.delete(target);
  }

  disconnect() {
    this.targets.clear();
  }

  notify() {
    this.callback([], this);
  }
}

let height: number;
let visible: number;

vitest.beforeEach(() => {
  height = 1000;
  visible = 400;
  const tops = new WeakMap<Element, number>();
  ObservedResize.instances = [];
  vitest.vi.stubGlobal("ResizeObserver", ObservedResize);
  vitest.vi
    .spyOn(Element.prototype, "scrollHeight", "get")
    .mockImplementation(() => height);
  vitest.vi
    .spyOn(Element.prototype, "clientHeight", "get")
    .mockImplementation(() => visible);
  vitest.vi
    .spyOn(Element.prototype, "scrollTop", "get")
    .mockImplementation(function (this: Element) {
      return Math.min(tops.get(this) ?? 0, Math.max(0, height - visible));
    });
  vitest.vi
    .spyOn(Element.prototype, "scrollTop", "set")
    .mockImplementation(function (this: Element, value: number) {
      tops.set(this, Math.max(0, Math.min(value, height - visible)));
    });
});

vitest.afterEach(() => vitest.vi.unstubAllGlobals());

function resize() {
  act(() => {
    for (const observer of ObservedResize.instances) {
      if (observer.targets.size) observer.notify();
    }
  });
}

function scrollUp(area: HTMLElement, top: number, notify = true) {
  fireEvent.wheel(area, { deltaY: -100 });
  area.scrollTop = top;
  if (notify) fireEvent.scroll(area);
}

function message(position: number): HistoryItem {
  return {
    position,
    step: {
      id: `message-${position}`,
      workflow_id: "root",
      operation: { kind: "llm.generate" },
      outcome: { kind: "completed", result: `Message ${position}` },
    },
  };
}

function workspaceProps(): ComponentProps<typeof Workspace> {
  return {
    sessions: [],
    selected: "session-a",
    onSelect: vitest.vi.fn(),
    history: [message(1)],
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
}

vitest.test(
  "rapid appends and delayed content growth keep the feed following",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    vitest.expect(area.scrollTop).toBe(600);

    for (let index = 2; index <= 8; index++) {
      height += 200;
      fireEvent.scroll(area);
      view.rerender(
        <Workspace {...props} history={[message(1), message(index)]} />,
      );
      height += 100;
      resize();
      fireEvent.scroll(area);
      vitest.expect(area.scrollTop).toBe(height - visible);
      vitest
        .expect(view.queryByRole("button", { name: /Jump to latest/ }))
        .toBeNull();
    }
  },
);

vitest.test(
  "viewport changes follow the bottom without requiring new history",
  () => {
    const view = render(<Workspace {...workspaceProps()} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    visible = 250;
    resize();
    vitest.expect(area.scrollTop).toBe(750);
    visible = 600;
    resize();
    vitest.expect(area.scrollTop).toBe(400);
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();
  },
);

vitest.test(
  "scrolling away preserves position and only new activity shows the jump button",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    scrollUp(area, 200);
    view.rerender(<Workspace {...props} busy />);
    visible = 300;
    resize();
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();

    height = 1500;
    view.rerender(<Workspace {...props} history={[message(1), message(2)]} />);
    resize();
    vitest.expect(area.scrollTop).toBe(200);
    fireEvent.click(view.getByRole("button", { name: /Jump to latest/ }));
    vitest.expect(area.scrollTop).toBe(1200);
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();

    visible = 400;
    height = 1800;
    resize();
    vitest.expect(area.scrollTop).toBe(1400);
  },
);

vitest.test("user scroll intent wins even while new content is growing", () => {
  const props = workspaceProps();
  const view = render(<Workspace {...props} />);
  const area = view.getByRole("region", { name: "Conversation history" });
  height = 1500;
  scrollUp(area, 300);
  resize();
  view.rerender(<Workspace {...props} history={[message(1), message(2)]} />);
  vitest.expect(area.scrollTop).toBe(300);
  vitest
    .expect(view.getByRole("button", { name: /Jump to latest/ }))
    .toBeTruthy();
});

vitest.test(
  "a resize callback does not pull a reader back before their scroll event arrives",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    scrollUp(area, 250, false);
    height = 1500;
    resize();
    fireEvent.scroll(area);
    view.rerender(<Workspace {...props} history={[message(1), message(2)]} />);
    vitest.expect(area.scrollTop).toBe(250);
    vitest
      .expect(view.getByRole("button", { name: /Jump to latest/ }))
      .toBeTruthy();
  },
);

vitest.test(
  "a history render respects user movement before the queued scroll event",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    scrollUp(area, 250, false);
    height = 1500;
    view.rerender(<Workspace {...props} history={[message(1), message(2)]} />);
    vitest.expect(area.scrollTop).toBe(250);
    fireEvent.scroll(area);
    vitest
      .expect(view.getByRole("button", { name: /Jump to latest/ }))
      .toBeTruthy();
  },
);

vitest.test(
  "manually reaching the bottom clears new activity and resumes following",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    scrollUp(area, 200);
    height = 1300;
    view.rerender(<Workspace {...props} history={[message(1), message(2)]} />);
    area.scrollTop = height - visible;
    fireEvent.scroll(area);
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();
    height = 1800;
    resize();
    vitest.expect(area.scrollTop).toBe(1400);
  },
);

vitest.test(
  "switching workflows always starts at the bottom and clears new activity",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    scrollUp(area, 200);
    height = 1200;
    view.rerender(<Workspace {...props} history={[message(1), message(2)]} />);
    vitest
      .expect(view.getByRole("button", { name: /Jump to latest/ }))
      .toBeTruthy();
    height = 1400;
    view.rerender(<Workspace {...props} selected="session-b" />);
    vitest.expect(area.scrollTop).toBe(1000);
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();
    scrollUp(area, 300);
    height = 1000;
    view.rerender(<Workspace {...props} />);
    vitest.expect(area.scrollTop).toBe(600);
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();
    height = 1800;
    view.rerender(<Workspace {...props} selected="session-b" />);
    vitest.expect(area.scrollTop).toBe(1400);
    height = 2000;
    resize();
    vitest.expect(area.scrollTop).toBe(1600);
  },
);

vitest.test(
  "workflow switching follows the bottom through empty history and delayed loading",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    scrollUp(area, 200);
    height = 400;
    view.rerender(<Workspace {...props} selected="session-b" history={[]} />);
    resize();
    vitest.expect(area.scrollTop).toBe(0);
    height = 1000;
    view.rerender(
      <Workspace {...props} selected="session-b" history={[message(2)]} />,
    );
    resize();
    vitest.expect(area.scrollTop).toBe(600);
    scrollUp(area, 200);
    height = 400;
    view.rerender(<Workspace {...props} history={[]} />);
    vitest.expect(area.scrollTop).toBe(0);
    height = 1200;
    view.rerender(<Workspace {...props} />);
    resize();
    vitest.expect(area.scrollTop).toBe(800);
    height = 1600;
    resize();
    vitest.expect(area.scrollTop).toBe(1200);
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();
  },
);

vitest.test(
  "the current workflow preserves reading position during temporary content shrinkage",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    scrollUp(area, 200);
    height = 400;
    view.rerender(<Workspace {...props} history={[]} />);
    resize();
    vitest.expect(area.scrollTop).toBe(0);
    height = 1000;
    view.rerender(<Workspace {...props} />);
    resize();
    vitest.expect(area.scrollTop).toBe(200);
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();
  },
);

vitest.test(
  "layout clamping between callbacks does not stop following a switched workflow",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    scrollUp(area, 200);
    height = 2000;
    view.rerender(<Workspace {...props} selected="session-b" />);
    vitest.expect(area.scrollTop).toBe(1600);
    height = 400;
    area.scrollTop = area.scrollTop;
    height = 3000;
    fireEvent.scroll(area);
    resize();
    vitest.expect(area.scrollTop).toBe(2600);
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();
    scrollUp(area, 500);
    view.rerender(<Workspace {...props} />);
    height = 400;
    area.scrollTop = area.scrollTop;
    height = 2400;
    resize();
    fireEvent.scroll(area);
    vitest.expect(area.scrollTop).toBe(2000);
  },
);

vitest.test(
  "a history render follows layout movement before the queued scroll event",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    height = 400;
    area.scrollTop = area.scrollTop;
    height = 1800;
    view.rerender(<Workspace {...props} history={[message(1), message(2)]} />);
    fireEvent.scroll(area);
    vitest.expect(area.scrollTop).toBe(1400);
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();
  },
);

vitest.test.each(["ArrowUp", "PageUp", "Home", " "])(
  "%s keyboard scrolling pauses following",
  (key) => {
    const view = render(<Workspace {...workspaceProps()} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    fireEvent.keyDown(area, { key, shiftKey: key === " " });
    area.scrollTop = 200;
    height = 1800;
    resize();
    fireEvent.scroll(area);
    vitest.expect(area.scrollTop).toBe(200);
  },
);

vitest.test("touch scrolling pauses following before resize delivery", () => {
  const view = render(<Workspace {...workspaceProps()} />);
  const area = view.getByRole("region", { name: "Conversation history" });
  fireEvent.touchStart(area, { touches: [{ clientY: 100 }] });
  fireEvent.touchMove(area, { touches: [{ clientY: 200 }] });
  area.scrollTop = 200;
  height = 1800;
  resize();
  fireEvent.scroll(area);
  vitest.expect(area.scrollTop).toBe(200);
});

vitest.test("dragging the scrollbar pauses following", () => {
  vitest.vi.stubGlobal("PointerEvent", MouseEvent);
  vitest.vi
    .spyOn(HTMLElement.prototype, "offsetWidth", "get")
    .mockReturnValue(420);
  const view = render(<Workspace {...workspaceProps()} />);
  const area = view.getByRole("region", { name: "Conversation history" });
  fireEvent.pointerDown(area, { button: 0, clientX: 410 });
  area.scrollTop = 200;
  height = 1800;
  resize();
  fireEvent.scroll(area);
  vitest.expect(area.scrollTop).toBe(200);
});

vitest.test(
  "pending decisions update the feed without falsely stopping follow mode",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    height = 1400;
    view.rerender(
      <Workspace
        {...props}
        pending={[
          {
            request_id: "question",
            execution_id: "session-a",
            workflow_id: "root",
            channel: "agent.question",
            response_type: "text",
            payload: "What should I do next?",
          },
        ]}
      />,
    );
    visible = 300;
    resize();
    vitest.expect(area.scrollTop).toBe(1100);
    vitest
      .expect(view.queryByRole("button", { name: /Jump to latest/ }))
      .toBeNull();
  },
);

vitest.test(
  "scope changes and unmount disconnect observers and ignore stale callbacks",
  () => {
    const props = workspaceProps();
    const view = render(<Workspace {...props} />);
    const area = view.getByRole("region", { name: "Conversation history" });
    const previous = ObservedResize.instances.find((observer) =>
      observer.targets.has(area),
    )!;
    view.rerender(<Workspace {...props} selected="session-b" />);
    vitest.expect(previous.targets.size).toBe(0);
    scrollUp(area, 200);
    height = 1500;
    act(() => previous.notify());
    vitest.expect(area.scrollTop).toBe(200);
    view.unmount();
    vitest
      .expect(
        ObservedResize.instances.every(
          (observer) => observer.targets.size === 0,
        ),
      )
      .toBe(true);
  },
);
