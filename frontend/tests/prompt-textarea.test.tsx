import * as React from "react";
import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { PromptTextarea } from "../src/components/PromptTextarea";
import { PromptHistoryApi } from "../src/communication/prompt-history";
import type {
  PromptEntry,
  PromptPage,
} from "../src/communication/prompt-history";
import { RunnerConnection } from "../src/communication/websocket";

function entry(id: number, text: string): PromptEntry {
  return {
    id,
    text,
    project_id: "/host/project",
    execution_id: "old-chat",
    timestamp: "2026-01-01T00:00:00Z",
  };
}

function setup(initial = "Current draft") {
  const api = new PromptHistoryApi(
    new RunnerConnection("ws://localhost", "test"),
  );
  const list = vitest.vi
    .spyOn(api, "list")
    .mockResolvedValue({ entries: [], next_before: null });
  const submit = vitest.vi.fn();
  function Input({ scope = "chat-a" }: { scope?: string }) {
    const [value, change] = React.useState(initial);
    const ref = React.useRef<HTMLTextAreaElement>(null);
    return (
      <>
        <label htmlFor="message-draft">Prompt</label>
        <PromptTextarea
          inputRef={ref}
          value={value}
          change={change}
          submit={submit}
          selected
          historyApi={api}
          executionId={scope}
          scope={scope}
        />
      </>
    );
  }
  const view = testing.render(<Input />);
  const input = view.getByLabelText("Prompt") as HTMLTextAreaElement;
  return { ...view, Input, input, api, list, submit };
}

vitest.afterEach(() => {
  vitest.vi.restoreAllMocks();
  vitest.vi.unstubAllGlobals();
});

vitest.test(
  "recalls global prompts lazily and restores the unsent draft",
  async () => {
    const view = setup();
    view.list.mockResolvedValueOnce({
      entries: [entry(8, "Newest")],
      next_before: 8,
    });
    view.list.mockResolvedValueOnce({
      entries: [entry(3, "Older\nmultiline")],
      next_before: null,
    });
    vitest.expect(view.list).not.toHaveBeenCalled();
    view.input.setSelectionRange(
      view.input.value.length,
      view.input.value.length,
    );
    await testing.act(async () =>
      testing.fireEvent.keyDown(view.input, { key: "ArrowUp" }),
    );
    vitest.expect(view.input.value).toBe("Newest");
    await testing.act(async () =>
      testing.fireEvent.keyDown(view.input, { key: "ArrowUp" }),
    );
    vitest.expect(view.input.value).toBe("Older\nmultiline");
    view.input.setSelectionRange(
      view.input.value.length,
      view.input.value.length,
    );
    testing.fireEvent.keyDown(view.input, { key: "ArrowDown" });
    vitest.expect(view.input.value).toBe("Newest");
    testing.fireEvent.keyDown(view.input, { key: "ArrowDown" });
    vitest.expect(view.input.value).toBe("Current draft");
    vitest
      .expect(view.list.mock.calls)
      .toEqual([[{ before: undefined, limit: 1 }], [{ before: 8, limit: 1 }]]);
    vitest.expect(view.submit).not.toHaveBeenCalled();
  },
);

vitest.test(
  "arrows inside multiline text or a selection do not recall history",
  () => {
    const view = setup("first\nmiddle\nlast");
    view.input.setSelectionRange(8, 8);
    testing.fireEvent.keyDown(view.input, { key: "ArrowUp" });
    testing.fireEvent.keyDown(view.input, { key: "ArrowDown" });
    view.input.setSelectionRange(0, 4);
    testing.fireEvent.keyDown(view.input, { key: "ArrowUp" });
    vitest.expect(view.list).not.toHaveBeenCalled();
    vitest.expect(view.input.value).toBe("first\nmiddle\nlast");
  },
);

vitest.test.each([
  ["one two three", 13, "ArrowLeft", 8],
  ["one two three", 8, "ArrowLeft", 4],
  ["one two three", 6, "ArrowLeft", 4],
  ["one two three", 0, "ArrowRight", 3],
  ["one two three", 3, "ArrowRight", 7],
  ["one two three", 5, "ArrowRight", 7],
  ["one,  \ntwo", 10, "ArrowLeft", 7],
  ["one,  \ntwo", 3, "ArrowRight", 10],
  ["café привет", 11, "ArrowLeft", 5],
  ["café привет", 0, "ArrowRight", 4],
  ["one", 0, "ArrowLeft", 0],
  ["one", 3, "ArrowRight", 3],
  ["", 0, "ArrowLeft", 0],
  ["", 0, "ArrowRight", 0],
] as const)(
  "moves by word in %j from %i with Alt+%s to %i without navigating history",
  (text, position, key, destination) => {
    const view = setup(text);
    view.input.setSelectionRange(position, position);
    const event = testing.createEvent.keyDown(view.input, {
      key,
      altKey: true,
    });
    testing.fireEvent(view.input, event);
    vitest.expect(event.defaultPrevented).toBe(true);
    vitest.expect(view.input.selectionStart).toBe(destination);
    vitest.expect(view.input.selectionEnd).toBe(destination);
    vitest.expect(view.input.value).toBe(text);
    vitest.expect(view.list).not.toHaveBeenCalled();
    vitest.expect(view.submit).not.toHaveBeenCalled();
  },
);

vitest.test("Shift+Alt+arrows extend and reverse word selections", () => {
  const view = setup("one two three");
  view.input.setSelectionRange(7, 7);
  const move = (key: string) =>
    testing.fireEvent.keyDown(view.input, {
      key,
      altKey: true,
      shiftKey: true,
    });
  move("ArrowLeft");
  vitest
    .expect([view.input.selectionStart, view.input.selectionEnd])
    .toEqual([4, 7]);
  vitest.expect(view.input.selectionDirection).toBe("backward");
  move("ArrowLeft");
  vitest
    .expect([view.input.selectionStart, view.input.selectionEnd])
    .toEqual([0, 7]);
  move("ArrowRight");
  move("ArrowRight");
  move("ArrowRight");
  vitest
    .expect([view.input.selectionStart, view.input.selectionEnd])
    .toEqual([7, 13]);
  vitest.expect(view.input.selectionDirection).toBe("forward");
});

vitest.test.each([
  { key: "ArrowLeft" },
  { key: "ArrowRight", ctrlKey: true },
  { key: "ArrowLeft", altKey: true, metaKey: true },
  { key: "ArrowRight", altKey: true, ctrlKey: true },
  { key: "ArrowLeft", altKey: true, isComposing: true },
  { key: "ArrowRight", altKey: true, keyCode: 229 },
])("leaves other arrow shortcuts and IME input untouched: %j", (keys) => {
  const view = setup("one two");
  view.input.setSelectionRange(4, 4);
  const event = testing.createEvent.keyDown(view.input, keys);
  testing.fireEvent(view.input, event);
  vitest.expect(event.defaultPrevented).toBe(false);
  vitest.expect(view.input.selectionStart).toBe(4);
});

vitest.test.each(["edit", "switch chat"])(
  "ignores an in-flight recall after %s",
  async (action) => {
    const view = setup();
    let finish = (_page: PromptPage) => {};
    view.list.mockReturnValue(new Promise((resolve) => (finish = resolve)));
    testing.fireEvent.keyDown(view.input, { key: "ArrowUp" });
    if (action === "switch chat") view.rerender(<view.Input scope="chat-b" />);
    testing.fireEvent.change(view.input, { target: { value: "New draft" } });
    await testing.act(async () =>
      finish({ entries: [entry(1, "Stale")], next_before: null }),
    );
    vitest.expect(view.input.value).toBe("New draft");
  },
);

vitest.test(
  "Down cancels a pending recall and keeps the original draft",
  async () => {
    const view = setup();
    let finish = (_page: PromptPage) => {};
    view.list.mockReturnValue(new Promise((resolve) => (finish = resolve)));
    testing.fireEvent.keyDown(view.input, { key: "ArrowUp" });
    view.input.setSelectionRange(
      view.input.value.length,
      view.input.value.length,
    );
    testing.fireEvent.keyDown(view.input, { key: "ArrowDown" });
    await testing.act(async () =>
      finish({ entries: [entry(1, "Late result")], next_before: null }),
    );
    vitest.expect(view.input.value).toBe("Current draft");
  },
);

vitest.test("a failed recall leaves the draft untouched", async () => {
  const view = setup();
  view.list.mockRejectedValue(new Error("Host unavailable"));
  await testing.act(async () =>
    testing.fireEvent.keyDown(view.input, { key: "ArrowUp" }),
  );
  vitest.expect(view.input.value).toBe("Current draft");
  vitest.expect(view.getByRole("alert").textContent).toContain("host");
});

vitest.test(
  "modified Enter replaces a selection, Ctrl+J adds a line, and IME/repeats never submit",
  () => {
    const view = setup("Hello world");
    view.input.setSelectionRange(5, 11);
    testing.fireEvent.keyDown(view.input, { key: "Enter", altKey: true });
    vitest.expect(view.input.value).toBe("Hello\n");
    vitest.expect(view.input.selectionStart).toBe(6);
    testing.fireEvent.keyDown(view.input, { key: "j", ctrlKey: true });
    vitest.expect(view.input.value).toBe("Hello\n\n");
    testing.fireEvent.keyDown(view.input, { key: "Enter", keyCode: 229 });
    testing.fireEvent.keyDown(view.input, { key: "Enter", isComposing: true });
    testing.fireEvent.keyDown(view.input, { key: "Enter", repeat: true });
    vitest.expect(view.submit).not.toHaveBeenCalled();
  },
);

vitest.test("multiline paste remains one edit and never submits", async () => {
  const view = setup("");
  const user = userEvent.setup();
  await user.click(view.input);
  await user.paste("one\ntwo\nthree");
  vitest.expect(view.input.value).toBe("one\ntwo\nthree");
  vitest.expect(view.submit).not.toHaveBeenCalled();
});

vitest.test(
  "grows, caps, scrolls, shrinks, and responds to viewport resizing",
  () => {
    let content = 28;
    vitest.vi
      .spyOn(HTMLTextAreaElement.prototype, "scrollHeight", "get")
      .mockImplementation(() => content);
    const view = setup("");
    const minimum = parseFloat(view.input.style.height);
    content = 100;
    testing.fireEvent.change(view.input, { target: { value: "wrapped text" } });
    vitest.expect(parseFloat(view.input.style.height)).toBeGreaterThan(minimum);
    content = 10000;
    testing.fireEvent.change(view.input, {
      target: { value: "long pasted text" },
    });
    vitest.expect(view.input.style.height).toBe(view.input.style.maxHeight);
    vitest.expect(view.input.style.overflowY).toBe("auto");
    vitest.vi.stubGlobal("innerHeight", 360);
    testing.fireEvent(window, new Event("resize"));
    vitest.expect(parseFloat(view.input.style.maxHeight)).toBe(120);
    content = 0;
    testing.fireEvent.change(view.input, { target: { value: "" } });
    vitest.expect(parseFloat(view.input.style.height)).toBe(minimum);
    vitest.expect(view.input.style.overflowY).toBe("hidden");
  },
);

vitest.test(
  "Ctrl+R searches host history, cycles matches, and recalls text without submitting",
  async () => {
    const view = setup();
    view.list.mockResolvedValue({
      entries: [entry(9, "Newest result"), entry(4, "Older result")],
      next_before: null,
    });
    const user = userEvent.setup();
    await user.click(view.input);
    await user.keyboard("{Control>}r{/Control}");
    const search = view.getByLabelText("Search submitted prompts");
    vitest.expect(document.activeElement).toBe(search);
    await testing.waitFor(() =>
      vitest
        .expect(
          view
            .getAllByRole("option")
            .filter((node) => node.tagName === "BUTTON"),
        )
        .toHaveLength(2),
    );
    await user.type(search, "result");
    await testing.waitFor(() =>
      vitest.expect(view.list).toHaveBeenLastCalledWith(
        vitest.expect.objectContaining({
          query: "result",
          scope: "global",
          execution_id: "chat-a",
        }),
      ),
    );
    await testing.waitFor(() =>
      vitest.expect(view.queryByText("Loading prompt history…")).toBeNull(),
    );
    await user.keyboard("{Control>}r{/Control}{Enter}");
    vitest.expect(view.queryByRole("dialog")).toBeNull();
    vitest.expect(view.input.value).toBe("Older result");
    vitest.expect(document.activeElement).toBe(view.input);
    vitest.expect(view.submit).not.toHaveBeenCalled();
  },
);

vitest.test(
  "search scopes are host filters and Escape preserves the draft and restores focus",
  async () => {
    const view = setup();
    const user = userEvent.setup();
    testing.fireEvent.keyDown(view.input, { key: "r", ctrlKey: true });
    await user.selectOptions(view.getByLabelText("History scope"), "project");
    await testing.waitFor(() =>
      vitest
        .expect(view.list)
        .toHaveBeenCalledWith(
          vitest.expect.objectContaining({ scope: "project" }),
        ),
    );
    await user.selectOptions(view.getByLabelText("History scope"), "session");
    await testing.waitFor(() =>
      vitest.expect(view.list).toHaveBeenCalledWith(
        vitest.expect.objectContaining({
          scope: "session",
          execution_id: "chat-a",
        }),
      ),
    );
    await user.keyboard("{Escape}");
    vitest.expect(view.queryByRole("dialog")).toBeNull();
    vitest.expect(view.input.value).toBe("Current draft");
    vitest.expect(document.activeElement).toBe(view.input);
  },
);

vitest.test(
  "stale search responses cannot replace results for a newer query",
  async () => {
    const view = setup();
    let finish = (_page: PromptPage) => {};
    view.list.mockImplementation(async (query) => {
      if (query?.query === "old")
        return await new Promise<PromptPage>((resolve) => (finish = resolve));
      return { entries: [entry(7, "Fresh match")], next_before: null };
    });
    testing.fireEvent.keyDown(view.input, { key: "r", ctrlKey: true });
    const search = view.getByLabelText("Search submitted prompts");
    testing.fireEvent.change(search, { target: { value: "old" } });
    await testing.waitFor(() =>
      vitest
        .expect(view.list)
        .toHaveBeenCalledWith(vitest.expect.objectContaining({ query: "old" })),
    );
    testing.fireEvent.change(search, { target: { value: "new" } });
    await testing.waitFor(() =>
      vitest
        .expect(view.getByRole("option", { name: /Fresh match/ }))
        .toBeDefined(),
    );
    await testing.act(async () =>
      finish({ entries: [entry(2, "Stale match")], next_before: null }),
    );
    vitest
      .expect(view.queryByRole("option", { name: /Stale match/ }))
      .toBeNull();
    vitest
      .expect(view.getByRole("option", { name: /Fresh match/ }))
      .toBeDefined();
  },
);

vitest.test(
  "result-button arrow navigation and Enter recall the selected match",
  async () => {
    const view = setup();
    view.list.mockResolvedValue({
      entries: [entry(9, "First result"), entry(4, "Second result")],
      next_before: null,
    });
    testing.fireEvent.keyDown(view.input, { key: "r", ctrlKey: true });
    const first = await view.findByRole("option", { name: /First result/ });
    testing.act(() => first.focus());
    const user = userEvent.setup();
    await user.keyboard("{ArrowDown}{Enter}");
    vitest.expect(view.input.value).toBe("Second result");
    vitest.expect(view.queryByRole("dialog")).toBeNull();
    vitest.expect(view.submit).not.toHaveBeenCalled();
  },
);

vitest.test(
  "compact history previews retain complete multiline text for recall",
  async () => {
    const view = setup();
    const text =
      "First line\n" +
      "A long prompt that should not enlarge the row. ".repeat(30);
    view.list.mockResolvedValue({
      entries: Array.from({ length: 12 }, (_, index) =>
        entry(20 - index, index ? `Prompt ${index}` : text),
      ),
      next_before: null,
    });
    testing.fireEvent.keyDown(view.input, { key: "r", ctrlKey: true });
    const results = await view.findByRole("listbox", {
      name: "Submitted prompts",
    });
    await testing.waitFor(() =>
      vitest
        .expect(testing.within(results).getAllByRole("option"))
        .toHaveLength(12),
    );
    const dialog = view.getByRole("dialog");
    vitest
      .expect(
        dialog
          .querySelector(".prompt-history-controls")
          ?.contains(view.getByLabelText("History scope")),
      )
      .toBe(true);
    const project = results.querySelector(".prompt-history-project");
    vitest.expect(project?.getAttribute("title")).toBe("/host/project");
    vitest
      .expect(results.querySelector("time")?.getAttribute("datetime"))
      .toBe("2026-01-01T00:00:00Z");
    await userEvent.setup().keyboard("{Enter}");
    vitest.expect(view.input.value).toBe(text);
    vitest.expect(view.submit).not.toHaveBeenCalled();
  },
);

vitest.test(
  "prompt history RPCs carry text/session metadata but no browser storage",
  async () => {
    const connection = new RunnerConnection("ws://localhost", "test");
    const request = vitest.vi
      .spyOn(connection, "request")
      .mockResolvedValue({});
    const api = new PromptHistoryApi(connection);
    await api.append("Text only", "chat", "retry-key");
    await api.list({ scope: "project", query: "Text" });
    vitest.expect(request.mock.calls).toEqual([
      [
        "prompt_history.append",
        {
          text: "Text only",
          execution_id: "chat",
          idempotency_key: "retry-key",
        },
      ],
      ["prompt_history.list", { scope: "project", query: "Text" }],
    ]);
  },
);
