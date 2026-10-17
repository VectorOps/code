import * as React from "react";
import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { PromptTextarea } from "../src/components/PromptTextarea";
import { FileSearchApi } from "../src/communication/file-search";
import type { FileSearchSnapshot } from "../src/communication/file-search";
import { PromptHistoryApi } from "../src/communication/prompt-history";
import { RunnerConnection } from "../src/communication/websocket";

function snapshot(
  paths = ["src/Composer.tsx", "tests/composer.test.tsx"],
  extra: Partial<FileSearchSnapshot> = {},
): FileSearchSnapshot {
  return {
    results: paths.map((path) => ({ path, highlights: [[0, 3]] })),
    index_revision: 1,
    refreshing: false,
    incomplete: false,
    truncated: false,
    refresh_id: 1,
    refresh_error: null,
    ...extra,
  };
}

function setup(initial = "") {
  const connection = new RunnerConnection("ws://localhost", "test");
  const api = new FileSearchApi(connection);
  const search = vitest.vi.spyOn(api, "search").mockResolvedValue(snapshot());
  const history = new PromptHistoryApi(connection);
  const recall = vitest.vi
    .spyOn(history, "list")
    .mockResolvedValue({ entries: [], next_before: null });
  const submit = vitest.vi.fn();
  function Input({
    scope = "chat-a",
    selected = true,
    supported = true,
  }: {
    scope?: string;
    selected?: boolean;
    supported?: boolean;
  }) {
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
          selected={selected}
          scope={scope}
          historyApi={history}
          fileSearch={supported ? api : undefined}
        />
      </>
    );
  }
  const view = testing.render(<Input />);
  return {
    ...view,
    Input,
    input: view.getByLabelText("Prompt") as HTMLTextAreaElement,
    api,
    search,
    recall,
    submit,
    user: userEvent.setup(),
  };
}

vitest.afterEach(() => vitest.vi.restoreAllMocks());

vitest.test.each(["Enter", "Tab"])(
  "@ completion inserts the selected path with %s, not a message",
  async (key) => {
    const view = setup("Fix ");
    vitest.expect(view.search).not.toHaveBeenCalled();
    await view.user.type(view.input, "@comp");
    await view.findByRole("option", { name: "src/Composer.tsx" });
    vitest.expect(view.search).toHaveBeenCalledWith("comp", true);
    await view.user.keyboard(`{ArrowDown}{${key}}`);
    vitest.expect(view.input.value).toBe("Fix @tests/composer.test.tsx ");
    vitest.expect(view.input.selectionStart).toBe(view.input.value.length);
    vitest.expect(document.activeElement).toBe(view.input);
    vitest.expect(view.queryByRole("listbox")).toBeNull();
    vitest.expect(view.submit).not.toHaveBeenCalled();
    vitest.expect(view.recall).not.toHaveBeenCalled();
    await view.user.keyboard("{Enter}");
    vitest.expect(view.submit).toHaveBeenCalledOnce();
  },
);

vitest.test(
  "click selection quotes file paths with spaces and preserves text after the caret",
  async () => {
    const view = setup("Read @fi then fix");
    view.search.mockResolvedValue(snapshot(["docs/my file.md"]));
    await view.user.click(view.input);
    testing.act(() => {
      view.input.setSelectionRange(8, 8);
      testing.fireEvent.select(view.input);
    });
    await view.user.click(
      await view.findByRole("option", { name: "docs/my file.md" }),
    );
    vitest.expect(view.input.value).toBe('Read @"docs/my file.md" then fix');
    vitest.expect(view.submit).not.toHaveBeenCalled();
    vitest.expect(document.activeElement).toBe(view.input);
  },
);

vitest.test(
  "Escape dismisses without clearing the draft; changing the mention reopens",
  async () => {
    const view = setup();
    await view.user.type(view.input, "@src");
    await view.findByRole("option", { name: "src/Composer.tsx" });
    await view.user.keyboard("{Escape}");
    vitest.expect(view.queryByRole("listbox")).toBeNull();
    vitest.expect(view.input.value).toBe("@src");
    await view.user.type(view.input, "/");
    await view.findByRole("option", { name: "src/Composer.tsx" });
  },
);

vitest.test.each(["new query", "switch chat", "disconnect", "dismiss"])(
  "stale results cannot return after %s",
  async (action) => {
    const view = setup();
    let finish: (result: FileSearchSnapshot) => void = () => {};
    view.search.mockImplementation(async (query) =>
      query === "old"
        ? await new Promise<FileSearchSnapshot>((resolve) => {
            finish = resolve;
          })
        : snapshot(["new.ts"]),
    );
    await view.user.type(view.input, "@old");
    await testing.waitFor(() =>
      vitest.expect(view.search).toHaveBeenCalledWith("old", true),
    );
    if (action === "new query") {
      testing.fireEvent.change(view.input, { target: { value: "@new" } });
      await view.findByRole("option", { name: "new.ts" });
    } else if (action === "switch chat")
      view.rerender(<view.Input scope="chat-b" />);
    else if (action === "disconnect")
      view.rerender(<view.Input supported={false} />);
    else await view.user.keyboard("{Escape}");
    await testing.act(async () => finish(snapshot(["stale.ts"])));
    vitest.expect(view.queryByRole("option", { name: "stale.ts" })).toBeNull();
    if (action !== "new query")
      vitest.expect(view.queryByRole("listbox")).toBeNull();
    vitest.expect(view.submit).not.toHaveBeenCalled();
  },
);

vitest.test(
  "keeps previous results and selection throughout the debounce and replacement request",
  async () => {
    const view = setup();
    await view.user.type(view.input, "@comp");
    const first = await view.findByRole("option", { name: "src/Composer.tsx" });
    await view.user.keyboard("{ArrowDown}");
    const chosen = view.getByRole("option", {
      name: "tests/composer.test.tsx",
    });
    const popup = view.getByRole("listbox");
    let finish: (result: FileSearchSnapshot) => void = () => {};
    view.search.mockReturnValueOnce(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    testing.fireEvent.change(view.input, { target: { value: "@compo" } });
    vitest.expect(view.getByRole("listbox")).toBe(popup);
    vitest
      .expect(view.getByRole("option", { name: "src/Composer.tsx" }))
      .toBe(first);
    vitest.expect(chosen.getAttribute("aria-selected")).toBe("true");
    vitest.expect(view.queryByText("Searching files…")).toBeNull();
    vitest.expect(view.search).toHaveBeenCalledTimes(1);
    await testing.waitFor(() =>
      vitest.expect(view.search).toHaveBeenCalledWith("compo", false),
    );
    vitest
      .expect(view.getByRole("option", { name: "src/Composer.tsx" }))
      .toBe(first);
    await testing.act(async () =>
      finish(snapshot(["tests/composer.test.tsx", "new.ts"])),
    );
    vitest
      .expect(view.queryByRole("option", { name: "src/Composer.tsx" }))
      .toBeNull();
    vitest
      .expect(
        view
          .getByRole("option", { name: "tests/composer.test.tsx" })
          .getAttribute("aria-selected"),
      )
      .toBe("true");
    vitest.expect(view.getByRole("listbox")).toBe(popup);
  },
);

vitest.test(
  "retained suggestions can be selected while a replacement query is pending",
  async () => {
    const view = setup();
    await view.user.type(view.input, "@comp");
    await view.findByRole("option", { name: "src/Composer.tsx" });
    let finish: (result: FileSearchSnapshot) => void = () => {};
    view.search.mockReturnValueOnce(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    testing.fireEvent.change(view.input, { target: { value: "@compo" } });
    await testing.waitFor(() =>
      vitest.expect(view.search).toHaveBeenCalledWith("compo", false),
    );
    await view.user.keyboard("{Tab}");
    vitest.expect(view.input.value).toBe("@src/Composer.tsx ");
    await testing.act(async () => finish(snapshot(["late.ts"])));
    vitest.expect(view.queryByRole("listbox")).toBeNull();
    vitest.expect(view.submit).not.toHaveBeenCalled();
  },
);

vitest.test(
  "refresh snapshots retain keyboard selection by path as the ranking changes",
  async () => {
    const view = setup();
    let finish: (result: FileSearchSnapshot) => void = () => {};
    view.search.mockResolvedValueOnce(
      snapshot(["first.ts", "chosen.ts"], { refreshing: true }),
    );
    view.search.mockReturnValueOnce(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    await view.user.type(view.input, "@ts");
    await view.findByRole("option", { name: "first.ts" });
    await view.user.keyboard("{ArrowDown}");
    await testing.waitFor(() =>
      vitest.expect(view.search).toHaveBeenCalledTimes(2),
    );
    await testing.act(async () =>
      finish(snapshot(["chosen.ts", "first.ts", "new.ts"])),
    );
    vitest
      .expect(
        view
          .getByRole("option", { name: "chosen.ts" })
          .getAttribute("aria-selected"),
      )
      .toBe("true");
    await view.user.keyboard("{Enter}");
    vitest.expect(view.input.value).toBe("@chosen.ts ");
    vitest.expect(view.search.mock.calls).toEqual([
      ["ts", true],
      ["ts", false],
    ]);
  },
);

vitest.test(
  "empty results and unavailable search never prevent normal editing or sending",
  async () => {
    const view = setup();
    view.search.mockResolvedValueOnce(snapshot([]));
    await view.user.type(view.input, "@missing");
    await view.findByText("No matching files");
    await view.user.keyboard("{Enter}");
    vitest.expect(view.submit).toHaveBeenCalledOnce();
    view.search.mockRejectedValue(new Error("unsupported_method"));
    await view.user.type(view.input, "/more");
    await view.findByText(/File search unavailable/);
    await view.user.keyboard("{Enter}");
    vitest.expect(view.submit).toHaveBeenCalledTimes(2);
    vitest.expect(view.input.value).toBe("@missing/more");
  },
);

vitest.test(
  "overlong queries do not hit the host or block sending",
  async () => {
    const view = setup();
    await view.user.click(view.input);
    testing.fireEvent.change(view.input, {
      target: { value: `@${"x".repeat(257)}` },
    });
    await view.findByText("Keep file queries to 256 characters or fewer.");
    vitest.expect(view.search).not.toHaveBeenCalled();
    await view.user.keyboard("{Enter}");
    vitest.expect(view.submit).toHaveBeenCalledOnce();
  },
);

vitest.test(
  "modifier Enter, IME, repeats, email addresses and selections do not accidentally select files",
  async () => {
    const view = setup();
    await view.user.type(view.input, "me@example.com");
    vitest.expect(view.queryByRole("listbox")).toBeNull();
    vitest.expect(view.search).not.toHaveBeenCalled();
    await view.user.clear(view.input);
    await view.user.type(view.input, "@src");
    await view.findByRole("option", { name: "src/Composer.tsx" });
    testing.fireEvent.keyDown(view.input, { key: "Enter", isComposing: true });
    testing.fireEvent.keyDown(view.input, { key: "Enter", repeat: true });
    vitest.expect(view.input.value).toBe("@src");
    testing.fireEvent.keyDown(view.input, { key: "Enter", shiftKey: true });
    vitest.expect(view.input.value).toBe("@src\n");
    vitest.expect(view.submit).not.toHaveBeenCalled();
  },
);

vitest.test(
  "file highlights use Python codepoint offsets, including non-BMP paths",
  async () => {
    const view = setup();
    view.search.mockResolvedValue(snapshot(["😀file.ts"]));
    await view.user.type(view.input, "@fi");
    const option = await view.findByRole("option", { name: "😀file.ts" });
    vitest.expect(option.querySelector("mark")?.textContent).toBe("😀fi");
  },
);

vitest.test(
  "autocomplete RPC is bounded and never sends a client filesystem root",
  async () => {
    const connection = new RunnerConnection("ws://localhost", "test");
    const request = vitest.vi
      .spyOn(connection, "request")
      .mockResolvedValue(snapshot());
    await new FileSearchApi(connection).search("src", true);
    vitest
      .expect(request)
      .toHaveBeenCalledExactlyOnceWith("file_search.search", {
        query: "src",
        limit: 20,
        refresh: true,
      });
  },
);
