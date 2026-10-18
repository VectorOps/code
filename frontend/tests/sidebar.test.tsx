import assert from "node:assert/strict";
import { test } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { SessionSidebar } from "../src/components/SessionSidebar";
import * as testing from "@testing-library/react";
import * as vitest from "vitest";
import * as models from "../src/models/runner";

test("collapsed sidebar retains an accessible navigation rail", () => {
  const noop = () => {};
  const props = {
    sessions: [],
    workflows: [],
    selected: "",
    status: "Connected",
    select: noop,
    collapse: noop,
    create: noop,
    configure: noop,
  };
  const rail = renderToStaticMarkup(
    createElement(SessionSidebar, { ...props, collapsed: true }),
  );
  assert.match(rail, /session-sidebar-collapsed/);
  assert.match(rail, /aria-label="Expand navigation"/);
  assert.match(rail, /aria-expanded="false"/);
  assert.match(rail, /aria-label="New conversation"/);
  assert.match(rail, /aria-label="Connection settings"/);
  assert.doesNotMatch(rail, /Find a session|Filter sessions/);
  const expanded = renderToStaticMarkup(createElement(SessionSidebar, props));
  assert.match(expanded, /aria-expanded="true"/);
  assert.match(expanded, /Find a session/);
});

test("sidebar shows workflow names instead of references", () => {
  const noop = () => {};
  const html = renderToStaticMarkup(
    createElement(SessionSidebar, {
      sessions: [
        {
          execution_id: "execution-123",
          reference: "builtin:agent",
          name: "A Joke, Please",
          status: "failed",
          active: false,
          pending_inputs: 0,
        },
      ],
      workflows: [
        {
          descriptor: {
            reference: "builtin:agent",
            name: "Coding agent",
            description: null,
            input_schema: {},
          },
        },
      ],
      selected: "execution-123",
      status: "Connected",
      select: noop,
      collapse: noop,
      create: noop,
      configure: noop,
    }),
  );
  assert.match(html, />A Joke, Please</);
  assert.doesNotMatch(html, />Coding agent</);
  assert.doesNotMatch(html, />builtin:agent</);
});

function sidebarProps(count: number) {
  const sessions: models.Session[] = Array.from(
    { length: count },
    (_, index) => ({
      execution_id: `execution-${index}`,
      reference: "builtin:agent",
      name: `Conversation ${index}`,
      status: "completed",
      active: false,
      pending_inputs: 0,
    }),
  );
  return {
    sessions,
    workflows: [],
    selected: "",
    status: "Connected",
    select: vitest.vi.fn(),
    collapse: vitest.vi.fn(),
    create: vitest.vi.fn(),
    configure: vitest.vi.fn(),
  };
}

test.each([false, true])(
  "connection controls stay compact and accessible when collapsed=%s",
  (collapsed) => {
    const props = sidebarProps(0);
    const view = testing.render(
      <SessionSidebar {...props} collapsed={collapsed} />,
    );
    const configure = view.getByRole("button", { name: "Connection settings" });
    vitest.expect(configure.textContent).toBe("");
    vitest.expect(configure.querySelector("svg")).not.toBeNull();
    vitest
      .expect(view.getByRole("img", { name: "Connection status: Connected" }))
      .toBeTruthy();
    testing.fireEvent.click(configure);
    vitest.expect(props.configure).toHaveBeenCalledOnce();
    view.rerender(
      <SessionSidebar {...props} status="Connecting" collapsed={collapsed} />,
    );
    vitest
      .expect(view.getByRole("img", { name: "Connection status: Connecting" }))
      .toBeTruthy();
  },
);

test("large loaded lists render only one page and search all loaded sessions", () => {
  const props = sidebarProps(10000);
  const view = testing.render(<SessionSidebar {...props} />);
  const rows = () =>
    testing.within(view.getByRole("navigation", { name: "Conversations" }));
  vitest.expect(rows().getAllByRole("button")).toHaveLength(100);
  vitest.expect(rows().getByText("Conversation 0")).toBeTruthy();
  testing.fireEvent.click(
    view.getByRole("button", { name: "Older conversations" }),
  );
  vitest.expect(rows().getAllByRole("button")).toHaveLength(100);
  vitest.expect(rows().getByText("Conversation 100")).toBeTruthy();
  testing.fireEvent.click(rows().getByText("Conversation 100"));
  vitest.expect(props.select).toHaveBeenCalledWith("execution-100");
  testing.fireEvent.click(
    view.getByRole("button", { name: "Newer conversations" }),
  );
  vitest.expect(rows().getByText("Conversation 0")).toBeTruthy();
  testing.fireEvent.change(view.getByRole("searchbox"), {
    target: { value: "Conversation 9999" },
  });
  vitest.expect(rows().getAllByRole("button")).toHaveLength(1);
  vitest.expect(rows().getByText("Conversation 9999")).toBeTruthy();
});

test("unrelated rerenders and search reuse projected session labels", () => {
  const props = sidebarProps(200);
  const display = vitest.vi.spyOn(models, "workflowDisplayName");
  const view = testing.render(<SessionSidebar {...props} />);
  vitest.expect(display).toHaveBeenCalledTimes(200);
  view.rerender(
    <SessionSidebar {...props} status="Updating" create={() => {}} />,
  );
  vitest.expect(display).toHaveBeenCalledTimes(200);
  testing.fireEvent.change(view.getByRole("searchbox"), {
    target: { value: "Conversation 9" },
  });
  vitest.expect(display).toHaveBeenCalledTimes(200);
});

test("collapsed navigation skips session projection", () => {
  const props = sidebarProps(10000);
  const display = vitest.vi.spyOn(models, "workflowDisplayName");
  const view = testing.render(<SessionSidebar {...props} collapsed />);
  vitest.expect(display).not.toHaveBeenCalled();
  view.rerender(<SessionSidebar {...props} />);
  vitest.expect(display).toHaveBeenCalledTimes(10000);
});

test("older sessions load on demand with clear loaded-only search scope", async () => {
  const props = sidebarProps(200);
  const loadMore = vitest.vi.fn(async () => {
    view.rerender(<SessionSidebar {...props} loadMore={loadMore} />);
    return true;
  });
  const view = testing.render(
    <SessionSidebar
      {...props}
      sessions={props.sessions.slice(0, 100)}
      hasMore
      loadMore={loadMore}
    />,
  );
  vitest
    .expect(view.getByText(/Search and filters cover loaded conversations/))
    .toBeTruthy();
  vitest.expect(loadMore).not.toHaveBeenCalled();
  testing.fireEvent.click(
    view.getByRole("button", { name: "Load older conversations" }),
  );
  await testing.waitFor(() =>
    vitest.expect(view.getByText("Conversation 100")).toBeTruthy(),
  );
  vitest.expect(loadMore).toHaveBeenCalledTimes(1);
  vitest
    .expect(
      testing
        .within(view.getByRole("navigation", { name: "Conversations" }))
        .getAllByRole("button"),
    )
    .toHaveLength(100);
});

test("loading disables duplicate older requests and does not show an empty archive", () => {
  const props = sidebarProps(0);
  const loadMore = vitest.vi.fn(async () => true);
  const view = testing.render(
    <SessionSidebar {...props} hasMore loading loadMore={loadMore} />,
  );
  vitest
    .expect(
      view
        .getByRole("button", { name: "Load older conversations" })
        .hasAttribute("disabled"),
    )
    .toBe(true);
  vitest.expect(view.queryByText("No conversations yet.")).toBeNull();
  testing.fireEvent.click(
    view.getByRole("button", { name: "Load older conversations" }),
  );
  vitest.expect(loadMore).not.toHaveBeenCalled();
});

test("new live sessions do not shift the currently displayed older page", () => {
  const props = sidebarProps(200);
  const view = testing.render(<SessionSidebar {...props} />);
  testing.fireEvent.click(
    view.getByRole("button", { name: "Older conversations" }),
  );
  const newest = {
    ...props.sessions[0],
    execution_id: "new",
    name: "New conversation",
  };
  view.rerender(
    <SessionSidebar {...props} sessions={[newest, ...props.sessions]} />,
  );
  const rows = testing.within(
    view.getByRole("navigation", { name: "Conversations" }),
  );
  vitest.expect(rows.getAllByRole("button")).toHaveLength(100);
  vitest.expect(rows.getByText("Conversation 100")).toBeTruthy();
  vitest.expect(rows.getByText("Conversation 199")).toBeTruthy();
  vitest.expect(rows.queryByText("Conversation 99")).toBeNull();
});

test("changing search during a page request does not navigate away from its results", async () => {
  const props = sidebarProps(200);
  let finish = (_value: boolean) => {};
  const loadMore = vitest.vi.fn(
    () => new Promise<boolean>((resolve) => (finish = resolve)),
  );
  const view = testing.render(
    <SessionSidebar
      {...props}
      sessions={props.sessions.slice(0, 100)}
      hasMore
      loadMore={loadMore}
    />,
  );
  testing.fireEvent.click(
    view.getByRole("button", { name: "Load older conversations" }),
  );
  testing.fireEvent.change(view.getByRole("searchbox"), {
    target: { value: "Conversation" },
  });
  view.rerender(<SessionSidebar {...props} loadMore={loadMore} />);
  await testing.act(async () => finish(true));
  vitest.expect(view.getByText("Conversation 0")).toBeTruthy();
  vitest.expect(view.queryByText("Conversation 100")).toBeNull();
});

test("a page with no additional matches keeps the current older results visible", async () => {
  const props = sidebarProps(200);
  const loadMore = vitest.vi.fn(async () => {
    view.rerender(<SessionSidebar {...props} loadMore={loadMore} />);
    return true;
  });
  const view = testing.render(
    <SessionSidebar {...props} hasMore loadMore={loadMore} />,
  );
  testing.fireEvent.click(
    view.getByRole("button", { name: "Older conversations" }),
  );
  testing.fireEvent.click(
    view.getByRole("button", { name: "Load older conversations" }),
  );
  await testing.waitFor(() => vitest.expect(loadMore).toHaveBeenCalledTimes(1));
  vitest.expect(view.getByText("Conversation 100")).toBeTruthy();
  vitest.expect(view.queryByText("Conversation 0")).toBeNull();
});
