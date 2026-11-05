import assert from "node:assert/strict";
import { test } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { SessionSidebar } from "../src/components/SessionSidebar";
import * as testing from "@testing-library/react";
import * as vitest from "vitest";
import * as models from "../src/models/runner";
import userEvent from "@testing-library/user-event";

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
  assert.match(expanded, /<strong>VOCODE<\/strong>/);
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

test.each([
  "starting",
  "running",
  "paused",
  "stopping",
  "completed",
  "failed",
  "cancelled",
])(
  "sidebar shows an inline %s icon instead of status and ID text",
  (status) => {
    const props = sidebarProps(1);
    props.sessions[0] = { ...props.sessions[0], status };
    const view = testing.render(<SessionSidebar {...props} />);
    const row = testing
      .within(view.getByRole("navigation", { name: "Conversations" }))
      .getByRole("button");
    const icon = testing.within(row).getByRole("img", {
      name: `Workflow status: ${status}`,
    });
    vitest
      .expect(icon.parentElement)
      .toBe(view.getByText("Conversation 0").parentElement);
    vitest.expect(row.textContent).toBe("Conversation 0");
    vitest
      .expect(view.getByText("Conversation 0").title)
      .toBe("Conversation 0");
    testing.fireEvent.click(row);
    vitest.expect(props.select).toHaveBeenCalledWith("execution-0");
  },
);

test("pending input takes icon priority and clears back to the workflow status", () => {
  const props = sidebarProps(1);
  props.sessions[0] = {
    ...props.sessions[0],
    status: "running",
    pending_inputs: 2,
  };
  const activitySessions = new Set(["execution-0"]);
  const view = testing.render(
    <SessionSidebar {...props} activitySessions={activitySessions} />,
  );
  const bubble = view.getByRole("img", {
    name: "Workflow status: Needs input",
  });
  vitest.expect(bubble.querySelectorAll('circle[r="1"]')).toHaveLength(3);
  vitest.expect(bubble.querySelector('circle[r="9"]')).toBeNull();
  vitest.expect(view.getByLabelText("New activity")).toBeTruthy();
  view.rerender(
    <SessionSidebar
      {...props}
      sessions={[{ ...props.sessions[0], pending_inputs: 0 }]}
    />,
  );
  vitest
    .expect(view.queryByRole("img", { name: "Workflow status: Needs input" }))
    .toBeNull();
  vitest
    .expect(view.getByRole("img", { name: "Workflow status: running" }))
    .toBeTruthy();
  vitest
    .expect(
      view
        .getByRole("img", { name: "Workflow status: running" })
        .querySelector('circle[r="9"]'),
    )
    .not.toBeNull();
});

test("sidebar shows a second line only for worktree sessions", () => {
  const props = sidebarProps(2);
  props.sessions[0] = { ...props.sessions[0], worktree_id: "old-tree" };
  const view = testing.render(<SessionSidebar {...props} />);
  const label = view.getByText("Worktree: Unavailable");
  vitest
    .expect(label.previousElementSibling?.className)
    .toBe("session-list-heading");
  vitest
    .expect(label.closest("button"))
    .toBe(view.getByText("Conversation 0").closest("button"));
  vitest
    .expect(view.getByText("Conversation 1").closest("button")?.textContent)
    .toBe("Conversation 1");
});

test.each([undefined, null, "", " \t ", " Investigate the sidebar title "])(
  "sidebar resolves generated description %j before the workflow name",
  (description) => {
    const props = sidebarProps(1);
    props.sessions[0] = {
      ...props.sessions[0],
      name: "Coding agent",
      description,
    };
    const view = testing.render(<SessionSidebar {...props} />);
    const expected = description?.trim() || "Coding agent";
    vitest.expect(view.getByText(expected)).toBeTruthy();
    testing.fireEvent.change(view.getByRole("searchbox"), {
      target: { value: expected },
    });
    vitest.expect(view.getByText(expected)).toBeTruthy();
  },
);

test("sidebar replaces the default name when a generated description arrives", () => {
  const props = sidebarProps(1);
  props.sessions[0] = { ...props.sessions[0], name: "Coding agent" };
  const view = testing.render(<SessionSidebar {...props} />);
  vitest.expect(view.getByText("Coding agent")).toBeTruthy();
  const description =
    "Investigate generated session descriptions in the sidebar";
  view.rerender(
    <SessionSidebar
      {...props}
      sessions={[{ ...props.sessions[0], description }]}
    />,
  );
  vitest.expect(view.getByText(description)).toBeTruthy();
  vitest.expect(view.queryByText("Coding agent")).toBeNull();
});

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

test("collapsed navigation renders one page and reuses session projection on expansion", () => {
  const props = sidebarProps(10000);
  const display = vitest.vi.spyOn(models, "workflowDisplayName");
  const view = testing.render(<SessionSidebar {...props} collapsed />);
  vitest.expect(display).toHaveBeenCalledTimes(10000);
  vitest
    .expect(
      testing
        .within(view.getByRole("navigation", { name: "Conversations" }))
        .getAllByRole("button"),
    )
    .toHaveLength(100);
  view.rerender(<SessionSidebar {...props} />);
  vitest.expect(display).toHaveBeenCalledTimes(10000);
});

test("collapsed status buttons identify conversations and support keyboard selection", async () => {
  const props = sidebarProps(7);
  const statuses = [
    "starting",
    "running",
    "paused",
    "stopping",
    "completed",
    "failed",
    "cancelled",
  ];
  props.sessions = props.sessions.map((session, index) => ({
    ...session,
    status: statuses[index],
    worktree_id: "tree",
  }));
  const view = testing.render(
    <SessionSidebar {...props} collapsed selected="execution-1" />,
  );
  const rows = testing.within(
    view.getByRole("navigation", { name: "Conversations" }),
  );
  for (const [index, status] of statuses.entries()) {
    const label = `Conversation ${index} — ${status}`;
    const row = rows.getByRole("button", { name: label });
    vitest.expect(row.title).toBe(label);
    vitest.expect(row.textContent).toBe("");
    vitest
      .expect(
        testing
          .within(row)
          .getByRole("img", { name: `Conversation ${index}: ${status}` }),
      )
      .toBeTruthy();
  }
  vitest.expect(view.queryByText("Worktree: Unavailable")).toBeNull();
  vitest
    .expect(
      rows
        .getByRole("button", { name: "Conversation 1 — running" })
        .getAttribute("aria-current"),
    )
    .toBe("page");
  rows.getByRole("button", { name: "Conversation 5 — failed" }).focus();
  await userEvent.setup().keyboard("{Enter}");
  vitest.expect(props.select).toHaveBeenCalledExactlyOnceWith("execution-5");
});

test("collapsed icons update pending input and activity without expanding", () => {
  const props = sidebarProps(1);
  props.sessions[0] = {
    ...props.sessions[0],
    status: "running",
    pending_inputs: 1,
  };
  const view = testing.render(
    <SessionSidebar
      {...props}
      collapsed
      activitySessions={new Set(["execution-0"])}
    />,
  );
  vitest
    .expect(
      view.getByRole("button", {
        name: "Conversation 0 — Needs input — New activity",
      }),
    )
    .toBeTruthy();
  vitest
    .expect(view.getByRole("img", { name: "Conversation 0: Needs input" }))
    .toBeTruthy();
  vitest.expect(view.getByLabelText("New activity")).toBeTruthy();
  view.rerender(
    <SessionSidebar
      {...props}
      collapsed
      sessions={[
        { ...props.sessions[0], pending_inputs: 0, status: "completed" },
      ]}
    />,
  );
  vitest
    .expect(view.getByRole("button", { name: "Conversation 0 — completed" }))
    .toBeTruthy();
  vitest.expect(view.queryByLabelText("New activity")).toBeNull();
});

test("collapsing preserves pagination and filters", () => {
  const props = sidebarProps(201);
  const view = testing.render(<SessionSidebar {...props} />);
  testing.fireEvent.click(
    view.getByRole("button", { name: "Older conversations" }),
  );
  view.rerender(<SessionSidebar {...props} collapsed />);
  vitest
    .expect(view.getByRole("button", { name: "Conversation 100 — completed" }))
    .toBeTruthy();
  testing.fireEvent.click(
    view.getByRole("button", { name: "Older conversations" }),
  );
  vitest
    .expect(view.getByRole("button", { name: "Conversation 200 — completed" }))
    .toBeTruthy();
  view.rerender(<SessionSidebar {...props} />);
  vitest.expect(view.getByText("Conversation 200")).toBeTruthy();
  testing.fireEvent.change(view.getByRole("searchbox"), {
    target: { value: "Conversation 19" },
  });
  view.rerender(<SessionSidebar {...props} collapsed />);
  const rows = testing.within(
    view.getByRole("navigation", { name: "Conversations" }),
  );
  vitest.expect(rows.getAllByRole("button")).toHaveLength(11);
  vitest
    .expect(rows.getByRole("button", { name: "Conversation 19 — completed" }))
    .toBeTruthy();
  testing.fireEvent.click(
    view.getByRole("button", { name: "Expand navigation" }),
  );
  vitest.expect(props.collapse).toHaveBeenCalledOnce();
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
