import assert from "node:assert/strict";
import { test } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { SessionSidebar } from "../src/components/SessionSidebar";

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
