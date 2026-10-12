import assert from "node:assert/strict";
import { test } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { SessionSidebar } from "../src/components/SessionSidebar";

test("collapsed sidebar retains an accessible navigation rail", () => {
  const noop = () => {};
  const props = {
    sessions: [],
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
