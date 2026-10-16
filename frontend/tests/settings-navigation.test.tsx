import assert from "node:assert/strict";
import { test } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { SessionSidebar } from "../src/components/SessionSidebar";
import { Settings } from "../src/screens/Settings";

test("theme selection lives in settings, not the session sidebar", () => {
  const noop = () => {};
  const sidebar = renderToStaticMarkup(
    createElement(SessionSidebar, {
      sessions: [],
      workflows: [],
      selected: "",
      status: "Disconnected",
      select: noop,
      collapse: noop,
      create: noop,
      configure: noop,
    }),
  );
  assert.doesNotMatch(sidebar, /Appearance|value="dark"/);
  const settings = renderToStaticMarkup(
    createElement(Settings, {
      connected: false,
      connection: { endpoint: "", token: "" },
    }),
  );
  assert.match(settings, /Settings categories/);
  assert.match(settings, /settings-nav/);
  assert.match(settings, /aria-current="page"/);
  assert.match(settings, /value="dark"/);
  assert.doesNotMatch(
    settings,
    /Choose a light or dark theme|saved on this device/,
  );
});
