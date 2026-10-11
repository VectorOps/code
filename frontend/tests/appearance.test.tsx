import assert from "node:assert/strict";
import { test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  AppearanceControl,
  parseAppearance,
} from "../src/components/Appearance";

test("appearance defaults to system and accepts explicit overrides", () => {
  for (const value of [null, "", "system", "invalid"]) {
    assert.equal(parseAppearance(value), "system");
  }
  for (const value of ["light", "dark"]) {
    assert.equal(parseAppearance(value), value);
  }
});

test("appearance control labels all modes and defaults to system", () => {
  const html = renderToStaticMarkup(createElement(AppearanceControl));
  assert.match(html, /Appearance/);
  assert.match(html, /value="system" selected=""/);
  assert.match(html, /value="light"/);
  assert.match(html, /value="dark"/);
});
