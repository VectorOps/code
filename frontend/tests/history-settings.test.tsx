import * as testing from "@testing-library/react";
import { beforeEach, expect, test } from "vitest";
import {
  historyLimits,
  defaultHistoryLimits,
} from "../src/models/history-limits";
import { useHistoryLimits } from "../src/hooks/useHistoryLimits";
import { Settings } from "../src/screens/Settings";

beforeEach(() => localStorage.clear());

test("history limits validate integer page sizes and retention relationships", () => {
  expect(historyLimits(defaultHistoryLimits)).toEqual(defaultHistoryLimits);
  expect(defaultHistoryLimits).toEqual({
    initial: 1000,
    maximum: 2000,
    target: 1000,
    older: 200,
  });
  for (const value of [
    null,
    {},
    { ...defaultHistoryLimits, older: 501 },
    { ...defaultHistoryLimits, initial: 0 },
    { ...defaultHistoryLimits, target: 1.5 },
    { ...defaultHistoryLimits, maximum: 499 },
    { ...defaultHistoryLimits, target: 2001 },
  ])
    expect(() => historyLimits(value)).toThrow();
});

test("history limits persist locally and invalid stored settings fall back to defaults", () => {
  localStorage.setItem("runner.historyLimits", '{"initial":-1}');
  const view = testing.renderHook(useHistoryLimits);
  expect(view.result.current.limits).toEqual(defaultHistoryLimits);
  const limits = { initial: 200, maximum: 800, target: 300, older: 50 };
  testing.act(() => view.result.current.change(limits));
  expect(view.result.current.limits).toEqual(limits);
  view.unmount();
  expect(testing.renderHook(useHistoryLimits).result.current.limits).toEqual(
    limits,
  );
});

test("conversation settings expose all history counts and apply them together", () => {
  function Preferences() {
    const settings = useHistoryLimits();
    return (
      <Settings
        connected={false}
        connection={{ endpoint: "", token: "" }}
        historyLimits={settings.limits}
        onHistoryLimitsChange={settings.change}
      />
    );
  }
  const view = testing.render(<Preferences />);
  const { fireEvent } = testing;
  fireEvent.click(view.getByRole("button", { name: "Conversations" }));
  expect(
    view.getByLabelText("Initial history steps").getAttribute("max"),
  ).toBeNull();
  fireEvent.change(view.getByLabelText("Initial history steps"), {
    target: { value: "200" },
  });
  fireEvent.change(view.getByLabelText("History retention ceiling"), {
    target: { value: "800" },
  });
  fireEvent.change(view.getByLabelText("History trim target"), {
    target: { value: "300" },
  });
  fireEvent.change(view.getByLabelText("Earlier steps per request"), {
    target: { value: "50" },
  });
  fireEvent.click(view.getByRole("button", { name: "Save history settings" }));
  expect(JSON.parse(localStorage.getItem("runner.historyLimits")!)).toEqual({
    initial: 200,
    maximum: 800,
    target: 300,
    older: 50,
  });
  fireEvent.change(view.getByLabelText("History trim target"), {
    target: { value: "900" },
  });
  fireEvent.click(view.getByRole("button", { name: "Save history settings" }));
  expect(view.getByRole("alert").textContent).toContain("retention ceiling");
  fireEvent.click(view.getByRole("button", { name: "Reset history settings" }));
  expect(JSON.parse(localStorage.getItem("runner.historyLimits")!)).toEqual(
    defaultHistoryLimits,
  );
});
