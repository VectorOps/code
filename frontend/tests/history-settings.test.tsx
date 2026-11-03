import * as testing from "@testing-library/react";
import { beforeEach, expect, test } from "vitest";
import {
  historyLimits,
  defaultHistoryLimits,
} from "../src/models/history-limits";
import { useHistoryLimits } from "../src/hooks/useHistoryLimits";
import { Settings } from "../src/screens/Settings";
import { settingsHost, SettingsHarness } from "./desktop-settings-fixture";
import { useDesktopSettings } from "../src/components/DesktopSettings";

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

test("history limits persist on the host and ignore browser storage", async () => {
  localStorage.setItem("runner.historyLimits", '{"initial":-1}');
  const host = settingsHost();
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <SettingsHarness connection={host.connection}>{children}</SettingsHarness>
  );
  const view = testing.renderHook(
    () => ({ ...useHistoryLimits(), ready: useDesktopSettings().ready }),
    { wrapper },
  );
  await testing.waitFor(() => expect(view.result.current.ready).toBe(true));
  expect(view.result.current.limits).toEqual(defaultHistoryLimits);
  const limits = { initial: 200, maximum: 800, target: 300, older: 50 };
  await testing.act(async () => view.result.current.change(limits));
  expect(view.result.current.limits).toEqual(limits);
  expect(host.read().history_limits).toEqual(limits);
  view.unmount();
  const next = testing.renderHook(useHistoryLimits, { wrapper });
  await testing.waitFor(() =>
    expect(next.result.current.limits).toEqual(limits),
  );
  expect(localStorage.getItem("runner.historyLimits")).toBe('{"initial":-1}');
});

test("conversation settings expose all history counts and apply them together", async () => {
  const host = settingsHost();
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
  const view = testing.render(
    <SettingsHarness connection={host.connection}>
      <Preferences />
    </SettingsHarness>,
  );
  await testing.act(async () => {});
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
  await testing.waitFor(() =>
    expect(host.read().history_limits).toEqual({
      initial: 200,
      maximum: 800,
      target: 300,
      older: 50,
    }),
  );
  fireEvent.change(view.getByLabelText("History trim target"), {
    target: { value: "900" },
  });
  fireEvent.click(view.getByRole("button", { name: "Save history settings" }));
  expect(view.getByRole("alert").textContent).toContain("retention ceiling");
  fireEvent.click(view.getByRole("button", { name: "Reset history settings" }));
  await testing.waitFor(() =>
    expect(host.read().history_limits).toEqual(defaultHistoryLimits),
  );
});
