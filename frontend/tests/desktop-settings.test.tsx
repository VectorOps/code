import * as testing from "@testing-library/react";
import { expect, test } from "vitest";
import * as preferences from "../src/components/DesktopSettings";
import { AppearanceProvider } from "../src/components/Appearance";
import { settingsHost, SettingsHarness } from "./desktop-settings-fixture";
import { useDebug } from "../src/components/Diagnostics";
import { Settings } from "../src/screens/Settings";

test("backend appearance and all preferences survive a new frontend instance", async () => {
  const host = settingsHost({
    appearance: "dark",
    default_workflow: "custom:agent",
  });
  const view = testing.renderHook(preferences.useDesktopSettings, {
    wrapper: ({ children }) => (
      <SettingsHarness connection={host.connection}>
        <AppearanceProvider>{children}</AppearanceProvider>
      </SettingsHarness>
    ),
  });
  await testing.waitFor(() => expect(view.result.current.ready).toBe(true));
  expect(document.documentElement.dataset.appearance).toBe("dark");
  expect(view.result.current.value.default_workflow).toBe("custom:agent");
  await testing.act(async () => {
    view.result.current.save({ appearance: "light" });
    view.result.current.save({ default_workflow: "review" });
  });
  expect(host.read().appearance).toBe("light");
  expect(host.read().default_workflow).toBe("review");
  view.unmount();
  const next = testing.renderHook(preferences.useDesktopSettings, {
    wrapper: ({ children }) => (
      <SettingsHarness connection={host.connection}>{children}</SettingsHarness>
    ),
  });
  await testing.waitFor(() => expect(next.result.current.ready).toBe(true));
  expect(next.result.current.value).toEqual(host.read());
});

test("failed saves report errors without pretending to persist or changing effective values", async () => {
  const host = settingsHost();
  const view = testing.renderHook(preferences.useDesktopSettings, {
    wrapper: ({ children }) => (
      <SettingsHarness connection={host.connection}>{children}</SettingsHarness>
    ),
  });
  await testing.waitFor(() => expect(view.result.current.ready).toBe(true));
  host.request.mockRejectedValueOnce(new Error("disk full"));
  await testing.act(async () =>
    view.result.current.save({ appearance: "dark" }),
  );
  expect(view.result.current.error).toContain("disk full");
  expect(view.result.current.value.appearance).toBe("system");
  await testing.act(async () =>
    view.result.current.save({ appearance: "light" }),
  );
  expect(view.result.current.value.appearance).toBe("light");
  expect(view.result.current.error).toBe("");
});

test("settings load failures stay visible and do not silently enable defaults", async () => {
  const host = settingsHost();
  host.request.mockRejectedValueOnce(new Error("unavailable"));
  const view = testing.renderHook(preferences.useDesktopSettings, {
    wrapper: ({ children }) => (
      <SettingsHarness connection={host.connection}>{children}</SettingsHarness>
    ),
  });
  await testing.waitFor(() =>
    expect(view.result.current.error).toContain("unavailable"),
  );
  expect(view.result.current.ready).toBe(false);
});

test("late responses from a previous host cannot replace the current host settings", async () => {
  const first = settingsHost();
  const second = settingsHost({ appearance: "dark" });
  let finish!: (value: preferences.DesktopSettings) => void;
  first.request.mockReturnValueOnce(
    new Promise((resolve) => {
      finish = resolve;
    }),
  );
  const view = testing.renderHook(preferences.useDesktopSettings, {
    wrapper: preferences.DesktopSettingsProvider,
  });
  testing.act(() => {
    view.result.current.attach(first.connection);
  });
  await testing.act(async () => {
    view.result.current.attach(second.connection);
  });
  expect(view.result.current.value.appearance).toBe("dark");
  await testing.act(async () =>
    finish({ ...preferences.defaultDesktopSettings, appearance: "light" }),
  );
  expect(view.result.current.value.appearance).toBe("dark");
});

test("debug visibility follows the backend snapshot and resets during reconnect", async () => {
  const host = settingsHost({ debug_enabled: true, development: true });
  const view = testing.renderHook(
    () => ({ ...preferences.useDesktopSettings(), debug: useDebug() }),
    {
      wrapper: preferences.DesktopSettingsProvider,
    },
  );
  expect(view.result.current.debug).toBe(false);
  await testing.act(async () => {
    view.result.current.attach(host.connection);
  });
  expect(view.result.current.debug).toBe(true);
  testing.act(() => {
    view.result.current.attach(undefined);
  });
  expect(view.result.current.debug).toBe(false);
  host.request.mockResolvedValueOnce({
    ...preferences.defaultDesktopSettings,
    development: true,
    debug_enabled: false,
  });
  await testing.act(async () => {
    view.result.current.attach(host.connection);
  });
  expect(view.result.current.debug).toBe(false);
});

test("production settings can enable and disable diagnostics without a dev client", async () => {
  const host = settingsHost({ development: false, debug_enabled: false });
  function Probe() {
    return <p>{useDebug() ? "Diagnostics visible" : "Diagnostics hidden"}</p>;
  }
  const view = testing.render(
    <SettingsHarness connection={host.connection}>
      <Settings
        connected
        connection={{ endpoint: "ws://localhost", token: "test" }}
      />
      <Probe />
    </SettingsHarness>,
  );
  testing.fireEvent.click(
    view.getByRole("button", { name: "Developer tools" }),
  );
  await testing.waitFor(() =>
    expect(view.getByLabelText("Debug interface")).toHaveProperty(
      "disabled",
      false,
    ),
  );
  expect(view.getByText("Diagnostics hidden")).toBeTruthy();
  host.request.mockResolvedValueOnce({
    ...preferences.defaultDesktopSettings,
    debug_mode: "on",
    development: false,
    debug_enabled: true,
  });
  testing.fireEvent.change(view.getByLabelText("Debug interface"), {
    target: { value: "on" },
  });
  await view.findByText("Diagnostics visible");
  expect(host.request).toHaveBeenLastCalledWith("desktop.settings.update", {
    debug_mode: "on",
  });
  host.request.mockResolvedValueOnce({
    ...preferences.defaultDesktopSettings,
    debug_mode: "off",
    development: false,
    debug_enabled: false,
  });
  testing.fireEvent.change(view.getByLabelText("Debug interface"), {
    target: { value: "off" },
  });
  await view.findByText("Diagnostics hidden");
});
