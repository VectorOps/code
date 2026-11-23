import * as testing from "@testing-library/react";
import { expect, test } from "vitest";
import { SessionVisibilitySettings } from "../src/components/SessionVisibilitySettings";
import { useDesktopSettings } from "../src/components/DesktopSettings";
import { Settings } from "../src/screens/Settings";
import { settingsHost, SettingsHarness } from "./desktop-settings-fixture";

test("conversation inactivity defaults to seven days and saves a validated host setting", async () => {
  const host = settingsHost();
  const view = testing.render(
    <SettingsHarness connection={host.connection}>
      <Settings
        connected
        connection={{ endpoint: "ws://localhost", token: "test" }}
      />
    </SettingsHarness>,
  );
  testing.fireEvent.click(view.getByRole("button", { name: "Conversations" }));
  const input = view.getByLabelText("Hide after inactivity (days)");
  const save = view.getByRole("button", { name: "Save inactivity limit" });
  await testing.waitFor(() => expect(input).toHaveProperty("disabled", false));
  expect(input).toHaveProperty("value", "7");
  expect(save).toHaveProperty("disabled", true);
  for (const invalid of ["", "-1", "0.5", "36501"]) {
    testing.fireEvent.change(input, { target: { value: invalid } });
    expect(input.getAttribute("aria-invalid")).toBe("true");
    expect(save).toHaveProperty("disabled", true);
  }
  for (const days of [14, 0]) {
    testing.fireEvent.change(input, { target: { value: String(days) } });
    testing.fireEvent.click(save);
    await testing.waitFor(() =>
      expect(host.read().auto_hide_after_days).toBe(days),
    );
    expect(host.request).toHaveBeenLastCalledWith("desktop.settings.update", {
      auto_hide_after_days: days,
    });
    await testing.waitFor(() => expect(save).toHaveProperty("disabled", true));
  }
});

test("inactivity settings require a connection and older snapshots use the default", async () => {
  const host = settingsHost();
  host.request.mockResolvedValueOnce({ appearance: "system" });
  const view = testing.render(
    <SettingsHarness>
      <SessionVisibilitySettings />
    </SettingsHarness>,
  );
  expect(view.getByLabelText("Hide after inactivity (days)")).toHaveProperty(
    "disabled",
    true,
  );
  view.rerender(
    <SettingsHarness connection={host.connection}>
      <SessionVisibilitySettings />
    </SettingsHarness>,
  );
  await testing.waitFor(() =>
    expect(view.getByLabelText("Hide after inactivity (days)")).toHaveProperty(
      "disabled",
      false,
    ),
  );
  expect(view.getByLabelText("Hide after inactivity (days)")).toHaveProperty(
    "value",
    "7",
  );
});

test("failed inactivity saves retain the effective value and can be retried", async () => {
  const host = settingsHost();
  function Probe() {
    const { value, error } = useDesktopSettings();
    return <p>{error || `Saved: ${value.auto_hide_after_days}`}</p>;
  }
  const view = testing.render(
    <SettingsHarness connection={host.connection}>
      <SessionVisibilitySettings />
      <Probe />
    </SettingsHarness>,
  );
  const input = view.getByLabelText("Hide after inactivity (days)");
  await testing.waitFor(() => expect(input).toHaveProperty("disabled", false));
  host.request.mockRejectedValueOnce(new Error("disk full"));
  testing.fireEvent.change(input, { target: { value: "30" } });
  testing.fireEvent.click(
    view.getByRole("button", { name: "Save inactivity limit" }),
  );
  await view.findByText(/disk full/);
  expect(host.read().auto_hide_after_days).toBe(7);
  testing.fireEvent.click(
    view.getByRole("button", { name: "Save inactivity limit" }),
  );
  await view.findByText("Saved: 30");
});
