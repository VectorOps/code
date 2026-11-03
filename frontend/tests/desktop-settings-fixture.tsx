import * as React from "react";
import { vi } from "vitest";
import { RunnerConnection } from "../src/communication/websocket";
import * as preferences from "../src/components/DesktopSettings";

export function settingsHost(
  initial: Partial<preferences.DesktopSettings> = {},
) {
  let value = { ...preferences.defaultDesktopSettings, ...initial };
  const connection = new RunnerConnection("ws://127.0.0.1:1234", "test");
  const request = vi
    .spyOn(connection, "request")
    .mockImplementation(async (method, params) => {
      if (method === "desktop.settings.update") value = { ...value, ...params };
      if (method.startsWith("desktop.settings.")) return structuredClone(value);
      if (method === "runner.info") return { workspace: "/project" };
      if (method === "worktrees.list")
        return { worktrees: [], next_after: null };
      throw new Error(`Unexpected request: ${method}`);
    });
  return { connection, request, read: () => value };
}

function Binding({
  connection,
  children,
}: {
  connection?: RunnerConnection;
  children: React.ReactNode;
}) {
  const { attach } = preferences.useDesktopSettings();
  React.useEffect(() => attach(connection), [attach, connection]);
  return <>{children}</>;
}

export function SettingsHarness({
  connection,
  children,
}: {
  connection?: RunnerConnection;
  children: React.ReactNode;
}) {
  return (
    <preferences.DesktopSettingsProvider>
      <Binding connection={connection}>{children}</Binding>
    </preferences.DesktopSettingsProvider>
  );
}
