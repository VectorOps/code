import * as React from "react";
import type { RunnerConnection } from "../communication/websocket";
import { DebugProvider } from "./Diagnostics";
import {
  defaultHistoryLimits,
  type HistoryLimits,
} from "../models/history-limits";

export type DesktopSettings = {
  appearance: "system" | "light" | "dark";
  default_workflow: string;
  history_limits: HistoryLimits;
  debug_mode: "auto" | "on" | "off";
};

export type SettingsSnapshot = DesktopSettings & {
  development?: boolean;
  debug_enabled?: boolean;
};

export const defaultDesktopSettings: DesktopSettings = {
  appearance: "system",
  default_workflow: "agent",
  history_limits: defaultHistoryLimits,
  debug_mode: "auto",
};

type Preferences = {
  value: SettingsSnapshot;
  ready: boolean;
  error: string;
  attach: (connection?: RunnerConnection) => () => void;
  save: (patch: Partial<DesktopSettings>) => void;
};

const Context = React.createContext<Preferences>({
  value: defaultDesktopSettings,
  ready: false,
  error: "",
  attach: () => () => {},
  save: () => {},
});

export const useDesktopSettings = () => React.useContext(Context);

export function DesktopSettingsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [value, setValue] = React.useState<SettingsSnapshot>(
    defaultDesktopSettings,
  );
  const [ready, setReady] = React.useState(false);
  const [error, setError] = React.useState("");
  const active = React.useRef<{
    connection: RunnerConnection;
    queue: Promise<void>;
  }>(undefined);

  const attach = React.useCallback((connection?: RunnerConnection) => {
    setReady(false);
    setError("");
    if (!connection) {
      active.current = undefined;
      return () => {};
    }
    const session = { connection, queue: Promise.resolve() };
    active.current = session;
    session.queue = connection
      .request<SettingsSnapshot>("desktop.settings.get")
      .then(
        (settings) => {
          if (active.current !== session) return;
          setValue({ ...defaultDesktopSettings, ...settings });
          setReady(true);
        },
        (reason) => {
          if (active.current === session)
            setError(`Cannot load desktop settings: ${reason}`);
        },
      );
    return () => {
      if (active.current === session) active.current = undefined;
    };
  }, []);

  const save = React.useCallback((patch: Partial<DesktopSettings>) => {
    const session = active.current;
    if (!session) {
      setError("Connect to the runner before saving settings.");
      return;
    }
    session.queue = session.queue.then(async () => {
      if (active.current !== session) return;
      try {
        const settings = await session.connection.request<SettingsSnapshot>(
          "desktop.settings.update",
          patch,
        );
        if (active.current !== session) return;
        setValue({ ...defaultDesktopSettings, ...settings });
        setError("");
      } catch (reason) {
        if (active.current === session)
          setError(`Cannot save desktop settings: ${reason}`);
      }
    });
  }, []);

  return (
    <Context.Provider value={{ value, ready, error, attach, save }}>
      <DebugProvider value={ready && value.debug_enabled === true}>
        {children}
      </DebugProvider>
    </Context.Provider>
  );
}

export function DebugSettings() {
  const { value, ready, save } = useDesktopSettings();
  return (
    <>
      <label>
        Debug interface
        <select
          value={value.debug_mode}
          disabled={!ready}
          onChange={(event) => {
            const debug_mode = event.target.value;
            if (
              debug_mode === "auto" ||
              debug_mode === "on" ||
              debug_mode === "off"
            )
              save({ debug_mode });
          }}
        >
          <option value="auto">Auto (follow backend mode)</option>
          <option value="on">On</option>
          <option value="off">Off</option>
        </select>
      </label>
      <p>
        Show raw steps, deep inspection, diagnostic metadata, and internal IDs.
        This does not change tool permissions or workflow execution.
      </p>
      <p>
        Saved on the runner host and shared across projects. No development
        client or restart is required.
      </p>
      <p role="status">
        {ready
          ? `Backend mode: ${value.development ? "development" : "production"}. Debug interface: ${value.debug_enabled ? "on" : "off"}.`
          : "Connect to load developer settings."}
      </p>
    </>
  );
}
