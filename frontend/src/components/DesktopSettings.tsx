import * as React from "react";
import type { RunnerConnection } from "../communication/websocket";
import {
  defaultHistoryLimits,
  type HistoryLimits,
} from "../models/history-limits";

export type DesktopSettings = {
  appearance: "system" | "light" | "dark";
  default_workflow: string;
  history_limits: HistoryLimits;
};

export const defaultDesktopSettings: DesktopSettings = {
  appearance: "system",
  default_workflow: "agent",
  history_limits: defaultHistoryLimits,
};

type Preferences = {
  value: DesktopSettings;
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
  const [value, setValue] = React.useState(defaultDesktopSettings);
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
      .request<DesktopSettings>("desktop.settings.get")
      .then(
        (settings) => {
          if (active.current !== session) return;
          setValue(settings);
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
        const settings = await session.connection.request<DesktopSettings>(
          "desktop.settings.update",
          patch,
        );
        if (active.current !== session) return;
        setValue(settings);
        setError("");
      } catch (reason) {
        if (active.current === session)
          setError(`Cannot save desktop settings: ${reason}`);
      }
    });
  }, []);

  return (
    <Context.Provider value={{ value, ready, error, attach, save }}>
      {children}
    </Context.Provider>
  );
}
