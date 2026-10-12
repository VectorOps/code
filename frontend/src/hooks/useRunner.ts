import * as React from "react";
import { RunnerApi } from "../communication/runner-api";
import { RunnerConnection } from "../communication/websocket";
import { HistorySync } from "../communication/history-sync";
import { object, text } from "../models/json";
import type { Json, Params } from "../models/json";
import type * as models from "../models/runner";

export function useRunner(initial: models.ConnectionSettings) {
  const [settings, setSettings] = React.useState(initial);
  const [api, setApi] = React.useState<RunnerApi>();
  const [status, setStatus] = React.useState("Disconnected");
  const [error, setError] = React.useState("");
  const [workflows, setWorkflows] = React.useState<models.Workflow[]>([]);
  const [sessions, setSessions] = React.useState<models.Session[]>([]);
  const [selected, setSelected] = React.useState("");
  const [history, setHistory] = React.useState<models.HistoryPage>({
    items: [],
    next_cursor: null,
  });
  const [pending, setPending] = React.useState<models.InputRequest[]>([]);
  const [busy, setBusy] = React.useState(false);
  const [revision, setRevision] = React.useState(0);
  const selectedRef = React.useRef(selected);
  selectedRef.current = selected;
  const startKey = React.useRef({ signature: "", key: "" });
  const resumeKey = React.useRef({ execution: "", key: "" });
  const ready = React.useRef(false);
  const refreshHistory = React.useRef(
    (_reconcile: boolean, _event?: Params) => {},
  );

  function connect(next: models.ConnectionSettings) {
    try {
      const nextApi = new RunnerApi(
        new RunnerConnection(next.endpoint, next.token),
      );
      setSettings(next);
      setApi(nextApi);
      setError("");
    } catch (reason) {
      setError(String(reason));
    }
  }

  React.useEffect(() => {
    if (initial.token) connect(initial);
  }, []);
  React.useEffect(() => {
    if (!api) return;
    let alive = true,
      running = false,
      dirty = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const active = () => alive;
    async function refresh() {
      if (running) {
        dirty = true;
        return;
      }
      running = true;
      try {
        do {
          dirty = false;
          const list = await api!.sessions(active);
          if (!alive) return;
          setSessions(list);
        } while (dirty && alive);
      } catch (reason) {
        if (alive) setError(String(reason));
      } finally {
        running = false;
      }
    }
    const connection = api.connection;
    connection.onStatus = (value) => {
      if (value !== "Connected") ready.current = false;
      if (alive) setStatus(value);
    };
    connection.onReady = () => {
      void (async () => {
        await api.subscribe();
        if (!alive) return;
        ready.current = true;
        refreshHistory.current(true);
        const list = await api.workflows(active);
        if (alive) {
          setWorkflows(list);
          setError("");
          await refresh();
        }
      })().catch((reason) => {
        if (alive) setError(String(reason));
      });
    };
    connection.onEvent = (event) => {
      refreshHistory.current(false, event);
      dirty = true;
      if (!timer)
        timer = setTimeout(() => {
          timer = undefined;
          void refresh();
        }, 150);
    };
    connection.connect();
    const poll = setInterval(() => {
      if (!ready.current) return;
      void refresh();
      refreshHistory.current(false);
    }, 5000);
    return () => {
      alive = false;
      ready.current = false;
      clearTimeout(timer);
      clearInterval(poll);
      connection.close();
    };
  }, [api]);

  React.useEffect(() => {
    setHistory({ items: [], next_cursor: null });
    setPending([]);
    if (!api || !selected) return;
    const sync = new HistorySync(api, selected);
    let alive = true;
    let running = false;
    let dirty = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const active = () => alive && selectedRef.current === selected;
    async function refresh() {
      if (!active() || !ready.current) return;
      if (running) {
        dirty = true;
        return;
      }
      running = true;
      try {
        do {
          dirty = false;
          const [history, pending] = await Promise.all([
            sync.refresh(active),
            api!.pending(selected, active),
          ]);
          if (!active()) return;
          setHistory(history);
          setPending((previous) =>
            JSON.stringify(previous) === JSON.stringify(pending)
              ? previous
              : pending,
          );
        } while (dirty && active() && ready.current);
      } catch (reason) {
        if (active()) setError(String(reason));
      } finally {
        running = false;
      }
    }
    const trigger = (reconcile: boolean, event?: Params) => {
      if (event) {
        if (event.execution_id !== selected) return;
        sync.invalidate(text(object(event.step).id));
      }
      if (reconcile) sync.reconcile();
      dirty = true;
      if (!timer)
        timer = setTimeout(() => {
          timer = undefined;
          void refresh();
        }, 150);
    };
    refreshHistory.current = trigger;
    if (ready.current) trigger(false);
    return () => {
      alive = false;
      clearTimeout(timer);
      if (refreshHistory.current === trigger) refreshHistory.current = () => {};
    };
  }, [api, selected]);
  React.useEffect(() => {
    if (revision) refreshHistory.current(false);
  }, [revision]);

  const action = React.useCallback(
    async function action(
      operation: (api: RunnerApi) => Promise<void>,
    ): Promise<boolean> {
      if (!api) return false;
      setBusy(true);
      setError("");
      try {
        await operation(api);
        setRevision((value) => value + 1);
        return true;
      } catch (reason) {
        setError(String(reason));
        return false;
      } finally {
        setBusy(false);
      }
    },
    [api],
  );
  const reply = React.useCallback(
    (request: models.InputRequest, response: Json, key: string) =>
      action(async (api) => {
        await api.respond(request, response, key);
      }),
    [action],
  );

  function start(reference: string, input: Params) {
    return action(async (api) => {
      const signature = JSON.stringify([reference, input]);
      if (startKey.current.signature !== signature)
        startKey.current = { signature, key: crypto.randomUUID() };
      const result = await api.start(reference, input, startKey.current.key);
      setSelected(result.execution_id);
      startKey.current = { signature: "", key: "" };
      setSessions(await api.sessions());
    });
  }

  function resume() {
    if (!selected) return Promise.resolve(false);
    return action(async (api) => {
      if (resumeKey.current.execution !== selected)
        resumeKey.current = { execution: selected, key: crypto.randomUUID() };
      await api.resume(selected, resumeKey.current.key);
      setSessions(await api.sessions());
      resumeKey.current = { execution: "", key: "" };
    });
  }

  return {
    api,
    settings,
    connect,
    status,
    error,
    dismiss: () => setError(""),
    workflows,
    sessions,
    selected,
    select: setSelected,
    history,
    pending,
    busy,
    enabled: status === "Connected" && !busy,
    start,
    resume,
    stop: () =>
      action(async (api) => {
        await api.stop(selected);
        setSessions(await api.sessions());
      }),
    reply,
  };
}
