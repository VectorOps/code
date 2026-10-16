import * as React from "react";
import { RunnerApi } from "../communication/runner-api";
import { RunnerConnection } from "../communication/websocket";
import { HistorySync } from "../communication/history-sync";
import { SessionSync } from "../communication/session-sync";
import { object, text } from "../models/json";
import type { Json, Params } from "../models/json";
import type * as models from "../models/runner";
import * as chatInput from "../models/chat-input";

export function useRunner(initial: models.ConnectionSettings) {
  const [settings, setSettings] = React.useState(initial);
  const [api, setApi] = React.useState<RunnerApi>();
  const [status, setStatus] = React.useState("Disconnected");
  const [error, setError] = React.useState("");
  const [workflows, setWorkflows] = React.useState<models.Workflow[]>([]);
  const [sessions, setSessions] = React.useState<models.Session[]>([]);
  const sessionSync = React.useMemo(
    () => (api ? new SessionSync(api, setSessions) : undefined),
    [api],
  );
  const [activitySessions, setActivitySessions] = React.useState<Set<string>>(
    () => new Set(),
  );
  const [selected, setSelected] = React.useState("");
  const [execution, setExecution] = React.useState<models.Execution>();
  const [branching, setBranching] = React.useState<models.BranchAccepted>();
  const [history, setHistory] = React.useState<models.HistoryPage>({
    items: [],
    next_cursor: null,
  });
  const [pending, setPending] = React.useState<models.InputRequest[]>([]);
  const [steering, setSteering] = React.useState<{
    address: models.InboxAddress;
    messages: models.InboxMessage[];
    count: number;
  }>();
  const [busy, setBusy] = React.useState(false);
  const [revision, setRevision] = React.useState(0);
  const selectedRef = React.useRef(selected);
  selectedRef.current = selected;
  const startKey = React.useRef({ signature: "", key: "" });
  const resumeKey = React.useRef({ execution: "", key: "" });
  const branchKey = React.useRef({ signature: "", key: "" });
  const branchOutcome = React.useRef("");
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
    if (!api || !sessionSync) return;
    let alive = true;
    let generation = 0;
    const connection = api.connection;
    connection.onStatus = (value) => {
      if (value !== "Connected") {
        generation++;
        sessionSync.invalidate();
        ready.current = false;
        setBranching(undefined);
      }
      if (alive) setStatus(value);
    };
    connection.onReady = () => {
      const currentGeneration = ++generation;
      const current = () => alive && generation === currentGeneration;
      void (async () => {
        if (!(await sessionSync.synchronize(current))) return;
        ready.current = true;
        refreshHistory.current(true);
        const list = await api.workflows(current);
        if (current()) {
          setWorkflows(list);
          setError("");
        }
      })().catch((reason) => {
        if (current()) setError(String(reason));
      });
    };
    connection.onEvent = (event) => {
      if (!alive || event.kind === "workflow.updated") return;
      const eventExecution = text(event.execution_id);
      if (eventExecution && eventExecution !== selectedRef.current) {
        setActivitySessions((previous) =>
          new Set(previous).add(eventExecution),
        );
      }
      if (event.kind === "execution.updated") {
        const update = sessionSync.apply(event);
        if (update && eventExecution === selectedRef.current) {
          setExecution((previous) =>
            previous?.id === eventExecution
              ? { ...previous, status: update.summary.status }
              : previous,
          );
          if (update.previous?.pending_inputs !== update.summary.pending_inputs)
            refreshHistory.current(false, event);
        }
        return;
      }
      if (
        event.kind === "execution.branch_applied" ||
        event.kind === "execution.branch_failed"
      ) {
        const operation = text(event.operation_id);
        branchOutcome.current = JSON.stringify([event.execution_id, operation]);
        setBranching((previous) =>
          previous?.operation_id === operation &&
          previous.execution_id === event.execution_id
            ? undefined
            : previous,
        );
        if (
          event.kind === "execution.branch_failed" &&
          event.execution_id === selectedRef.current
        )
          setError(text(event.error) || "Branch preparation failed.");
      }
      refreshHistory.current(false, event);
    };
    connection.connect();
    return () => {
      alive = false;
      sessionSync.invalidate();
      ready.current = false;
      connection.close();
    };
  }, [api, sessionSync]);

  React.useEffect(() => {
    if (selected)
      setActivitySessions((previous) => {
        const next = new Set(previous);
        next.delete(selected);
        return next;
      });
  }, [selected]);

  React.useEffect(() => {
    setHistory({ items: [], next_cursor: null });
    setPending([]);
    setSteering(undefined);
    setExecution(undefined);
    if (!api || !sessionSync || !selected) return;
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
          const previousSummary = sessionSync!.get(selected);
          let execution = await api!.execution(selected);
          const [history, pending, inboxes] = await Promise.all([
            sync.refresh(active),
            api!.pending(selected, active),
            ["running", "paused", "starting"].includes(execution.status)
              ? api!.inboxes(selected).catch(() => [])
              : Promise.resolve([]),
          ]);
          if (!active()) return;
          const latestSummary = sessionSync!.get(selected);
          if (latestSummary && latestSummary !== previousSummary)
            execution = { ...execution, status: latestSummary.status };
          setExecution((previous) =>
            JSON.stringify(previous) === JSON.stringify(execution)
              ? previous
              : execution,
          );
          setHistory(history);
          setPending((previous) =>
            JSON.stringify(previous) === JSON.stringify(pending)
              ? previous
              : pending,
          );
          const mailbox = inboxes.find(
            (inbox) =>
              inbox.address.workflow_id === execution.root_workflow_id &&
              inbox.address.conversation_id === "steer" &&
              inbox.status === "open",
          );
          if (!mailbox) setSteering(undefined);
          else {
            const messages = await api!.inboxMessages(mailbox.address);
            if (active())
              setSteering({
                address: mailbox.address,
                messages,
                count: mailbox.queued_messages,
              });
          }
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
        sync.invalidate(text(event.input_step_id));
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
  }, [api, sessionSync, selected]);
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
  const steer = React.useCallback(
    (value: string) => {
      if (!api || !steering || !value.trim()) return Promise.resolve(false);
      return action(async (client) => {
        await client.sendSteering(
          steering.address,
          chatInput.chatMessage(value),
        );
      });
    },
    [action, api, steering],
  );

  const branch = React.useCallback(
    (stepId: string, branchAction: models.BranchAction) => {
      if (!selected) return Promise.resolve(false);
      return action(async (api) => {
        const signature = JSON.stringify([selected, stepId, branchAction]);
        if (branchKey.current.signature !== signature)
          branchKey.current = { signature, key: crypto.randomUUID() };
        const accepted = await api.branch(
          selected,
          stepId,
          branchAction,
          branchKey.current.key,
        );
        branchKey.current = { signature: "", key: "" };
        if (
          branchOutcome.current !==
          JSON.stringify([accepted.execution_id, accepted.operation_id])
        )
          setBranching(accepted);
      });
    },
    [action, selected],
  );

  function start(reference: string, input: Params) {
    return action(async (api) => {
      const signature = JSON.stringify([reference, input]);
      if (startKey.current.signature !== signature)
        startKey.current = { signature, key: crypto.randomUUID() };
      const result = await api.start(reference, input, startKey.current.key);
      setSelected(result.execution_id);
      startKey.current = { signature: "", key: "" };
    });
  }

  function resume() {
    if (!selected) return Promise.resolve(false);
    return action(async (api) => {
      if (resumeKey.current.execution !== selected)
        resumeKey.current = { execution: selected, key: crypto.randomUUID() };
      await api.resume(selected, resumeKey.current.key);
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
    activitySessions,
    selected,
    select: setSelected,
    history,
    execution,
    branching,
    pending,
    steering,
    busy,
    enabled: status === "Connected" && !busy,
    start,
    resume,
    stop: () =>
      action(async (api) => {
        await api.stop(selected);
      }),
    reply,
    steer,
    branch,
  };
}
