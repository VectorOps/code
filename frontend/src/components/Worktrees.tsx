import * as React from "react";
import { WorktreesApi } from "../communication/worktrees-api";
import type { RunnerApi } from "../communication/runner-api";
import type { Worktree } from "../models/worktrees";
import "./worktrees.css";

type Catalog = {
  api?: WorktreesApi;
  records: Worktree[];
  loading: boolean;
  error: string;
  refresh: () => Promise<void>;
  remember: (record: Worktree) => void;
};

const Context = React.createContext<Catalog>({
  records: [],
  loading: false,
  error: "",
  refresh: async () => {},
  remember: () => {},
});

export function WorktreesProvider({
  runner,
  children,
}: {
  runner?: RunnerApi;
  children: React.ReactNode;
}) {
  const api = React.useMemo(
    () => runner && new WorktreesApi(runner.connection),
    [runner],
  );
  const [records, setRecords] = React.useState<Worktree[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");
  const generation = React.useRef(0);
  const currentApi = React.useRef(api);
  currentApi.current = api;
  const remember = React.useCallback(
    (record: Worktree) => {
      if (currentApi.current !== api) return;
      setRecords((previous) => [
        ...previous.filter((item) => item.id !== record.id),
        record,
      ]);
    },
    [api],
  );
  const refresh = React.useCallback(async () => {
    if (!api) return;
    const current = ++generation.current;
    setLoading(true);
    setError("");
    try {
      const next = await api.list(true);
      if (current === generation.current) setRecords(next);
    } catch (reason) {
      if (current === generation.current) setError(String(reason));
    } finally {
      if (current === generation.current) setLoading(false);
    }
  }, [api]);
  React.useLayoutEffect(() => {
    setRecords([]);
    setError("");
    setLoading(false);
    return () => {
      generation.current++;
    };
  }, [api]);
  return (
    <Context.Provider
      value={{ api, records, loading, error, refresh, remember }}
    >
      {children}
    </Context.Provider>
  );
}

export function useWorktrees() {
  return React.useContext(Context);
}

export function useWorktreeList() {
  const catalog = useWorktrees();
  React.useEffect(() => {
    void catalog.refresh();
  }, [catalog.refresh]);
  return catalog;
}

export function WorktreeLabel({ id, path }: { id: string; path?: string }) {
  const { api, records, remember } = useWorktrees();
  const record = records.find((item) => item.id === id);
  const [error, setError] = React.useState("");
  React.useEffect(() => {
    let active = true;
    setError("");
    if (api && !record)
      void api.get(id).then(
        (next) => {
          if (active) remember(next);
        },
        (reason) => {
          if (active) setError(String(reason));
        },
      );
    return () => {
      active = false;
    };
  }, [api, id, Boolean(record), remember]);
  return (
    <span
      className="worktree-label"
      title={
        error ||
        `${record?.path || path || id}${record ? ` · ${record.branch} · ${record.status}` : ""}`
      }
    >
      Worktree: {record?.name || id}
    </span>
  );
}
