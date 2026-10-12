import * as React from "react";
import type { InputRequest } from "../models/runner";

type RequestScope = Pick<InputRequest, "execution_id" | "request_id">;
type Retained = Map<string, Map<string, object>>;

function requestKey(request: RequestScope) {
  return JSON.stringify([request.execution_id, request.request_id]);
}

type FeedState = {
  expanded: Map<string, boolean>;
  expand: (id: string, open: boolean) => void;
  retained: Retained;
};

const State = React.createContext<FeedState | null>(null);

export function FeedStateProvider({
  children,
  requests,
}: {
  children: React.ReactNode;
  requests: RequestScope[];
}) {
  const [expanded, setExpanded] = React.useState(new Map<string, boolean>());
  const retained = React.useRef<Retained>(new Map());
  React.useEffect(() => {
    const active = new Set(requests.map(requestKey));
    for (const key of retained.current.keys())
      if (!active.has(key)) retained.current.delete(key);
  }, [requests]);
  const expand = React.useCallback((id: string, open: boolean) => {
    setExpanded((previous) => {
      if (previous.get(id) === open) return previous;
      return new Map(previous).set(id, open);
    });
  }, []);
  const value = React.useMemo(
    () => ({ expanded, expand, retained: retained.current }),
    [expanded, expand],
  );
  return <State.Provider value={value}>{children}</State.Provider>;
}

export function useFeedExpansion(id: string, initiallyOpen = false) {
  const state = React.useContext(State);
  const [local, setLocal] = React.useState(initiallyOpen);
  const expanded = state?.expanded.get(id) ?? local;
  const expand = React.useCallback(
    (open: boolean) => {
      if (state) state.expand(id, open);
      else setLocal(open);
    },
    [id, state?.expand],
  );
  return [expanded, expand] as const;
}

export function useDecisionDraft(request: RequestScope) {
  return useRetainedState(request, "draft", () => "");
}

export function useExpandWorkflows() {
  const state = React.useContext(State);
  return state?.expand;
}

export function useRetainedValue<T extends object>(
  request: RequestScope,
  slot: string,
  create: () => T,
): T {
  const state = React.useContext(State);
  const local = React.useRef<Retained>(new Map());
  const retained = state?.retained ?? local.current;
  const key = requestKey(request);
  let scope = retained.get(key);
  if (!scope) {
    scope = new Map();
    retained.set(key, scope);
  }
  const cached = scope.get(slot);
  if (cached) return cached as T;
  const value = create();
  scope.set(slot, value);
  return value;
}

export function useRetainedState<T>(
  request: RequestScope,
  slot: string,
  create: () => T,
) {
  const store = useRetainedValue(request, slot, () => ({
    value: create(),
    listeners: new Set<() => void>(),
  }));
  const subscribe = React.useCallback(
    (listener: () => void) => {
      store.listeners.add(listener);
      return () => {
        store.listeners.delete(listener);
      };
    },
    [store],
  );
  const snapshot = React.useCallback(() => store.value, [store]);
  const value = React.useSyncExternalStore(subscribe, snapshot, snapshot);
  const change = React.useCallback(
    (next: T) => {
      if (Object.is(store.value, next)) return;
      store.value = next;
      store.listeners.forEach((listener) => listener());
    },
    [store],
  );
  return [value, change] as const;
}
