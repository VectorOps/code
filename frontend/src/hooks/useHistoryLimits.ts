import { useCallback, useState } from "react";
import * as history from "../models/history-limits";

export function useHistoryLimits() {
  const [limits, setLimits] = useState(() => {
    try {
      const stored = localStorage.getItem("runner.historyLimits");
      return stored
        ? history.historyLimits(JSON.parse(stored))
        : history.defaultHistoryLimits;
    } catch {
      return history.defaultHistoryLimits;
    }
  });
  const change = useCallback((value: history.HistoryLimits) => {
    const next = history.historyLimits(value);
    setLimits(next);
    try {
      localStorage.setItem("runner.historyLimits", JSON.stringify(next));
    } catch {}
  }, []);
  return { limits, change };
}
