import { useCallback } from "react";
import { useDesktopSettings } from "../components/DesktopSettings";
import * as history from "../models/history-limits";

export function useHistoryLimits() {
  const { value, save } = useDesktopSettings();
  const change = useCallback(
    (value: history.HistoryLimits) => {
      const next = history.historyLimits(value);
      save({ history_limits: next });
    },
    [save],
  );
  return { limits: value.history_limits, change };
}
