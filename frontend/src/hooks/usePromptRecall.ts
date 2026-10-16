import * as React from "react";
import type {
  PromptHistoryApi,
  PromptEntry,
} from "../communication/prompt-history";

type Navigation = {
  draft: string;
  entries: PromptEntry[];
  index: number;
  before?: number | null;
  recalled?: string;
};

export function usePromptRecall(
  api: PromptHistoryApi | undefined,
  scope: string,
  value: string,
  restore: (text: string, edge: "start" | "end") => void,
) {
  const navigation = React.useRef<Navigation | undefined>(undefined);
  const generation = React.useRef(0);
  const loading = React.useRef(false);
  const [error, setError] = React.useState("");

  function reset() {
    generation.current++;
    loading.current = false;
    navigation.current = undefined;
  }

  React.useEffect(() => {
    reset();
    setError("");
    return reset;
  }, [api, scope]);
  React.useLayoutEffect(() => {
    if (navigation.current && navigation.current.recalled !== value) reset();
  }, [value]);

  async function older() {
    if (!api || loading.current) return;
    const current = (navigation.current ??= {
      draft: value,
      entries: [],
      index: -1,
    });
    if (current.index + 1 === current.entries.length) {
      if (current.before === null) return;
      const request = generation.current;
      loading.current = true;
      try {
        const page = await api.list({ before: current.before, limit: 1 });
        if (request !== generation.current) return;
        current.entries.push(...page.entries);
        current.before = page.next_before;
        setError("");
      } catch {
        if (request === generation.current)
          setError("Could not load prompt history from the host.");
        return;
      } finally {
        if (request === generation.current) loading.current = false;
      }
    }
    const entry = current.entries[current.index + 1];
    if (!entry) return;
    current.index++;
    current.recalled = entry.text;
    restore(entry.text, "start");
  }

  function newer() {
    const current = navigation.current;
    if (!current) return;
    if (loading.current) {
      generation.current++;
      loading.current = false;
    }
    if (current.index <= 0) {
      const draft = current.draft;
      reset();
      restore(draft, "end");
      return;
    }
    current.index--;
    current.recalled = current.entries[current.index].text;
    restore(current.recalled, "end");
  }

  return { older, newer, reset, error };
}
