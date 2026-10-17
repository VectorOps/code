import * as React from "react";
import type {
  FileSearchApi,
  FileSearchSnapshot,
} from "../communication/file-search";
import { fileMention, insertFileMention } from "../models/file-mention";

export function useFileCompletion({
  api,
  scope,
  value,
  enabled,
  replace,
}: {
  api?: FileSearchApi;
  scope: string;
  value: string;
  enabled: boolean;
  replace: (text: string, caret: number) => void;
}) {
  const [caret, setCaret] = React.useState({ value: "", start: 0, end: 0 });
  const [focused, setFocused] = React.useState(false);
  const [dismissed, setDismissed] = React.useState("");
  const [response, setResponse] = React.useState<{
    key: string;
    context: string;
    api: FileSearchApi;
    snapshot?: FileSearchSnapshot;
    error?: string;
  }>();
  const [selected, setSelected] = React.useState("");
  const refresh = React.useRef(true);
  const signature = JSON.stringify([scope, value, caret.start, caret.end]);
  const mention =
    enabled &&
    api &&
    focused &&
    caret.value === value &&
    dismissed !== signature
      ? fileMention(value, caret.start, caret.end)
      : undefined;
  const key = mention
    ? JSON.stringify([scope, mention.start, mention.query])
    : "";
  const context = mention ? JSON.stringify([scope, mention.start]) : "";
  const query = mention?.query;
  const tooLong = query !== undefined && Array.from(query).length > 256;
  const current =
    response?.key === key && response.api === api ? response : undefined;
  const visible =
    !tooLong && response?.context === context && response.api === api
      ? response
      : undefined;
  const results = visible?.snapshot?.results ?? [];
  const activePath = results.some((result) => result.path === selected)
    ? selected
    : results[0]?.path;

  React.useEffect(() => {
    if (!api || query === undefined || tooLong) {
      refresh.current = true;
      setResponse(undefined);
      return;
    }
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function search() {
      if (!api || query === undefined) return;
      const shouldRefresh = refresh.current;
      refresh.current = false;
      try {
        const snapshot = await api.search(query, shouldRefresh);
        if (!active) return;
        setResponse({ key, context, api, snapshot });
        setSelected((previous) =>
          snapshot.results.some((result) => result.path === previous)
            ? previous
            : (snapshot.results[0]?.path ?? ""),
        );
        if (snapshot.refreshing) timer = setTimeout(() => void search(), 150);
      } catch {
        if (active)
          setResponse({
            key,
            context,
            api,
            error:
              "File search unavailable. You can still type a path or send your message.",
          });
      }
    }
    timer = setTimeout(() => void search(), 100);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [api, key, context, query, tooLong]);

  React.useEffect(() => {
    setFocused(false);
    setDismissed("");
    refresh.current = true;
  }, [api, scope]);

  function track(element: HTMLTextAreaElement) {
    setCaret({
      value: element.value,
      start: element.selectionStart,
      end: element.selectionEnd,
    });
  }

  function close() {
    setDismissed(signature);
  }

  function choose(path: string) {
    if (!mention || !results.some((result) => result.path === path)) return;
    const next = insertFileMention(value, mention, path);
    setDismissed(JSON.stringify([scope, next.text, next.caret, next.caret]));
    replace(next.text, next.caret);
  }

  function keyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (
      !mention ||
      event.shiftKey ||
      event.altKey ||
      event.metaKey ||
      event.ctrlKey
    )
      return false;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close();
      return true;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const index = results.findIndex((result) => result.path === activePath);
      const next =
        (index + (event.key === "ArrowDown" ? 1 : -1) + results.length) %
        results.length;
      if (results[next]) setSelected(results[next].path);
      return true;
    }
    if ((event.key === "Enter" || event.key === "Tab") && activePath) {
      event.preventDefault();
      if (!event.repeat) choose(activePath);
      return true;
    }
    if (
      event.key === "Enter" &&
      !tooLong &&
      (!current || current.snapshot?.refreshing)
    ) {
      event.preventDefault();
      return true;
    }
    return false;
  }

  return {
    open: Boolean(mention),
    results,
    activePath,
    snapshot: visible?.snapshot,
    error: tooLong
      ? "Keep file queries to 256 characters or fewer."
      : current?.error,
    loading: Boolean(mention && !tooLong && !current),
    track,
    focus: (element: HTMLTextAreaElement) => {
      track(element);
      setFocused(true);
      setDismissed("");
    },
    blur: () => setFocused(false),
    close,
    choose,
    keyDown,
  };
}
