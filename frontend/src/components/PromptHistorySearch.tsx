import * as React from "react";
import type * as history from "../communication/prompt-history";
import { useFocusBlocker } from "../focus/FocusProvider";

export function PromptHistorySearch({
  api,
  executionId,
  recall,
  close,
}: {
  api: history.PromptHistoryApi;
  executionId: string;
  recall: (text: string) => void;
  close: () => void;
}) {
  const [query, setQuery] = React.useState("");
  const [scope, setScope] = React.useState<history.HistoryScope>("global");
  const [entries, setEntries] = React.useState<history.PromptEntry[]>([]);
  const [before, setBefore] = React.useState<number | null>(null);
  const [selected, setSelected] = React.useState(0);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");
  const generation = React.useRef(0);
  const loadingRef = React.useRef(false);
  const input = React.useRef<HTMLInputElement>(null);
  useFocusBlocker(true);

  async function load(append: boolean, request: number) {
    loadingRef.current = true;
    setLoading(true);
    setError("");
    try {
      const page = await api.list({
        scope,
        execution_id: executionId,
        query,
        before: append ? before : null,
      });
      if (generation.current !== request) return;
      setEntries((previous) =>
        append ? [...previous, ...page.entries] : page.entries,
      );
      setBefore(page.next_before);
      if (append)
        setSelected(
          page.entries.length
            ? entries.length
            : Math.max(0, entries.length - 1),
        );
    } catch {
      if (generation.current === request)
        setError("Could not search prompt history on the host.");
    } finally {
      if (generation.current === request) {
        loadingRef.current = false;
        setLoading(false);
      }
    }
  }

  React.useEffect(() => {
    const request = ++generation.current;
    setEntries([]);
    setBefore(null);
    setSelected(0);
    loadingRef.current = true;
    setLoading(true);
    const timer = setTimeout(() => void load(false, request), 150);
    return () => {
      clearTimeout(timer);
      generation.current++;
    };
  }, [api, executionId, query, scope]);
  React.useEffect(() => input.current?.focus(), []);
  React.useEffect(() => {
    const entry = entries[selected];
    if (entry)
      document
        .getElementById(`prompt-result-${entry.id}`)
        ?.scrollIntoView?.({ block: "nearest" });
    if (entry && document.activeElement?.getAttribute("role") === "option")
      document.getElementById(`prompt-result-${entry.id}`)?.focus();
  }, [entries, selected]);

  function next() {
    if (loadingRef.current) return;
    if (selected + 1 < entries.length) setSelected(selected + 1);
    else if (before !== null) void load(true, generation.current);
  }

  return (
    <div className="prompt-history-overlay">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="prompt-history-title"
        className="prompt-history-dialog"
        onPaste={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (
            event.nativeEvent.isComposing ||
            event.nativeEvent.keyCode === 229
          )
            return;
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            close();
          } else if (
            (event.ctrlKey || event.metaKey) &&
            event.key.toLowerCase() === "r"
          ) {
            event.preventDefault();
            next();
          } else if (event.target instanceof HTMLSelectElement) {
            return;
          } else if (event.key === "ArrowDown") {
            event.preventDefault();
            next();
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setSelected((index) => Math.max(0, index - 1));
          } else if (
            event.key === "Enter" &&
            (event.target === input.current ||
              (event.target instanceof HTMLElement &&
                event.target.getAttribute("role") === "option"))
          ) {
            event.preventDefault();
            if (entries[selected]) recall(entries[selected].text);
          } else if (event.key === "Tab") {
            const controls = Array.from(
              event.currentTarget.querySelectorAll<HTMLElement>(
                "input, select, button:not(:disabled)",
              ),
            );
            const first = controls[0],
              last = controls[controls.length - 1];
            if (event.shiftKey && document.activeElement === first) {
              event.preventDefault();
              last?.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
              event.preventDefault();
              first?.focus();
            }
          }
        }}
      >
        <h2 id="prompt-history-title">Prompt history</h2>
        <label>
          Search submitted prompts
          <input
            ref={input}
            value={query}
            onChange={(event) => {
              generation.current++;
              setQuery(event.target.value);
            }}
            aria-controls="prompt-history-results"
            aria-activedescendant={
              entries[selected]
                ? `prompt-result-${entries[selected].id}`
                : undefined
            }
          />
        </label>
        <label>
          History scope
          <select
            value={scope}
            onChange={(event) => {
              generation.current++;
              setScope(event.target.value as history.HistoryScope);
            }}
          >
            <option value="global">Global</option>
            <option value="project">This project</option>
            <option value="session">This chat</option>
          </select>
        </label>
        {error && <p role="alert">{error}</p>}
        {loading && <p role="status">Loading prompt history…</p>}
        {!loading && !entries.length && !error && (
          <p role="status">No matching prompts.</p>
        )}
        <div
          id="prompt-history-results"
          role="listbox"
          aria-label="Submitted prompts"
          className="prompt-history-results"
        >
          {entries.map((entry, index) => (
            <button
              type="button"
              role="option"
              id={`prompt-result-${entry.id}`}
              key={entry.id}
              aria-selected={index === selected}
              onFocus={() => setSelected(index)}
              onClick={() => recall(entry.text)}
            >
              <span>{entry.text}</span>
              <small>
                {entry.project_id} ·{" "}
                {new Date(entry.timestamp).toLocaleString()}
              </small>
            </button>
          ))}
        </div>
        <p>
          Enter recalls text only. Escape keeps your draft. Ctrl+R selects an
          older match.
        </p>
        <div className="prompt-history-actions">
          {before !== null && (
            <button
              type="button"
              disabled={loading}
              onClick={() => void load(true, generation.current)}
            >
              Load older
            </button>
          )}
          <button type="button" onClick={close}>
            Cancel
          </button>
        </div>
      </section>
    </div>
  );
}
