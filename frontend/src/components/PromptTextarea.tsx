import * as React from "react";
import { createPortal, flushSync } from "react-dom";
import type { PromptHistoryApi } from "../communication/prompt-history";
import { usePromptRecall } from "../hooks/usePromptRecall";
import { PromptHistorySearch } from "./PromptHistorySearch";

function atBoundary(
  element: HTMLTextAreaElement,
  edge: "start" | "end",
): boolean {
  const position = element.selectionStart;
  const part =
    edge === "start"
      ? element.value.slice(0, position)
      : element.value.slice(position);
  if (part.includes("\n")) return false;
  if (!element.clientWidth) return true;
  const style = getComputedStyle(element);
  const mirror = document.createElement("div");
  Object.assign(mirror.style, {
    position: "fixed",
    visibility: "hidden",
    whiteSpace: "pre-wrap",
    overflowWrap: "break-word",
    boxSizing: "border-box",
    width: `${element.clientWidth}px`,
    font: style.font,
    lineHeight: style.lineHeight,
    letterSpacing: style.letterSpacing,
    tabSize: style.tabSize,
    padding: style.padding,
  });
  const start = document.createElement("span");
  const caret = document.createElement("span");
  const end = document.createElement("span");
  for (const marker of [start, caret, end]) marker.textContent = "\u200b";
  mirror.append(
    start,
    element.value.slice(0, position),
    caret,
    element.value.slice(position),
    end,
  );
  document.body.append(mirror);
  try {
    return (
      caret.offsetTop === (edge === "start" ? start.offsetTop : end.offsetTop)
    );
  } finally {
    mirror.remove();
  }
}

export function PromptTextarea({
  inputRef,
  value,
  change,
  submit,
  selected,
  historyApi,
  executionId = "",
  scope = "",
}: {
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  value: string;
  change: (text: string) => void;
  submit: () => void;
  selected: boolean;
  historyApi?: PromptHistoryApi;
  executionId?: string;
  scope?: string;
}) {
  const [search, setSearch] = React.useState(false);
  function restore(text: string, edge: "start" | "end") {
    flushSync(() => change(text));
    const element = inputRef.current;
    const position = edge === "start" ? 0 : text.length;
    element?.focus({ preventScroll: true });
    element?.setSelectionRange(position, position);
  }
  const recall = usePromptRecall(historyApi, scope, value, restore);
  React.useEffect(() => setSearch(false), [scope, historyApi]);
  React.useLayoutEffect(() => {
    const element = inputRef.current;
    if (!element) return;
    function size() {
      if (!element) return;
      const style = getComputedStyle(element);
      const line = parseFloat(style.lineHeight) || 24;
      const padding =
        (parseFloat(style.paddingTop) || 0) +
        (parseFloat(style.paddingBottom) || 0);
      const border =
        (parseFloat(style.borderTopWidth) || 0) +
        (parseFloat(style.borderBottomWidth) || 0);
      const minimum = line + padding + border;
      const maximum = Math.min(
        line * 10 + padding + border,
        Math.max(line * 3 + padding + border, window.innerHeight / 3),
      );
      element.style.height = "auto";
      element.style.maxHeight = `${maximum}px`;
      const content = Math.max(minimum, element.scrollHeight + border);
      element.style.height = `${Math.min(content, maximum)}px`;
      element.style.overflowY = content > maximum ? "auto" : "hidden";
    }
    size();
    let width = element.clientWidth;
    const observer =
      typeof ResizeObserver === "undefined"
        ? undefined
        : new ResizeObserver(() => {
            if (element.clientWidth !== width) {
              width = element.clientWidth;
              size();
            }
          });
    observer?.observe(element);
    window.addEventListener("resize", size);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", size);
    };
  }, [inputRef, value]);

  function newline() {
    const element = inputRef.current;
    if (!element) return;
    const position = element.selectionStart;
    const text =
      value.slice(0, position) + "\n" + value.slice(element.selectionEnd);
    recall.reset();
    flushSync(() => change(text));
    element.setSelectionRange(position + 1, position + 1);
  }

  function closeSearch() {
    flushSync(() => setSearch(false));
    inputRef.current?.focus({ preventScroll: true });
  }

  return (
    <>
      <textarea
        ref={inputRef}
        id="message-draft"
        rows={1}
        aria-describedby="message-draft-help"
        placeholder={
          selected
            ? "Write an instruction..."
            : "Select a conversation to write a message"
        }
        disabled={!selected}
        value={value}
        onChange={(event) => {
          recall.reset();
          change(event.target.value);
        }}
        onKeyDown={(event) => {
          if (
            event.nativeEvent.isComposing ||
            event.nativeEvent.keyCode === 229
          )
            return;
          if (
            (event.ctrlKey || event.metaKey) &&
            event.key.toLowerCase() === "r" &&
            historyApi
          ) {
            event.preventDefault();
            recall.reset();
            setSearch(true);
          } else if (event.ctrlKey && event.key.toLowerCase() === "j") {
            event.preventDefault();
            newline();
          } else if (event.key === "Enter") {
            event.preventDefault();
            if (
              event.shiftKey ||
              event.altKey ||
              event.ctrlKey ||
              event.metaKey
            )
              newline();
            else if (!event.repeat) {
              submit();
            }
          } else if (
            historyApi &&
            !event.shiftKey &&
            !event.ctrlKey &&
            !event.metaKey &&
            !event.altKey &&
            event.currentTarget.selectionStart ===
              event.currentTarget.selectionEnd
          ) {
            if (
              event.key === "ArrowUp" &&
              atBoundary(event.currentTarget, "start")
            ) {
              event.preventDefault();
              void recall.older();
            } else if (
              event.key === "ArrowDown" &&
              atBoundary(event.currentTarget, "end")
            ) {
              event.preventDefault();
              recall.newer();
            }
          }
        }}
      />
      <span className="sr-only" id="message-draft-help">
        Enter sends. Shift, Alt, Ctrl, or Cmd+Enter adds a line. Up on the first
        line and Down on the last line recall global prompts. Ctrl+R searches
        prompt history.
      </span>
      {recall.error && <p role="alert">{recall.error}</p>}
      {search &&
        historyApi &&
        createPortal(
          <PromptHistorySearch
            api={historyApi}
            executionId={executionId}
            close={closeSearch}
            recall={(text) => {
              recall.reset();
              closeSearch();
              restore(text, "end");
            }}
          />,
          document.body,
        )}
    </>
  );
}
