import * as React from "react";
import { createPortal, flushSync } from "react-dom";
import type { PromptHistoryApi } from "../communication/prompt-history";
import type { FileSearchApi } from "../communication/file-search";
import { usePromptRecall } from "../hooks/usePromptRecall";
import { useFileCompletion } from "../hooks/useFileCompletion";
import { PromptHistorySearch } from "./PromptHistorySearch";
import { FileCompletions } from "./FileCompletions";
import { useComposerTarget } from "../focus/FocusProvider";
import { isHistoryShortcut } from "../focus/shortcuts";
import * as chatInput from "../models/chat-input";

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
  fileSearch,
  executionId = "",
  scope = "",
  attach,
}: {
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  value: string;
  change: (text: string) => void;
  submit: () => void;
  selected: boolean;
  historyApi?: PromptHistoryApi;
  fileSearch?: FileSearchApi;
  executionId?: string;
  scope?: string;
  attach?: (files: File[]) => void;
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
  const completionId = React.useId();
  const files = useFileCompletion({
    api: fileSearch,
    scope,
    value,
    enabled: selected && !search,
    replace: (text, caret) => {
      recall.reset();
      flushSync(() => change(text));
      const element = inputRef.current;
      element?.focus({ preventScroll: true });
      element?.setSelectionRange(caret, caret);
      if (element) files.track(element);
    },
  });
  function openHistory() {
    if (!selected || !historyApi) return false;
    recall.reset();
    setSearch(true);
    return true;
  }
  useComposerTarget(inputRef, {
    isEditable: () =>
      Boolean(
        inputRef.current &&
        !inputRef.current.disabled &&
        !inputRef.current.readOnly,
      ),
    focus: () => inputRef.current?.focus({ preventScroll: true }),
    beginTyping: (text) => {
      recall.reset();
      restore(value + text, "end");
    },
    openHistory,
    paste: (clipboard) => {
      const { text, files } = chatInput.clipboardInput(clipboard);
      if (!selected || (!text && (!files.length || !attach))) return false;
      recall.reset();
      if (text) restore(value + text, "end");
      else inputRef.current?.focus({ preventScroll: true });
      if (files.length) attach?.(files);
      return true;
    },
  });
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
      <div className="prompt-input">
        {files.open && (
          <FileCompletions
            id={completionId}
            results={files.results}
            activePath={files.activePath}
            choose={files.choose}
            loading={files.loading}
            snapshot={files.snapshot}
            error={files.error}
          />
        )}
        <textarea
          ref={inputRef}
          id="message-draft"
          rows={1}
          aria-describedby="message-draft-help"
          role={files.open ? "combobox" : undefined}
          aria-autocomplete={fileSearch ? "list" : undefined}
          aria-expanded={files.open ? true : undefined}
          aria-controls={files.open ? completionId : undefined}
          aria-activedescendant={
            files.open && files.activePath
              ? `${completionId}-${files.results.findIndex((result) => result.path === files.activePath)}`
              : undefined
          }
          placeholder={
            selected
              ? "Write an instruction..."
              : "Select a conversation to write a message"
          }
          disabled={!selected}
          value={value}
          onChange={(event) => {
            recall.reset();
            files.track(event.currentTarget);
            change(event.target.value);
          }}
          onSelect={(event) => files.track(event.currentTarget)}
          onFocus={(event) => files.focus(event.currentTarget)}
          onBlur={files.blur}
          onPaste={(event) => {
            if (event.defaultPrevented || !selected || !attach) return;
            const { text, files } = chatInput.clipboardInput(
              event.clipboardData,
            );
            if (!files.length) return;
            if (!text) event.preventDefault();
            attach(files);
          }}
          onKeyDown={(event) => {
            if (
              event.nativeEvent.isComposing ||
              event.nativeEvent.keyCode === 229
            )
              return;
            if (files.keyDown(event)) return;
            if (isHistoryShortcut(event.nativeEvent) && historyApi) {
              event.preventDefault();
              openHistory();
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
      </div>
      <span className="sr-only" id="message-draft-help">
        Enter sends. Shift, Alt, Ctrl, or Cmd+Enter adds a line. Up on the first
        line and Down on the last line recall global prompts. Ctrl+R searches
        prompt history.
        {fileSearch &&
          " Type @ to find workspace files. Use Up and Down to select, Tab or Enter to insert, and Escape to dismiss."}
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
