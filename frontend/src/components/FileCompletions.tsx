import * as React from "react";
import type {
  FileMatch,
  FileSearchSnapshot,
} from "../communication/file-search";
import "./file-completions.css";

function HighlightedPath({ result }: { result: FileMatch }) {
  const characters = Array.from(result.path);
  const parts: React.ReactNode[] = [];
  let offset = 0;
  for (const [start, end] of result.highlights) {
    parts.push(characters.slice(offset, start).join(""));
    parts.push(
      <mark key={start}>{characters.slice(start, end).join("")}</mark>,
    );
    offset = end;
  }
  parts.push(characters.slice(offset).join(""));
  return <>{parts}</>;
}

export function FileCompletions({
  id,
  results,
  activePath,
  choose,
  loading,
  snapshot,
  error,
}: {
  id: string;
  results: FileMatch[];
  activePath?: string;
  choose: (path: string) => void;
  loading: boolean;
  snapshot?: FileSearchSnapshot;
  error?: string;
}) {
  const activeIndex = results.findIndex((result) => result.path === activePath);
  React.useEffect(() => {
    document
      .getElementById(`${id}-${activeIndex}`)
      ?.scrollIntoView?.({ block: "nearest" });
  }, [id, activeIndex]);
  const status =
    error ??
    (loading && !snapshot
      ? "Searching files…"
      : snapshot?.refreshing
        ? "Indexing workspace…"
        : snapshot?.incomplete
          ? "File index incomplete. Some files may be missing."
          : results.length
            ? snapshot?.truncated
              ? "More matches · keep typing to narrow"
              : "Workspace files"
            : "No matching files");
  return (
    <div className="file-completions">
      <div className="file-completions-heading">
        <span role="status">{status}</span>
        <span>↑↓ navigate · Tab / Enter insert · Esc close</span>
      </div>
      <div
        id={id}
        role="listbox"
        aria-label="Workspace files"
        aria-busy={loading || snapshot?.refreshing}
      >
        {results.map((result, index) => (
          <button
            key={result.path}
            id={`${id}-${index}`}
            type="button"
            role="option"
            aria-label={result.path}
            aria-selected={result.path === activePath}
            tabIndex={-1}
            title={result.path}
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => choose(result.path)}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M14 3H5v18h14V8Z M14 3v5h5" />
            </svg>
            <span>
              <HighlightedPath result={result} />
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
