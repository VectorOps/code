import * as React from "react";
import "./code-block.css";

export function CodeBlock({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const preview = React.useRef<HTMLPreElement>(null);
  const [status, setStatus] = React.useState("");

  React.useEffect(() => {
    if (!status) return;
    const timer = window.setTimeout(() => setStatus(""), 2500);
    return () => window.clearTimeout(timer);
  }, [status]);

  async function copy() {
    setStatus("");
    try {
      await navigator.clipboard.writeText(preview.current?.textContent ?? "");
      setStatus("Copied");
    } catch {
      setStatus("Copy failed. Please try again.");
    }
  }

  return (
    <div className="code-block">
      <div className="code-block-actions">
        <span role="status">{status}</span>
        <button
          type="button"
          className="code-block-copy"
          aria-label="Copy code to clipboard"
          title="Copy code to clipboard"
          onClick={(event) => {
            event.stopPropagation();
            void copy();
          }}
        >
          <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="8" y="8" width="12" height="13" rx="2" />
            <path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" />
          </svg>
        </button>
      </div>
      <pre ref={preview} className={className}>
        {children}
      </pre>
    </div>
  );
}
