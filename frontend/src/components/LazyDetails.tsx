import { useState, type ReactNode } from "react";

export function LazyDetails({
  summary,
  children,
}: {
  summary: ReactNode;
  children: () => ReactNode;
}) {
  const [expanded, setExpanded] = useState(false);
  return (
    <details onToggle={(event) => setExpanded(event.currentTarget.open)}>
      <summary>{summary}</summary>
      {expanded && children()}
    </details>
  );
}
