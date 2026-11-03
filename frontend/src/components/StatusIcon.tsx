const paths: Record<string, string> = {
  running: "m10 8 6 4-6 4Z",
  starting: "M12 7v5l3 2",
  connecting: "M12 7v5l3 2",
  paused: "M9 8v8m6-8v8",
  "needs input":
    "M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-6 3V6a2 2 0 0 1 2-2Z",
  stopping: "M8 8h8v8H8Z",
  completed: "m7 12 3 3 7-7",
  connected: "m7 12 3 3 7-7",
  failed: "m8 8 8 8m0-8-8 8",
  cancelled: "M8 12h8",
};

export function StatusIcon({
  status,
  label,
}: {
  status: string;
  label: string;
}) {
  const normalized = status.toLowerCase();
  return (
    <span
      className="status-icon"
      data-status={normalized}
      role="img"
      aria-label={`${label}: ${status}`}
      title={`${label}: ${status}`}
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        aria-hidden="true"
      >
        {normalized !== "needs input" && <circle cx="12" cy="12" r="9" />}
        <path d={paths[normalized] ?? "M12 7v6m0 3v1"} />
        {normalized === "needs input" &&
          [8, 12, 16].map((cx) => (
            <circle
              key={cx}
              cx={cx}
              cy="11"
              r="1"
              fill="currentColor"
              stroke="none"
            />
          ))}
      </svg>
    </span>
  );
}
