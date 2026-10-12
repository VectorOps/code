const statusPaths: Readonly<Record<string, string>> = {
  completed: "m5 12 4 4L19 6",
  failed: "m6 6 12 12M6 18 18 6",
  streaming: "M20 12a8 8 0 1 1-8-8",
  pending: "M12 8v4l3 2M20 12a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  "needs approval": "M12 8v5m0 3h.01M10 4 2 18h20L14 4a2 2 0 0 0-4 0",
  authorized: "m9 12 2 2 4-4M12 3l8 3v6c0 5-8 9-8 9s-8-4-8-9V6Z",
  cancelled: "M20 12a8 8 0 1 1-16 0 8 8 0 0 1 16 0M6 6l12 12",
};

export function StepIndicators({
  state,
  inspect,
}: {
  state?: string;
  inspect?: () => void;
}) {
  const normalized = state?.toLowerCase();
  const label =
    state && !["completed", "authorized"].includes(normalized ?? "")
      ? state[0].toUpperCase() + state.slice(1)
      : undefined;
  const details = (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v6m0-10h.01" />
    </svg>
  );
  return (
    <span className="entry-state inline-flex items-center gap-2">
      {label && (
        <span role="img" aria-label={label} title={label}>
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            className={
              normalized === "streaming" || normalized === "pending"
                ? "animate-spin motion-reduce:animate-none"
                : undefined
            }
          >
            <path
              d={
                statusPaths[
                  normalized === "pending" ? "streaming" : (normalized ?? "")
                ] ?? "M12 8v4m0 4h.01M20 12a8 8 0 1 1-16 0 8 8 0 0 1 16 0"
              }
            />
          </svg>
        </span>
      )}
      {inspect ? (
        <button
          type="button"
          className="step-details-button"
          aria-label="View details"
          title="View details"
          onClick={inspect}
        >
          {details}
        </button>
      ) : (
        <span role="img" aria-label="View details" title="View details">
          {details}
        </span>
      )}
    </span>
  );
}
