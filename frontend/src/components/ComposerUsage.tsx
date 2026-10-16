import { object, text } from "../models/json";
import type { HistoryItem, Session } from "../models/runner";

function tokens(value: number | null | undefined): string {
  if (value == null) return "—";
  for (const [scale, suffix] of [
    [1_000_000_000, "B"],
    [1_000_000, "M"],
    [1_000, "k"],
  ] as const) {
    if (value >= scale) return `${Number((value / scale).toFixed(1))}${suffix}`;
  }
  return String(value);
}

function cost(value: string | undefined): string {
  if (value == null) return "—";
  const amount = Number(value);
  if (!Number.isFinite(amount)) return "—";
  if (amount >= 1000) return tokens(amount);
  return String(Number(amount.toFixed(amount >= 1 ? 2 : 4)));
}

export function ComposerUsage({
  session,
  history,
}: {
  session: Pick<Session, "usage">;
  history: HistoryItem[];
}) {
  const usage = session.usage;
  const context = usage?.root_context;
  const percentage =
    context?.input_tokens_estimate != null &&
    context.context_window != null &&
    context.context_window > 0
      ? Math.round(
          (context.input_tokens_estimate / context.context_window) * 100,
        )
      : null;
  const step = history.find(
    (item) => object(item.step).id === context?.step_id,
  );
  const metrics = object(object(object(step?.step).metrics).llm);
  const model = text(metrics.catalog_model);
  const details = [
    "Latest main workflow input estimate / model context window.",
    "Estimated total cost for this execution, including child workflows and retained branches.",
    usage?.totals.incomplete_calls
      ? "Some calls have incomplete usage or pricing; cost may be understated."
      : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <span
      className="composer-usage"
      aria-label="Workflow usage"
      title={details}
    >
      <span className="composer-model">{model || "—"}</span>{" "}
      <span className="composer-tokens">
        {tokens(context?.input_tokens_estimate)} /{" "}
        {tokens(context?.context_window)}
        {percentage != null && ` (${percentage}%)`}
      </span>{" "}
      <span className="composer-cost">
        ~${cost(usage?.totals.cost_estimate_usd)}
      </span>
    </span>
  );
}
