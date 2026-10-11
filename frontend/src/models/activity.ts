import { object, text, type Json } from "./json";
import type { HistoryItem } from "./runner";

export function activity(item: HistoryItem) {
  const step = object(item.step),
    operation = object(step.operation),
    outcome = object(step.outcome);
  return {
    id: text(step.id) || String(item.position),
    workflow: text(step.workflow_id),
    kind: text(operation.kind),
    name: text(operation.name),
    operation,
    outcome,
    state: text(outcome.kind),
  };
}

export function messageText(value: Json | undefined): string {
  const record = object(value);
  if (typeof value === "string") return value;
  if (Array.isArray(record.content))
    return record.content
      .map((block) => text(object(block).text))
      .filter(Boolean)
      .join("\n\n");
  if (record.message) return messageText(record.message);
  if (record.response) return messageText(record.response);
  return "";
}
