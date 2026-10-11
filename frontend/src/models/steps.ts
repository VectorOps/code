import { activity, messageText } from "./activity";
import { object, text, type Json } from "./json";
import type { HistoryItem, InputRequest } from "./runner";

export type StepView = ReturnType<typeof activity> & {
  item: HistoryItem;
  output: Json | undefined;
  message: string;
  input: Record<string, Json>;
  toolCall?: Json;
};

const toolCalls = new WeakMap<HistoryItem[], Map<HistoryItem, Json>>();
const views = new WeakMap<HistoryItem, StepView>();

function toolCallIndex(history: HistoryItem[]): Map<HistoryItem, Json> {
  const cached = toolCalls.get(history);
  if (cached) return cached;
  const result = new Map<HistoryItem, Json>();
  const latest = new Map<string, Json>();
  for (const item of [...history].sort((a, b) => a.position - b.position)) {
    const entry = activity(item);
    if (entry.kind === "llm.generate") {
      const generation = object(entry.outcome.result);
      const response = object(
        generation.response ?? generation.message ?? generation,
      );
      const seen = new Set<string>();
      for (const block of Array.isArray(response.content)
        ? response.content
        : []) {
        const call = object(block);
        if (call.type !== "tool_call") continue;
        const key = JSON.stringify([entry.workflow, call.id, call.name]);
        if (!seen.has(key)) latest.set(key, block);
        seen.add(key);
      }
    } else if (entry.kind === "llm.tool") {
      const output = object(
        entry.state === "streaming"
          ? entry.outcome.state
          : entry.outcome.result,
      );
      const call = latest.get(
        JSON.stringify([entry.workflow, output.tool_call_id, output.tool_name]),
      );
      if (call !== undefined) result.set(item, call);
    }
  }
  toolCalls.set(history, result);
  return result;
}

export function stepView(
  item: HistoryItem,
  history: HistoryItem[] = [],
): StepView {
  const entry = activity(item);
  const output =
    entry.state === "streaming" ? entry.outcome.state : entry.outcome.result;
  const callId = text(object(output).tool_call_id);
  const toolCall =
    entry.kind === "llm.tool" && callId
      ? toolCallIndex(history).get(item)
      : undefined;
  const cached = views.get(item);
  if (cached && cached.toolCall === toolCall) return cached;
  const view = {
    ...entry,
    item,
    output,
    message: messageText(output),
    input: object(entry.operation.input),
    toolCall,
  };
  views.set(item, view);
  return view;
}

export function isUserResponse(step: StepView): boolean {
  return (
    step.kind === "mailbox.input" &&
    step.state === "completed" &&
    (step.input.request_type === "chat.message.v1" ||
      step.input.channel === "user.message")
  );
}

export function approvalFor(
  step: StepView,
  pending: InputRequest[],
): InputRequest | undefined {
  return pending.find(
    (request) =>
      request.request_type === "tool.approval.v1" &&
      request.request_id === step.id &&
      request.workflow_id === step.workflow,
  );
}

export function toolDescription(step: StepView, approval?: InputRequest) {
  const presentation = object(object(step.item.step).presentation);
  const call = object(
    object(approval?.payload).call ??
      presentation.call ??
      step.input.call ??
      step.toolCall,
  );
  const output = presentation.result ?? step.output;
  const result = object(output);
  return {
    name: text(call.name) || text(result.tool_name) || "Tool call",
    arguments: call.arguments,
    failed: step.state === "failed" || result.is_error === true,
    output,
    state:
      step.input.gate_only === true &&
      presentation.result === undefined &&
      result.is_error !== true
        ? "Authorized"
        : step.state,
    error: text(object(step.outcome.error).message),
  };
}
