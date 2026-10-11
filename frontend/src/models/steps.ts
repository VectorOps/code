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

export function stepView(
  item: HistoryItem,
  history: HistoryItem[] = [],
): StepView {
  const entry = activity(item);
  const output =
    entry.state === "streaming" ? entry.outcome.state : entry.outcome.result;
  const callId = text(object(output).tool_call_id);
  let toolCall: Json | undefined;
  if (entry.kind === "llm.tool" && callId) {
    const candidates = history
      .filter((candidate) => candidate.position < item.position)
      .sort((a, b) => b.position - a.position);
    for (const candidate of candidates) {
      const source = activity(candidate);
      if (source.workflow !== entry.workflow || source.kind !== "llm.generate")
        continue;
      const generation = object(source.outcome.result);
      const response = object(
        generation.response ?? generation.message ?? generation,
      );
      const blocks = Array.isArray(response.content) ? response.content : [];
      toolCall = blocks.find((block) => {
        const call = object(block);
        return (
          call.type === "tool_call" &&
          call.id === callId &&
          call.name === object(output).tool_name
        );
      });
      if (toolCall) break;
    }
  }
  return {
    ...entry,
    item,
    output,
    message: messageText(output),
    input: object(entry.operation.input),
    toolCall,
  };
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
