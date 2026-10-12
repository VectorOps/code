import { activity, messageText } from "./activity";
import { object, text, type Json } from "./json";
import type { HistoryItem, InputRequest } from "./runner";

export type StepView = ReturnType<typeof activity> & {
  item: HistoryItem;
  output: Json | undefined;
  message: string;
  input: Record<string, Json>;
  toolCall?: Json;
  processCommand?: Json;
};

type HistoryIndex = {
  calls: Map<HistoryItem, Json>;
  commands: Map<HistoryItem, Json>;
};

const historyIndexes = new WeakMap<HistoryItem[], HistoryIndex>();
const views = new WeakMap<HistoryItem, StepView>();

function historyIndex(history: HistoryItem[]): HistoryIndex {
  const cached = historyIndexes.get(history);
  if (cached) return cached;
  const result: HistoryIndex = { calls: new Map(), commands: new Map() };
  const latest = new Map<string, Json>();
  const processes = new Map<string, Json>();
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
      if (call !== undefined) result.calls.set(item, call);
      const tool = toolDescription(createStepView(item, call));
      const args = object(tool.arguments);
      if (tool.name === "exec") {
        const processId = text(object(toolPayload(tool.output)).process_id);
        if (processId && args.command !== undefined)
          processes.set(
            JSON.stringify([entry.workflow, processId]),
            args.command,
          );
      } else if (
        ["poll_process", "write_stdin", "kill_process"].includes(tool.name)
      ) {
        const command = processes.get(
          JSON.stringify([entry.workflow, text(args.process_id)]),
        );
        if (command !== undefined) result.commands.set(item, command);
      }
    }
  }
  historyIndexes.set(history, result);
  return result;
}

function createStepView(
  item: HistoryItem,
  toolCall?: Json,
  processCommand?: Json,
): StepView {
  const entry = activity(item);
  const output =
    entry.state === "streaming" ? entry.outcome.state : entry.outcome.result;
  return {
    ...entry,
    item,
    output,
    message: messageText(output),
    input: object(entry.operation.input),
    toolCall,
    processCommand,
  };
}

export function stepView(
  item: HistoryItem,
  history: HistoryItem[] = [],
): StepView {
  const index = historyIndex(history);
  const toolCall = index.calls.get(item);
  const processCommand = index.commands.get(item);
  const cached = views.get(item);
  if (
    cached &&
    cached.toolCall === toolCall &&
    cached.processCommand === processCommand
  )
    return cached;
  const view = createStepView(item, toolCall, processCommand);
  views.set(item, view);
  return view;
}

export function toolPayload(output: Json | undefined): Json | undefined {
  const content = object(output).content;
  const first = text(
    object(Array.isArray(content) ? content[0] : undefined).text,
  );
  if (!first) return output;
  try {
    return JSON.parse(first) as Json;
  } catch {
    return first;
  }
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
    processCommand: step.processCommand,
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
