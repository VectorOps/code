import type { Json } from "./json";
import { chatMessage, type ImageInput } from "./chat-input";

export type ExecutionUsage = {
  totals: {
    input_tokens: number;
    output_tokens: number;
    cost_estimate_usd: string;
    estimated_calls: number;
    incomplete_calls: number;
  };
  root_context: {
    step_id: string;
    input_tokens_estimate: number | null;
    context_window: number | null;
  } | null;
};

export type Session = {
  execution_id: string;
  root_workflow_id?: string | null;
  reference: string | null;
  name?: string | null;
  status: string;
  active: boolean;
  pending_inputs: number;
  usage?: ExecutionUsage | null;
};
export type SessionPage = {
  executions: Session[];
  next_after: string | null;
};
export type Workflow = {
  descriptor: {
    reference: string;
    name: string;
    description: string | null;
    input_schema: Json;
  };
};

export function workflowDisplayName(
  reference: string | null,
  workflows: Workflow[] | ReadonlyMap<string, string>,
  name?: string | null,
): string {
  return (
    name ||
    (Array.isArray(workflows)
      ? workflows.find(
          (workflow) => workflow.descriptor.reference === reference,
        )?.descriptor.name
      : workflows.get(reference ?? "")) ||
    reference ||
    "Workflow"
  );
}
export type InputRequest = {
  request_id: string;
  execution_id: string;
  workflow_id?: string;
  request_type?: string;
  channel: string;
  payload: Json;
  response_type: "text" | "json";
  response_schema?: Json;
};
export type InboxAddress = {
  execution_id: string;
  workflow_id: string;
  conversation_id: string;
};
export type InboxStatus = {
  address: InboxAddress;
  status: "open" | "closed";
  queued_messages: number;
};
export type InboxMessage = { id: string; created_at: string; payload: Json };
export type HistoryItem = { position: number; step: Json };
export type HistoryPage = { items: HistoryItem[]; next_cursor: string | null };
export type Execution = {
  id: string;
  root_workflow_id: string;
  status: string;
  checkpoint: { heads: Record<string, string> };
};
export type BranchAction =
  { kind: "restart" } | { kind: "insert_input"; response: Json };
export type BranchAccepted = {
  execution_id: string;
  operation_id: string;
  status: "accepted";
};
export type ToolRuleScope = {
  scope: "workspace" | "workflow";
  execution_id?: string;
  workflow_id?: string;
};
export type ToolRuleInput = {
  tool_name: string;
  decision: "approve" | "deny";
  properties: Record<string, Json>;
};
export type ToolRule = ToolRuleInput & { id: string };
export type ConnectionSettings = { endpoint: string; token: string };

export function isChatRequest(request: InputRequest): boolean {
  return (
    request.request_type === "chat.message.v1" ||
    (request.channel === "user.message" && request.response_type === "text")
  );
}

export function inputResponse(
  request: Pick<InputRequest, "request_type" | "response_type">,
  value: string,
  images: readonly ImageInput[] = [],
): Json {
  if (request.request_type === "chat.message.v1")
    return chatMessage(value, images);
  if (images.length)
    throw new Error(
      "This workflow accepts text only. Remove the images to send.",
    );
  return request.response_type === "json" ? JSON.parse(value) : value;
}
