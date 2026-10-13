import type { Json } from "./json";
import { chatMessage, type ImageInput } from "./chat-input";

export type Session = {
  execution_id: string;
  root_workflow_id?: string | null;
  reference: string | null;
  status: string;
  active: boolean;
  pending_inputs: number;
};
export type Workflow = {
  descriptor: {
    reference: string;
    name: string;
    description: string | null;
    input_schema: Json;
  };
};
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
