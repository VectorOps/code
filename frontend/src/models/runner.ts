import type { Json } from "./json";

export type Session = {
  execution_id: string;
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
export type ConnectionSettings = { endpoint: string; token: string };

export function isChatRequest(request: InputRequest): boolean {
  return (
    request.request_type === "chat.message.v1" ||
    (request.channel === "user.message" && request.response_type === "text")
  );
}

export function inputResponse(request: InputRequest, value: string): Json {
  if (request.request_type === "chat.message.v1")
    return { content: [{ type: "text", text: value }] };
  return request.response_type === "json" ? JSON.parse(value) : value;
}
