import type { RunnerConnection } from "./websocket";

export type HistoryScope = "global" | "project" | "session";
export type PromptEntry = {
  id: number;
  project_id: string;
  execution_id: string;
  timestamp: string;
  text: string;
};
export type PromptPage = { entries: PromptEntry[]; next_before: number | null };
export type PromptQuery = {
  scope?: HistoryScope;
  execution_id?: string;
  query?: string;
  before?: number | null;
  limit?: number;
};

export class PromptHistoryApi {
  constructor(private connection: RunnerConnection) {}

  list(query: PromptQuery = {}): Promise<PromptPage> {
    return this.connection.request("prompt_history.list", query);
  }

  append(text: string, executionId: string, key: string): Promise<PromptEntry> {
    return this.connection.request("prompt_history.append", {
      text,
      execution_id: executionId,
      idempotency_key: key,
    });
  }
}
