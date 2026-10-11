import { RunnerConnection } from "./websocket";
import type { Json, Params } from "../models/json";
import { object, text } from "../models/json";
import type * as models from "../models/runner";

export class RunnerApi {
  constructor(readonly connection: RunnerConnection) {}

  private async pages<T>(
    method: string,
    field: string,
    params: Params = {},
    active = () => true,
  ): Promise<T[]> {
    const result: T[] = [];
    let after: string | null = null;
    do {
      const page: { next_after: string | null; [key: string]: unknown } =
        await this.connection.request(method, { ...params, after, limit: 100 });
      result.push(...(page[field] as T[]));
      after = page.next_after;
    } while (after && active());
    return result;
  }

  subscribe() {
    return this.connection.request("events.subscribe");
  }
  sessions(active?: () => boolean) {
    return this.pages<models.Session>(
      "executions.summaries",
      "executions",
      {},
      active,
    );
  }
  workflows(active?: () => boolean) {
    return this.pages<models.Workflow>(
      "workflows.list",
      "workflows",
      {},
      active,
    );
  }
  pending(execution_id: string, active?: () => boolean) {
    return this.pages<models.InputRequest>(
      "mailbox.list",
      "requests",
      { execution_id },
      active,
    );
  }
  async history(execution_id: string): Promise<models.HistoryPage> {
    const items: models.HistoryItem[] = [];
    let cursor: string | null = null;
    do {
      const page: models.HistoryPage = await this.connection.request(
        "history.query",
        { execution_id, limit: 500, cursor },
      );
      items.push(...page.items);
      cursor = page.next_cursor;
    } while (cursor);
    return {
      items: items.sort((a, b) => a.position - b.position),
      next_cursor: null,
    };
  }
  start(reference: string, input: Params, idempotency_key: string) {
    return this.connection.request<{ execution_id: string }>(
      "executions.start",
      { reference, input, idempotency_key },
    );
  }
  stop(execution_id: string) {
    return this.connection.request("executions.stop", { execution_id });
  }
  async respond(
    request: models.InputRequest,
    response: Json,
    idempotency_key: string,
  ): Promise<unknown> {
    if (object(response).scope === "workspace") {
      if (
        request.request_type !== "tool.approval.v1" ||
        object(response).approved !== true
      )
        throw new Error("Workspace approval requires an approved tool request");
      const current = await this.connection.request<models.InputRequest>(
        "mailbox.get",
        {
          execution_id: request.execution_id,
          request_id: request.request_id,
        },
      );
      const payload = object(current.payload);
      const name = text(object(payload.call).name);
      if (!name || name !== text(object(payload.tool).name))
        throw new Error("Cannot identify the tool for workspace approval");
      const configuration = await this.connection.request<{
        exists: boolean;
        value: Json;
      }>("configuration.get", {
        name: "tools.approval.whitelist",
      });
      const existing = configuration.exists ? configuration.value : [];
      if (
        !Array.isArray(existing) ||
        !existing.every((value) => typeof value === "string")
      )
        throw new Error("Workspace tool whitelist must be a list of names");
      await this.connection.request("configuration.set", {
        name: "tools.approval.whitelist",
        value: [...new Set([...existing, name])],
        permanent: true,
      });
      try {
        return await this.respond(
          request,
          { approved: true, scope: "call" },
          idempotency_key,
        );
      } catch (error) {
        throw new Error(
          `Workspace permission was saved, but the current call's approval was not confirmed. Retry or inspect pending decisions. ${String(error)}`,
        );
      }
    }
    return this.connection.request("mailbox.respond", {
      execution_id: request.execution_id,
      request_id: request.request_id,
      response,
      idempotency_key,
    });
  }
}
