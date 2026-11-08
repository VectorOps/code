import { RunnerConnection } from "./websocket";
import type { Json, Params } from "../models/json";
import type * as models from "../models/runner";

export class RunnerApi {
  constructor(readonly connection: RunnerConnection) {}

  async workspace(): Promise<string> {
    const info = await this.connection.request<{ workspace: string }>(
      "runner.info",
    );
    return info.workspace;
  }

  diskUsage() {
    return this.connection.request<models.DiskUsageReport>(
      "runner.disk_usage",
      { top: 10 },
    );
  }

  executionDiskUsage(execution_id: string) {
    return this.connection.request<models.WorkflowDiskUsage>(
      "executions.disk_usage",
      { execution_id },
    );
  }

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
  sessions(after: string | null = null) {
    return this.connection.request<models.SessionPage>("executions.summaries", {
      after,
      limit: 100,
    });
  }
  session(execution_id: string) {
    return this.connection.request<models.Session>("executions.summary", {
      execution_id,
    });
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
  async inboxes(execution_id: string): Promise<models.InboxStatus[]> {
    const result = await this.connection.request<{
      mailboxes: models.InboxStatus[];
    }>("mailbox.inboxes.list", { execution_id });
    return result.mailboxes;
  }
  async inboxMessages(
    address: models.InboxAddress,
  ): Promise<models.InboxMessage[]> {
    const messages: models.InboxMessage[] = [];
    let after: string | null = null;
    do {
      const page: {
        messages: models.InboxMessage[];
        next_after: string | null;
      } = await this.connection.request("mailbox.inboxes.messages", {
        ...address,
        after,
        limit: 500,
      });
      messages.push(...page.messages);
      after = page.next_after;
    } while (after);
    return messages;
  }
  sendSteering(address: models.InboxAddress, payload: Json) {
    return this.connection.request<models.InboxMessage>(
      "mailbox.inboxes.send",
      {
        ...address,
        payload,
      },
    );
  }
  removeSteering(address: models.InboxAddress, message_id: string) {
    return this.connection.request<models.InboxMessage | null>(
      "mailbox.inboxes.remove",
      { ...address, message_id },
    );
  }
  history(
    execution_id: string,
    query: {
      limit: number;
      before_step_id?: string;
      cursor?: string;
    },
  ): Promise<models.HistoryPage> {
    return this.connection.request("history.query", {
      execution_id,
      selection: "tail",
      ...query,
    });
  }
  start(
    reference: string,
    input: Params,
    idempotency_key: string,
    worktree_id?: string,
  ) {
    return this.connection.request<{ execution_id: string }>(
      "executions.start",
      {
        reference,
        input,
        idempotency_key,
        ...(worktree_id ? { worktree_id } : {}),
      },
    );
  }
  stop(execution_id: string) {
    return this.connection.request("executions.stop", { execution_id });
  }
  deleteExecution(execution_id: string) {
    return this.connection.request<{
      execution_id: string;
      deleted: boolean;
    }>("executions.delete", { execution_id });
  }
  execution(execution_id: string) {
    return this.connection.request<models.Execution>("executions.get", {
      execution_id,
    });
  }
  workflowState(execution_id: string, workflow_id: string) {
    return this.connection.request<models.WorkflowState>("workflows.state", {
      execution_id,
      workflow_id,
    });
  }
  branch(
    execution_id: string,
    step_id: string,
    action: models.BranchAction,
    idempotency_key: string,
  ) {
    return this.connection.request<models.BranchAccepted>("executions.branch", {
      execution_id,
      step_id,
      action,
      idempotency_key,
    });
  }
  resume(execution_id: string, idempotency_key: string) {
    return this.connection.request<{
      execution_id: string;
      status: "starting";
    }>("executions.resume", { execution_id, idempotency_key });
  }
  rules(scope: models.ToolRuleScope, tool_name?: string) {
    return this.connection
      .request<{ rules: models.ToolRule[] }>("tools.rules.list", {
        ...scope,
        ...(tool_name ? { tool_name } : {}),
      })
      .then((result) => result.rules);
  }
  rule(scope: models.ToolRuleScope, rule_id: string) {
    return this.connection.request<models.ToolRule>("tools.rules.get", {
      ...scope,
      rule_id,
    });
  }
  createRule(scope: models.ToolRuleScope, rule: models.ToolRuleInput) {
    return this.connection.request<models.ToolRule>("tools.rules.create", {
      ...scope,
      rule,
    });
  }
  updateRule(
    scope: models.ToolRuleScope,
    rule_id: string,
    rule: models.ToolRuleInput,
  ) {
    return this.connection.request<models.ToolRule>("tools.rules.update", {
      ...scope,
      rule_id,
      rule,
    });
  }
  deleteRule(scope: models.ToolRuleScope, rule_id: string) {
    return this.connection.request<null>("tools.rules.delete", {
      ...scope,
      rule_id,
    });
  }
  respond(
    request: models.InputRequest,
    response: Json,
    idempotency_key: string,
  ): Promise<unknown> {
    return this.connection.request("mailbox.respond", {
      execution_id: request.execution_id,
      request_id: request.request_id,
      response,
      idempotency_key,
    });
  }
}
