import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import type { Params } from "../src/models/json";

class Connection extends RunnerConnection {
  calls: { method: string; params: Params }[] = [];

  override async request<T>(method: string, params: Params = {}): Promise<T> {
    this.calls.push({ method, params });
    if (method === "tools.rules.list") return { rules: [] } as T;
    return {} as T;
  }
}

vitest.test("rule API sends workspace and workflow contracts", async () => {
  const connection = new Connection("ws://localhost", "test");
  const api = new RunnerApi(connection);
  const workspace = { scope: "workspace" } as const;
  const workflow = {
    scope: "workflow",
    execution_id: "execution-1",
    workflow_id: "workflow-1",
  } as const;
  const rule = {
    tool_name: "read_file",
    decision: "approve",
    properties: { repository: "runner" },
  } as const;
  await api.rules(workspace, "read_file");
  await api.rule(workflow, "rule-1");
  await api.createRule(workspace, rule);
  await api.updateRule(workflow, "rule-1", rule);
  await api.deleteRule(workspace, "rule-1");
  vitest.expect(connection.calls).toEqual([
    {
      method: "tools.rules.list",
      params: { scope: "workspace", tool_name: "read_file" },
    },
    {
      method: "tools.rules.get",
      params: { ...workflow, rule_id: "rule-1" },
    },
    {
      method: "tools.rules.create",
      params: { scope: "workspace", rule },
    },
    {
      method: "tools.rules.update",
      params: { ...workflow, rule_id: "rule-1", rule },
    },
    {
      method: "tools.rules.delete",
      params: { scope: "workspace", rule_id: "rule-1" },
    },
  ]);
});
