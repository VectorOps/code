import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import type { Params } from "../src/models/json";
import { ToolPermissions } from "../src/screens/ToolPermissions";

class Connection extends RunnerConnection {
  calls: { method: string; params: Params }[] = [];

  override async request<T>(method: string, params: Params = {}): Promise<T> {
    this.calls.push({ method, params });
    if (method === "tools.rules.list") return { rules: [] } as T;
    if (method === "configuration.get")
      return { exists: true, value: false } as T;
    if (method === "tools.rules.create")
      return { id: "rule-1", ...params } as T;
    return null as T;
  }
}

vitest.test(
  "workspace permissions manage global rules and approve all",
  async () => {
    const connection = new Connection("ws://localhost", "test");
    const view = testing.render(
      <ToolPermissions api={new RunnerApi(connection)} />,
    );
    await view.findByText("No authorization rules in this scope.");
    vitest
      .expect(view.getByRole("checkbox", { name: /Approve all/ }))
      .toBeTruthy();
    vitest.expect(view.queryByLabelText("Execution ID")).toBeNull();
    vitest.expect(view.queryByLabelText("Workflow ID")).toBeNull();
    const editor = view.getByText("Create rule manually").closest("details");
    vitest.expect(editor).toHaveProperty("open", false);
    vitest.expect(connection.calls.slice(0, 2)).toEqual([
      { method: "tools.rules.list", params: { scope: "workspace" } },
      {
        method: "configuration.get",
        params: { name: "tools.approval.approve_all" },
      },
    ]);
  },
);

vitest.test(
  "workflow permissions use contextual execution and workflow IDs",
  async () => {
    const connection = new Connection("ws://localhost", "test");
    const user = userEvent.setup();
    const scope = {
      scope: "workflow",
      execution_id: "execution-1",
      workflow_id: "workflow-1",
    } as const;
    const view = testing.render(
      <ToolPermissions api={new RunnerApi(connection)} scope={scope} />,
    );
    await view.findByText("No authorization rules in this scope.");
    vitest
      .expect(view.queryByRole("checkbox", { name: /Approve all/ }))
      .toBeNull();
    vitest.expect(view.getByText("Workflow workflow-1")).toBeTruthy();
    await user.click(view.getByText("Create rule manually"));
    await user.type(view.getByLabelText("Tool name"), "exec");
    await user.click(view.getByRole("button", { name: "Save rule" }));
    await view.findByText("Authorization rule created.");
    vitest.expect(connection.calls).toContainEqual({
      method: "tools.rules.create",
      params: {
        ...scope,
        rule: { tool_name: "exec", decision: "approve", properties: {} },
      },
    });
    vitest
      .expect(
        connection.calls.filter((call) => call.method === "tools.rules.list"),
      )
      .toEqual([
        { method: "tools.rules.list", params: scope },
        { method: "tools.rules.list", params: scope },
      ]);
  },
);
