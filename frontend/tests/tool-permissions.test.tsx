import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import type { Params } from "../src/models/json";
import type { ToolRuleScope } from "../src/models/runner";
import { ToolPermissions } from "../src/screens/ToolPermissions";

class Connection extends RunnerConnection {
  calls: { method: string; params: Params }[] = [];
  rules: Params[] = [];

  override async request<T>(method: string, params: Params = {}): Promise<T> {
    this.calls.push({ method, params });
    if (method === "tools.rules.list") return { rules: this.rules } as T;
    if (method === "configuration.get")
      return { exists: true, value: false } as T;
    if (method === "tools.rules.create")
      return { id: "rule-1", ...params } as T;
    return null as T;
  }
}

vitest.test.each<ToolRuleScope>([
  { scope: "workspace" },
  { scope: "workflow", execution_id: "execution-1", workflow_id: "workflow-1" },
])("$scope permissions sort rules by tool name", async (scope) => {
  const connection = new Connection("ws://localhost", "test");
  const user = userEvent.setup();
  connection.rules = ["write_file", "read_file", "exec", "read_file"].map(
    (tool_name, index) => ({
      id: `rule-${index}`,
      tool_name,
      decision: "approve",
      properties: { index },
    }),
  );
  const originalRules = [...connection.rules];
  const view = testing.render(
    <ToolPermissions api={new RunnerApi(connection)} scope={scope} />,
  );
  await view.findByText("write_file");
  const displayedNames = () =>
    view
      .getAllByRole("article")
      .map((rule) => rule.querySelector("strong")?.textContent);
  vitest
    .expect(displayedNames())
    .toEqual(["exec", "read_file", "read_file", "write_file"]);
  vitest.expect(connection.rules).toEqual(originalRules);
  vitest
    .expect(
      view
        .getAllByRole("article")
        .map((rule) => rule.querySelector("code")?.textContent),
    )
    .toEqual(["2", "1", "3", "0"]);

  connection.rules = [
    ...connection.rules,
    {
      id: "rule-new",
      tool_name: "apply_patch",
      decision: "deny",
      properties: {},
    },
  ];
  await user.click(view.getByRole("button", { name: "Refresh permissions" }));
  await view.findByText("apply_patch");
  vitest
    .expect(displayedNames())
    .toEqual(["apply_patch", "exec", "read_file", "read_file", "write_file"]);
});

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
    vitest
      .expect(
        view.container.querySelector(".tool-permissions.settings-content"),
      )
      .toBeTruthy();
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
    vitest
      .expect(
        view.container.querySelector(".tool-permissions.settings-content"),
      )
      .toBeTruthy();
    await user.click(view.getByText("Create rule manually"));
    await user.type(view.getByLabelText("Tool name"), "exec");
    await user.click(view.getByRole("button", { name: "Add property" }));
    await user.type(view.getByLabelText("Property 1 name"), "scope");
    await user.clear(view.getByLabelText("Property 1 value"));
    await user.type(view.getByLabelText("Property 1 value"), '"internal"');
    await user.click(view.getByRole("button", { name: "Add property" }));
    await user.type(view.getByLabelText("Property 2 name"), "priority");
    await user.clear(view.getByLabelText("Property 2 value"));
    await user.type(view.getByLabelText("Property 2 value"), "3");
    await user.click(view.getByRole("button", { name: "Save rule" }));
    await view.findByText("Authorization rule created.");
    vitest.expect(connection.calls).toContainEqual({
      method: "tools.rules.create",
      params: {
        ...scope,
        rule: {
          tool_name: "exec",
          decision: "approve",
          properties: { scope: "internal", priority: 3 },
        },
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

vitest.test("invalid property values do not create rules", async () => {
  const connection = new Connection("ws://localhost", "test");
  const user = userEvent.setup();
  const view = testing.render(
    <ToolPermissions api={new RunnerApi(connection)} />,
  );
  await view.findByText("No authorization rules in this scope.");
  await user.click(view.getByText("Create rule manually"));
  await user.type(view.getByLabelText("Tool name"), "exec");
  await user.click(view.getByRole("button", { name: "Add property" }));
  await user.type(view.getByLabelText("Property 1 name"), "scope");
  await user.clear(view.getByLabelText("Property 1 value"));
  await user.type(view.getByLabelText("Property 1 value"), "internal");
  await user.click(view.getByRole("button", { name: "Save rule" }));
  vitest
    .expect((await view.findByRole("alert")).textContent)
    .toContain("valid JSON value");
  vitest
    .expect(
      connection.calls.some((call) => call.method === "tools.rules.create"),
    )
    .toBe(false);
  await user.click(view.getByRole("switch", { name: "Match property 1" }));
  await user.click(view.getByRole("button", { name: "Save rule" }));
  await view.findByText("Authorization rule created.");
  vitest.expect(connection.calls).toContainEqual({
    method: "tools.rules.create",
    params: {
      scope: "workspace",
      rule: { tool_name: "exec", decision: "approve", properties: {} },
    },
  });
});

vitest.test(
  "rule previews render property rows instead of raw JSON",
  async () => {
    const connection = new Connection("ws://localhost", "test");
    connection.rules = [
      {
        id: "rule-1",
        tool_name: "read_file",
        decision: "approve",
        properties: { scope: "internal", priority: 3 },
      },
    ];
    const view = testing.render(
      <ToolPermissions api={new RunnerApi(connection)} />,
    );
    await view.findByText("read_file");
    const preview = view.container.querySelector<HTMLElement>(
      ".permission-property-preview",
    );
    vitest.expect(preview).toBeTruthy();
    vitest.expect(testing.within(preview!).getByText("scope")).toBeTruthy();
    vitest
      .expect(testing.within(preview!).getByText('"internal"'))
      .toBeTruthy();
    vitest.expect(testing.within(preview!).getByText("priority")).toBeTruthy();
    vitest.expect(testing.within(preview!).getByText("3")).toBeTruthy();
    vitest.expect(preview?.querySelector("pre")).toBeNull();
  },
);
