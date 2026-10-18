import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { ApprovalActions } from "../src/components/ApprovalActions";
import type { InputRequest } from "../src/models/runner";

const request: InputRequest = {
  request_id: "step-1",
  execution_id: "execution-1",
  workflow_id: "workflow-1",
  request_type: "tool.approval.v1",
  channel: "tools.approval",
  response_type: "json",
  payload: {
    call: { name: "read_file", arguments: { path: "README.md" } },
    tool: { name: "read_file" },
    tool_version: 1,
    auth_context: { properties: { repository: "runner", access: "read" } },
  },
};

vitest.test(
  "approval submits decisions with the selected metadata mask",
  async () => {
    const reply = vitest.vi.fn(async () => true);
    const user = userEvent.setup();
    const view = testing.render(
      <ApprovalActions request={request} enabled reply={reply} />,
    );
    vitest.expect(view.getByText("read_file")).toBeTruthy();
    vitest.expect(view.getByText(/README.md/)).toBeTruthy();
    vitest.expect(view.getByText(/repository/)).toBeTruthy();
    await user.click(view.getByRole("radio", { name: "This workflow" }));
    await user.click(view.getByRole("checkbox", { name: /access/ }));
    await user.click(view.getByRole("button", { name: "Approve" }));
    vitest.expect(reply).toHaveBeenCalledExactlyOnceWith(
      request,
      {
        decision: "approve",
        scope: "workflow",
        properties: { repository: "runner" },
      },
      vitest.expect.any(String),
    );
  },
);

vitest.test(
  "approval explains that an empty mask affects every tool call",
  async () => {
    const user = userEvent.setup();
    const view = testing.render(
      <ApprovalActions request={request} enabled reply={vitest.vi.fn()} />,
    );
    await user.click(view.getByRole("radio", { name: "This workspace" }));
    await user.click(view.getByRole("checkbox", { name: /access/ }));
    await user.click(view.getByRole("checkbox", { name: /repository/ }));
    vitest
      .expect(view.getByRole("status").textContent)
      .toContain("every call to this tool");
  },
);
