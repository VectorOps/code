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
    vitest.expect(view.getByText("Read File")).toBeTruthy();
    vitest.expect(view.getByText("README.md")).toBeTruthy();
    vitest
      .expect(view.getByRole("switch", { name: "Match repository" }))
      .toHaveProperty("checked", true);
    await user.click(view.getByRole("switch", { name: "Match access" }));
    await user.click(
      view.getByRole("button", { name: "Approve for workflow" }),
    );
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
  "workspace approval can create a broad rule with every metadata key disabled",
  async () => {
    const reply = vitest.vi.fn(async () => true);
    const user = userEvent.setup();
    const view = testing.render(
      <ApprovalActions request={request} enabled reply={reply} />,
    );
    await user.click(view.getByRole("switch", { name: "Match access" }));
    await user.click(view.getByRole("switch", { name: "Match repository" }));
    await user.click(
      view.getByRole("button", { name: "Approve for workspace" }),
    );
    vitest
      .expect(reply)
      .toHaveBeenCalledWith(
        request,
        { decision: "approve", scope: "workspace", properties: {} },
        vitest.expect.any(String),
      );
  },
);

vitest.test(
  "approval omits metadata controls when no metadata is provided",
  () => {
    const withoutMetadata = {
      ...request,
      payload: {
        call: { name: "read_file", arguments: { path: "README.md" } },
        tool: { name: "read_file" },
        tool_version: 1,
      },
    };
    const view = testing.render(
      <ApprovalActions
        request={withoutMetadata}
        enabled
        reply={vitest.vi.fn()}
      />,
    );
    vitest.expect(view.queryByRole("switch")).toBeNull();
    vitest.expect(view.queryByText(/metadata/i)).toBeNull();
  },
);
