import { DebugProvider } from "../src/components/Diagnostics";
import * as React from "react";
import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import { WorktreesApi } from "../src/communication/worktrees-api";
import { WorktreeLabel, WorktreesProvider } from "../src/components/Worktrees";
import {
  NewSessionForm,
  type NewSessionDraft,
} from "../src/components/NewSessionForm";
import { Settings } from "../src/screens/Settings";
import { WorktreeCreate } from "../src/components/WorktreeCreate";
import { Workspace } from "../src/screens/Workspace";
import { SessionSidebar } from "../src/components/SessionSidebar";
import type { Worktree } from "../src/models/worktrees";
import type { Params } from "../src/models/json";
import { workflowTree, workflowWorktree } from "../src/models/workflow-tree";

const record: Worktree = {
  id: "tree-1",
  name: "Review sandbox",
  description: "Isolated review",
  path: "/repo/worktrees/review",
  repository: { root: "/repo", identity: "repo" },
  branch: "review",
  base_commit: "abc",
  origin: { kind: "user" },
  status: "ready",
  created_at: "2026-01-01",
  updated_at: "2026-01-01",
  inspection: { head: "abc", dirty: true, inspected_at: "2026-01-01" },
};

function client() {
  const connection = new RunnerConnection("ws://localhost", "test");
  const request = vitest.vi.spyOn(connection, "request");
  request.mockImplementation(async (method, params = {}) => {
    if (method === "worktrees.list")
      return { worktrees: [record], next_after: null };
    if (method === "worktrees.get") return record;
    if (method === "worktrees.create")
      return { ...record, id: "new-tree", ...params };
    if (method === "worktrees.update") return { ...record, ...params };
    if (method === "worktrees.delete") return { ...record, status: "deleted" };
    throw new Error(`Unexpected method ${method}`);
  });
  return { api: new RunnerApi(connection), request };
}

vitest.test(
  "worktree API paginates and sends exact CRUD and launch contracts",
  async () => {
    const { api, request } = client();
    const worktrees = new WorktreesApi(api.connection);
    request.mockResolvedValueOnce({
      worktrees: [record],
      next_after: record.id,
    });
    request.mockResolvedValueOnce({
      worktrees: [{ ...record, id: "tree-2" }],
      next_after: null,
    });
    vitest.expect(await worktrees.list(true)).toHaveLength(2);
    vitest.expect(request).toHaveBeenNthCalledWith(2, "worktrees.list", {
      after: record.id,
      limit: 100,
      include_deleted: true,
      refresh: true,
    });
    await worktrees.get(record.id);
    vitest
      .expect(request)
      .toHaveBeenLastCalledWith("worktrees.get", { worktree_id: record.id });
    const input = { name: "New", description: "", base_ref: "main" };
    await worktrees.create(input, "create-key");
    vitest.expect(request).toHaveBeenLastCalledWith("worktrees.create", {
      ...input,
      idempotency_key: "create-key",
    });
    await worktrees.update(record.id, "Renamed", "Description");
    vitest.expect(request).toHaveBeenLastCalledWith("worktrees.update", {
      worktree_id: record.id,
      name: "Renamed",
      description: "Description",
    });
    await worktrees.delete(record.id);
    vitest.expect(request).toHaveBeenLastCalledWith("worktrees.delete", {
      worktree_id: record.id,
      force: false,
    });
    request.mockResolvedValue({ execution_id: "execution" });
    await api.start("agent", {}, "start-key", record.id);
    vitest.expect(request).toHaveBeenLastCalledWith("executions.start", {
      reference: "agent",
      input: {},
      idempotency_key: "start-key",
      worktree_id: record.id,
    });
    await api.start("agent", {}, "default-key");
    vitest.expect(request).toHaveBeenLastCalledWith("executions.start", {
      reference: "agent",
      input: {},
      idempotency_key: "default-key",
    });
  },
);

vitest.test(
  "settings manager inspects, creates, renames and confirms deletion",
  async () => {
    const { api, request } = client();
    const user = userEvent.setup();
    const view = testing.render(
      <WorktreesProvider runner={api}>
        <Settings
          api={api}
          connected
          connection={{ endpoint: "ws://localhost", token: "test" }}
        />
        <WorktreeLabel id={record.id} />
      </WorktreesProvider>,
    );
    await user.click(view.getByRole("button", { name: "Worktrees" }));
    const card = testing.within(
      await view.findByRole("article", { name: record.name }),
    );
    vitest.expect(card.getByText(/Uncommitted changes/)).toBeDefined();
    vitest.expect(card.getByText(record.path)).toBeDefined();
    await user.click(card.getByRole("button", { name: "Edit" }));
    await user.clear(card.getByLabelText("Worktree name"));
    await user.type(card.getByLabelText("Worktree name"), "Renamed");
    await user.click(card.getByRole("button", { name: "Save worktree" }));
    await view.findByText("Worktree: Renamed");
    await user.click(card.getByRole("button", { name: "Delete" }));
    vitest
      .expect(
        request.mock.calls.filter(([method]) => method === "worktrees.delete"),
      )
      .toHaveLength(0);
    await user.click(card.getByRole("button", { name: "Cancel" }));
    await user.click(card.getByRole("button", { name: "Delete" }));
    await user.click(card.getByLabelText(/Force deletion/));
    vitest
      .expect(card.getByRole("alert").textContent)
      .toContain("permanently lost");
    await user.click(card.getByRole("button", { name: "Confirm deletion" }));
    vitest.expect(request).toHaveBeenLastCalledWith("worktrees.delete", {
      worktree_id: record.id,
      force: true,
    });
    vitest.expect(view.queryByRole("article")).toBeNull();
    await user.click(view.getByLabelText("Show deleted worktrees"));
    await view.findByRole("article", { name: record.name });
    await user.click(view.getByRole("button", { name: "New worktree" }));
    await user.type(view.getByLabelText("Worktree name"), "Created here");
    await user.click(view.getByRole("button", { name: "Create worktree" }));
    await view.findByRole("article", { name: "Created here" });
  },
);

function Launch({
  start,
}: {
  start: (
    reference: string,
    input: Params,
    worktreeId?: string,
  ) => Promise<boolean>;
}) {
  const [value, change] = React.useState<NewSessionDraft>({
    reference: "agent",
    input: "{}",
  });
  return (
    <NewSessionForm
      workflows={[
        {
          descriptor: {
            reference: "agent",
            name: "Agent",
            description: null,
            input_schema: { type: "object", properties: {} },
          },
        },
      ]}
      value={value}
      change={change}
      enabled
      start={start}
    />
  );
}

vitest.test(
  "advanced launch selects a worktree and inline creation retries safely without launching",
  async () => {
    const { api, request } = client();
    const start = vitest.vi.fn(async () => true);
    const user = userEvent.setup();
    const view = testing.render(
      <WorktreesProvider runner={api}>
        <Launch start={start} />
      </WorktreesProvider>,
    );
    await view.findByRole("option", { name: "Review sandbox — review" });
    await user.selectOptions(view.getByLabelText("Worktree"), record.id);
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest.expect(start).toHaveBeenLastCalledWith("agent", {}, record.id);
    start.mockClear();
    await user.click(view.getByRole("button", { name: "New worktree" }));
    vitest
      .expect(view.getByRole("button", { name: "Start conversation" }))
      .toHaveProperty("disabled", true);
    await user.type(view.getByLabelText("Worktree name"), "Inline");
    await user.type(view.getByLabelText("Base ref"), "main");
    request.mockRejectedValueOnce(new Error("Connection lost"));
    await user.click(view.getByRole("button", { name: "Create worktree" }));
    vitest
      .expect(view.getByRole("alert").textContent)
      .toContain("Connection lost");
    const failed = request.mock.calls.at(-1);
    await user.click(view.getByRole("button", { name: "Create worktree" }));
    vitest.expect(request.mock.calls.at(-1)).toEqual(failed);
    vitest
      .expect(view.getByLabelText("Worktree"))
      .toHaveProperty("value", "new-tree");
    vitest.expect(start).not.toHaveBeenCalled();
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest.expect(start).toHaveBeenLastCalledWith("agent", {}, "new-tree");
    await user.selectOptions(view.getByLabelText("Worktree"), "");
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest.expect(start).toHaveBeenLastCalledWith("agent", {});
  },
);

vitest.test(
  "refresh failure is visible and a missing selection never silently launches at the default path",
  async () => {
    const { api, request } = client();
    const user = userEvent.setup();
    const start = vitest.vi.fn(async () => true);
    const view = testing.render(
      <WorktreesProvider runner={api}>
        <Launch start={start} />
      </WorktreesProvider>,
    );
    await view.findByRole("option", { name: "Review sandbox — review" });
    await user.selectOptions(view.getByLabelText("Worktree"), record.id);
    request.mockRejectedValueOnce(new Error("Offline"));
    await user.click(view.getByRole("button", { name: "Refresh worktrees" }));
    vitest.expect(view.getByRole("alert").textContent).toContain("Offline");
    vitest
      .expect(view.getByRole("button", { name: "Start conversation" }))
      .toHaveProperty("disabled", true);
    request.mockResolvedValueOnce({
      worktrees: [{ ...record, status: "missing" }],
      next_after: null,
    });
    await user.click(view.getByRole("button", { name: "Refresh worktrees" }));
    vitest
      .expect(view.getByRole("option", { name: "Review sandbox (missing)" }))
      .toBeDefined();
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest.expect(start).not.toHaveBeenCalled();
    await user.selectOptions(view.getByLabelText("Worktree"), "");
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest.expect(start).toHaveBeenCalledExactlyOnceWith("agent", {});
  },
);

vitest.test(
  "chat header and subagent starts show explicit and inherited worktree names",
  async () => {
    vitest.vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    );
    const { api, request } = client();
    request.mockImplementation(async (_method, params = {}) =>
      params.worktree_id === "other"
        ? { ...record, id: "other", name: "Other sandbox" }
        : record,
    );
    try {
      const view = testing.render(
        <WorktreesProvider runner={api}>
          <Workspace
            sessions={[
              {
                execution_id: "execution",
                reference: "agent",
                status: "running",
                active: true,
                pending_inputs: 0,
              },
            ]}
            selected="execution"
            onSelect={() => {}}
            pending={[]}
            status="Connected"
            endpoint="ws://localhost"
            enabled
            busy={false}
            onStop={() => {}}
            onReply={async () => true}
            newSession={null}
            connection={null}
            error=""
            dismiss={() => {}}
            hasOlder={false}
            execution={{
              id: "execution",
              root_workflow_id: "root",
              status: "running",
              checkpoint: { heads: { root: "spawn-2" } },
              target: {
                kind: "local",
                root: record.path,
                worktree_id: record.id,
              },
            }}
            history={[
              {
                position: 1,
                step: {
                  id: "spawn-1",
                  workflow_id: "root",
                  operation: {
                    kind: "workflow.spawn",
                    child_workflow_id: "child-1",
                    description: "Inherited",
                  },
                  outcome: { kind: "completed" },
                },
              },
              {
                position: 2,
                step: {
                  id: "spawn-2",
                  previous_step_id: "spawn-1",
                  workflow_id: "root",
                  operation: {
                    kind: "workflow.spawn",
                    child_workflow_id: "child-2",
                    description: "Explicit",
                    worktree_id: "other",
                  },
                  outcome: { kind: "completed" },
                },
              },
            ]}
          />
        </WorktreesProvider>,
      );
      await testing.waitFor(() =>
        vitest
          .expect(view.getAllByText("Worktree: Review sandbox"))
          .toHaveLength(2),
      );
      await view.findByText("Worktree: Other sandbox");
      vitest
        .expect(
          view.getByText("Agent: Inherited").closest("summary")?.textContent,
        )
        .toContain("Worktree: Review sandbox");
      vitest
        .expect(
          view.getByText("Agent: Explicit").closest("summary")?.textContent,
        )
        .toContain("Worktree: Other sandbox");
    } finally {
      vitest.vi.unstubAllGlobals();
    }
  },
);

vitest.test(
  "unavailable worktree metadata hides its ID unless debugging is enabled",
  async () => {
    const { api, request } = client();
    request.mockRejectedValue(new Error("Worktree not found"));
    const view = testing.render(
      <WorktreesProvider runner={api}>
        <WorktreeLabel id="old-tree" />
      </WorktreesProvider>,
    );
    await testing.waitFor(() =>
      vitest
        .expect(view.getByText("Worktree: Unavailable").title)
        .toContain("Worktree not found"),
    );
    vitest.expect(view.queryByText("Worktree: old-tree")).toBeNull();
    view.rerender(
      <DebugProvider value={true}>
        <WorktreesProvider runner={api}>
          <WorktreeLabel id="old-tree" />
        </WorktreesProvider>
      </DebugProvider>,
    );
    vitest.expect(view.getByText("Worktree: old-tree")).toBeTruthy();
  },
);

vitest.test(
  "sidebar resolves worktree names and keeps full metadata in the tooltip",
  async () => {
    const { api, request } = client();
    const name = "A very long worktree name that must stay on a single line";
    request.mockResolvedValue({ ...record, name });
    const noop = () => {};
    const view = testing.render(
      <WorktreesProvider runner={api}>
        <SessionSidebar
          sessions={[
            {
              execution_id: "execution",
              reference: "agent",
              name: "Review changes",
              status: "running",
              active: true,
              pending_inputs: 0,
              worktree_id: record.id,
            },
          ]}
          workflows={[]}
          selected="execution"
          status="Connected"
          select={noop}
          collapse={noop}
          create={noop}
          configure={noop}
        />
      </WorktreesProvider>,
    );
    const label = await view.findByText(`Worktree: ${name}`);
    vitest.expect(label.title).toContain(name);
    vitest.expect(label.title).toContain(record.path);
    vitest
      .expect(request)
      .toHaveBeenCalledWith("worktrees.get", { worktree_id: record.id });
  },
);

vitest.test(
  "late creation does not select a worktree after dismissal or leak it to another runner",
  async () => {
    const first = client();
    const second = client();
    let finish!: (record: Worktree) => void;
    first.request.mockReturnValue(
      new Promise<Worktree>((resolve) => {
        finish = resolve;
      }),
    );
    second.request.mockResolvedValue({ worktrees: [], next_after: null });
    const created = vitest.vi.fn();
    const user = userEvent.setup();
    const view = testing.render(
      <WorktreesProvider runner={first.api}>
        <WorktreeCreate created={created} cancel={() => {}} />
      </WorktreesProvider>,
    );
    await user.type(view.getByLabelText("Worktree name"), "Pending");
    await user.click(view.getByRole("button", { name: "Create worktree" }));
    view.rerender(
      <WorktreesProvider runner={second.api}>
        <Launch start={async () => true} />
      </WorktreesProvider>,
    );
    await testing.act(async () => finish(record));
    vitest.expect(created).not.toHaveBeenCalled();
    vitest
      .expect(view.queryByRole("option", { name: /Review sandbox/ }))
      .toBeNull();
  },
);

vitest.test(
  "nested subagents inherit the nearest explicit worktree rather than the root",
  () => {
    const roots = workflowTree(
      [
        {
          position: 1,
          step: {
            id: "spawn-parent",
            workflow_id: "root",
            operation: {
              kind: "workflow.spawn",
              child_workflow_id: "parent",
              worktree_id: "parent-tree",
            },
            outcome: { kind: "completed" },
          },
        },
        {
          position: 2,
          step: {
            id: "spawn-child",
            workflow_id: "parent",
            operation: { kind: "workflow.spawn", child_workflow_id: "child" },
            outcome: { kind: "completed" },
          },
        },
      ],
      [],
    );
    const parent = roots[0].entries[0];
    vitest.expect("entries" in parent).toBe(true);
    if (!("entries" in parent)) throw new Error("Missing parent workflow");
    const child = parent.entries[0];
    if (!("entries" in child)) throw new Error("Missing child workflow");
    vitest.expect(workflowWorktree(child, "root-tree")).toBe("parent-tree");
  },
);

vitest.test(
  "inline creation preserves workflow input edits made while creation is pending",
  async () => {
    const { api, request } = client();
    const start = vitest.vi.fn(async () => true);
    const user = userEvent.setup();
    const view = testing.render(
      <WorktreesProvider runner={api}>
        <Launch start={start} />
      </WorktreesProvider>,
    );
    await view.findByRole("option", { name: "Review sandbox — review" });
    await user.click(view.getByRole("button", { name: "Edit as JSON" }));
    await user.click(view.getByRole("button", { name: "New worktree" }));
    await user.type(view.getByLabelText("Worktree name"), "Pending");
    let finish!: (record: Worktree) => void;
    request.mockReturnValueOnce(
      new Promise<Worktree>((resolve) => {
        finish = resolve;
      }),
    );
    await user.click(view.getByRole("button", { name: "Create worktree" }));
    testing.fireEvent.change(view.getByLabelText("Workflow input (JSON)"), {
      target: { value: '{"task":"Updated"}' },
    });
    await testing.act(async () => finish({ ...record, id: "new-tree" }));
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest
      .expect(start)
      .toHaveBeenCalledExactlyOnceWith(
        "agent",
        { task: "Updated" },
        "new-tree",
      );
  },
);
