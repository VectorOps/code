import * as testing from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import { WorkflowStats } from "../src/components/WorkflowStats";
import { formatBytes } from "../src/models/disk-usage";
import type { Params } from "../src/models/json";
import type {
  DiskUsageReport,
  Session,
  WorkflowDiskUsage,
} from "../src/models/runner";
import { DiskUsage } from "../src/screens/DiskUsage";
import { Settings } from "../src/screens/Settings";

const report: DiskUsageReport = {
  total_bytes: 8192,
  workflow_count: 12,
  top_workflows: Array.from({ length: 10 }, (_, index) => ({
    execution_id: `exec-${index}`,
    size_bytes: 1024 - index,
  })),
};
const session: Session = {
  execution_id: "exec-0",
  root_workflow_id: "root",
  reference: "agent",
  status: "completed",
  active: false,
  pending_inputs: 0,
};

class Connection extends RunnerConnection {
  calls: { method: string; params: Params }[] = [];
  read = vi.fn(async (): Promise<DiskUsageReport> => report);
  summary = { ...session, description: "Review changes" };

  constructor() {
    super("ws://localhost:1234", "test");
  }

  override async request<T>(method: string, params: Params = {}): Promise<T> {
    this.calls.push({ method, params });
    if (method === "runner.disk_usage") return (await this.read()) as T;
    if (method === "executions.disk_usage")
      return { execution_id: params.execution_id, size_bytes: 2048 } as T;
    if (method === "executions.summary") {
      if (params.execution_id !== "exec-0") throw new Error("not_found");
      return this.summary as T;
    }
    throw new Error(`Unexpected method: ${method}`);
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test.each([
  [0, "0 B"],
  [1, "1 B"],
  [1023, "1,023 B"],
  [1024, "1 KiB"],
  [1536, "1.5 KiB"],
  [1024 ** 2, "1 MiB"],
  [1024 ** 3, "1 GiB"],
  [1024 ** 4, "1 TiB"],
])("formats %s bytes as %s", (bytes, expected) => {
  expect(formatBytes(bytes as number)).toBe(expected);
});

test("disk usage API delegates ranking and targeted size measurement to Runner", async () => {
  const connection = new Connection();
  const api = new RunnerApi(connection);
  expect(await api.diskUsage()).toEqual(report);
  expect(await api.executionDiskUsage("outside-top-ten")).toEqual({
    execution_id: "outside-top-ten",
    size_bytes: 2048,
  });
  expect(connection.calls).toEqual([
    { method: "runner.disk_usage", params: { top: 10 } },
    {
      method: "executions.disk_usage",
      params: { execution_id: "outside-top-ten" },
    },
  ]);
});

test("settings loads total usage and the top ten only when the disk usage tab opens", async () => {
  const connection = new Connection();
  const api = new RunnerApi(connection);
  const view = testing.render(
    <Settings
      api={api}
      connected
      connection={{ endpoint: "ws://localhost:1234", token: "test" }}
    />,
  );
  expect(connection.calls).toEqual([]);
  testing.fireEvent.click(view.getByRole("button", { name: "Disk usage" }));
  expect(view.getByText("Loading disk usage…")).toBeTruthy();
  expect(await view.findByText("8 KiB")).toBeTruthy();
  expect(await view.findByText("Review changes")).toBeTruthy();
  expect(view.getByText("completed")).toBeTruthy();
  expect(view.getByText("12")).toBeTruthy();
  const rows = testing.within(view.getByRole("table")).getAllByRole("row");
  expect(rows).toHaveLength(11);
  report.top_workflows.forEach((workflow, index) => {
    expect(rows[index + 1].textContent).toContain(workflow.execution_id);
    expect(rows[index + 1].textContent).toContain(
      formatBytes(workflow.size_bytes),
    );
  });
  expect(view.queryByRole("alert")).toBeNull();
  expect(
    connection.calls.filter((call) => call.method === "executions.summary"),
  ).toHaveLength(10);
  connection.read.mockResolvedValue({
    total_bytes: 0,
    workflow_count: 0,
    top_workflows: [],
  });
  testing.fireEvent.click(
    view.getByRole("button", { name: "Refresh disk usage" }),
  );
  expect(await view.findByText("No workflows stored yet.")).toBeTruthy();
  expect(view.getByText("0 B")).toBeTruthy();
  expect(view.queryByRole("table")).toBeNull();
});

test("disk usage errors can be retried and disconnected settings do not scan", async () => {
  const connection = new Connection();
  connection.read.mockRejectedValueOnce(new Error("Scan denied"));
  const props = {
    api: new RunnerApi(connection),
    connection: { endpoint: "ws://localhost:1234", token: "test" },
  };
  const view = testing.render(<Settings {...props} connected={false} />);
  testing.fireEvent.click(view.getByRole("button", { name: "Disk usage" }));
  expect(view.getByText("Runner disconnected.")).toBeTruthy();
  expect(connection.calls).toEqual([]);
  view.rerender(<Settings {...props} connected />);
  expect((await view.findByRole("alert")).textContent).toContain("Scan denied");
  expect(view.queryByText("0 B")).toBeNull();
  testing.fireEvent.click(
    view.getByRole("button", { name: "Refresh disk usage" }),
  );
  expect(await view.findByText("8 KiB")).toBeTruthy();
  expect(view.queryByRole("alert")).toBeNull();
});

test("disk usage ignores a scan from a previous runner", async () => {
  const old = deferred<DiskUsageReport>();
  const connection = new Connection();
  connection.read.mockReturnValue(old.promise);
  const view = testing.render(<DiskUsage api={new RunnerApi(connection)} />);
  const next = new Connection();
  next.read.mockResolvedValue({
    total_bytes: 0,
    workflow_count: 0,
    top_workflows: [],
  });
  view.rerender(<DiskUsage api={new RunnerApi(next)} />);
  expect(await view.findByText("0 B")).toBeTruthy();
  await testing.act(async () => {
    old.resolve(report);
  });
  expect(view.queryByText("8 KiB")).toBeNull();
  expect(connection.calls).toHaveLength(1);
});

test.each([
  { status: "completed", active: false },
  { status: "failed", active: false },
  { status: "failed", active: true },
])(
  "$status workflow (active=$active) confirms beside its row before deletion and refreshes usage",
  async ({ status, active }) => {
    const connection = new Connection();
    connection.summary.status = status;
    connection.summary.active = active;
    const onDelete = vi.fn(async () => true);
    const view = testing.render(
      <Settings
        api={new RunnerApi(connection)}
        connected
        connection={{ endpoint: "ws://localhost:1234", token: "test" }}
        onDelete={onDelete}
      />,
    );
    testing.fireEvent.click(view.getByRole("button", { name: "Disk usage" }));
    const remove = await view.findByRole("button", {
      name: "Delete Review changes",
    });
    expect(remove.querySelector("svg")).toBeTruthy();
    expect(
      (view.getByRole("button", { name: "Delete exec-1" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    testing.fireEvent.click(remove);
    expect(onDelete).not.toHaveBeenCalled();
    const confirmation = view.getByRole("group", {
      name: "Confirm workflow deletion",
    });
    expect(confirmation.closest("tr")).toBe(
      remove.closest("tr")?.nextElementSibling,
    );
    const cancel = view.getByRole("button", { name: "Cancel" });
    expect(document.activeElement).toBe(cancel);
    expect(
      (
        view.getByRole("button", {
          name: "Permanently delete workflow",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
    testing.fireEvent.click(cancel);
    expect(document.activeElement).toBe(remove);
    expect(onDelete).not.toHaveBeenCalled();
    testing.fireEvent.click(remove);
    connection.read.mockResolvedValue({
      total_bytes: 0,
      workflow_count: 0,
      top_workflows: [],
    });
    testing.fireEvent.click(
      view.getByRole("button", { name: "Permanently delete workflow" }),
    );
    expect(await view.findByText("No workflows stored yet.")).toBeTruthy();
    expect(onDelete).toHaveBeenCalledExactlyOnceWith("exec-0");
    expect(connection.read).toHaveBeenCalledTimes(2);
    expect(
      view.queryByRole("group", { name: "Confirm workflow deletion" }),
    ).toBeNull();
  },
);

test.each(["running", "starting", "stopping"])(
  "%s workflows show status and cannot be deleted",
  async (status) => {
    const connection = new Connection();
    connection.summary.status = status;
    const onDelete = vi.fn(async () => true);
    const view = testing.render(
      <DiskUsage api={new RunnerApi(connection)} onDelete={onDelete} />,
    );
    expect(await view.findByText(status)).toBeTruthy();
    testing.fireEvent.click(
      view.getByRole("button", { name: "Delete Review changes" }),
    );
    expect(view.getByText(/Stop the workflow and wait/)).toBeTruthy();
    const confirm = view.getByRole("button", {
      name: "Permanently delete workflow",
    });
    expect((confirm as HTMLButtonElement).disabled).toBe(true);
    testing.fireEvent.click(confirm);
    expect(onDelete).not.toHaveBeenCalled();
  },
);

test("failed deletion keeps the workflow and allows retry without a duplicate in-flight request", async () => {
  const pending = deferred<boolean>();
  const onDelete = vi.fn(() => pending.promise);
  const connection = new Connection();
  const view = testing.render(
    <DiskUsage api={new RunnerApi(connection)} onDelete={onDelete} />,
  );
  testing.fireEvent.click(
    await view.findByRole("button", { name: "Delete Review changes" }),
  );
  const confirm = view.getByRole("button", {
    name: "Permanently delete workflow",
  });
  testing.fireEvent.click(confirm);
  testing.fireEvent.click(confirm);
  expect(onDelete).toHaveBeenCalledTimes(1);
  await testing.act(async () => {
    pending.resolve(false);
  });
  expect((await view.findByRole("alert")).textContent).toContain(
    "Workflow was not deleted",
  );
  expect(view.getByText("8 KiB")).toBeTruthy();
  expect(connection.read).toHaveBeenCalledTimes(1);
  expect(
    (
      view.getByRole("button", {
        name: "Permanently delete workflow",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(false);
});

test("workflow info loads the selected execution size, ignores stale responses and reloads on reopening", async () => {
  const pending = deferred<WorkflowDiskUsage>();
  const load = vi.fn(
    async (executionId: string): Promise<WorkflowDiskUsage> => ({
      execution_id: executionId,
      size_bytes: 0,
    }),
  );
  load.mockReturnValueOnce(pending.promise);
  const view = testing.render(
    <WorkflowStats session={session} loadDiskUsage={load} />,
  );
  expect(view.getByText("Loading…")).toBeTruthy();
  expect(load).toHaveBeenCalledWith("exec-0");
  view.rerender(
    <WorkflowStats
      session={{ ...session, execution_id: "another" }}
      loadDiskUsage={load}
    />,
  );
  expect(await view.findByText("0 B")).toBeTruthy();
  expect(load).toHaveBeenCalledWith("another");
  await testing.act(async () => {
    pending.resolve({ execution_id: "exec-0", size_bytes: 8192 });
  });
  expect(view.queryByText("8 KiB")).toBeNull();
  view.unmount();
  const reopened = testing.render(
    <WorkflowStats session={session} loadDiskUsage={load} />,
  );
  expect(await reopened.findByText("0 B")).toBeTruthy();
  expect(load).toHaveBeenCalledTimes(3);
});

test("workflow size errors and disconnection preserve other workflow information", async () => {
  const load = vi.fn().mockRejectedValue(new Error("not_found"));
  const view = testing.render(
    <WorkflowStats session={session} loadDiskUsage={load} />,
  );
  expect((await view.findByRole("alert")).textContent).toContain("not_found");
  expect(view.getByText("Unavailable")).toBeTruthy();
  expect(view.getByText("completed")).toBeTruthy();
  expect(view.getByText("root")).toBeTruthy();
  view.rerender(<WorkflowStats session={session} />);
  expect(view.getByText("Runner disconnected.")).toBeTruthy();
  expect(view.queryByRole("alert")).toBeNull();
  expect(load).toHaveBeenCalledTimes(1);
});
