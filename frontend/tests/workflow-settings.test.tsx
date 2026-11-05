import * as testing from "@testing-library/react";
import { DebugProvider } from "../src/components/Diagnostics";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { RunnerConnection } from "../src/communication/websocket";
import { RunnerApi } from "../src/communication/runner-api";
import type { Json, Params } from "../src/models/json";
import { object } from "../src/models/json";
import { Settings } from "../src/screens/Settings";

class Connection extends RunnerConnection {
  calls: { method: string; params: Params }[] = [];
  value: Json = {
    agent: { agents: ["builtin:helper", "missing:main"] },
    helper: { max_rounds: 30 },
  };
  exists = true;
  failDiscovery = false;
  read = vitest.vi.fn(async () => ({
    name: "components.workflows",
    exists: this.exists,
    value: structuredClone(this.value),
  }));
  write = vitest.vi.fn(async (params: Params) => {
    const parts = String(params.name).split(".");
    const group = parts[2];
    this.value = {
      ...object(this.value),
      [group]: {
        ...object(object(this.value)[group]),
        [parts[3]]: params.value,
      },
    };
    return null;
  });
  reset = vitest.vi.fn(async (_params: Params) => null);

  constructor() {
    super("ws://127.0.0.1:1234", "test");
  }

  override async request<T>(method: string, params: Params = {}): Promise<T> {
    this.calls.push({ method, params });
    if (method === "configuration.get") return (await this.read()) as T;
    if (method === "configuration.set") return (await this.write(params)) as T;
    if (method === "configuration.reset")
      return (await this.reset(params)) as T;
    if (method === "configuration.suggest")
      return { values: ["low", "high"] } as T;
    if (method === "workflows.list") {
      if (this.failDiscovery) throw new Error("offline");
      const references = params.after
        ? ["checks:main"]
        : ["builtin:helper", "research:main"];
      return {
        workflows: references.map((reference) => ({
          descriptor: {
            reference,
            name: reference.split(":")[0],
            description: null,
            input_schema: {},
          },
        })),
        next_after: params.after ? null : "page-2",
        errors: [],
      } as T;
    }
    throw new Error(`Unexpected method: ${method}`);
  }
}

async function open(connection = new Connection(), debug = false) {
  const user = userEvent.setup();
  const view = testing.render(
    <DebugProvider value={debug}>
      <Settings
        api={new RunnerApi(connection)}
        connected
        connection={{ endpoint: "ws://127.0.0.1:1234", token: "test" }}
      />
    </DebugProvider>,
  );
  await user.click(view.getByRole("button", { name: "Workflows" }));
  await view.findByRole("form", { name: "Edit Allowed agents" });
  return { connection, view, user };
}

function lifetime(view: testing.RenderResult, label: string) {
  return testing.within(
    view.getByRole("group", { name: `${label} override lifetime` }),
  );
}

vitest.test("workflow settings require a runner", async () => {
  const view = testing.render(
    <Settings connected={false} connection={{ endpoint: "", token: "" }} />,
  );
  await userEvent
    .setup()
    .click(view.getByRole("button", { name: "Workflows" }));
  vitest
    .expect(view.getByRole("status").textContent)
    .toContain("Runner disconnected");
  vitest.expect(view.queryByRole("form")).toBeNull();
});

vitest.test(
  "discovers paginated workflows, searches and saves unique agent tags",
  async () => {
    const { connection, view, user } = await open();
    await user.click(view.getByLabelText("Allowed agents"));
    await user.click(await view.findByRole("checkbox", { name: /research/ }));
    await user.type(view.getByLabelText("Allowed agents"), "checks");
    vitest
      .expect(view.queryByRole("checkbox", { name: /research/ }))
      .toBeNull();
    await user.click(view.getByRole("checkbox", { name: /checks/ }));
    await user.click(view.getByRole("button", { name: "Remove missing:main" }));
    await user.click(
      lifetime(view, "Allowed agents").getByRole("radio", {
        name: "Workspace",
      }),
    );
    await user.click(
      view.getByRole("button", { name: "Apply allowed agents" }),
    );
    await view.findByText("Allowed agents applied as a permanent override.");
    vitest.expect(connection.write).toHaveBeenCalledExactlyOnceWith({
      name: "components.workflows.agent.agents",
      value: ["builtin:helper", "research:main", "checks:main"],
      permanent: true,
    });
    vitest
      .expect(
        connection.calls.filter((call) => call.method === "workflows.list"),
      )
      .toEqual([
        { method: "workflows.list", params: { after: null, limit: 100 } },
        { method: "workflows.list", params: { after: "page-2", limit: 100 } },
      ]);
    await user.clear(view.getByLabelText("Allowed agents"));
    await user.click(view.getByLabelText("Allowed agents"));
    const research = view.getByRole("checkbox", { name: /research/ });
    await user.click(research);
    await user.click(research);
    vitest
      .expect(view.getAllByRole("button", { name: "Remove research:main" }))
      .toHaveLength(1);
  },
);

vitest.test(
  "workflow picker supports keyboard selection, dismissal, and tag removal",
  async () => {
    const { connection, view, user } = await open();
    const search = view.getByLabelText("Allowed agents");
    await user.click(search);
    await user.keyboard("{ArrowDown}");
    vitest
      .expect(document.activeElement)
      .toBe(view.getByRole("checkbox", { name: /builtin/ }));
    await user.keyboard(" ");
    vitest
      .expect(view.queryByRole("button", { name: "Remove builtin:helper" }))
      .toBeNull();
    await user.keyboard("{Escape}");
    vitest.expect(document.activeElement).toBe(search);
    vitest
      .expect(view.queryByRole("group", { name: "Available workflows" }))
      .toBeNull();
    view.getByRole("button", { name: "Remove missing:main" }).focus();
    await user.keyboard("{Enter}");
    vitest
      .expect(view.queryByRole("button", { name: "Remove missing:main" }))
      .toBeNull();
    vitest.expect(connection.write).not.toHaveBeenCalled();
  },
);

vitest.test(
  "each setting has an independent two-sided radio toggle",
  async () => {
    const { connection, view, user } = await open();
    vitest
      .expect(view.queryByRole("group", { name: "Shared defaults" }))
      .toBeNull();
    vitest
      .expect(view.queryByLabelText(/Strong model|Medium model|Weak model/))
      .toBeNull();
    const toggle = view.getByRole("group", {
      name: "Agent tool collection override lifetime",
    });
    vitest.expect(toggle.textContent).toContain("Temporary");
    vitest.expect(toggle.textContent).toContain("Workspace");
    for (const button of [
      view.getByRole("button", { name: "Apply agent tool collection" }),
      view.getByRole("button", { name: "Reset agent tool collection to TOML" }),
    ]) {
      vitest.expect(button.textContent?.trim()).toMatch(/Apply|Reset/);
      vitest.expect(button.querySelector("svg")).not.toBeNull();
      vitest.expect(button.title).not.toBe("");
    }
    await user.click(
      lifetime(view, "Agent tool collection").getByRole("radio", {
        name: "Workspace",
      }),
    );
    vitest
      .expect(
        lifetime(view, "Helper tool collection").getByRole("radio", {
          name: "Temporary",
        }),
      )
      .toHaveProperty("checked", true);
    vitest
      .expect(
        lifetime(view, "Allowed agents").getByRole("radio", {
          name: "Temporary",
        }),
      )
      .toHaveProperty("checked", true);
    await user.type(
      view.getByLabelText("Agent tool collection"),
      "agent_tools",
    );
    await user.type(
      view.getByLabelText("Helper tool collection"),
      "helper_tools",
    );
    await user.click(
      view.getByRole("button", { name: "Apply agent tool collection" }),
    );
    await view.findByText(
      "Agent tool collection applied as a permanent override.",
    );
    await user.click(
      view.getByRole("button", { name: "Apply helper tool collection" }),
    );
    await view.findByText(
      "Helper tool collection applied as a temporary override.",
    );
    vitest
      .expect(connection.write.mock.calls.map(([params]) => params))
      .toEqual([
        {
          name: "components.workflows.agent.tool_collection",
          value: "agent_tools",
          permanent: true,
        },
        {
          name: "components.workflows.helper.tool_collection",
          value: "helper_tools",
          permanent: false,
        },
      ]);
    await user.click(
      view.getByRole("button", { name: "Reset agent tool collection to TOML" }),
    );
    await view.findByText("Agent tool collection reset to startup TOML.");
    vitest.expect(connection.reset).toHaveBeenCalledExactlyOnceWith({
      name: "components.workflows.agent.tool_collection",
    });
  },
);

vitest.test(
  "empty agent selection is saved explicitly without losing unavailable references on discovery failure",
  async () => {
    const connection = new Connection();
    connection.failDiscovery = true;
    const { view, user } = await open(connection);
    vitest
      .expect((await view.findByRole("alert")).textContent)
      .toContain("Existing selections are preserved");
    await user.click(
      view.getByRole("button", { name: "Remove builtin:helper" }),
    );
    await user.click(view.getByRole("button", { name: "Remove missing:main" }));
    await user.click(
      view.getByRole("button", { name: "Apply allowed agents" }),
    );
    await view.findByText("Allowed agents applied as a temporary override.");
    vitest.expect(connection.write).toHaveBeenCalledExactlyOnceWith({
      name: "components.workflows.agent.agents",
      value: [],
      permanent: false,
    });
  },
);

vitest.test(
  "missing settings explain the built-in allowlist without writing defaults",
  async () => {
    const connection = new Connection();
    connection.exists = false;
    connection.value = null;
    const { view } = await open(connection);
    vitest
      .expect(
        view.getByText(
          "Not configured. The built-in default is builtin:helper.",
        ),
      )
      .toBeTruthy();
    vitest
      .expect(view.getByRole("button", { name: "Remove builtin:helper" }))
      .toBeTruthy();
    vitest.expect(connection.write).not.toHaveBeenCalled();
    vitest.expect(view.queryByRole("alert")).toBeNull();
  },
);

vitest.test(
  "invalid agent values remain inspectable and resettable",
  async () => {
    const connection = new Connection();
    connection.value = { agent: { agents: "invalid" } };
    const { view, user } = await open(connection, true);
    vitest
      .expect(view.getByRole("button", { name: "Apply allowed agents" }))
      .toHaveProperty("disabled", true);
    await user.click(view.getByText("Configuration JSON"));
    vitest.expect(view.getByText(/"agents": "invalid"/)).toBeTruthy();
    await user.click(
      view.getByRole("button", { name: "Reset allowed agents to TOML" }),
    );
    await view.findByText("Allowed agents reset to startup TOML.");
    vitest.expect(connection.reset).toHaveBeenCalledExactlyOnceWith({
      name: "components.workflows.agent.agents",
    });
  },
);

vitest.test(
  "numeric helper settings enforce constraints and prompts allow empty strings",
  async () => {
    const { connection, view, user } = await open();
    await user.clear(view.getByLabelText("Maximum rounds"));
    await user.type(view.getByLabelText("Maximum rounds"), "1.5");
    await user.click(
      view.getByRole("button", { name: "Apply maximum rounds" }),
    );
    vitest.expect(connection.write).not.toHaveBeenCalled();
    await user.clear(view.getByLabelText("Maximum rounds"));
    await user.type(view.getByLabelText("Maximum rounds"), "10");
    await user.click(
      view.getByRole("button", { name: "Apply maximum rounds" }),
    );
    await view.findByText("Maximum rounds applied as a temporary override.");
    const agent = testing.within(view.getByRole("group", { name: "Agent" }));
    await user.click(
      agent.getByRole("button", { name: "Apply system prompt" }),
    );
    await view.findByText("System prompt applied as a temporary override.");
    vitest.expect(connection.write).toHaveBeenLastCalledWith({
      name: "components.workflows.agent.system_prompt",
      value: "",
      permanent: false,
    });
  },
);

vitest.test(
  "load and mutation failures can be retried and applied changes are distinguished",
  async () => {
    const connection = new Connection();
    connection.read.mockRejectedValueOnce(new Error("offline"));
    const user = userEvent.setup();
    const view = testing.render(
      <Settings
        api={new RunnerApi(connection)}
        connected
        connection={{ endpoint: "", token: "" }}
      />,
    );
    await user.click(view.getByRole("button", { name: "Workflows" }));
    vitest
      .expect((await view.findByRole("alert")).textContent)
      .toContain("Unable to load workflow settings");
    await user.click(view.getByRole("button", { name: "Refresh settings" }));
    await view.findByRole("form", { name: "Edit Allowed agents" });
    connection.write.mockRejectedValueOnce(new Error("rejected"));
    await user.click(
      view.getByRole("button", { name: "Apply allowed agents" }),
    );
    vitest
      .expect((await view.findByRole("alert")).textContent)
      .toContain("could not be confirmed");
    connection.read.mockRejectedValueOnce(new Error("offline"));
    await user.click(
      view.getByRole("button", { name: "Reset allowed agents to TOML" }),
    );
    vitest
      .expect((await view.findByRole("alert")).textContent)
      .toContain("Change applied, but reloading failed");
  },
);
