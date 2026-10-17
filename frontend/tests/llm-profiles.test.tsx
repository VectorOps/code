import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { RunnerConnection } from "../src/communication/websocket";
import { RunnerApi } from "../src/communication/runner-api";
import type { Json, Params } from "../src/models/json";
import { Settings } from "../src/screens/Settings";

class Connection extends RunnerConnection {
  calls: { method: string; params: Params }[] = [];
  value: Json = {
    model: "openai/base",
    provider: null,
    temperature: 0,
    retry: { max_retries: 2 },
    profiles: {
      strong: { model: "openai/powerful" },
      "custom.name": { model: "router/vendor/custom" },
    },
  };
  exists = true;
  read = vitest.vi.fn(async () => ({
    name: "components.llm",
    exists: this.exists,
    value: structuredClone(this.value),
  }));
  write = vitest.vi.fn(async (_params: Params) => null);
  reset = vitest.vi.fn(async (_params: Params) => null);
  suggestions = vitest.vi.fn(
    async (_params: Params): Promise<{ values: Json[] }> => ({
      values: [
        "openai/catalog-model",
        "other/catalog-model",
        "router/vendor/nested:tag",
      ],
    }),
  );

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
      return (await this.suggestions(params)) as T;
    throw new Error(`Unexpected method: ${method}`);
  }
}

async function open(connection = new Connection()) {
  const user = userEvent.setup();
  const view = testing.render(
    <Settings
      api={new RunnerApi(connection)}
      connected
      connection={{ endpoint: "ws://127.0.0.1:1234", token: "test" }}
    />,
  );
  await user.click(view.getByRole("button", { name: "LLM profiles" }));
  await view.findByRole("form", { name: "Edit Model" });
  return { connection, view, user };
}

vitest.test("profile settings require a runner connection", async () => {
  const view = testing.render(
    <Settings connected={false} connection={{ endpoint: "", token: "" }} />,
  );
  await userEvent
    .setup()
    .click(view.getByRole("button", { name: "LLM profiles" }));
  vitest
    .expect(view.getByRole("status").textContent)
    .toContain("Runner disconnected");
  vitest.expect(view.queryByRole("form")).toBeNull();
});

vitest.test(
  "shows raw shared values, configured profiles, and no invented inheritance",
  async () => {
    const { connection, view, user } = await open();
    vitest
      .expect(view.getByLabelText("Model"))
      .toHaveProperty("value", "openai/base");
    vitest
      .expect(view.getByLabelText("Model"))
      .toHaveProperty("placeholder", "provider/model");
    vitest
      .expect(view.getByLabelText("Temperature"))
      .toHaveProperty("value", "0");
    vitest.expect(view.queryByLabelText("Provider")).toBeNull();
    vitest.expect(view.queryByLabelText("Legacy provider")).toBeNull();
    const target = view.getByLabelText("Edit settings for");
    vitest
      .expect(
        Array.from(
          target.querySelectorAll("option"),
          (option) => option.textContent,
        ),
      )
      .toEqual(["Shared settings", "custom.name", "strong"]);
    await user.selectOptions(target, "profile:strong");
    vitest
      .expect(view.getByLabelText("Model"))
      .toHaveProperty("value", "openai/powerful");
    vitest
      .expect(view.getByLabelText("Temperature"))
      .toHaveProperty("value", "");
    vitest
      .expect(
        testing
          .within(view.getByRole("form", { name: "Edit Temperature" }))
          .getByText("Not configured"),
      )
      .toBeTruthy();
    await user.selectOptions(target, "shared");
    await user.click(view.getByText("Configuration JSON"));
    vitest.expect(view.getByText(/"max_retries": 2/)).toBeTruthy();
    vitest.expect(view.getByText(/"provider": null/)).toBeTruthy();
    vitest.expect(connection.write).not.toHaveBeenCalled();
    vitest.expect(connection.reset).not.toHaveBeenCalled();
    vitest
      .expect(
        view.queryByText(
          /Raw effective configuration|Temporary writes replace|Suggestions are advisory/,
        ),
      )
      .toBeNull();
  },
);

vitest.test(
  "applies each field independently and rereads without losing other drafts",
  async () => {
    const { connection, view, user } = await open();
    await user.clear(view.getByLabelText("Model"));
    await user.type(view.getByLabelText("Model"), "other/unlisted-model");
    await user.clear(view.getByLabelText("Temperature"));
    await user.type(view.getByLabelText("Temperature"), "0.5");
    connection.value = {
      model: "other/unlisted-model",
      temperature: 0,
      provider: null,
    };
    await user.click(view.getByRole("button", { name: "Apply model" }));
    await view.findByText("Model applied as a temporary override.");
    vitest.expect(connection.write).toHaveBeenCalledExactlyOnceWith({
      name: "components.llm.model",
      value: "other/unlisted-model",
      permanent: false,
    });
    vitest.expect(connection.read).toHaveBeenCalledTimes(2);
    vitest
      .expect(view.getByLabelText("Temperature"))
      .toHaveProperty("value", "0.5");
    vitest
      .expect(
        connection.calls.filter((call) => call.method === "configuration.set"),
      )
      .toHaveLength(1);
  },
);

vitest.test(
  "qualified profile models are applied without separate provider writes",
  async () => {
    const { connection, view, user } = await open();
    await user.selectOptions(
      view.getByLabelText("Edit settings for"),
      "profile:strong",
    );
    await user.clear(view.getByLabelText("Model"));
    await user.type(
      view.getByLabelText("Model"),
      "openrouter/anthropic/claude:tag",
    );
    await user.click(
      testing
        .within(view.getByRole("group", { name: "Model override lifetime" }))
        .getByRole("radio", { name: "Workspace" }),
    );
    connection.value = {
      profiles: { strong: { model: "openrouter/anthropic/claude:tag" } },
    };
    await user.click(view.getByRole("button", { name: "Apply model" }));
    await view.findByText("Model applied as a permanent override.");
    vitest.expect(connection.write).toHaveBeenCalledExactlyOnceWith({
      name: "components.llm.profiles.strong.model",
      value: "openrouter/anthropic/claude:tag",
      permanent: true,
    });
    vitest.expect(connection.reset).not.toHaveBeenCalled();
    vitest
      .expect(view.getByLabelText("Model"))
      .toHaveProperty("value", "openrouter/anthropic/claude:tag");
  },
);

vitest.test(
  "provider controls are absent even for legacy snapshots",
  async () => {
    const connection = new Connection();
    connection.value = { model: "bare-model", provider: "legacy" };
    const { view } = await open(connection);
    vitest
      .expect(view.getByLabelText("Model"))
      .toHaveProperty("value", "bare-model");
    vitest.expect(view.queryByLabelText("Provider")).toBeNull();
    vitest.expect(view.queryByLabelText("Legacy provider")).toBeNull();
    vitest.expect(view.queryByRole("button", { name: /provider/i })).toBeNull();
    vitest.expect(connection.write).not.toHaveBeenCalled();
    vitest.expect(connection.reset).not.toHaveBeenCalled();
  },
);

vitest.test(
  "supports permanent numeric profile edits and field resets",
  async () => {
    const { connection, view, user } = await open();
    await user.selectOptions(
      view.getByLabelText("Edit settings for"),
      "profile:custom.name",
    );
    await user.type(view.getByLabelText("Maximum output tokens"), "2048");
    await user.click(
      testing
        .within(
          view.getByRole("group", {
            name: "Maximum output tokens override lifetime",
          }),
        )
        .getByRole("radio", { name: "Workspace" }),
    );
    connection.value = {
      profiles: {
        "custom.name": {
          model: "router/vendor/custom",
          max_output_tokens: 2048,
        },
      },
    };
    await user.click(
      view.getByRole("button", { name: "Apply maximum output tokens" }),
    );
    await view.findByText(
      "Maximum output tokens applied as a permanent override.",
    );
    vitest.expect(connection.write).toHaveBeenCalledExactlyOnceWith({
      name: "components.llm.profiles.custom~1name.max_output_tokens",
      value: 2048,
      permanent: true,
    });
    connection.value = {
      profiles: {
        "custom.name": { model: "openai/toml-model", max_output_tokens: 2048 },
      },
    };
    await user.click(view.getByRole("button", { name: "Reset model to TOML" }));
    await view.findByText("Model reset to startup TOML.");
    vitest.expect(connection.reset).toHaveBeenCalledExactlyOnceWith({
      name: "components.llm.profiles.custom~1name.model",
    });
    vitest.expect(connection.read).toHaveBeenCalledTimes(3);
    vitest
      .expect(view.getByLabelText("Model"))
      .toHaveProperty("value", "openai/toml-model");
  },
);

vitest.test(
  "suggestions remain advisory and failures do not block editing",
  async () => {
    const { connection, view, user } = await open();
    await testing.waitFor(() =>
      vitest
        .expect(
          view.container.querySelector('option[value="openai/catalog-model"]'),
        )
        .not.toBeNull(),
    );
    vitest
      .expect(
        view.container.querySelector('option[value="other/catalog-model"]'),
      )
      .not.toBeNull();
    vitest
      .expect(
        view.container.querySelector(
          'option[value="router/vendor/nested:tag"]',
        ),
      )
      .not.toBeNull();
    connection.suggestions.mockRejectedValue(new Error("unknown_method"));
    await user.clear(view.getByLabelText("Model"));
    await user.type(view.getByLabelText("Model"), "router/vendor/custom:tag");
    await testing
      .within(view.getByRole("form", { name: "Edit Model" }))
      .findByText("Suggestions unavailable.");
    await user.click(view.getByRole("button", { name: "Apply model" }));
    await view.findByText("Model applied as a temporary override.");
    vitest.expect(connection.write).toHaveBeenCalledWith({
      name: "components.llm.model",
      value: "router/vendor/custom:tag",
      permanent: false,
    });
  },
);

vitest.test.each([
  { label: "Maximum output tokens", invalid: "0", valid: "1" },
  { label: "Maximum output tokens", invalid: "-1", valid: "1" },
  { label: "Maximum output tokens", invalid: "1.5", valid: "1" },
  { label: "Temperature", invalid: "-0.1", valid: "0" },
  { label: "Temperature", invalid: "2.1", valid: "2" },
])(
  "rejects invalid $label input: $invalid",
  async ({ label, invalid, valid }) => {
    const { connection, view, user } = await open();
    const input = view.getByLabelText(label);
    const apply = view.getByRole("button", {
      name: `Apply ${label.toLowerCase()}`,
    });
    await user.clear(input);
    await user.type(input, invalid);
    await user.click(apply);
    vitest.expect(connection.write).not.toHaveBeenCalled();
    await user.clear(input);
    await user.type(input, valid);
    await user.click(apply);
    await view.findByText(`${label} applied as a temporary override.`);
    vitest.expect(connection.write).toHaveBeenCalledExactlyOnceWith({
      name:
        label === "Temperature"
          ? "components.llm.temperature"
          : "components.llm.max_output_tokens",
      value: Number(valid),
      permanent: false,
    });
  },
);

vitest.test(
  "late suggestions do not replace suggestions for the current draft",
  async () => {
    const connection = new Connection();
    let resolve: (value: { values: Json[] }) => void = () => {};
    const pending = new Promise<{ values: Json[] }>((done) => {
      resolve = done;
    });
    connection.suggestions.mockImplementation(async (params) => {
      if (
        params.name === "components.llm.model" &&
        params.prefix === "openai/base"
      )
        return pending;
      return { values: ["latest"] };
    });
    const { view, user } = await open(connection);
    await testing.waitFor(() =>
      vitest.expect(connection.suggestions).toHaveBeenCalledWith({
        name: "components.llm.model",
        prefix: "openai/base",
      }),
    );
    await user.clear(view.getByLabelText("Model"));
    await user.type(view.getByLabelText("Model"), "new");
    const form = view.getByRole("form", { name: "Edit Model" });
    await testing.waitFor(() =>
      vitest
        .expect(form.querySelector('option[value="latest"]'))
        .not.toBeNull(),
    );
    await testing.act(async () => resolve({ values: ["outdated"] }));
    vitest.expect(form.querySelector('option[value="outdated"]')).toBeNull();
    vitest.expect(view.getByLabelText("Model")).toHaveProperty("value", "new");
  },
);

vitest.test(
  "reports load and write failures with refresh available",
  async () => {
    const connection = new Connection();
    connection.read.mockRejectedValueOnce(new Error("disconnected"));
    const user = userEvent.setup();
    const view = testing.render(
      <Settings
        api={new RunnerApi(connection)}
        connected
        connection={{ endpoint: "ws://127.0.0.1:1234", token: "test" }}
      />,
    );
    await user.click(view.getByRole("button", { name: "LLM profiles" }));
    vitest
      .expect((await view.findByRole("alert")).textContent)
      .toContain("disconnected");
    await user.click(view.getByRole("button", { name: "Refresh settings" }));
    await view.findByRole("form", { name: "Edit Model" });
    connection.write.mockRejectedValueOnce(new Error("configuration_error"));
    await user.click(view.getByRole("button", { name: "Apply model" }));
    vitest
      .expect((await view.findByRole("alert")).textContent)
      .toContain("could not be confirmed");
    vitest
      .expect(view.getByRole("alert").textContent)
      .toContain("configuration_error");
    vitest.expect(connection.read).toHaveBeenCalledTimes(2);
  },
);

vitest.test(
  "distinguishes an applied change from a failed reread",
  async () => {
    const { connection, view, user } = await open();
    connection.read.mockRejectedValueOnce(new Error("connection lost"));
    await user.click(view.getByRole("button", { name: "Reset model to TOML" }));
    vitest
      .expect((await view.findByRole("alert")).textContent)
      .toContain("Change applied, but reloading failed.");
    vitest.expect(connection.reset).toHaveBeenCalledTimes(1);
  },
);

vitest.test.each([null, "malformed", { profiles: [] }])(
  "malformed raw configuration stays inspectable: %j",
  async (value) => {
    const connection = new Connection();
    connection.value = value;
    const { view } = await open(connection);
    vitest.expect(view.getByLabelText("Model")).toHaveProperty("value", "");
    vitest
      .expect(view.getByRole("button", { name: "Reset model to TOML" }))
      .toBeTruthy();
    if (value === null || typeof value === "string")
      vitest
        .expect(view.getByRole("alert").textContent)
        .toBe("Invalid settings object.");
    else
      vitest
        .expect(view.getByRole("alert").textContent)
        .toBe("Invalid profile catalog.");
  },
);

vitest.test(
  "missing configuration offers shared fields without implicit profiles",
  async () => {
    const connection = new Connection();
    connection.value = null;
    connection.exists = false;
    const { view } = await open(connection);
    vitest.expect(view.getByText("No configured LLM settings.")).toBeTruthy();
    vitest.expect(view.queryByRole("alert")).toBeNull();
    vitest
      .expect(
        view.getByLabelText("Edit settings for").querySelectorAll("option"),
      )
      .toHaveLength(1);
  },
);
