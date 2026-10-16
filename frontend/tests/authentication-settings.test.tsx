import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import type { Params } from "../src/models/json";
import { Settings } from "../src/screens/Settings";

class Connection extends RunnerConnection {
  calls: { method: string; params: Params }[] = [];
  profiles = ["default", "work"];
  active = "default";

  constructor() {
    super("ws://127.0.0.1:1234", "test");
  }

  override async request<T>(method: string, params: Params = {}): Promise<T> {
    this.calls.push({ method, params });
    if (method === "auth.profile.add") this.profiles.push(String(params.name));
    if (method === "auth.profile.switch") this.active = String(params.name);
    if (method === "auth.profile.delete")
      this.profiles = this.profiles.filter((name) => name !== params.name);
    if (method.startsWith("auth.profile."))
      return { profiles: [...this.profiles], active_profile: this.active } as T;
    if (method === "auth.providers")
      return {
        providers: [
          {
            provider: "openai",
            display_name: "OpenAI",
            credential_kinds: ["api_key"],
            oauth_login: false,
          },
          {
            provider: "chatgpt",
            display_name: "ChatGPT",
            credential_kinds: ["oauth2"],
            oauth_login: true,
          },
        ],
        custom_api_key_providers: false,
      } as T;
    if (method === "auth.credentials.list")
      return { credentials: [], next_after: null } as T;
    if (method === "auth.list") return { requests: [] } as T;
    if (method === "auth.provision") return null as T;
    throw new Error(`Unexpected method: ${method}`);
  }
}

async function open() {
  const connection = new Connection();
  const user = userEvent.setup();
  const view = testing.render(
    <Settings
      api={new RunnerApi(connection)}
      connected
      connection={{ endpoint: "ws://127.0.0.1:1234", token: "test" }}
    />,
  );
  await user.click(view.getByRole("button", { name: "LLM authentication" }));
  await testing.waitFor(() =>
    vitest
      .expect(view.getByLabelText("Credential profile"))
      .toHaveProperty("value", "default"),
  );
  return { connection, view, user };
}

vitest.test(
  "authentication presents controls and feedback without help blocks",
  async () => {
    const { view } = await open();
    vitest
      .expect(view.getByRole("button", { name: "Set active profile" }))
      .toBeTruthy();
    vitest
      .expect(view.getByRole("button", { name: "Save API key" }))
      .toBeTruthy();
    vitest
      .expect(view.getByRole("button", { name: "Sign in with ChatGPT" }))
      .toBeTruthy();
    vitest.expect(view.getByText("No stored credentials.")).toBeTruthy();
    vitest
      .expect(
        view.queryByText(
          /outside workspace history|selection is shared|Status reflects|Browser authorization/,
        ),
      )
      .toBeNull();
  },
);

vitest.test(
  "credential profile controls still switch, add, and delete profiles",
  async () => {
    const { connection, view, user } = await open();
    await user.selectOptions(view.getByLabelText("Credential profile"), "work");
    await user.click(view.getByRole("button", { name: "Set active profile" }));
    await view.findByText("Active profile changed.");
    vitest.expect(connection.calls).toContainEqual({
      method: "auth.profile.switch",
      params: { name: "work" },
    });
    await user.type(view.getByLabelText("New credential profile"), "personal");
    await user.click(view.getByRole("button", { name: "Add profile" }));
    await view.findByText("Profile added.");
    vitest.expect(connection.calls).toContainEqual({
      method: "auth.profile.add",
      params: { name: "personal" },
    });
    await user.selectOptions(
      view.getByLabelText("Credential profile"),
      "personal",
    );
    const confirm = vitest.vi.spyOn(window, "confirm").mockReturnValue(true);
    await user.click(view.getByRole("button", { name: "Delete profile" }));
    await view.findByText("Profile deleted.");
    vitest
      .expect(confirm)
      .toHaveBeenCalledWith(
        'Delete credential profile "personal" and its stored credentials?',
      );
    vitest.expect(connection.calls).toContainEqual({
      method: "auth.profile.delete",
      params: { name: "personal" },
    });
  },
);

vitest.test(
  "API key form saves credentials and clears the secret",
  async () => {
    const { connection, view, user } = await open();
    await user.selectOptions(view.getByLabelText("Provider"), "openai");
    await user.type(view.getByLabelText("API key"), "placeholder");
    await user.click(view.getByRole("button", { name: "Save API key" }));
    await view.findByText("API key saved.");
    vitest.expect(connection.calls).toContainEqual({
      method: "auth.provision",
      params: {
        provider: "openai",
        profile: "default",
        kind: "api_key",
        secret: "placeholder",
      },
    });
    vitest.expect(view.getByLabelText("API key")).toHaveProperty("value", "");
  },
);
