import assert from "node:assert/strict";
import { test } from "node:test";
import { AuthApi } from "../src/communication/auth-api";
import { RunnerConnection } from "../src/communication/websocket";
import type { Params } from "../src/models/json";

test("provider discovery preserves labels and authentication capabilities", async () => {
  const result = {
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
        credential_kinds: ["api_key", "oauth2"],
        oauth_login: true,
      },
    ],
    custom_api_key_providers: true,
  };
  class Connection extends RunnerConnection {
    override async request<T>(method: string, params: Params = {}): Promise<T> {
      assert.equal(method, "auth.providers");
      assert.deepEqual(params, {});
      return result as T;
    }
  }
  const api = new AuthApi(new Connection("ws://127.0.0.1:1234", "test"));
  assert.deepEqual(await api.providers(), result);
  result.providers = [];
  assert.deepEqual((await api.providers()).providers, []);
});
