import assert from "node:assert/strict";
import { test } from "node:test";
import { AuthApi } from "../src/communication/auth-api";
import { RunnerConnection } from "../src/communication/websocket";
import type { Params } from "../src/models/json";

class Connection extends RunnerConnection {
  calls: { method: string; params: Params }[] = [];
  results: unknown[] = [];
  constructor() {
    super("ws://127.0.0.1:1234", "test");
  }
  override async request<T>(method: string, params: Params = {}): Promise<T> {
    this.calls.push({ method, params });
    return this.results.shift() as T;
  }
}

test("credential listing paginates within an explicitly selected profile", async () => {
  const connection = new Connection();
  connection.results = [
    { credentials: [{ provider: "chatgpt" }], next_after: "chatgpt" },
    { credentials: [{ provider: "openai" }], next_after: null },
  ];
  const entries = await new AuthApi(connection).credentials("work");
  assert.deepEqual(
    entries.map((entry) => entry.provider),
    ["chatgpt", "openai"],
  );
  assert.deepEqual(connection.calls, [
    {
      method: "auth.credentials.list",
      params: { profile: "work", after: null, limit: 100 },
    },
    {
      method: "auth.credentials.list",
      params: { profile: "work", after: "chatgpt", limit: 100 },
    },
  ]);
});

test("authentication operations preserve runner wire contracts", async () => {
  const connection = new Connection();
  const api = new AuthApi(connection);
  await api.profiles();
  await api.profile("add", "work");
  await api.profile("switch", "work");
  await api.profile("delete", "work");
  await api.provision("openai", "work", "placeholder");
  await api.check("openai", "work");
  await api.remove("openai", "work");
  await api.login("chatgpt", "work");
  await api.logins();
  await api.cancel("login-1");
  assert.deepEqual(connection.calls, [
    { method: "auth.profile.list", params: {} },
    { method: "auth.profile.add", params: { name: "work" } },
    { method: "auth.profile.switch", params: { name: "work" } },
    { method: "auth.profile.delete", params: { name: "work" } },
    {
      method: "auth.provision",
      params: {
        provider: "openai",
        profile: "work",
        kind: "api_key",
        secret: "placeholder",
      },
    },
    {
      method: "auth.credentials.check",
      params: { provider: "openai", profile: "work" },
    },
    {
      method: "auth.credentials.delete",
      params: { provider: "openai", profile: "work" },
    },
    { method: "auth.login", params: { provider: "chatgpt", profile: "work" } },
    { method: "auth.list", params: {} },
    { method: "auth.cancel", params: { id: "login-1" } },
  ]);
});
