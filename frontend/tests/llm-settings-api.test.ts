import assert from "node:assert/strict";
import { test } from "vitest";
import { LlmSettingsApi } from "../src/communication/llm-settings-api";
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

test("LLM settings use existing configuration RPCs with explicit lifetimes", async () => {
  const connection = new Connection();
  const snapshot = {
    name: "components.llm",
    exists: true,
    value: { model: "openai/base" },
  };
  connection.results = [
    snapshot,
    null,
    null,
    null,
    { values: ["high", null, 1] },
  ];
  const api = new LlmSettingsApi(connection);
  assert.deepEqual(await api.get(), snapshot);
  await api.set(null, "model", "router/vendor/custom:tag", false);
  await api.set("strong", "temperature", 0, true);
  await api.reset("strong", "model");
  assert.deepEqual(await api.suggest("strong", "reasoning.effort", "hi"), [
    "high",
  ]);
  assert.deepEqual(connection.calls, [
    { method: "configuration.get", params: { name: "components.llm" } },
    {
      method: "configuration.set",
      params: {
        name: "components.llm.model",
        value: "router/vendor/custom:tag",
        permanent: false,
      },
    },
    {
      method: "configuration.set",
      params: {
        name: "components.llm.profiles.strong.temperature",
        value: 0,
        permanent: true,
      },
    },
    {
      method: "configuration.reset",
      params: { name: "components.llm.profiles.strong.model" },
    },
    {
      method: "configuration.suggest",
      params: {
        name: "components.llm.profiles.strong.reasoning.effort",
        prefix: "hi",
      },
    },
  ]);
});

test("profile names remain literal path segments", async () => {
  const connection = new Connection();
  const api = new LlmSettingsApi(connection);
  await api.set("custom.~1", "max_output_tokens", 100, true);
  await api.reset("~", "model");
  assert.deepEqual(connection.calls, [
    {
      method: "configuration.set",
      params: {
        name: "components.llm.profiles.custom~1~01.max_output_tokens",
        value: 100,
        permanent: true,
      },
    },
    {
      method: "configuration.reset",
      params: { name: "components.llm.profiles.~0.model" },
    },
  ]);
});

test("qualified model suggestions and updates preserve provider prefixes and nested names", async () => {
  const connection = new Connection();
  const values = ["first/shared", "second/shared", "router/vendor/nested:tag"];
  connection.results = [{ values }, { values: [values[2]] }, null];
  const api = new LlmSettingsApi(connection);
  assert.deepEqual(await api.suggest("strong", "model", ""), values);
  assert.deepEqual(await api.suggest("strong", "model", "router/"), [
    values[2],
  ]);
  await api.set("strong", "model", "router/vendor/nested:tag", true);
  assert.deepEqual(connection.calls, [
    {
      method: "configuration.suggest",
      params: { name: "components.llm.profiles.strong.model", prefix: "" },
    },
    {
      method: "configuration.suggest",
      params: {
        name: "components.llm.profiles.strong.model",
        prefix: "router/",
      },
    },
    {
      method: "configuration.set",
      params: {
        name: "components.llm.profiles.strong.model",
        value: "router/vendor/nested:tag",
        permanent: true,
      },
    },
  ]);
});
