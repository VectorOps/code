import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, onTestFinished, test, vi } from "vitest";
import { RunnerConnection } from "../src/communication/websocket";
import { ProjectLlmSetup } from "../src/components/ProjectLlmSetup";
import type { Json } from "../src/models/json";

function host(value: Json = null) {
  const connection = new RunnerConnection("ws://localhost", "test");
  const request = vi
    .spyOn(connection, "request")
    .mockImplementation(async (method, params) => {
      if (method === "configuration.get")
        return { name: params?.name, exists: value !== null, value };
      if (method === "configuration.set") return null;
      throw new Error(`Unexpected request: ${method}`);
    });
  const writes = () =>
    request.mock.calls.filter(([method]) => method === "configuration.set");
  return { connection, request, writes };
}

const completeModels = {
  model: "example/default",
  profiles: {
    strong: { model: "example/strong" },
    medium: { model: "example/medium" },
    weak: { model: "example/weak" },
  },
};

beforeEach(() => {
  const prototype = HTMLDialogElement.prototype;
  for (const [name, open] of [
    ["showModal", true],
    ["close", false],
  ] as const) {
    const original = Object.getOwnPropertyDescriptor(prototype, name);
    Object.defineProperty(prototype, name, {
      configurable: true,
      value(this: HTMLDialogElement) {
        this.open = open;
      },
    });
    onTestFinished(() => {
      if (original) Object.defineProperty(prototype, name, original);
      else Reflect.deleteProperty(prototype, name);
    });
  }
});

test("prompts on connection and saves independent permanent model overrides", async () => {
  const { connection, request, writes } = host({ temperature: 0.2 });
  const view = testing.render(
    <ProjectLlmSetup connection={connection} connected={false} />,
  );
  expect(request).not.toHaveBeenCalled();
  view.rerender(<ProjectLlmSetup connection={connection} connected />);

  const dialog = testing.within(
    await view.findByRole("dialog", { name: "Set up LLM models" }),
  );
  expect(request).toHaveBeenCalledExactlyOnceWith("configuration.get", {
    name: "components.llm",
  });
  const user = userEvent.setup();
  const save = dialog.getByRole("button", { name: "Save models" });
  expect(save).toHaveProperty("disabled", true);
  await user.type(
    dialog.getByRole("textbox", { name: "Default model" }),
    "  example/default  ",
  );
  await user.type(
    dialog.getByRole("textbox", { name: "Strong model" }),
    "example/strong",
  );
  await user.type(
    dialog.getByRole("textbox", { name: "Weak model" }),
    "example/weak",
  );
  expect(dialog.getByRole("textbox", { name: "Medium model" })).toHaveProperty(
    "value",
    "",
  );
  await user.click(save);

  expect(view.queryByRole("dialog")).toBeNull();
  expect(writes()).toEqual([
    [
      "configuration.set",
      {
        name: "components.llm.model",
        value: "example/default",
        permanent: true,
      },
    ],
    [
      "configuration.set",
      {
        name: "components.llm.profiles.strong.model",
        value: "example/strong",
        permanent: true,
      },
    ],
    [
      "configuration.set",
      {
        name: "components.llm.profiles.medium.model",
        value: "example/default",
        permanent: true,
      },
    ],
    [
      "configuration.set",
      {
        name: "components.llm.profiles.weak.model",
        value: "example/weak",
        permanent: true,
      },
    ],
  ]);
});

test("does not prompt when all four models are configured", async () => {
  const { connection, request, writes } = host(completeModels);
  const view = testing.render(
    <ProjectLlmSetup connection={connection} connected />,
  );
  await testing.act(async () => {});
  expect(request).toHaveBeenCalledOnce();
  expect(view.queryByRole("dialog")).toBeNull();
  expect(writes()).toEqual([]);
});

test("shared-only settings prompt and create every named profile", async () => {
  const { connection, writes } = host({ model: "example/shared" });
  const view = testing.render(
    <ProjectLlmSetup connection={connection} connected />,
  );
  await view.findByRole("dialog");
  expect(view.getByRole("textbox", { name: "Default model" })).toHaveProperty(
    "value",
    "example/shared",
  );
  await userEvent
    .setup()
    .click(view.getByRole("button", { name: "Save models" }));
  expect(writes().map(([, params]) => params)).toEqual(
    [
      "model",
      "profiles.strong.model",
      "profiles.medium.model",
      "profiles.weak.model",
    ].map((field) => ({
      name: `components.llm.${field}`,
      value: "example/shared",
      permanent: true,
    })),
  );
  expect(view.queryByRole("dialog")).toBeNull();
});

test.each(["default", "strong", "medium", "weak"])(
  "prompts when only the %s model is missing and prefills the rest",
  async (missing) => {
    const value = structuredClone(completeModels);
    if (missing === "default") value.model = "";
    else value.profiles[missing as keyof typeof value.profiles].model = "  ";
    const { connection } = host(value);
    const view = testing.render(
      <ProjectLlmSetup connection={connection} connected />,
    );
    await view.findByRole("dialog");
    for (const name of ["default", "strong", "medium", "weak"]) {
      expect(
        view.getByRole("textbox", { name: new RegExp(`^${name} model$`, "i") }),
      ).toHaveProperty(
        "value",
        name === missing ? (name === "default" ? "" : "  ") : `example/${name}`,
      );
    }
  },
);

test("ignores a stale settings response from another project", async () => {
  const previous = host();
  const current = host(completeModels);
  let resolve: (value: unknown) => void = () => {};
  previous.request.mockImplementationOnce(
    () =>
      new Promise((done) => {
        resolve = done;
      }),
  );
  const view = testing.render(
    <ProjectLlmSetup connection={previous.connection} connected />,
  );
  view.rerender(<ProjectLlmSetup connection={current.connection} connected />);
  await testing.act(async () => {
    resolve({ name: "components.llm", exists: false, value: null });
  });
  expect(view.queryByRole("dialog")).toBeNull();
});

test("Not now does not write or re-open on reconnect, but a different project is checked", async () => {
  const previous = host();
  const next = host();
  const view = testing.render(
    <ProjectLlmSetup connection={previous.connection} connected />,
  );
  await view.findByRole("dialog");
  await userEvent.setup().click(view.getByRole("button", { name: "Not now" }));
  view.rerender(
    <ProjectLlmSetup connection={previous.connection} connected={false} />,
  );
  view.rerender(<ProjectLlmSetup connection={previous.connection} connected />);
  await testing.act(async () => {});
  expect(view.queryByRole("dialog")).toBeNull();
  expect(previous.request).toHaveBeenCalledOnce();
  expect(previous.writes()).toEqual([]);

  view.rerender(<ProjectLlmSetup connection={next.connection} connected />);
  await view.findByRole("dialog");
});

test("keeps the draft open and lets the user retry a failed save", async () => {
  const { connection, request, writes } = host();
  const view = testing.render(
    <ProjectLlmSetup connection={connection} connected />,
  );
  await view.findByRole("dialog");
  request.mockRejectedValueOnce(new Error("Failed to persist"));
  const user = userEvent.setup();
  const model = view.getByRole("textbox", { name: "Default model" });
  await user.type(model, "example/default");
  await user.click(view.getByRole("button", { name: "Save models" }));
  await view.findByRole("alert");
  expect(model).toHaveProperty("value", "example/default");
  await user.click(view.getByRole("button", { name: "Save models" }));
  expect(view.queryByRole("dialog")).toBeNull();
  expect(writes()).toHaveLength(5);
});

test("does not assume missing configuration when the check fails", async () => {
  const { connection, request } = host();
  request.mockRejectedValueOnce(new Error("Connection lost"));
  const view = testing.render(
    <ProjectLlmSetup connection={connection} connected />,
  );
  await testing.act(async () => {});
  expect(view.queryByRole("dialog")).toBeNull();
});
