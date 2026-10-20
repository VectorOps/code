import { useState } from "react";
import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { RunnerScreen } from "../src/screens/RunnerScreen";
import * as runnerHook from "../src/hooks/useRunner";
import { resolveDefaultWorkflow } from "../src/hooks/useDefaultWorkflow";
import {
  NewSessionForm,
  type NewSessionDraft,
} from "../src/components/NewSessionForm";
import type { Workflow } from "../src/models/runner";
import type { Params } from "../src/models/json";

const workflows: Workflow[] = [
  {
    descriptor: {
      reference: "builtin:agent",
      name: "Agent",
      description: null,
      input_schema: { type: "object", properties: {} },
    },
  },
  {
    descriptor: {
      reference: "custom:review",
      name: "Review",
      description: "Review changes",
      input_schema: {
        type: "object",
        properties: { rounds: { type: "integer", default: 2 } },
      },
    },
  },
];

vitest.beforeEach(() => {
  localStorage.clear();
  vitest.vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
});
vitest.afterEach(() => vitest.vi.unstubAllGlobals());

function mockRunner() {
  const runner: ReturnType<typeof runnerHook.useRunner> = {
    api: undefined,
    settings: { endpoint: "ws://localhost", token: "" },
    connect: vitest.vi.fn(),
    status: "Connected",
    error: "",
    dismiss: vitest.vi.fn(),
    workflows,
    sessions: [],
    hasMoreSessions: false,
    sessionsLoading: false,
    loadMoreSessions: vitest.vi.fn(async () => false),
    activitySessions: new Set(),
    selected: "",
    select: vitest.vi.fn(),
    history: { items: [], hasOlder: false, loadingOlder: false },
    followHistory: vitest.vi.fn(),
    trimHistory: vitest.vi.fn(),
    loadOlderHistory: vitest.vi.fn(async () => {}),
    execution: undefined,
    branching: undefined,
    pending: [],
    steering: undefined,
    busy: false,
    enabled: true,
    start: vitest.vi.fn(async () => true),
    resume: vitest.vi.fn(async () => true),
    stop: vitest.vi.fn(async () => true),
    reply: vitest.vi.fn(async () => true),
    steer: vitest.vi.fn(async () => true),
    branch: vitest.vi.fn(async () => true),
  };
  vitest.vi.spyOn(runnerHook, "useRunner").mockImplementation(() => runner);
  return runner;
}

async function shortcut(options: KeyboardEventInit) {
  const event = new KeyboardEvent("keydown", {
    key: "n",
    bubbles: true,
    cancelable: true,
    ...options,
  });
  await testing.act(async () => {
    (document.activeElement ?? document.body).dispatchEvent(event);
  });
  return event;
}

vitest.test.each(["ctrlKey", "metaKey"])(
  "%s+N starts agent and Shift opens advanced without starting",
  async (modifier) => {
    const runner = mockRunner();
    const view = testing.render(<RunnerScreen initial={runner.settings} />);
    const event = await shortcut({ [modifier]: true });
    vitest.expect(event.defaultPrevented).toBe(true);
    vitest
      .expect(runner.start)
      .toHaveBeenCalledExactlyOnceWith("builtin:agent", {});
    vitest
      .expect(
        view.queryByRole("heading", { name: "Advanced new conversation" }),
      )
      .toBeNull();
    const repeat = await shortcut({ [modifier]: true, repeat: true });
    vitest.expect(repeat.defaultPrevented).toBe(true);
    await shortcut({ [modifier]: true, altKey: true });
    await shortcut({ [modifier]: true, isComposing: true });
    vitest.expect(runner.start).toHaveBeenCalledOnce();
    await shortcut({ [modifier]: true, shiftKey: true, key: "N" });
    vitest
      .expect(view.getByRole("heading", { name: "Advanced new conversation" }))
      .toBeDefined();
    vitest.expect(runner.start).toHaveBeenCalledOnce();
  },
);

vitest.test(
  "default workflow setting is saved and used by the main button after remount",
  async () => {
    const runner = mockRunner();
    const user = userEvent.setup();
    const view = testing.render(<RunnerScreen initial={runner.settings} />);
    await user.click(view.getByRole("button", { name: "Settings" }));
    await user.click(view.getByRole("button", { name: "Conversations" }));
    vitest
      .expect(view.getByLabelText("Default workflow"))
      .toHaveProperty("value", "agent");
    await user.selectOptions(
      view.getByLabelText("Default workflow"),
      "custom:review",
    );
    vitest
      .expect(localStorage.getItem("runner.defaultWorkflow"))
      .toBe("custom:review");
    view.unmount();
    const next = testing.render(<RunnerScreen initial={runner.settings} />);
    await user.click(
      next.getAllByRole("button", { name: "New conversation" })[0],
    );
    vitest
      .expect(runner.start)
      .toHaveBeenCalledExactlyOnceWith("custom:review", { rounds: 2 });
  },
);

vitest.test(
  "missing defaults report an error rather than silently starting another workflow",
  async () => {
    localStorage.setItem("runner.defaultWorkflow", "missing:workflow");
    const runner = mockRunner();
    const view = testing.render(<RunnerScreen initial={runner.settings} />);
    await shortcut({ metaKey: true });
    vitest
      .expect(view.getByRole("alert").textContent)
      .toContain("is unavailable");
    vitest.expect(runner.start).not.toHaveBeenCalled();
    await userEvent
      .setup()
      .click(view.getByRole("button", { name: "Advanced new conversation" }));
    vitest.expect(view.getByLabelText("Workflow")).toBeDefined();
  },
);

vitest.test(
  "advanced drafts survive closing and a successful custom launch selects the conversation",
  async () => {
    const runner = mockRunner();
    const user = userEvent.setup();
    const view = testing.render(<RunnerScreen initial={runner.settings} />);
    await shortcut({ metaKey: true, shiftKey: true });
    await user.selectOptions(view.getByLabelText("Workflow"), "custom:review");
    const rounds = view.getByLabelText(/rounds/);
    await user.clear(rounds);
    await user.type(rounds, "4");
    await user.click(view.getByRole("button", { name: "Close inspector" }));
    await shortcut({ ctrlKey: true, shiftKey: true });
    vitest.expect(view.getByLabelText(/rounds/)).toHaveProperty("value", "4");
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest
      .expect(runner.start)
      .toHaveBeenCalledExactlyOnceWith("custom:review", { rounds: 4 });
    runner.selected = "new-execution";
    view.rerender(<RunnerScreen initial={runner.settings} />);
    vitest
      .expect(
        view.queryByRole("heading", { name: "Advanced new conversation" }),
      )
      .toBeNull();
  },
);

vitest.test(
  "duplicate starts are guarded while a request is in flight",
  async () => {
    const runner = mockRunner();
    let finish!: (value: boolean) => void;
    vitest.vi.mocked(runner.start).mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    testing.render(<RunnerScreen initial={runner.settings} />);
    await shortcut({ ctrlKey: true });
    await shortcut({ metaKey: true });
    vitest.expect(runner.start).toHaveBeenCalledOnce();
    await testing.act(async () => finish(true));
    await shortcut({ ctrlKey: true });
    vitest.expect(runner.start).toHaveBeenCalledTimes(2);
    await testing.act(async () => finish(true));
  },
);

vitest.test(
  "disconnected shortcuts do not launch, but advanced is available in the collapsed rail",
  async () => {
    const runner = mockRunner();
    runner.enabled = false;
    runner.status = "Disconnected";
    const view = testing.render(<RunnerScreen initial={runner.settings} />);
    await shortcut({ ctrlKey: true });
    vitest.expect(runner.start).not.toHaveBeenCalled();
    const user = userEvent.setup();
    await user.click(view.getByRole("button", { name: "Collapse navigation" }));
    vitest
      .expect(
        testing
          .within(
            view.getByRole("complementary", { name: "Session navigation" }),
          )
          .getByRole("button", { name: "New conversation" }),
      )
      .toHaveProperty("disabled", true);
    await user.click(
      view.getByRole("button", { name: "Advanced new conversation" }),
    );
    vitest
      .expect(view.getByRole("heading", { name: "Advanced new conversation" }))
      .toBeDefined();
    vitest
      .expect(view.getByRole("button", { name: "Start conversation" }))
      .toHaveProperty("disabled", true);
  },
);

vitest.test(
  "agent resolution prefers references and never guesses an ambiguous name",
  () => {
    vitest
      .expect(resolveDefaultWorkflow(workflows, "agent"))
      .toBe(workflows[0]);
    vitest
      .expect(resolveDefaultWorkflow(workflows, "custom:review"))
      .toBe(workflows[1]);
    vitest
      .expect(
        resolveDefaultWorkflow(
          [
            workflows[0],
            { descriptor: { ...workflows[0].descriptor, reference: "agent" } },
          ],
          "agent",
        )?.descriptor.reference,
      )
      .toBe("agent");
    vitest
      .expect(
        resolveDefaultWorkflow(
          [
            {
              descriptor: {
                ...workflows[0].descriptor,
                reference: "custom:agent",
              },
            },
          ],
          "agent",
        )?.descriptor.reference,
      )
      .toBe("custom:agent");
    vitest
      .expect(
        resolveDefaultWorkflow(
          ["a", "b"].map((reference) => ({
            descriptor: { ...workflows[0].descriptor, reference },
          })),
          "agent",
        ),
      )
      .toBeUndefined();
  },
);

const formWorkflow: Workflow = {
  descriptor: {
    reference: "custom:form",
    name: "Form workflow",
    description: "Configure a task",
    input_schema: {
      type: "object",
      required: ["task"],
      properties: {
        task: {
          type: "string",
          title: "Task",
          description: "What should happen?",
        },
        retries: { type: "integer", title: "Retries", default: 3 },
        enabled: { type: "boolean", title: "Enabled", default: false },
        mode: {
          type: "string",
          title: "Mode",
          enum: ["fast", "thorough"],
          default: "fast",
        },
      },
    },
  },
};

function FormHarness({
  start,
  available = [formWorkflow],
}: {
  start: (reference: string, input: Params) => Promise<boolean>;
  available?: Workflow[];
}) {
  const [value, change] = useState<NewSessionDraft>({
    reference: "",
    input: "{}",
  });
  return (
    <NewSessionForm
      workflows={available}
      enabled
      value={value}
      change={change}
      start={start}
    />
  );
}

vitest.test(
  "advanced inputs use labeled typed fields, defaults, descriptions, and enum choices",
  async () => {
    const start = vitest.vi.fn(async () => true);
    const view = testing.render(<FormHarness start={start} />);
    const user = userEvent.setup();
    await user.selectOptions(view.getByLabelText("Workflow"), "custom:form");
    vitest.expect(view.queryByLabelText("Workflow input (JSON)")).toBeNull();
    vitest.expect(view.getByLabelText(/Retries/)).toHaveProperty("value", "3");
    vitest
      .expect(view.getByLabelText(/Enabled/))
      .toHaveProperty("value", "false");
    vitest.expect(view.getByText("What should happen?")).toBeDefined();
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest.expect(start).not.toHaveBeenCalled();
    await user.type(view.getByLabelText(/Task/), "Review changes");
    await user.selectOptions(view.getByLabelText(/Mode/), '"thorough"');
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest.expect(start).toHaveBeenCalledExactlyOnceWith("custom:form", {
      task: "Review changes",
      retries: 3,
      enabled: false,
      mode: "thorough",
    });
  },
);

vitest.test(
  "JSON and form mode round-trip inputs without dropping extra properties",
  async () => {
    const start = vitest.vi.fn(async () => true);
    const view = testing.render(<FormHarness start={start} />);
    const user = userEvent.setup();
    await user.selectOptions(view.getByLabelText("Workflow"), "custom:form");
    await user.click(view.getByRole("button", { name: "Edit as JSON" }));
    const raw = view.getByLabelText("Workflow input (JSON)");
    testing.fireEvent.change(raw, {
      target: {
        value: JSON.stringify({ task: "Saved", extra: { keep: true } }),
      },
    });
    await user.click(view.getByRole("button", { name: "Use form fields" }));
    await user.type(view.getByLabelText(/Task/), " task");
    await user.click(view.getByRole("button", { name: "Edit as JSON" }));
    vitest
      .expect(
        JSON.parse(
          (view.getByLabelText("Workflow input (JSON)") as HTMLTextAreaElement)
            .value,
        ),
      )
      .toEqual({ task: "Saved task", extra: { keep: true } });
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest.expect(start).toHaveBeenCalledWith("custom:form", {
      task: "Saved task",
      extra: { keep: true },
    });
  },
);

vitest.test(
  "complex schemas retain JSON editing and invalid objects are rejected locally",
  async () => {
    const start = vitest.vi.fn(async () => true);
    const complex: Workflow = {
      descriptor: {
        ...formWorkflow.descriptor,
        input_schema: {
          type: "object",
          properties: { config: { type: "object", default: { keep: true } } },
        },
      },
    };
    const view = testing.render(
      <FormHarness start={start} available={[complex]} />,
    );
    const user = userEvent.setup();
    vitest.expect(view.queryByLabelText("Workflow input (JSON)")).toBeNull();
    await user.selectOptions(view.getByLabelText("Workflow"), "custom:form");
    vitest
      .expect(view.queryByRole("button", { name: "Use form fields" }))
      .toBeNull();
    const raw = view.getByLabelText("Workflow input (JSON)");
    vitest
      .expect(JSON.parse((raw as HTMLTextAreaElement).value))
      .toEqual({ config: { keep: true } });
    testing.fireEvent.change(raw, { target: { value: "[]" } });
    await user.click(view.getByRole("button", { name: "Start conversation" }));
    vitest
      .expect(view.getByRole("alert").textContent)
      .toContain("must be a JSON object");
    vitest.expect(start).not.toHaveBeenCalled();
    await user.selectOptions(view.getByLabelText("Workflow"), "");
    vitest.expect(view.queryByLabelText("Workflow input (JSON)")).toBeNull();
  },
);
