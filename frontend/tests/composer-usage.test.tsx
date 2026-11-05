import { render } from "@testing-library/react";
import * as vitest from "vitest";
import { Workspace } from "../src/screens/Workspace";
import type { HistoryItem, Session } from "../src/models/runner";

const { expect, test } = vitest;

vitest.beforeEach(() => {
  vitest.vi.stubGlobal(
    "ResizeObserver",
    class implements ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
});

vitest.afterEach(() => vitest.vi.unstubAllGlobals());

const session: Session = {
  execution_id: "execution",
  root_workflow_id: "root",
  reference: "workflow",
  status: "running",
  active: true,
  pending_inputs: 0,
  usage: {
    totals: {
      input_tokens: 999_999,
      output_tokens: 100_000,
      cost_estimate_usd: "1.234567",
      estimated_calls: 1,
      incomplete_calls: 0,
    },
    root_context: {
      step_id: "root-turn",
      input_tokens_estimate: 12_345,
      context_window: 200_000,
    },
  },
};

const history: HistoryItem[] = [
  {
    position: 1,
    step: {
      id: "root-turn",
      workflow_id: "root",
      operation: { kind: "llm.generate" },
      metrics: { llm: { catalog_model: "main-model" } },
    },
  },
  {
    position: 2,
    step: {
      id: "child-turn",
      workflow_id: "child",
      operation: { kind: "llm.generate" },
      metrics: { llm: { catalog_model: "child-model" } },
    },
  },
];

function workspace(current: Session, items = history) {
  return (
    <Workspace
      sessions={[current]}
      selected={current.execution_id}
      onSelect={() => {}}
      history={items}
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
    />
  );
}

test("usage shares the attachment row and uses root context, not cumulative or child tokens", () => {
  const view = render(workspace(session));
  const usage = view.getByLabelText("Workflow usage");
  expect(usage.textContent).toBe("main-model 12.3k / 200k (6%) ~$1.23");
  expect(usage.parentElement).toBe(
    view.getByRole("button", { name: "Attach images" }).parentElement,
  );
  expect(usage.parentElement?.lastElementChild).toBe(
    view.getByRole("button", { name: "Stop workflow" }),
  );
  const help = view.getByText("Draft freely while the workflow runs.");
  expect(help.className).toBe("composer-help");
  expect(help.nextElementSibling).toBe(usage);
  expect(usage.nextElementSibling).toBe(usage.parentElement?.lastElementChild);
  expect(usage.title).toContain(
    "including child workflows and retained branches",
  );
});

test.each([
  {
    requested: "provider/main-model:fast",
    expected: "provider/main-model:fast",
  },
  { requested: "", expected: "main-model" },
  { requested: null, expected: "main-model" },
  { requested: 42, expected: "main-model" },
])(
  "model display prefers a valid requested reference: $requested",
  ({ requested, expected }) => {
    const view = render(
      workspace(session, [
        {
          position: 1,
          step: {
            id: "root-turn",
            workflow_id: "root",
            operation: { kind: "llm.generate" },
            metrics: {
              llm: { requested_model: requested, catalog_model: "main-model" },
            },
          },
        },
        history[1],
      ]),
    );
    expect(
      view.getByLabelText("Workflow usage").querySelector(".composer-model")
        ?.textContent,
    ).toBe(expected);
  },
);

test("usage refreshes from durable summaries and indicates incomplete cost coverage", () => {
  const view = render(workspace(session));
  view.rerender(
    workspace(
      {
        ...session,
        usage: {
          totals: {
            ...session.usage!.totals,
            cost_estimate_usd: "2.456",
            incomplete_calls: 1,
          },
          root_context: {
            step_id: "next-root-turn",
            input_tokens_estimate: 450,
            context_window: 128_000,
          },
        },
      },
      [
        ...history,
        {
          position: 3,
          step: {
            id: "next-root-turn",
            workflow_id: "root",
            operation: { kind: "llm.generate" },
            metrics: { llm: { catalog_model: "new-main-model" } },
          },
        },
      ],
    ),
  );
  const usage = view.getByLabelText("Workflow usage");
  expect(usage.textContent).toBe("new-main-model 450 / 128k (0%) ~$2.46");
  expect(usage.title).toContain("cost may be understated");
});

test("missing usage is unknown rather than zero or stale values from a different session", () => {
  const view = render(workspace(session));
  view.rerender(
    workspace({ ...session, execution_id: "other", usage: null }, []),
  );
  expect(view.getByLabelText("Workflow usage").textContent).toBe("— — / — ~$—");
});

test("known zero usage and unknown model or context capacity remain distinct", () => {
  const view = render(
    workspace({
      ...session,
      usage: {
        totals: { ...session.usage!.totals, cost_estimate_usd: "0" },
        root_context: {
          step_id: "missing",
          input_tokens_estimate: 0,
          context_window: null,
        },
      },
    }),
  );
  expect(view.getByLabelText("Workflow usage").textContent).toBe("— 0 / — ~$0");
});

test.each([
  [0, 200_000, "0 / 200k (0%)"],
  [50_000, 200_000, "50k / 200k (25%)"],
  [250_000, 200_000, "250k / 200k (125%)"],
  [100, 0, "100 / 0"],
  [null, 200_000, "— / 200k"],
])("context %s / %s has a safe percentage", (input, limit, expected) => {
  const view = render(
    workspace({
      ...session,
      usage: {
        totals: session.usage!.totals,
        root_context: {
          step_id: "root-turn",
          input_tokens_estimate: input,
          context_window: limit,
        },
      },
    }),
  );
  expect(
    view.getByLabelText("Workflow usage").querySelector(".composer-tokens")
      ?.textContent,
  ).toBe(expected);
});
