import { useState, type ComponentProps } from "react";
import { act, fireEvent, render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { Composer } from "../src/components/Composer";

type Props = ComponentProps<typeof Composer>;

function DraftComposer(props: Props) {
  const [value, change] = useState(props.value);
  return <Composer {...props} value={value} change={change} />;
}

function compose(overrides: Partial<Props> = {}) {
  const send = vitest.vi.fn(async () => {});
  const props: Props = {
    selected: true,
    pendingCount: 0,
    ready: true,
    enabled: true,
    busy: false,
    value: "Hello",
    change: vitest.vi.fn(),
    send,
    review: vitest.vi.fn(),
    ...overrides,
  };
  return { ...render(<DraftComposer {...props} />), send: props.send };
}

vitest.test.each(["ctrlKey", "metaKey"])(
  "%s + Enter submits an enabled draft",
  async (modifier) => {
    const view = compose();
    await act(async () => {
      fireEvent.keyDown(view.getByRole("textbox"), {
        key: "Enter",
        [modifier]: true,
      });
    });
    vitest.expect(view.send).toHaveBeenCalledOnce();
  },
);

vitest.test("Enter adds a newline instead of sending", async () => {
  const view = compose();
  const user = userEvent.setup();
  const draft = view.getByRole("textbox");
  await user.click(draft);
  await user.keyboard("{End}{Enter}Next line");
  vitest.expect(draft).toHaveProperty("value", "Hello\nNext line");
  vitest.expect(view.send).not.toHaveBeenCalled();
});

vitest.test.each([
  { enabled: false },
  { ready: false },
  { busy: true },
  { selected: false },
  { value: "  " },
])(
  "keyboard and form submission honor unavailable state %j",
  async (overrides) => {
    const view = compose(overrides);
    const draft = view.getByRole("textbox");
    await act(async () => {
      fireEvent.keyDown(draft, { key: "Enter", ctrlKey: true });
      fireEvent.submit(draft.closest("form")!);
    });
    vitest.expect(view.send).not.toHaveBeenCalled();
    vitest
      .expect(view.getByRole("button", { name: /Send/ }))
      .toHaveProperty("disabled", true);
  },
);

vitest.test("IME composition does not submit a draft", async () => {
  const view = compose();
  await act(async () => {
    fireEvent.keyDown(view.getByRole("textbox"), {
      key: "Enter",
      ctrlKey: true,
      isComposing: true,
    });
  });
  vitest.expect(view.send).not.toHaveBeenCalled();
});

vitest.test(
  "rapid keyboard and form submissions send only once while a reply is in flight",
  async () => {
    let finish: () => void = () => {};
    const send = vitest.vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    );
    const view = compose({ send });
    const draft = view.getByRole("textbox");
    fireEvent.keyDown(draft, { key: "Enter", ctrlKey: true });
    fireEvent.keyDown(draft, { key: "Enter", metaKey: true });
    fireEvent.submit(draft.closest("form")!);
    vitest.expect(send).toHaveBeenCalledOnce();
    vitest
      .expect(view.getByRole("button", { name: "Sending..." }))
      .toHaveProperty("disabled", true);
    await userEvent.setup().type(draft, " more");
    await act(async () => finish());
    vitest.expect(draft).toHaveProperty("value", "Hello more");
    vitest
      .expect(view.getByRole("button", { name: "Send" }))
      .toHaveProperty("disabled", false);
  },
);

vitest.test("the Send button uses the same submission path", async () => {
  const view = compose();
  await userEvent.setup().click(view.getByRole("button", { name: "Send" }));
  vitest.expect(view.send).toHaveBeenCalledOnce();
});
