import { useState, type ComponentProps } from "react";
import { act, fireEvent, render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { Composer } from "../src/components/Composer";
import type { ImageAttachment } from "../src/models/chat-input";

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
    stop: vitest.vi.fn(async () => {}),
    resume: vitest.vi.fn(async () => {}),
    review: vitest.vi.fn(),
    ...overrides,
  };
  return {
    ...render(<DraftComposer {...props} />),
    send: props.send,
    stop: props.stop,
    resume: props.resume,
    review: props.review,
  };
}

vitest.test.each(["shiftKey", "altKey", "ctrlKey", "metaKey"])(
  "%s + Enter adds a newline without submitting",
  async (modifier) => {
    const view = compose();
    const input = view.getByRole("textbox") as HTMLTextAreaElement;
    input.setSelectionRange(5, 5);
    await act(async () => {
      fireEvent.keyDown(input, {
        key: "Enter",
        [modifier]: true,
      });
    });
    vitest.expect(view.send).not.toHaveBeenCalled();
    vitest.expect(input.value).toBe("Hello\n");
    vitest.expect(input.selectionStart).toBe(6);
  },
);

vitest.test(
  "shows a steering queue control only when messages are queued",
  async () => {
    const empty = compose();
    vitest
      .expect(empty.queryByRole("button", { name: /queued steering messages/ }))
      .toBeNull();

    const reviewSteering = vitest.vi.fn();
    const queued = compose({ steeringCount: 2, reviewSteering });
    const control = queued.getByRole("button", {
      name: "2 queued steering messages",
    });
    vitest.expect(control.textContent).toBe("Steering · 2");
    await userEvent.setup().click(control);
    vitest.expect(reviewSteering).toHaveBeenCalledOnce();
  },
);

vitest.test(
  "image-only drafts can be submitted, previewed, and removed",
  async () => {
    const removeImage = vitest.vi.fn();
    const image: ImageAttachment = {
      id: "image",
      name: "shot.png",
      state: "ready",
      image: { type: "image", mime_type: "image/png", data: "cG5n" },
    };
    const view = compose({ value: "", images: [image], removeImage });
    vitest
      .expect(view.getByAltText("shot.png").getAttribute("src"))
      .toBe("data:image/png;base64,cG5n");
    await userEvent
      .setup()
      .click(view.getByRole("button", { name: "Send message" }));
    vitest.expect(view.send).toHaveBeenCalledOnce();
    await userEvent
      .setup()
      .click(view.getByRole("button", { name: "Remove shot.png" }));
    vitest.expect(removeImage).toHaveBeenCalledWith("image");
  },
);

vitest.test(
  "picker and clipboard files use the same attachment path without affecting plain-text paste",
  async () => {
    const attach = vitest.vi.fn();
    const view = compose({ value: "", attach });
    const file = new File(["png"], "shot.png", { type: "image/png" });
    const user = userEvent.setup();
    const button = view.getByRole("button", { name: "Attach images" });
    vitest.expect(button.textContent).toBe("");
    vitest.expect(button.querySelector("svg path")).not.toBeNull();
    const pickerClick = vitest.vi.spyOn(
      view.getByLabelText("Choose images"),
      "click",
    );
    await user.click(button);
    vitest.expect(pickerClick).toHaveBeenCalledOnce();
    await user.upload(view.getByLabelText("Choose images"), file);
    vitest.expect(attach).toHaveBeenCalledWith([file]);
    vitest
      .expect(view.getByLabelText("Choose images"))
      .toHaveProperty("value", "");
    const clipboardData = { files: [file], items: [], getData: () => "" };
    const event = new Event("paste", { bubbles: true, cancelable: true });
    Object.defineProperty(event, "clipboardData", { value: clipboardData });
    fireEvent(view.getByRole("textbox"), event);
    vitest.expect(event.defaultPrevented).toBe(true);
    vitest.expect(attach).toHaveBeenCalledTimes(2);
    await user.click(view.getByRole("textbox"));
    await user.paste("Plain text");
    vitest
      .expect(view.getByRole("textbox"))
      .toHaveProperty("value", "Plain text");
    vitest.expect(attach).toHaveBeenCalledTimes(2);
  },
);

vitest.test("clipboard file items work when the files list is empty", () => {
  const attach = vitest.vi.fn();
  const view = compose({ attach });
  const file = new File(["png"], "shot.png", { type: "image/png" });
  fireEvent.paste(view.getByRole("textbox"), {
    clipboardData: {
      files: [],
      items: [{ kind: "file", getAsFile: () => file }],
      getData: () => "",
    },
  });
  vitest.expect(attach).toHaveBeenCalledWith([file]);
});

vitest.test.each(["preparing", "failed"] as const)(
  "%s attachments block sending",
  async (state) => {
    const image: ImageAttachment =
      state === "preparing"
        ? { id: "x", name: "x.png", state }
        : { id: "x", name: "x.png", state, error: "Could not read" };
    const view = compose({ images: [image] });
    await userEvent
      .setup()
      .click(view.getByRole("button", { name: "Send message" }));
    vitest.expect(view.send).not.toHaveBeenCalled();
    vitest
      .expect(view.getByRole("button", { name: "Send message" }))
      .toHaveProperty("disabled", true);
  },
);

vitest.test("Enter submits without adding a newline", async () => {
  const view = compose();
  const user = userEvent.setup();
  const draft = view.getByRole("textbox");
  await user.click(draft);
  await user.keyboard("{End}{Enter}");
  vitest.expect(draft).toHaveProperty("value", "Hello");
  vitest.expect(view.send).toHaveBeenCalledOnce();
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
      fireEvent.keyDown(draft, { key: "Enter" });
      fireEvent.submit(draft.closest("form")!);
    });
    vitest.expect(view.send).not.toHaveBeenCalled();
    vitest.expect(view.getByRole("button")).toHaveProperty("disabled", true);
  },
);

vitest.test("IME composition does not submit a draft", async () => {
  const view = compose();
  await act(async () => {
    fireEvent.keyDown(view.getByRole("textbox"), {
      key: "Enter",
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
    fireEvent.keyDown(draft, { key: "Enter" });
    fireEvent.keyDown(draft, { key: "Enter" });
    fireEvent.submit(draft.closest("form")!);
    vitest.expect(send).toHaveBeenCalledOnce();
    vitest
      .expect(view.getByRole("button", { name: "Sending message" }))
      .toHaveProperty("disabled", true);
    await userEvent.setup().type(draft, " more");
    await act(async () => finish());
    vitest.expect(draft).toHaveProperty("value", "Hello more");
    vitest
      .expect(view.getByRole("button", { name: "Send message" }))
      .toHaveProperty("disabled", false);
  },
);

vitest.test("the play icon uses the same message submission path", async () => {
  const view = compose();
  const button = view.getByRole("button", { name: "Send message" });
  vitest.expect(button.textContent).toBe("");
  vitest.expect(button.querySelector("svg path")).not.toBeNull();
  vitest.expect(button).toHaveProperty("title", "Send message");
  await userEvent.setup().click(button);
  vitest.expect(view.send).toHaveBeenCalledOnce();
});

vitest.test.each(["running", "starting"])(
  "a %s workflow has a stop icon and stopping preserves the draft",
  async (status) => {
    const view = compose({
      ready: false,
      session: { status, active: true, pending_inputs: 0 },
    });
    const button = view.getByRole("button", { name: "Stop workflow" });
    vitest.expect(button.querySelector("svg rect")).not.toBeNull();
    fireEvent.keyDown(view.getByRole("textbox"), {
      key: "Enter",
    });
    fireEvent.submit(view.getByRole("textbox").closest("form")!);
    vitest.expect(view.stop).not.toHaveBeenCalled();
    await userEvent.setup().click(button);
    vitest.expect(view.stop).toHaveBeenCalledOnce();
    vitest.expect(view.send).not.toHaveBeenCalled();
    vitest.expect(view.resume).not.toHaveBeenCalled();
    vitest.expect(view.getByRole("textbox")).toHaveProperty("value", "Hello");
  },
);

vitest.test.each(["paused", "running"])(
  "a waiting message uses play even when the execution is active with status %s",
  async (status) => {
    const view = compose({
      session: { status, active: true, pending_inputs: 1 },
    });
    const button = view.getByRole("button", { name: "Send message" });
    vitest.expect(button.querySelector("svg path")).not.toBeNull();
    await userEvent.setup().click(button);
    vitest.expect(view.send).toHaveBeenCalledOnce();
    vitest.expect(view.stop).not.toHaveBeenCalled();
  },
);

vitest.test.each([
  { value: "" },
  { messageError: "Remove failed images before sending." },
])(
  "the single control stops an active workflow without a sendable draft %j",
  async (overrides) => {
    const view = compose({
      session: { status: "running", active: true, pending_inputs: 0 },
      ...overrides,
    });
    vitest
      .expect(view.queryByRole("button", { name: "Send message" }))
      .toBeNull();
    vitest.expect(view.getAllByRole("button")).toHaveLength(1);
    await userEvent
      .setup()
      .click(view.getByRole("button", { name: "Stop workflow" }));
    vitest.expect(view.stop).toHaveBeenCalledOnce();
    vitest.expect(view.send).not.toHaveBeenCalled();
    vitest
      .expect(view.getByRole("textbox"))
      .toHaveProperty("value", overrides.value ?? "Hello");
  },
);

vitest.test.each([{ enabled: false }, { busy: true }])(
  "the single stop control honors unavailable state %j",
  async (unavailable) => {
    const view = compose({
      session: { status: "running", active: true, pending_inputs: 0 },
      value: "",
      ...unavailable,
    });
    const stop = view.getByRole("button", { name: "Stop workflow" });
    vitest.expect(stop).toHaveProperty("disabled", true);
    await userEvent.setup().click(stop);
    vitest.expect(view.stop).not.toHaveBeenCalled();
  },
);

vitest.test.each(["stop", "send"] as const)(
  "%s disables the single composer action while in flight",
  async (activity) => {
    let finish: () => void = () => {};
    const callback = vitest.vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    );
    const view = compose({
      session: { status: "running", active: true, pending_inputs: 0 },
      value: activity === "stop" ? "" : "Hello",
      [activity]: callback,
    });
    const control = view.getByRole("button");
    fireEvent.click(control);
    fireEvent.click(control);
    fireEvent.submit(view.getByRole("textbox").closest("form")!);
    vitest.expect(callback).toHaveBeenCalledOnce();
    vitest
      .expect(view[activity === "stop" ? "send" : "stop"])
      .not.toHaveBeenCalled();
    vitest.expect(control).toHaveProperty("disabled", true);
    vitest.expect(view.getAllByRole("button")).toHaveLength(1);
    await act(async () => finish());
    vitest.expect(control).toHaveProperty("disabled", false);
    vitest
      .expect(view.getByRole("textbox"))
      .toHaveProperty("value", activity === "stop" ? "" : "Hello");
  },
);

vitest.test(
  "the right-most control switches between stop and send as the draft changes",
  async () => {
    const view = compose({
      value: "",
      attach: vitest.vi.fn(),
      session: { status: "running", active: true, pending_inputs: 0 },
    });
    const control = view.getByRole("button", { name: "Stop workflow" });
    const user = userEvent.setup();
    await user.type(view.getByRole("textbox"), "Steer the workflow");
    vitest
      .expect(view.getByRole("button", { name: "Send message" }))
      .toBe(control);
    vitest
      .expect(view.queryByRole("button", { name: "Stop workflow" }))
      .toBeNull();
    await user.clear(view.getByRole("textbox"));
    vitest
      .expect(view.getByRole("button", { name: "Stop workflow" }))
      .toBe(control);
    vitest.expect(control.parentElement?.lastElementChild).toBe(control);
    await user.click(control);
    vitest.expect(view.stop).toHaveBeenCalledOnce();
    vitest.expect(view.send).not.toHaveBeenCalled();
  },
);

vitest.test.each(["", "Saved instruction"])(
  "a cancelled workflow can start without submitting draft %j",
  async (value) => {
    const view = compose({
      ready: false,
      value,
      session: { status: "cancelled", active: false, pending_inputs: 0 },
    });
    const button = view.getByRole("button", { name: "Start workflow" });
    vitest.expect(button.querySelector("svg path")).not.toBeNull();
    await userEvent.setup().click(button);
    vitest.expect(view.resume).toHaveBeenCalledOnce();
    vitest.expect(view.send).not.toHaveBeenCalled();
    vitest.expect(view.stop).not.toHaveBeenCalled();
    vitest.expect(view.getByRole("textbox")).toHaveProperty("value", value);
  },
);

vitest.test(
  "a paused workflow awaiting a decision offers review instead of stopping",
  async () => {
    const view = compose({
      ready: false,
      value: "",
      pendingCount: 1,
      session: { status: "paused", active: false, pending_inputs: 1 },
    });
    const button = view.getByRole("button", { name: "Review pending input" });
    vitest.expect(button.querySelector("svg path")).not.toBeNull();
    await userEvent.setup().click(button);
    vitest.expect(view.review).toHaveBeenCalledOnce();
    vitest.expect(view.stop).not.toHaveBeenCalled();
    vitest.expect(view.resume).not.toHaveBeenCalled();
  },
);

vitest.test(
  "pending child decisions do not replace stop while the workflow is running",
  () => {
    const view = compose({
      ready: false,
      pendingCount: 1,
      session: { status: "running", active: true, pending_inputs: 1 },
    });
    vitest
      .expect(view.getByRole("button", { name: "Stop workflow" }))
      .toHaveProperty("disabled", false);
  },
);

vitest.test.each([
  { status: "stopping", active: true, pending_inputs: 0 },
  { status: "completed", active: false, pending_inputs: 0 },
  { status: "failed", active: false, pending_inputs: 0 },
  { status: "cancelled", active: false, pending_inputs: 1 },
  { status: "paused", active: false, pending_inputs: 1 },
])(
  "unavailable workflow state %j cannot trigger an action",
  async (session) => {
    const view = compose({ ready: false, session });
    vitest.expect(view.getByRole("button")).toHaveProperty("disabled", true);
    await userEvent.setup().click(view.getByRole("button"));
    vitest.expect(view.send).not.toHaveBeenCalled();
    vitest.expect(view.stop).not.toHaveBeenCalled();
    vitest.expect(view.resume).not.toHaveBeenCalled();
  },
);

vitest.test.each(["stop", "resume"] as const)(
  "%s obeys disconnected and busy guards",
  async (activity) => {
    const session =
      activity === "stop"
        ? { status: "running", active: true, pending_inputs: 0 }
        : { status: "cancelled", active: false, pending_inputs: 0 };
    for (const unavailable of [{ enabled: false }, { busy: true }]) {
      const view = compose({ ready: false, session, ...unavailable });
      await userEvent.setup().click(view.getByRole("button"));
      vitest.expect(view[activity]).not.toHaveBeenCalled();
      vitest.expect(view.getByRole("button")).toHaveProperty("disabled", true);
      view.unmount();
    }
  },
);

vitest.test.each(["stop", "resume"] as const)(
  "rapid %s clicks invoke the action once",
  async (activity) => {
    let finish: () => void = () => {};
    const callback = vitest.vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    );
    const session =
      activity === "stop"
        ? { status: "running", active: true, pending_inputs: 0 }
        : { status: "cancelled", active: false, pending_inputs: 0 };
    const view = compose({ ready: false, session, [activity]: callback });
    fireEvent.click(view.getByRole("button"));
    fireEvent.click(view.getByRole("button"));
    vitest.expect(callback).toHaveBeenCalledOnce();
    vitest.expect(view.getByRole("button")).toHaveProperty("disabled", true);
    vitest
      .expect(view.getByRole("button").getAttribute("aria-busy"))
      .toBe("true");
    await act(async () => finish());
    vitest.expect(view.getByRole("button")).toHaveProperty("disabled", false);
  },
);
