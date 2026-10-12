import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { ImageThumbnail } from "../src/components/ImageThumbnail";
import { Composer } from "../src/components/Composer";
import { FocusProvider, useFocusRegion } from "../src/focus/FocusProvider";
import type { ImageAttachment } from "../src/models/chat-input";

const image: ImageAttachment = {
  id: "image",
  name: "shot.png",
  state: "ready",
  image: { type: "image", mime_type: "image/png", data: "cG5n" },
};

vitest.beforeEach(() => {
  for (const [name, method] of [
    [
      "showModal",
      function (this: HTMLDialogElement) {
        this.open = true;
        this.querySelector<HTMLButtonElement>("button")?.focus();
      },
    ],
    [
      "close",
      function (this: HTMLDialogElement) {
        this.open = false;
      },
    ],
  ] as const) {
    const original = Object.getOwnPropertyDescriptor(
      HTMLDialogElement.prototype,
      name,
    );
    Object.defineProperty(HTMLDialogElement.prototype, name, {
      configurable: true,
      value: vitest.vi.fn(method),
    });
    vitest.onTestFinished(() => {
      if (original)
        Object.defineProperty(HTMLDialogElement.prototype, name, original);
      else Reflect.deleteProperty(HTMLDialogElement.prototype, name);
    });
  }
});

vitest.test(
  "compact thumbnail opens a full-image dialog outside its parent layout",
  async () => {
    const view = testing.render(
      <ImageThumbnail src="data:image/png;base64,cG5n" name="shot.png" />,
    );
    const user = userEvent.setup();
    const opener = view.getByRole("button", { name: "Preview shot.png" });
    vitest.expect(opener.classList.contains("image-thumbnail")).toBe(true);
    await user.click(opener);
    const dialog = view.getByRole("dialog", {
      name: "Image preview: shot.png",
    });
    vitest.expect(dialog.parentElement).toBe(document.body);
    vitest
      .expect(
        testing.within(dialog).getByAltText("shot.png").getAttribute("src"),
      )
      .toBe("data:image/png;base64,cG5n");
    await user.click(testing.within(dialog).getByAltText("shot.png"));
    vitest.expect(view.getByRole("dialog")).toBe(dialog);
    await user.click(view.getByRole("button", { name: "Close image preview" }));
    vitest.expect(view.queryByRole("dialog")).toBeNull();
  },
);

vitest.test.each(["cancel", "backdrop"])(
  "image preview closes on %s",
  async (action) => {
    const view = testing.render(
      <ImageThumbnail src="data:image/png;base64,cG5n" name="shot.png" />,
    );
    await userEvent
      .setup()
      .click(view.getByRole("button", { name: "Preview shot.png" }));
    const dialog = view.getByRole("dialog");
    if (action === "cancel")
      testing.fireEvent(
        dialog,
        new Event("cancel", { bubbles: false, cancelable: true }),
      );
    else testing.fireEvent.click(dialog);
    vitest.expect(view.queryByRole("dialog")).toBeNull();
  },
);

function PreviewWorkspace({
  change,
  send,
  removeImage,
}: {
  change: (value: string) => void;
  send: () => Promise<void>;
  removeImage: (id: string) => void;
}) {
  const ref = useFocusRegion<HTMLElement>("workspace");
  return (
    <main ref={ref}>
      <Composer
        selected
        ready
        enabled
        busy={false}
        pendingCount={0}
        value=""
        change={change}
        send={send}
        review={vitest.vi.fn()}
        images={[image]}
        removeImage={removeImage}
      />
    </main>
  );
}

vitest.test(
  "preview blocks passive typing, restores opener focus, and never submits the draft",
  async () => {
    const change = vitest.vi.fn();
    const send = vitest.vi.fn(async () => {});
    const removeImage = vitest.vi.fn();
    const view = testing.render(
      <FocusProvider scope="session">
        <PreviewWorkspace
          change={change}
          send={send}
          removeImage={removeImage}
        />
      </FocusProvider>,
    );
    const user = userEvent.setup();
    const opener = view.getByRole("button", { name: "Preview shot.png" });
    await user.click(opener);
    await user.keyboard("ignored");
    vitest.expect(change).not.toHaveBeenCalled();
    vitest.expect(send).not.toHaveBeenCalled();
    vitest
      .expect(view.getByLabelText("Message draft"))
      .toHaveProperty("value", "");
    await user.click(view.getByRole("button", { name: "Close image preview" }));
    vitest.expect(document.activeElement).toBe(opener);
    await user.keyboard("a");
    vitest.expect(change).toHaveBeenCalledWith("a");
    await user.click(view.getByRole("button", { name: "Remove shot.png" }));
    vitest.expect(removeImage).toHaveBeenCalledWith("image");
    vitest.expect(view.queryByRole("dialog")).toBeNull();
  },
);
