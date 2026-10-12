import * as testing from "@testing-library/react";
import * as vitest from "vitest";
import { useImageAttachments } from "../src/hooks/useImageAttachments";

function file(name: string) {
  return new File(["png"], name, { type: "image/png" });
}

vitest.test(
  "image preparation stays with its original session after navigation",
  async () => {
    const readers: FileReader[] = [];
    vitest.vi
      .spyOn(FileReader.prototype, "readAsDataURL")
      .mockImplementation(function (this: FileReader) {
        readers.push(this);
      });
    const view = testing.renderHook(({ scope }) => useImageAttachments(scope), {
      initialProps: { scope: "a" },
    });
    testing.act(() => view.result.current.add([file("a.png")]));
    view.rerender({ scope: "b" });
    testing.act(() => view.result.current.add([file("b.png")]));
    await testing.act(async () => {
      Object.defineProperty(readers[0], "result", {
        value: "data:image/png;base64,cG5n",
      });
      readers[0].dispatchEvent(new ProgressEvent("load"));
      readers[0].dispatchEvent(new ProgressEvent("loadend"));
    });
    vitest
      .expect(view.result.current.images)
      .toMatchObject([{ name: "b.png", state: "preparing" }]);
    view.rerender({ scope: "a" });
    vitest
      .expect(view.result.current.images)
      .toMatchObject([{ name: "a.png", state: "ready" }]);
  },
);

vitest.test(
  "removing a preparing image cancels it without resurrecting the attachment",
  async () => {
    const read = vitest.vi
      .spyOn(FileReader.prototype, "readAsDataURL")
      .mockImplementation(() => {});
    const abort = vitest.vi.spyOn(FileReader.prototype, "abort");
    const view = testing.renderHook(() => useImageAttachments("a"));
    testing.act(() => view.result.current.add([file("a.png")]));
    await testing.act(async () =>
      view.result.current.remove([view.result.current.images[0].id]),
    );
    vitest.expect(read).toHaveBeenCalledOnce();
    vitest.expect(abort).toHaveBeenCalledOnce();
    vitest.expect(view.result.current.images).toEqual([]);
  },
);

vitest.test("unmount cancels outstanding image reads", () => {
  vitest.vi
    .spyOn(FileReader.prototype, "readAsDataURL")
    .mockImplementation(() => {});
  const abort = vitest.vi.spyOn(FileReader.prototype, "abort");
  const view = testing.renderHook(() => useImageAttachments("a"));
  testing.act(() => view.result.current.add([file("a.png")]));
  view.unmount();
  vitest.expect(abort).toHaveBeenCalledOnce();
});

vitest.test(
  "selection batches cannot exceed four images or silently replace previous selections",
  async () => {
    const view = testing.renderHook(() => useImageAttachments("a"));
    testing.act(() => {
      view.result.current.add([file("a.png"), file("b.png")]);
      view.result.current.add([file("c.png"), file("d.png")]);
      view.result.current.add([file("e.png")]);
    });
    await testing.waitFor(() =>
      vitest
        .expect(
          view.result.current.images.every((image) => image.state === "ready"),
        )
        .toBe(true),
    );
    vitest.expect(view.result.current.images).toHaveLength(4);
    vitest.expect(view.result.current.error).toMatch(/at most four/);
  },
);
