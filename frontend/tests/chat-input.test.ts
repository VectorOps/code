import * as vitest from "vitest";
import * as chat from "../src/models/chat-input";
import { inputResponse, type InputRequest } from "../src/models/runner";

const image: chat.ImageInput = {
  type: "image",
  mime_type: "image/png",
  data: "cG5n",
};
const request: InputRequest = {
  request_id: "input",
  execution_id: "session",
  request_type: "chat.message.v1",
  channel: "user.message",
  payload: {},
  response_type: "json",
};
const attachment: chat.ImageAttachment = {
  id: "image",
  name: "pixel.png",
  state: "ready",
  image,
};

vitest.test(
  "chat responses support image-only and mixed messages without attachment metadata",
  () => {
    vitest
      .expect(inputResponse(request, "", [image]))
      .toEqual({ content: [image] });
    vitest
      .expect(inputResponse(request, "Look at this", [image]))
      .toEqual({ content: [{ type: "text", text: "Look at this" }, image] });
    vitest.expect(chat.imageSource(image)).toBe("data:image/png;base64,cG5n");
    vitest
      .expect(() =>
        inputResponse({ ...request, request_type: undefined }, "Hello", [
          image,
        ]),
      )
      .toThrow(/text only/);
    vitest
      .expect(
        inputResponse(
          { ...request, request_type: undefined, response_type: "text" },
          "Hello",
        ),
      )
      .toBe("Hello");
  },
);

vitest.test(
  "message limits count text as one block and stop pending or failed images",
  () => {
    const four = Array.from({ length: 4 }, () => attachment);
    vitest.expect(chat.messageError("", four)).toBe("");
    vitest
      .expect(chat.messageError("Hello", four))
      .toMatch(/three images with text/);
    vitest
      .expect(() => chat.chatMessage("Hello", [image, image, image, image]))
      .toThrow(/three images with text/);
    vitest.expect(chat.messageError("a".repeat(65536), [])).toBe("");
    vitest.expect(chat.messageError("😀".repeat(65536), [])).toBe("");
    vitest.expect(chat.messageError("a".repeat(65537), [])).toMatch(/65,536/);
    vitest
      .expect(
        chat.messageError("", [
          { id: "pending", name: "x", state: "preparing" },
        ]),
      )
      .toMatch(/Preparing/);
    vitest
      .expect(
        chat.messageError("", [
          { id: "failed", name: "x", state: "failed", error: "Failed" },
        ]),
      )
      .toMatch(/Remove failed/);
    vitest
      .expect(chat.messageError("", [attachment], "text"))
      .toMatch(/text only/);
    vitest
      .expect(() => chat.chatMessage("", []))
      .toThrow(/Add a message or an image/);
  },
);

vitest.test.each(chat.imageTypes)(
  "image reading supports %s and strips the data URL prefix",
  async (type) => {
    const result = await chat.readImage(
      new File(["png"], "image", { type }),
      new AbortController().signal,
    );
    vitest
      .expect(result)
      .toEqual({ type: "image", mime_type: type, data: "cG5n" });
  },
);

vitest.test(
  "image limits reject unsupported, empty, and oversized files before reading",
  async () => {
    const controller = new AbortController();
    await vitest
      .expect(
        chat.readImage(
          new File(["svg"], "x.svg", { type: "image/svg+xml" }),
          controller.signal,
        ),
      )
      .rejects.toThrow(/PNG, JPEG, GIF, or WebP/);
    await vitest
      .expect(
        chat.readImage(
          new File([], "x.png", { type: "image/png" }),
          controller.signal,
        ),
      )
      .rejects.toThrow(/nonempty/);
    await vitest
      .expect(
        chat.readImage(
          new File([new Uint8Array(chat.maxImageBytes + 1)], "x.png", {
            type: "image/png",
          }),
          controller.signal,
        ),
      )
      .rejects.toThrow(/768 KiB/);
    const boundary = await chat.readImage(
      new File([new Uint8Array(chat.maxImageBytes)], "x.png", {
        type: "image/png",
      }),
      controller.signal,
    );
    vitest.expect(boundary.data.length).toBe(1048576);
  },
);

vitest.test(
  "cancelled and unreadable images do not produce content",
  async () => {
    const controller = new AbortController();
    controller.abort();
    await vitest
      .expect(
        chat.readImage(
          new File(["png"], "x.png", { type: "image/png" }),
          controller.signal,
        ),
      )
      .rejects.toThrow(/cancelled/);
    vitest.vi
      .spyOn(FileReader.prototype, "readAsDataURL")
      .mockImplementation(function (this: FileReader) {
        this.dispatchEvent(new ProgressEvent("error"));
        this.dispatchEvent(new ProgressEvent("loadend"));
      });
    await vitest
      .expect(
        chat.readImage(
          new File(["png"], "x.png", { type: "image/png" }),
          new AbortController().signal,
        ),
      )
      .rejects.toThrow(/Could not read/);
  },
);

vitest.test("history previews accept only bounded inline images", () => {
  vitest
    .expect(
      chat.messageImages({ content: [image, { type: "text", text: "Hello" }] }),
    )
    .toEqual([image]);
  for (const invalid of [
    { ...image, mime_type: "image/svg+xml" },
    { ...image, data: "data:image/png;base64,cG5n" },
    { ...image, data: "bad" },
    { ...image, data: "a".repeat(1048580) },
  ])
    vitest.expect(chat.messageImages({ content: [invalid] })).toEqual([]);
});
