import { fireEvent } from "@testing-library/dom";
import * as vitest from "vitest";
import { FocusCoordinator } from "../src/focus/FocusCoordinator";

const cleanups: Array<() => void> = [];

vitest.afterEach(() => {
  document.getSelection()?.removeAllRanges();
  for (const cleanup of cleanups.splice(0).reverse()) cleanup();
});

function fixture() {
  const root = document.createElement("section");
  const text = document.createElement("p");
  text.textContent = "Conversation text";
  const composer = document.createElement("textarea");
  root.append(text, composer);
  document.body.append(root);
  const coordinator = new FocusCoordinator("session-a");
  const focus = vitest.vi.fn(() => composer.focus({ preventScroll: true }));
  const beginTyping = vitest.vi.fn((value: string) => {
    composer.value += value;
    focus();
  });
  cleanups.push(() => root.remove());
  cleanups.push(coordinator.registerRegion(root, "workspace"));
  cleanups.push(
    coordinator.registerComposer({
      element: composer,
      isEditable: () => !composer.disabled && !composer.readOnly,
      focus,
      beginTyping,
    }),
  );
  const disconnect = coordinator.connect(document);
  cleanups.push(disconnect);
  return { root, text, composer, coordinator, focus, beginTyping, disconnect };
}

vitest.test(
  "typing from body enters the composer without a preceding click",
  () => {
    const { beginTyping, composer } = fixture();
    vitest
      .expect(fireEvent.keyDown(document.body, { key: "A", shiftKey: true }))
      .toBe(false);
    vitest.expect(beginTyping).toHaveBeenCalledExactlyOnceWith("A");
    vitest.expect(document.activeElement).toBe(composer);
  },
);

vitest.test.each([
  "<input />",
  "<textarea></textarea>",
  "<select><option>Choice</option></select>",
  '<div contenteditable="true"><span>Editor</span></div>',
  '<div role="menu"><span>Menu content</span></div>',
  '<div role="slider">Custom control</div>',
])("native and accessible control boundaries protect %s", (markup) => {
  const { root, beginTyping } = fixture();
  const container = document.createElement("div");
  container.innerHTML = markup;
  root.append(container);
  const target =
    container.querySelector("span, summary") ?? container.firstElementChild!;
  if (!(target instanceof HTMLElement)) throw new Error("Expected a control");
  target.tabIndex = -1;
  target.focus();
  fireEvent.keyDown(target, { key: "a" });
  vitest.expect(beginTyping).not.toHaveBeenCalled();
});

vitest.test.each([
  "<button>Action</button>",
  "<button disabled>Disabled action</button>",
  '<a href="#">Link</a>',
  "<details><summary>Disclosure</summary></details>",
  '<section tabindex="-1">Navigation target</section>',
  '<div role="button" tabindex="0">Custom action</div>',
])("navigation controls allow ordinary typing from %s", (markup) => {
  const { root, beginTyping, composer } = fixture();
  const container = document.createElement("div");
  container.innerHTML = markup;
  root.append(container);
  const control =
    container.querySelector<HTMLElement>("summary") ??
    container.firstElementChild;
  if (!(control instanceof HTMLElement)) throw new Error("Expected a control");
  control.focus();
  vitest
    .expect(fireEvent.keyDown(document.activeElement!, { key: "a" }))
    .toBe(false);
  vitest.expect(beginTyping).toHaveBeenCalledExactlyOnceWith("a");
  vitest.expect(document.activeElement).toBe(composer);
});

vitest.test.each(["button", "a", "summary"])(
  "Space retains native activation for %s",
  (tag) => {
    const { root, beginTyping } = fixture();
    const control = document.createElement(tag);
    control.tabIndex = 0;
    if (control instanceof HTMLAnchorElement) control.href = "#";
    root.append(control);
    control.focus();
    vitest.expect(fireEvent.keyDown(control, { key: " " })).toBe(true);
    vitest.expect(beginTyping).not.toHaveBeenCalled();
    vitest.expect(document.activeElement).toBe(control);
  },
);

vitest.test("protected custom regions own typing while focused", () => {
  const { root, coordinator, beginTyping } = fixture();
  const widget = document.createElement("div");
  const text = document.createElement("span");
  widget.append(text);
  root.append(widget);
  cleanups.push(coordinator.registerRegion(widget, "protected"));
  widget.tabIndex = 0;
  widget.focus();
  fireEvent.keyDown(widget, { key: "a" });
  vitest.expect(beginTyping).not.toHaveBeenCalled();
});

vitest.test("typing outside the registered workspace is not redirected", () => {
  const { beginTyping } = fixture();
  const control = document.createElement("button");
  document.body.append(control);
  cleanups.push(() => control.remove());
  control.focus();
  fireEvent.keyDown(control, { key: "a" });
  vitest.expect(beginTyping).not.toHaveBeenCalled();
});

vitest.test("the focused composer keeps native input handling", () => {
  const { composer, beginTyping } = fixture();
  composer.focus();
  vitest.expect(fireEvent.keyDown(composer, { key: "a" })).toBe(true);
  vitest.expect(beginTyping).not.toHaveBeenCalled();
});

vitest.test(
  "shadow DOM input owners are protected through the composed event path",
  () => {
    const { root, beginTyping } = fixture();
    const host = document.createElement("div");
    const shadow = host.attachShadow({ mode: "open" });
    const input = document.createElement("input");
    shadow.append(input);
    root.append(host);
    input.focus();
    fireEvent.keyDown(input, { key: "a", composed: true });
    vitest.expect(beginTyping).not.toHaveBeenCalled();
  },
);

vitest.test(
  "removing another input does not move focus until a new key requests entry",
  () => {
    const { root, beginTyping, composer } = fixture();
    const input = document.createElement("input");
    root.append(input);
    input.focus();
    input.remove();
    vitest.expect(beginTyping).not.toHaveBeenCalled();
    vitest.expect(document.activeElement).not.toBe(composer);
    fireEvent.keyDown(document.body, { key: "a" });
    vitest.expect(beginTyping).toHaveBeenCalledExactlyOnceWith("a");
  },
);

vitest.test(
  "a focused control vetoes recovery even after passive pointer interaction",
  () => {
    const { root, text, beginTyping } = fixture();
    const input = document.createElement("input");
    root.append(input);
    input.focus();
    fireEvent.pointerDown(text);
    fireEvent.keyDown(input, { key: "a" });
    vitest.expect(beginTyping).not.toHaveBeenCalled();
  },
);

vitest.test.each([
  { key: "a", ctrlKey: true },
  { key: "a", metaKey: true },
  { key: "a", altKey: true },
  { key: "a", isComposing: true },
  { key: "a", keyCode: 229 },
  { key: "Enter" },
  { key: "Tab" },
  { key: "Escape" },
  { key: "Backspace" },
  { key: "ArrowDown" },
  { key: "Dead" },
  { key: "Process" },
  { key: "\u0000" },
])("does not redirect keyboard event %j", (event) => {
  const { text, beginTyping } = fixture();
  fireEvent.pointerDown(text);
  fireEvent.keyDown(document.body, event);
  vitest.expect(beginTyping).not.toHaveBeenCalled();
});

vitest.test("composition lifecycle blocks otherwise ordinary keys", () => {
  const { text, beginTyping } = fixture();
  fireEvent.pointerDown(text);
  fireEvent.compositionStart(document.body);
  fireEvent.keyDown(document.body, { key: "a" });
  vitest.expect(beginTyping).not.toHaveBeenCalled();
  fireEvent.compositionEnd(document.body);
  fireEvent.keyDown(document.body, { key: "a" });
  vitest.expect(beginTyping).toHaveBeenCalledOnce();
});

vitest.test(
  "AltGraph does not redirect even without ordinary modifier flags",
  () => {
    const { text, beginTyping } = fixture();
    fireEvent.pointerDown(text);
    const event = new KeyboardEvent("keydown", {
      key: "@",
      bubbles: true,
      cancelable: true,
    });
    vitest.vi
      .spyOn(event, "getModifierState")
      .mockImplementation((key) => key === "AltGraph");
    document.body.dispatchEvent(event);
    vitest.expect(beginTyping).not.toHaveBeenCalled();
  },
);

vitest.test.each(["disabled", "readOnly"] as const)(
  "a %s composer cannot receive passive typing",
  (property) => {
    const { text, composer, beginTyping } = fixture();
    composer[property] = true;
    fireEvent.pointerDown(text);
    fireEvent.keyDown(document.body, { key: "a" });
    vitest.expect(beginTyping).not.toHaveBeenCalled();
  },
);

vitest.test(
  "disconnect removes listeners and prevents restoration during unmount",
  () => {
    const { text, composer, coordinator, beginTyping, focus, disconnect } =
      fixture();
    composer.focus();
    const release = coordinator.block(true);
    composer.blur();
    disconnect();
    release();
    fireEvent.pointerDown(text);
    fireEvent.keyDown(document.body, { key: "a" });
    vitest.expect(focus).not.toHaveBeenCalled();
    vitest.expect(beginTyping).not.toHaveBeenCalled();
  },
);

vitest.test("handled events and selected text are preserved", () => {
  const { text, beginTyping } = fixture();
  fireEvent.pointerDown(text);
  const handled = new KeyboardEvent("keydown", {
    key: "a",
    bubbles: true,
    cancelable: true,
  });
  handled.preventDefault();
  document.body.dispatchEvent(handled);
  const range = document.createRange();
  range.selectNodeContents(text);
  document.getSelection()?.addRange(range);
  fireEvent.keyDown(document.body, { key: "a" });
  vitest.expect(beginTyping).not.toHaveBeenCalled();
  vitest.expect(document.getSelection()?.toString()).toBe("Conversation text");
});

vitest.test(
  "nested blockers remain effective until every token is released",
  () => {
    const { text, coordinator, beginTyping } = fixture();
    const outer = coordinator.block();
    const inner = coordinator.block();
    fireEvent.pointerDown(text);
    outer();
    fireEvent.pointerDown(text);
    fireEvent.keyDown(document.body, { key: "a" });
    vitest.expect(beginTyping).not.toHaveBeenCalled();
    inner();
    fireEvent.keyDown(document.body, { key: "a" });
    vitest.expect(beginTyping).toHaveBeenCalledOnce();
  },
);

vitest.test(
  "window reactivation and session changes allow fresh typing without a click",
  () => {
    const { composer, coordinator, beginTyping } = fixture();
    fireEvent(window, new Event("blur"));
    fireEvent.keyDown(document.body, { key: "a" });
    vitest.expect(beginTyping).not.toHaveBeenCalled();
    fireEvent(window, new Event("focus"));
    fireEvent.keyDown(document.body, { key: "a" });
    vitest.expect(beginTyping).toHaveBeenCalledExactlyOnceWith("a");
    composer.blur();
    coordinator.setScope("session-b");
    fireEvent.keyDown(document.body, { key: "b" });
    vitest.expect(beginTyping).toHaveBeenCalledTimes(2);
    vitest.expect(beginTyping).toHaveBeenLastCalledWith("b");
  },
);

vitest.test("replaced registrations cannot be removed by stale cleanup", () => {
  const { text, coordinator, beginTyping } = fixture();
  const old = coordinator.registerRegion(text, "protected");
  cleanups.push(coordinator.registerRegion(text, "protected"));
  old();
  text.tabIndex = 0;
  text.focus();
  fireEvent.keyDown(text, { key: "a" });
  vitest.expect(beginTyping).not.toHaveBeenCalled();
});

vitest.test(
  "composer recovery is invalidated by focus, blockers, and scope changes",
  () => {
    const { root, composer, coordinator, focus } = fixture();
    const input = document.createElement("input");
    root.append(input);
    composer.focus();
    const recover = coordinator.captureComposerRecovery(root);
    recover();
    vitest.expect(focus).toHaveBeenCalledOnce();
    focus.mockClear();
    const moved = coordinator.captureComposerRecovery(root);
    input.focus();
    moved();
    vitest.expect(focus).not.toHaveBeenCalled();
    composer.focus();
    const blocked = coordinator.captureComposerRecovery(root);
    const release = coordinator.block();
    release();
    blocked();
    const changed = coordinator.captureComposerRecovery(root);
    coordinator.setScope("session-b");
    changed();
    vitest.expect(focus).not.toHaveBeenCalled();
  },
);

vitest.test(
  "overlay restoration preserves its opener and ignores stale sessions",
  () => {
    const { root, coordinator, composer } = fixture();
    const button = document.createElement("button");
    root.append(button);
    button.focus();
    const release = coordinator.block(true);
    composer.focus();
    release();
    vitest.expect(document.activeElement).toBe(button);
    const stale = coordinator.block(true);
    composer.focus();
    coordinator.setScope("session-b");
    stale();
    vitest.expect(document.activeElement).toBe(composer);
  },
);
