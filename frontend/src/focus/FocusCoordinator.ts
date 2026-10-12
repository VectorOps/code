export type FocusMode = "workspace" | "protected";

export type ComposerTarget = {
  element: HTMLTextAreaElement;
  isEditable: () => boolean;
  focus: () => void;
  beginTyping: (text: string) => void;
};

const typingOwnerSelector = [
  "input",
  "textarea",
  "select",
  "iframe",
  "object",
  "embed",
  "audio[controls]",
  "video[controls]",
  '[contenteditable]:not([contenteditable="false"])',
  "[inert]",
  ...[
    "textbox",
    "searchbox",
    "combobox",
    "listbox",
    "option",
    "checkbox",
    "radio",
    "switch",
    "slider",
    "spinbutton",
    "tree",
    "treeitem",
    "grid",
    "menu",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "dialog",
    "application",
  ].map((role) => `[role~="${role}"]`),
].join(",");

const activationSelector =
  'button, a[href], summary, [role~="button"], [role~="link"], [role~="tab"]';

export class FocusCoordinator {
  private regions = new Map<Element, { mode: FocusMode; token: symbol }>();
  private blockers = new Set<symbol>();
  private composer: ComposerTarget | null = null;
  private revision = 0;
  private connected = false;
  private active = true;
  private composing = false;

  constructor(private scope: string) {}

  setScope(scope: string) {
    if (scope === this.scope) return;
    this.scope = scope;
    this.revision++;
  }

  registerRegion(element: Element, mode: FocusMode) {
    const token = Symbol();
    this.regions.set(element, { mode, token });
    this.revision++;
    return () => {
      if (this.regions.get(element)?.token !== token) return;
      this.regions.delete(element);
      this.revision++;
    };
  }

  registerComposer(target: ComposerTarget) {
    this.composer = target;
    this.revision++;
    return () => {
      if (this.composer !== target) return;
      this.composer = null;
      this.revision++;
    };
  }

  block(restore = false) {
    const token = Symbol();
    const scope = this.scope;
    const document = this.composer?.element.ownerDocument;
    const opener = document?.activeElement;
    this.blockers.add(token);
    this.revision++;
    return () => {
      if (!this.blockers.delete(token)) return;
      this.revision++;
      if (!restore || scope !== this.scope || !this.available()) return;
      if (
        opener instanceof HTMLElement &&
        opener !== opener.ownerDocument.body &&
        opener.isConnected &&
        !opener.closest('[inert], [hidden], [aria-hidden="true"]') &&
        !opener.matches(":disabled")
      ) {
        opener.focus({ preventScroll: true });
      }
    };
  }

  captureComposerRecovery(origin: HTMLElement) {
    const document = origin.ownerDocument;
    const revision = this.revision;
    const scope = this.scope;
    const eligible =
      this.available() && origin.contains(document.activeElement);
    return () => {
      if (
        !eligible ||
        revision !== this.revision ||
        scope !== this.scope ||
        !this.available()
      )
        return;
      if (!this.composer?.isEditable() || this.hasSelection(document)) return;
      this.composer.focus();
    };
  }

  connect(document: Document) {
    this.connected = true;
    this.active = document.visibilityState !== "hidden";
    const changed = () => {
      this.revision++;
    };
    const keydown = (event: KeyboardEvent) =>
      this.handleTyping(event, document);
    const compositionStart = () => {
      this.composing = true;
      changed();
    };
    const compositionEnd = () => {
      this.composing = false;
    };
    const deactivate = () => {
      this.active = false;
      this.composing = false;
      changed();
    };
    const activate = () => {
      this.active = true;
    };
    const visibility = () => {
      if (document.visibilityState === "hidden") deactivate();
      else this.active = document.hasFocus();
    };
    document.addEventListener("pointerdown", changed, true);
    document.addEventListener("focusin", changed, true);
    document.addEventListener("keydown", changed, true);
    document.addEventListener("keydown", keydown);
    document.addEventListener("compositionstart", compositionStart, true);
    document.addEventListener("compositionend", compositionEnd, true);
    document.addEventListener("visibilitychange", visibility);
    document.defaultView?.addEventListener("blur", deactivate);
    document.defaultView?.addEventListener("focus", activate);
    return () => {
      document.removeEventListener("pointerdown", changed, true);
      document.removeEventListener("focusin", changed, true);
      document.removeEventListener("keydown", changed, true);
      document.removeEventListener("keydown", keydown);
      document.removeEventListener("compositionstart", compositionStart, true);
      document.removeEventListener("compositionend", compositionEnd, true);
      document.removeEventListener("visibilitychange", visibility);
      document.defaultView?.removeEventListener("blur", deactivate);
      document.defaultView?.removeEventListener("focus", activate);
      this.connected = false;
      this.composing = false;
      changed();
    };
  }

  private available() {
    return (
      this.connected &&
      this.active &&
      this.blockers.size === 0 &&
      this.composer?.element.ownerDocument.visibilityState !== "hidden" &&
      Boolean(this.composer?.element.isConnected)
    );
  }

  private hasSelection(document: Document) {
    const selection = document.getSelection();
    return selection !== null && !selection.isCollapsed;
  }

  private ancestors(element: Element | null): EventTarget[] {
    const path: EventTarget[] = [];
    while (element) {
      path.push(element);
      const root = element.getRootNode();
      element =
        element.parentElement ??
        (root instanceof ShadowRoot ? root.host : null);
    }
    return path;
  }

  private ownsTyping(path: EventTarget[]) {
    return path.some(
      (target) =>
        target instanceof Element &&
        (this.regions.get(target)?.mode === "protected" ||
          target.matches(typingOwnerSelector)),
    );
  }

  private withinWorkspace(path: EventTarget[], document: Document) {
    const workspace = this.ancestors(this.composer?.element ?? null).find(
      (target): target is Element =>
        target instanceof Element &&
        this.regions.get(target)?.mode === "workspace",
    );
    if (!workspace?.isConnected) return false;
    const element = path.find(
      (target): target is Element => target instanceof Element,
    );
    return (
      path.includes(workspace) ||
      !element ||
      element === document.body ||
      element === document.documentElement ||
      workspace.contains(element)
    );
  }

  private handleTyping(event: KeyboardEvent, document: Document) {
    if (
      event.defaultPrevented ||
      event.isComposing ||
      event.keyCode === 229 ||
      this.composing ||
      event.ctrlKey ||
      event.metaKey ||
      event.altKey ||
      event.getModifierState("AltGraph") ||
      Array.from(event.key).length !== 1 ||
      /[\p{Cc}\p{Cs}]/u.test(event.key) ||
      !this.available() ||
      this.hasSelection(document) ||
      !this.composer?.isEditable()
    )
      return;
    const eventPath = event.composedPath();
    const focusedPath = this.ancestors(document.activeElement);
    if (
      !this.withinWorkspace(eventPath, document) ||
      !this.withinWorkspace(focusedPath, document) ||
      this.ownsTyping(eventPath) ||
      this.ownsTyping(focusedPath) ||
      (event.key === " " &&
        [...eventPath, ...focusedPath].some(
          (target) =>
            target instanceof Element && target.matches(activationSelector),
        ))
    )
      return;
    event.preventDefault();
    this.composer.beginTyping(event.key);
  }
}
