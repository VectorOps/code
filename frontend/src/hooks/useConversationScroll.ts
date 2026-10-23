import * as React from "react";

type Position = { top: number; following: boolean };
type HistoryScroll = {
  following?: (following: boolean) => void;
  trim?: (beforePosition: number) => void;
  protected?: boolean;
};

export function useConversationScroll(
  scope: string,
  revision: unknown,
  history: HistoryScroll = {},
) {
  const historyRef = React.useRef(history);
  historyRef.current = history;
  const viewport = React.useRef<HTMLDivElement>(null);
  const content = React.useRef<HTMLDivElement>(null);
  const boundary = React.useRef<HTMLDivElement>(null);
  const previousRevision = React.useRef(revision);
  const synchronize = React.useRef((_changed: boolean) => {});
  const resume = React.useRef(() => {});
  const pause = React.useRef(() => {});
  const [newActivity, setNewActivity] = React.useState(false);
  const pauseFollowing = React.useCallback(() => pause.current(), []);

  React.useLayoutEffect(() => {
    const element = viewport.current;
    const body = content.current;
    if (!element || !body) return;
    const area = element;
    const contents = body;
    const position: Position = {
      top: 0,
      following: true,
    };
    previousRevision.current = revision;
    setNewActivity(false);
    let top = position.top;
    let restoreTop: number | undefined;
    let active = true;
    let touchY = 0;
    let reportedFollowing: boolean | undefined;
    let anchor:
      { element: HTMLElement; key?: string; offset: number } | undefined;

    function reportFollowing() {
      const following = position.following && !historyRef.current.protected;
      if (following !== reportedFollowing) {
        reportedFollowing = following;
        historyRef.current.following?.(following);
      }
      return following;
    }

    function rememberAnchor() {
      const bounds = area.getBoundingClientRect();
      let beforePosition = Infinity;
      anchor = undefined;
      for (const element of contents.querySelectorAll<HTMLElement>(
        "[data-history-position]",
      )) {
        const rect = element.getBoundingClientRect();
        const visible = rect.bottom > bounds.top && rect.top < bounds.bottom;
        if (visible || element.contains(document.activeElement))
          beforePosition = Math.min(
            beforePosition,
            Number(element.dataset.historyPosition),
          );
        if (visible)
          anchor = {
            element,
            key: element.dataset.historyKey,
            offset: rect.top - bounds.top,
          };
      }
      if (boundary.current) {
        const rect = boundary.current.getBoundingClientRect();
        if (rect.bottom > bounds.top && rect.top < bounds.bottom)
          anchor = {
            element: boundary.current,
            offset: rect.top - bounds.top,
          };
      }
      if (reportFollowing()) historyRef.current.trim?.(beforePosition);
    }

    function restoreIfClamped() {
      const maximum = Math.max(0, area.scrollHeight - area.clientHeight);
      if (
        !position.following &&
        position.top > maximum &&
        area.scrollTop >= maximum
      ) {
        restoreTop = position.top;
        return true;
      }
      return false;
    }

    function atLatest() {
      return area.scrollHeight - area.clientHeight - area.scrollTop <= 2;
    }

    function update(changed: boolean) {
      if (!active) return;
      if (changed && !position.following && anchor && area.scrollTop === top) {
        const target = anchor.element.isConnected
          ? anchor.element
          : [
              ...contents.querySelectorAll<HTMLElement>("[data-history-key]"),
            ].find((element) => element.dataset.historyKey === anchor?.key);
        if (target) {
          const offset =
            target.getBoundingClientRect().top -
            area.getBoundingClientRect().top;
          area.scrollTop += offset - anchor.offset;
          position.top = area.scrollTop;
        }
      }
      restoreIfClamped();
      const restoring = restoreTop !== undefined;
      if (restoreTop !== undefined) {
        area.scrollTop = restoreTop;
        if (area.scrollHeight - area.clientHeight >= restoreTop)
          restoreTop = undefined;
      }
      if (position.following) {
        area.scrollTop = Math.max(0, area.scrollHeight - area.clientHeight);
        setNewActivity(false);
      } else if (changed && !restoring) {
        setNewActivity(true);
      }
      top = area.scrollTop;
      if (restoreTop === undefined) position.top = top;
      rememberAnchor();
    }

    area.scrollTop = position.top;
    update(false);
    synchronize.current = update;
    resume.current = () => {
      restoreTop = undefined;
      position.following = true;
      update(false);
    };
    function pauseFollowing() {
      if (!active) return;
      restoreTop = undefined;
      position.following = false;
      top = area.scrollTop;
      position.top = top;
      rememberAnchor();
    }
    pause.current = pauseFollowing;

    function wheeled(event: WheelEvent) {
      if (event.deltaY < 0) pauseFollowing();
    }

    function touched(event: TouchEvent) {
      touchY = event.touches[0]?.clientY ?? 0;
    }

    function dragged(event: TouchEvent) {
      const next = event.touches[0]?.clientY ?? touchY;
      if (next > touchY) pauseFollowing();
      touchY = next;
    }

    function keyed(event: KeyboardEvent) {
      if (
        event.defaultPrevented ||
        (event.target instanceof HTMLElement &&
          event.target.closest("input, textarea, select, [contenteditable]"))
      )
        return;
      if (
        event.key === "ArrowUp" ||
        event.key === "PageUp" ||
        event.key === "Home" ||
        (event.key === " " && event.shiftKey)
      )
        pauseFollowing();
    }

    function pointed(event: PointerEvent) {
      if (
        event.button === 0 &&
        event.target === area &&
        area.offsetWidth > area.clientWidth &&
        event.clientX >= area.getBoundingClientRect().left + area.clientWidth
      )
        pauseFollowing();
    }

    function scrolled() {
      if (!active) return;
      if (restoreIfClamped()) {
        update(false);
        return;
      }
      if (restoreTop !== undefined) {
        if (area.scrollTop === top) {
          update(false);
          return;
        }
        restoreTop = undefined;
      }
      if (atLatest()) {
        position.following = true;
        setNewActivity(false);
      }
      update(false);
    }

    const observer = new ResizeObserver(() => update(false));
    observer.observe(area);
    observer.observe(body);
    area.addEventListener("scroll", scrolled);
    area.addEventListener("wheel", wheeled, { passive: true });
    area.addEventListener("touchstart", touched, { passive: true });
    area.addEventListener("touchmove", dragged, { passive: true });
    area.addEventListener("keydown", keyed);
    area.addEventListener("pointerdown", pointed);
    return () => {
      active = false;
      observer.disconnect();
      area.removeEventListener("scroll", scrolled);
      area.removeEventListener("wheel", wheeled);
      area.removeEventListener("touchstart", touched);
      area.removeEventListener("touchmove", dragged);
      area.removeEventListener("keydown", keyed);
      area.removeEventListener("pointerdown", pointed);
      synchronize.current = () => {};
      resume.current = () => {};
      pause.current = () => {};
    };
  }, [scope]);

  React.useLayoutEffect(() => {
    if (previousRevision.current === revision) return;
    previousRevision.current = revision;
    synchronize.current(true);
  }, [scope, revision]);

  React.useLayoutEffect(() => {
    synchronize.current(false);
  }, [history.protected]);

  return {
    viewport,
    content,
    boundary,
    newActivity,
    jumpToLatest: () => resume.current(),
    pauseFollowing,
  };
}
