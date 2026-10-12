import { useLayoutEffect, useRef, useState } from "react";

type Position = { top: number; following: boolean };

export function useConversationScroll(scope: string, revision: unknown) {
  const viewport = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const positions = useRef(new Map<string, Position>());
  const previousRevision = useRef(revision);
  const synchronize = useRef((_changed: boolean) => {});
  const resume = useRef(() => {});
  const [newActivity, setNewActivity] = useState(false);

  useLayoutEffect(() => {
    const element = viewport.current;
    const body = content.current;
    if (!element || !body) return;
    const area = element;
    const position = positions.current.get(scope) ?? {
      top: 0,
      following: true,
    };
    positions.current.set(scope, position);
    previousRevision.current = revision;
    setNewActivity(false);
    let top = position.top;
    let restoreTop = position.following ? undefined : position.top;
    let active = true;

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

    function movedAway() {
      return area.scrollTop < top && !atLatest();
    }

    function update(changed: boolean, forceFollow = false) {
      if (!active) return;
      if (!forceFollow && movedAway()) position.following = false;
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
    }

    area.scrollTop = position.top;
    update(false);
    synchronize.current = update;
    resume.current = () => {
      restoreTop = undefined;
      position.following = true;
      update(false, true);
    };

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
      } else if (movedAway()) {
        position.following = false;
      }
      update(false);
    }

    const observer = new ResizeObserver(scrolled);
    observer.observe(area);
    observer.observe(body);
    area.addEventListener("scroll", scrolled);
    return () => {
      active = false;
      observer.disconnect();
      area.removeEventListener("scroll", scrolled);
      synchronize.current = () => {};
      resume.current = () => {};
    };
  }, [scope]);

  useLayoutEffect(() => {
    if (previousRevision.current === revision) return;
    previousRevision.current = revision;
    synchronize.current(true);
  }, [scope, revision]);

  return {
    viewport,
    content,
    newActivity,
    jumpToLatest: () => resume.current(),
  };
}
