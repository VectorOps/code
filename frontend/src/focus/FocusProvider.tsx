import * as React from "react";
import {
  FocusCoordinator,
  type ComposerTarget,
  type FocusMode,
} from "./FocusCoordinator";

const FocusContext = React.createContext<FocusCoordinator | null>(null);

export function FocusProvider({
  scope,
  children,
}: {
  scope: string;
  children: React.ReactNode;
}) {
  const [coordinator] = React.useState(() => new FocusCoordinator(scope));
  React.useLayoutEffect(
    () => coordinator.setScope(scope),
    [coordinator, scope],
  );
  React.useLayoutEffect(() => coordinator.connect(document), [coordinator]);
  return (
    <FocusContext.Provider value={coordinator}>
      {children}
    </FocusContext.Provider>
  );
}

export function useFocusCoordinator() {
  return React.useContext(FocusContext);
}

export function useFocusRegion<T extends HTMLElement>(mode?: FocusMode) {
  const coordinator = useFocusCoordinator();
  return React.useCallback(
    (element: T | null) => {
      if (element && mode) return coordinator?.registerRegion(element, mode);
    },
    [coordinator, mode],
  );
}

export function FocusRegion({
  mode,
  ...props
}: React.ComponentPropsWithoutRef<"section"> & { mode?: FocusMode }) {
  const ref = useFocusRegion<HTMLElement>(mode);
  return <section {...props} ref={ref} />;
}

export function useFocusBlocker(active: boolean, restore = false) {
  const coordinator = useFocusCoordinator();
  React.useLayoutEffect(() => {
    if (active) return coordinator?.block(restore);
  }, [coordinator, active, restore]);
}

export function useComposerTarget(
  element: React.RefObject<HTMLTextAreaElement | null>,
  controls: Omit<ComposerTarget, "element">,
) {
  const coordinator = useFocusCoordinator();
  const latest = React.useRef(controls);
  React.useLayoutEffect(() => {
    latest.current = controls;
  });
  React.useLayoutEffect(() => {
    if (!element.current) return;
    return coordinator?.registerComposer({
      element: element.current,
      isEditable: () => latest.current.isEditable(),
      focus: () => latest.current.focus(),
      beginTyping: (text) => latest.current.beginTyping(text),
    });
  }, [coordinator, element]);
}
