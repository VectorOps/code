import { useRef, type ReactNode } from "react";
import { useDebug } from "../Diagnostics";

export function StepCard({
  children,
  className,
  inspect,
  busy,
  diagnostic = true,
}: {
  children: ReactNode;
  className: string;
  inspect?: () => void;
  busy?: boolean;
  diagnostic?: boolean;
}) {
  const debug = useDebug();
  const open = !diagnostic || debug ? inspect : undefined;
  const details = useRef<HTMLButtonElement>(null);
  return (
    <section
      className={`feed-entry step-card ${className}`}
      aria-busy={busy}
      onClick={(event) => {
        if (!open) return;
        if (
          event.defaultPrevented ||
          window.getSelection()?.isCollapsed === false
        )
          return;
        if (!(event.target instanceof Element)) return;
        if (!event.currentTarget.contains(event.target)) return;
        const control = event.target.closest(
          "a, button, input, textarea, select, summary, [role='button'], [contenteditable], [tabindex]",
        );
        if (control && control !== event.currentTarget) return;
        details.current?.focus({ preventScroll: true });
        open();
      }}
    >
      {open && (
        <button
          ref={details}
          type="button"
          className="step-card-details"
          onClick={open}
        >
          View details
        </button>
      )}
      {children}
    </section>
  );
}
