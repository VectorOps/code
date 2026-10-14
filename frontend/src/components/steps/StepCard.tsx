import { useRef, type ReactNode } from "react";

export function StepCard({
  children,
  className,
  inspect,
  busy,
}: {
  children: ReactNode;
  className: string;
  inspect: () => void;
  busy?: boolean;
}) {
  const details = useRef<HTMLButtonElement>(null);
  return (
    <section
      className={`feed-entry step-card ${className}`}
      aria-busy={busy}
      onClick={(event) => {
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
        inspect();
      }}
    >
      <button
        ref={details}
        type="button"
        className="step-card-details"
        onClick={inspect}
      >
        View details
      </button>
      {children}
    </section>
  );
}
