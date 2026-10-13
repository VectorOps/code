import type { ReactNode } from "react";

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
  return (
    <section
      className={`feed-entry step-card ${className}`}
      role="button"
      tabIndex={0}
      aria-label="View details"
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
        inspect();
      }}
      onKeyDown={(event) => {
        if (
          event.target === event.currentTarget &&
          (event.key === "Enter" || event.key === " ")
        ) {
          event.preventDefault();
          inspect();
        }
      }}
    >
      {children}
    </section>
  );
}
